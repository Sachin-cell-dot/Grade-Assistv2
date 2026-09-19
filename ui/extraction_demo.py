from pathlib import Path
from time import perf_counter
from typing import Literal, MutableMapping

import streamlit as st
from pydantic import BaseModel

from config.settings import get_settings
from database.audit_store import AuditStore, LifecycleState, extraction_fingerprint
from database.db import initialize_database
from core.evidence_verification import verify_extraction
from core.local_result_cache import LocalGroqResultCache
from core.models import VisionExtraction
from core.partial_credit import (
    SemanticEmbeddingConfig,
    SuggestionStatus,
    TeacherDispositionAction,
    record_teacher_disposition,
    semantic_partial_credit_suggestions,
)
from core.provenance import ExtractionProvenance, build_provenance
from core.reference_models import QuestionPaperExtraction, RubricExtraction
from core.routing import ReviewRoute, route_extraction
from core.teacher_review import ReviewAction, TeacherCorrection, correction_value, resolve_teacher_review
from core.teacher_evidence_import import local_result_label
from services.groq_vision_service import (
    GroqVisionError,
    GroqVisionService,
    groq_dto_to_vision_extraction,
    groq_question_paper_to_extraction,
    groq_rubric_to_extraction,
)
from tools.scoring import deterministic_total
from tools.vision import validate_worksheet_image
from tools.worksheet_input import WorksheetInputError, prepare_worksheet_upload

DocumentType = Literal["student_answer_sheet", "question_paper", "rubric"]
POLICY_VERSION = "groq-json-prompt-v1"
VERIFIED_LOCAL_RESULT_LABEL = "Verified local result"
DOCUMENT_LABELS: dict[str, DocumentType] = {
    "Student answer sheet": "student_answer_sheet",
    "Question paper": "question_paper",
    "Rubric": "rubric",
}
UPLOAD_LABELS = {
    "Student answer sheet": "Upload English answer sheet",
    "Question paper": "Upload English question paper",
    "Rubric": "Upload English rubric",
}
DOCUMENT_STATE_NAMESPACE = "document_evidence_state"


def document_state_for(
    session_state: MutableMapping[str, object], document_type: DocumentType
) -> dict[str, object]:
    """Return the isolated session record for one evidence-document type."""
    states = session_state.setdefault(DOCUMENT_STATE_NAMESPACE, {})
    if not isinstance(states, dict):  # Defensive recovery from an old session value.
        states = {}
        session_state[DOCUMENT_STATE_NAMESPACE] = states
    state = states.setdefault(document_type, {})
    if not isinstance(state, dict):
        state = {}
        states[document_type] = state
    return state


def store_uploaded_document(
    state: dict[str, object],
    *,
    source_path: Path,
    image_path: Path,
    source_filename: str,
    label: str,
) -> bool:
    """Replace only this document's evidence when its uploaded image changes."""
    image_reference = str(image_path)
    replaced = state.get("image_path") != image_reference
    if replaced:
        state.clear()
    state.update({
        "source_path": str(source_path),
        "image_path": image_reference,
        "source_filename": source_filename,
        "label": label,
    })
    return replaced


def matching_local_result(
    cache: LocalGroqResultCache,
    image_path: Path,
    document_type: DocumentType,
    model: str,
    source_filename: str | None = None,
):
    """Find an explicitly selectable local result for this upload, if one exists."""
    cached = cache.get(image_path, document_type, model, POLICY_VERSION)
    if cached is not None:
        return cached
    # Teacher-verified/recovered evidence is keyed by its original source image
    # name, whereas browser uploads use a content-fingerprinted local filename.
    identifiers = [source_filename, image_path.name]
    for identifier in dict.fromkeys(item for item in identifiers if item):
        cached = cache.for_source_image_identifier(document_type, identifier, POLICY_VERSION)
        if cached is not None:
            return cached
    if document_type in {"question_paper", "rubric"}:
        return cache.latest_for_document_type_any_model(document_type, POLICY_VERSION)
    return None


def extract_uploaded_document(image_path: Path, document_type: DocumentType) -> tuple[BaseModel, ExtractionProvenance]:
    """Run one explicitly selected document type through its matching Groq method only."""
    service = GroqVisionService()
    started = perf_counter()
    if document_type == "student_answer_sheet":
        extraction: BaseModel = groq_dto_to_vision_extraction(service.extract_answer_sheet(image_path))
    elif document_type == "question_paper":
        extraction = groq_question_paper_to_extraction(service.extract_question_paper(image_path))
    else:
        extraction = groq_rubric_to_extraction(service.extract_rubric(image_path))
    provenance = build_provenance(
        image_path,
        provider="groq",
        model=service.settings.groq_vision_model,
        policy_version=POLICY_VERSION,
        elapsed_seconds=perf_counter() - started,
    )
    return extraction, provenance


def extract_uploaded_answer_sheet(image_path: Path) -> tuple[VisionExtraction, ExtractionProvenance]:
    """Backward-compatible answer-sheet helper used by the Groq UI integration test."""
    extraction, provenance = extract_uploaded_document(image_path, "student_answer_sheet")
    return VisionExtraction.model_validate(extraction), provenance


def teacher_readable_groq_error(error: GroqVisionError) -> str:
    if "HTTP 429" in str(error):
        return "Groq is temporarily rate-limited. Please wait briefly, then retry."
    return f"Groq Vision extraction failed: {error}"


def answer_evidence_rows(extraction: VisionExtraction) -> list[dict[str, object]]:
    """Teacher-facing worksheet evidence without internal extraction fields."""
    return [
        {
            "Question": question.identifier or "Unlabelled",
            "Student answer": question.student_answer.visible_text or "No extracted answer",
            "Teacher mark": (question.teacher_marking.visible_individual_score.obtained
                             if question.teacher_marking and question.teacher_marking.visible_individual_score else None),
            "Max mark": (question.teacher_marking.visible_individual_score.maximum
                         if question.teacher_marking and question.teacher_marking.visible_individual_score else None),
        }
        for section in extraction.sections
        for question in section.questions
    ]


def worksheet_reported_total_display(extraction: VisionExtraction) -> float | str:
    """Return only the visible obtained score for a Streamlit numeric display."""
    score = extraction.worksheet_reported_score
    return score.obtained if score is not None else "Not visible"


def reference_evidence_rows(extraction: BaseModel | dict, document_type: DocumentType) -> list[dict[str, object]]:
    """Compact reference-document summaries for the judge-facing interface."""
    if document_type == "question_paper":
        paper = QuestionPaperExtraction.model_validate(extraction)
        return [
            {"Question": question.identifier or "Unlabelled", "Printed question": question.printed_text or "Not extracted", "Maximum": question.printed_maximum_marks}
            for section in paper.sections for question in section.questions
        ]
    rubric = RubricExtraction.model_validate(extraction)
    return [
        {"Question": question.identifier or "Unlabelled", "Rubric criteria": "; ".join(question.visible_criteria) or "Not extracted", "Maximum": question.visible_maximum_marks}
        for section in rubric.sections for question in section.questions
    ]


def document_status_rows(session_state: MutableMapping[str, object]) -> list[dict[str, str]]:
    states = session_state.get(DOCUMENT_STATE_NAMESPACE, {})
    states = states if isinstance(states, dict) else {}
    return [
        {"Document": label, "Status": "✓ loaded" if isinstance(states.get(kind), dict) and states[kind].get("result") else "Not loaded"}
        for label, kind in DOCUMENT_LABELS.items()
    ]


def best_evidence_phrase(suggestion) -> tuple[str | None, float | None]:
    matches = [
        match for match in suggestion.matched_rubric_criteria + suggestion.missing_rubric_criteria
        if match.matched_answer_phrase
    ]
    if not matches:
        return None, None
    best = max(matches, key=lambda match: match.similarity_score)
    return best.matched_answer_phrase, best.similarity_score


def _render_technical_evidence(payload: object) -> None:
    with st.expander("Technical evidence / raw extraction", expanded=False):
        st.json(payload)


def _audit_for(extraction: VisionExtraction, provenance: ExtractionProvenance | None = None) -> tuple[AuditStore, int]:
    """Persist initial factual evaluation once per displayed immutable extraction."""
    fingerprint = extraction_fingerprint(extraction)
    store = AuditStore(initialize_database())
    audit_id = store.create_extraction(extraction, provenance)
    key = f"audit-initialized:{fingerprint}"
    if not st.session_state.get(key):
        verification = verify_extraction(extraction)
        store.record_verification(audit_id, verification)
        store.apply_route(audit_id, route_extraction(extraction, verification))
        st.session_state[key] = True
    return store, audit_id


def _render_audit_summary(store: AuditStore, audit_id: int) -> None:
    summary = store.summary(audit_id)
    st.caption("Audit summary")
    st.dataframe([{
        "Extraction route": summary.route or "Not routed",
        "Original teacher mark": summary.original_teacher_mark,
        "Semantic suggestion": summary.suggestion,
        "Teacher action": summary.teacher_action,
        "Lifecycle state": summary.lifecycle_state.value,
        "Accepted": summary.accepted_count,
        "Edited": summary.edited_count,
        "Kept teacher mark": summary.kept_teacher_mark_count,
    }], hide_index=True)


def _render_teacher_review(
    extraction: VisionExtraction,
    provenance: ExtractionProvenance | None = None,
    state: dict[str, object] | None = None,
) -> None:
    """Small evidence-only review panel; corrections are append-only session records."""
    audit_store, audit_id = _audit_for(extraction, provenance)
    state = state if state is not None else st.session_state
    corrections = state.setdefault("teacher_corrections", [])
    effective = extraction
    if corrections:
        effective = resolve_teacher_review(extraction, ReviewAction.CORRECT_EVIDENCE, corrections).effective_extraction
    verification = verify_extraction(effective)
    routing = route_extraction(effective, verification)
    visible_total = effective.worksheet_reported_score.obtained if effective.worksheet_reported_score else None
    individual_total = verification.evidence_summary.visible_individual_mark_sum
    if routing.route == ReviewRoute.TEACHER_REVIEW:
        st.error("REVIEW REQUIRED")
    elif routing.route == ReviewRoute.BLOCKED:
        st.error("REVIEW BLOCKED - no usable question evidence")
    else:
        st.success("Evidence is internally consistent")
    left, right = st.columns(2)
    left.metric("Extracted worksheet total", visible_total if visible_total is not None else "Not visible")
    right.metric("Sum of visible individual marks", individual_total if individual_total is not None else "Incomplete")
    st.subheader("Evidence extracted")
    st.dataframe(answer_evidence_rows(effective), hide_index=True, use_container_width=True)
    for item in routing.review_items:
        st.warning(item.summary)
    st.caption("GradeAssist extracted visible evidence and checked factual consistency. It did not grade, infer a correct mark, or change evidence.")

    mark_paths = [
        f"sections[{section_index}].questions[{question_index}].teacher_marking.visible_individual_score.obtained"
        for section_index, section in enumerate(effective.sections)
        for question_index, question in enumerate(section.questions)
        if question.teacher_marking and question.teacher_marking.visible_individual_score is not None
    ]
    field_paths = (["worksheet_reported_score.obtained"] if effective.worksheet_reported_score else []) + mark_paths
    if field_paths:
        selected_path = st.selectbox("Visible value to correct", field_paths, key="review_field_path")
        extracted_value = correction_value(effective, selected_path)
        st.caption(f"Extracted value: {extracted_value}")
        corrected_value = st.number_input("Teacher correction", min_value=0.0, value=float(extracted_value), step=0.5, key="review_corrected_value")
        note = st.text_input("Teacher note (optional)", key="review_note")
        if st.button("Correct evidence"):
            item = next((item for item in routing.review_items if selected_path in item.field_paths), None)
            review_item_id = item.id if item else "teacher-visible-evidence"
            correction = TeacherCorrection(review_item_id=review_item_id, field_path=selected_path, original_value=extracted_value, corrected_value=corrected_value, teacher_note=note or None)
            corrections.append(correction)
            audit_store.record_correction(audit_id, correction)
            st.rerun()
    confirm, defer = st.columns(2)
    if confirm.button("Confirm extracted evidence"):
        action = ReviewAction.CONFIRM_EXTRACTED if not corrections else ReviewAction.CORRECT_EVIDENCE
        resolution = resolve_teacher_review(extraction, action, corrections)
        state["teacher_review_resolution"] = resolution.model_dump(mode="json")
        if audit_store.lifecycle_state(audit_id) in {LifecycleState.EXTRACTED, LifecycleState.REVIEW_REQUIRED}:
            audit_store.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=action)
        st.success("Teacher action recorded; original extraction remains unchanged.")
    if defer.button("Defer"):
        state["teacher_review_resolution"] = resolve_teacher_review(extraction, ReviewAction.DEFER, corrections).model_dump(mode="json")
        if audit_store.lifecycle_state(audit_id) in {LifecycleState.EXTRACTED, LifecycleState.REVIEW_REQUIRED, LifecycleState.TEACHER_CONFIRMED}:
            audit_store.transition(audit_id, LifecycleState.DEFERRED, rationale="Teacher deferred review.")
        st.info("Review deferred. No verification decision was applied.")
    if corrections:
        st.markdown("**Append-only teacher corrections**")
        st.dataframe([
            {"Field": correction.field_path, "Original": correction.original_value, "Corrected": correction.corrected_value, "Note": correction.teacher_note}
            for correction in corrections
        ], hide_index=True, use_container_width=True)
    _render_technical_evidence(effective.model_dump(mode="json"))
    _render_audit_summary(audit_store, audit_id)


def _render_partial_credit_review(
    answer_sheet: VisionExtraction, model: str, state: dict[str, object] | None = None
) -> None:
    """Show semantic suggestions only when all three independently cached sources are available."""
    st.divider()
    st.subheader("Partial-Credit Review")
    audit_store, audit_id = _audit_for(answer_sheet)
    cache = LocalGroqResultCache()
    cached_rubric = cache.latest_for_document_type("rubric", model, POLICY_VERSION)
    cached_question_paper = cache.latest_for_document_type("question_paper", model, POLICY_VERSION)
    if cached_rubric is None or cached_question_paper is None:
        st.info("Partial-credit review is unavailable until verified local question-paper and rubric results are available.")
        return
    rubric = RubricExtraction.model_validate(cached_rubric.result_payload)
    question_paper = QuestionPaperExtraction.model_validate(cached_question_paper.result_payload)
    settings = get_settings()
    suggestions = semantic_partial_credit_suggestions(
        question_paper,
        rubric,
        answer_sheet,
        SemanticEmbeddingConfig(model_name=settings.embedding_model, semantic_threshold=settings.semantic_match_threshold),
    )
    for suggestion in suggestions:
        suggestion_key = f"audit-suggestion:{audit_id}:{suggestion.model_dump_json()}"
        if not st.session_state.get(suggestion_key):
            audit_store.record_suggestion(audit_id, suggestion)
            st.session_state[suggestion_key] = True
    st.caption("This is a suggestion based on rubric-concept matching. The teacher’s mark is unchanged until the teacher decides.")
    state = state if state is not None else st.session_state
    dispositions = state.setdefault("partial_credit_dispositions", [])
    for suggestion in suggestions:
        with st.expander(
            f"Question {suggestion.question_identifier or 'unlabelled'}",
            expanded=suggestion.status != SuggestionStatus.INSUFFICIENT_EVIDENCE,
        ):
            evaluation_maximum = suggestion.evaluation_maximum_marks or suggestion.rubric_maximum_marks
            overview = st.columns(4)
            overview[0].metric("Teacher mark", suggestion.teacher_awarded_mark if suggestion.teacher_awarded_mark is not None else "Not visible")
            overview[1].metric("Maximum mark", evaluation_maximum if evaluation_maximum is not None else "Not available")
            overview[2].metric("Suggested mark", suggestion.suggested_mark if suggestion.suggested_mark is not None else "Not available")
            overview[3].metric("Status", suggestion.status.value)
            st.write(f"Raw semantic coverage: {suggestion.raw_semantic_coverage if suggestion.raw_semantic_coverage is not None else 'Not available'}")
            st.write("Matched rubric criteria:", [match.criterion.text for match in suggestion.matched_rubric_criteria] or "None")
            st.write("Missing rubric criteria:", [match.criterion.text for match in suggestion.missing_rubric_criteria] or "None")
            phrase, similarity = best_evidence_phrase(suggestion)
            st.write("Best student evidence:", phrase or "No usable answer evidence")
            st.write("Similarity score:", similarity if similarity is not None else "Not available")
            st.write("Rationale:", suggestion.rationale)
            if suggestion.status == SuggestionStatus.INSUFFICIENT_EVIDENCE:
                st.info("Partial-credit suggestion unavailable: teacher review requires usable answer and reference evidence.")
                continue
            if suggestion.status == SuggestionStatus.AGREES_WITH_TEACHER:
                st.caption("Suggestion matches the visible teacher mark. No action is required.")
                continue
            if suggestion.teacher_awarded_mark is None:
                st.warning("No visible teacher mark was extracted, so no teacher-mark comparison action is available.")
                continue
            accept, edit, proceed = st.columns(3)
            if accept.button("Accept suggestion", key=f"accept-{suggestion.question_identifier}"):
                disposition = record_teacher_disposition(suggestion, TeacherDispositionAction.ACCEPT_SUGGESTION, "Teacher accepted the suggestion.")
                dispositions.append(disposition)
                audit_store.record_partial_credit_decision(audit_id, disposition)
                st.rerun()
            edited = edit.number_input("Final mark", min_value=0.0, max_value=float(suggestion.rubric_maximum_marks), value=float(suggestion.teacher_awarded_mark), key=f"edit-mark-{suggestion.question_identifier}")
            if edit.button("Edit final mark", key=f"edit-{suggestion.question_identifier}"):
                disposition = record_teacher_disposition(suggestion, TeacherDispositionAction.EDIT_FINAL_MARK, edited_final_mark=edited)
                dispositions.append(disposition)
                audit_store.record_partial_credit_decision(audit_id, disposition)
                st.rerun()
            if proceed.button("Proceed with teacher mark", key=f"proceed-{suggestion.question_identifier}"):
                disposition = record_teacher_disposition(suggestion, TeacherDispositionAction.PROCEED_WITH_TEACHER_MARK, "Teacher retained the visible extracted mark.")
                dispositions.append(disposition)
                audit_store.record_partial_credit_decision(audit_id, disposition)
                st.rerun()
    if dispositions:
        st.caption("Append-only partial-credit teacher dispositions")
        st.dataframe([
            {"Question": disposition.question_identifier, "Action": disposition.action.value, "Original teacher mark": disposition.original_teacher_mark, "Suggested mark": disposition.system_suggested_mark, "Teacher final mark": disposition.teacher_final_mark}
            for disposition in dispositions
        ], hide_index=True, use_container_width=True)
    _render_audit_summary(audit_store, audit_id)


def _choose_image(
    upload_label: str, document_type: DocumentType, state: dict[str, object]
) -> tuple[Path | None, str | None]:
    st.caption("Supported formats: JPEG, PNG, PDF")
    upload = st.file_uploader(
        upload_label,
        type=["jpeg", "jpg", "png", "pdf"],
        key=f"worksheet-upload-{document_type}",
    )
    if not upload:
        image_reference = state.get("image_path")
        label = state.get("label")
        return (Path(image_reference), str(label)) if image_reference and label else (None, None)
    try:
        prepared = prepare_worksheet_upload(upload.name, upload.getvalue())
    except WorksheetInputError as exc:
        st.error(str(exc))
        return None, None
    store_uploaded_document(
        state,
        source_path=prepared.source_path,
        image_path=prepared.image_path,
        source_filename=upload.name,
        label=prepared.label,
    )
    return prepared.image_path, prepared.label


def render_extraction_demo() -> None:
    selected_label = st.radio("Document type", list(DOCUMENT_LABELS), horizontal=True)
    document_type = DOCUMENT_LABELS[selected_label]
    document_state = document_state_for(st.session_state, document_type)
    st.subheader(f"English {selected_label.lower()} extraction")
    st.caption("Visible evidence only. GradeAssist does not solve, grade, or alter the worksheet.")
    st.caption("Provider: Groq Vision")
    status_rows = document_status_rows(st.session_state)
    st.dataframe(status_rows, hide_index=True, use_container_width=True)
    if all(row["Status"] == "✓ loaded" for row in status_rows):
        st.success("Partial-credit review ready")
    image_path, label = _choose_image(UPLOAD_LABELS[selected_label], document_type, document_state)
    if image_path is None:
        st.info(f"Upload one English {selected_label.lower()} image or one-page PDF to begin.")
        return
    ok, message = validate_worksheet_image(image_path)
    if not ok:
        st.error(message)
        return
    st.success(message)
    st.image(str(image_path), caption=label)
    settings_model = GroqVisionService().settings.groq_vision_model
    cache = LocalGroqResultCache()
    cached = matching_local_result(
        cache,
        image_path,
        document_type,
        settings_model,
        str(document_state.get("source_filename") or "") or None,
    )
    run_label = {
        "student_answer_sheet": "Run English evidence extraction",
        "question_paper": "Run question-paper extraction",
        "rubric": "Run rubric extraction",
    }[document_type]
    use_cached = False
    run_live = False
    if cached:
        st.info(f"{local_result_label(cached.provenance)} available")
        cached_column, live_column = st.columns(2)
        use_cached = cached_column.button("Use verified local result")
        run_live = live_column.button("Run Groq again", type="primary")
    else:
        run_live = st.button(run_label, type="primary")
    if use_cached:
        document_state["result"] = cached.result_payload
        document_state["provenance"] = cached.provenance.model_dump(mode="json")
        document_state["source_label"] = local_result_label(cached.provenance)
    if run_live:
        try:
            extraction, provenance = extract_uploaded_document(image_path, document_type)
            result_payload = extraction.model_dump(mode="json")
            cache.put(image_path, document_type, provenance.model, POLICY_VERSION, result_payload, provenance)
            document_state["result"] = result_payload
            document_state["provenance"] = provenance.model_dump(mode="json")
            document_state["source_label"] = "Live Groq result"
            document_state["teacher_corrections"] = []
        except GroqVisionError as exc:
            st.error(teacher_readable_groq_error(exc))
            return
    result_json = document_state.get("result")
    if result_json:
        provenance = document_state.get("provenance")
        if provenance:
            st.success(f"{document_state.get('source_label')} | Provider: Groq Vision | Model: {provenance['model']}")
        if document_type != "student_answer_sheet":
            st.subheader("Evidence extracted")
            st.dataframe(reference_evidence_rows(result_json, document_type), hide_index=True, use_container_width=True)
            st.info("Reference-document transcription only. GradeAssist did not grade or apply this reference to a student answer.")
            _render_technical_evidence(result_json)
            return
        result = VisionExtraction.model_validate(result_json)
        left, right = st.columns(2)
        left.metric("Deterministic total (visible individual marks)", deterministic_total(result))
        right.metric("Worksheet-reported total (as extracted)", worksheet_reported_total_display(result))
        st.divider()
        st.subheader("English evidence review")
        audit_provenance = ExtractionProvenance.model_validate(provenance) if provenance else None
        _render_teacher_review(result, audit_provenance, document_state)
        _render_partial_credit_review(result, settings_model, document_state)
