"""Append-only SQLite audit trail for evidence and teacher decisions."""
from __future__ import annotations

from enum import Enum
from hashlib import sha256
import json
import sqlite3
from dataclasses import dataclass

from core.evidence_verification import VerificationResult
from core.models import VisionExtraction
from core.partial_credit import PartialCreditSuggestion, TeacherDisposition
from core.provenance import ExtractionProvenance
from core.routing import RoutingDecision, ReviewRoute
from core.teacher_review import ReviewAction, TeacherCorrection


class LifecycleState(str, Enum):
    EXTRACTED = "EXTRACTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    TEACHER_CONFIRMED = "TEACHER_CONFIRMED"
    VERIFIED = "VERIFIED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"


_TRANSITIONS = {
    LifecycleState.EXTRACTED: {LifecycleState.REVIEW_REQUIRED, LifecycleState.BLOCKED, LifecycleState.TEACHER_CONFIRMED, LifecycleState.DEFERRED},
    LifecycleState.REVIEW_REQUIRED: {LifecycleState.TEACHER_CONFIRMED, LifecycleState.DEFERRED, LifecycleState.BLOCKED},
    LifecycleState.TEACHER_CONFIRMED: {LifecycleState.VERIFIED, LifecycleState.REVIEW_REQUIRED, LifecycleState.DEFERRED},
    LifecycleState.VERIFIED: set(),
    LifecycleState.DEFERRED: set(),
    LifecycleState.BLOCKED: set(),
}


@dataclass(frozen=True)
class AuditSummary:
    audit_id: int
    lifecycle_state: LifecycleState
    route: str | None
    original_teacher_mark: float | None
    suggestion: float | None
    teacher_action: str | None
    accepted_count: int
    edited_count: int
    kept_teacher_mark_count: int


def _json(model) -> str:
    value = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def extraction_fingerprint(extraction: VisionExtraction) -> str:
    return sha256(_json(extraction).encode("utf-8")).hexdigest()


class AuditStore:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def create_extraction(self, extraction: VisionExtraction, provenance: ExtractionProvenance | None = None) -> int:
        fingerprint = extraction_fingerprint(extraction)
        self.connection.execute(
            "INSERT OR IGNORE INTO assessment_audits(extraction_fingerprint, original_extraction_json, provenance_json, lifecycle_state) VALUES (?, ?, ?, ?)",
            (fingerprint, _json(extraction), _json(provenance) if provenance else None, LifecycleState.EXTRACTED.value),
        )
        row = self.connection.execute("SELECT id FROM assessment_audits WHERE extraction_fingerprint = ?", (fingerprint,)).fetchone()
        self.connection.commit()
        return int(row[0])

    def create_demo_extraction(self, extraction: VisionExtraction, provenance: ExtractionProvenance) -> int:
        """Create a distinct, explicitly labelled local-demo audit record.

        This deliberately does not reuse a normal extraction fingerprint: a
        REVIEW_REQUIRED source record must never be relabelled or promoted to
        VERIFIED merely because it is used in a demo seed.
        """
        source_fingerprint = extraction_fingerprint(extraction)
        fingerprint = sha256(f"local-demo|{source_fingerprint}".encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT OR IGNORE INTO assessment_audits("
            "extraction_fingerprint, original_extraction_json, provenance_json, lifecycle_state, is_demo, demo_label"
            ") VALUES (?, ?, ?, ?, 1, ?)",
            (fingerprint, _json(extraction), _json(provenance), LifecycleState.EXTRACTED.value, "LOCAL DEMO DATA"),
        )
        row = self.connection.execute("SELECT id FROM assessment_audits WHERE extraction_fingerprint = ?", (fingerprint,)).fetchone()
        self.connection.commit()
        return int(row[0])

    def original_extraction(self, audit_id: int) -> VisionExtraction:
        row = self.connection.execute("SELECT original_extraction_json FROM assessment_audits WHERE id = ?", (audit_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown audit id: {audit_id}")
        return VisionExtraction.model_validate_json(row[0])

    def append(self, audit_id: int, event_type: str, payload, rationale: str | None = None) -> None:
        self.connection.execute("INSERT INTO assessment_audit_events(assessment_audit_id, event_type, payload_json, teacher_rationale) VALUES (?, ?, ?, ?)", (audit_id, event_type, _json(payload), rationale))
        self.connection.commit()

    def record_verification(self, audit_id: int, result: VerificationResult) -> None: self.append(audit_id, "VERIFICATION", result)
    def record_routing(self, audit_id: int, decision: RoutingDecision) -> None: self.append(audit_id, "ROUTING", decision)
    def record_correction(self, audit_id: int, correction: TeacherCorrection) -> None: self.append(audit_id, "TEACHER_CORRECTION", correction, correction.teacher_note)
    def record_suggestion(self, audit_id: int, suggestion: PartialCreditSuggestion) -> None: self.append(audit_id, "SEMANTIC_SUGGESTION", suggestion, suggestion.rationale)
    def record_partial_credit_decision(self, audit_id: int, decision: TeacherDisposition) -> None: self.append(audit_id, "PARTIAL_CREDIT_DECISION", decision, decision.rationale)

    def lifecycle_state(self, audit_id: int) -> LifecycleState:
        row = self.connection.execute("SELECT lifecycle_state FROM assessment_audits WHERE id = ?", (audit_id,)).fetchone()
        if row is None: raise KeyError(f"Unknown audit id: {audit_id}")
        return LifecycleState(row[0])

    def transition(self, audit_id: int, target: LifecycleState, *, teacher_action: ReviewAction | None = None, rationale: str | None = None) -> None:
        current = self.lifecycle_state(audit_id)
        if target not in _TRANSITIONS[current]:
            raise ValueError(f"Invalid lifecycle transition: {current.value} -> {target.value}")
        if target in {LifecycleState.TEACHER_CONFIRMED, LifecycleState.VERIFIED} and teacher_action not in {ReviewAction.CONFIRM_EXTRACTED, ReviewAction.CORRECT_EVIDENCE}:
            raise ValueError("Teacher confirmation is required before this lifecycle transition.")
        if target == LifecycleState.VERIFIED and current != LifecycleState.TEACHER_CONFIRMED:
            raise ValueError("Only a teacher-confirmed record can become VERIFIED.")
        self.connection.execute("UPDATE assessment_audits SET lifecycle_state = ? WHERE id = ?", (target.value, audit_id))
        self.append(audit_id, "LIFECYCLE", {"from": current.value, "to": target.value, "teacher_action": teacher_action.value if teacher_action else None}, rationale)

    def apply_route(self, audit_id: int, decision: RoutingDecision) -> None:
        self.record_routing(audit_id, decision)
        if self.lifecycle_state(audit_id) != LifecycleState.EXTRACTED:
            return
        if decision.route == ReviewRoute.BLOCKED:
            self.transition(audit_id, LifecycleState.BLOCKED, rationale=decision.teacher_summary)
        elif decision.route == ReviewRoute.TEACHER_REVIEW:
            self.transition(audit_id, LifecycleState.REVIEW_REQUIRED, rationale=decision.teacher_summary)

    def summary(self, audit_id: int) -> AuditSummary:
        state = self.lifecycle_state(audit_id)
        rows = self.connection.execute("SELECT event_type, payload_json FROM assessment_audit_events WHERE assessment_audit_id = ? ORDER BY id", (audit_id,)).fetchall()
        route = mark = suggestion = action = None
        accepted = edited = kept = 0
        original = self.original_extraction(audit_id)
        for section in original.sections:
            for question in section.questions:
                if question.teacher_marking and question.teacher_marking.visible_individual_score:
                    mark = question.teacher_marking.visible_individual_score.obtained
                    break
            if mark is not None: break
        for event_type, payload_json in rows:
            payload = json.loads(payload_json)
            if event_type == "ROUTING": route = payload["route"]
            elif event_type == "SEMANTIC_SUGGESTION": suggestion = payload.get("suggested_mark")
            elif event_type == "PARTIAL_CREDIT_DECISION":
                action = payload["action"]
                if action == "ACCEPT_SUGGESTION": accepted += 1
                elif action == "EDIT_FINAL_MARK": edited += 1
                elif action == "PROCEED_WITH_TEACHER_MARK": kept += 1
        return AuditSummary(audit_id, state, route, mark, suggestion, action, accepted, edited, kept)
