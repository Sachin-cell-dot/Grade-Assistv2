from core.models import VisionExtraction

def deterministic_total(extraction: VisionExtraction) -> float | None:
    """Sum only visible individual awarded marks; None means no mark evidence exists."""
    total = 0.0
    has_visible_mark = False
    for section in extraction.sections:
        for question in section.questions:
            if question.teacher_marking and question.teacher_marking.visible_individual_score is not None:
                total += question.teacher_marking.visible_individual_score.obtained
                has_visible_mark = True
            for subquestion in question.subquestions:
                if subquestion.teacher_marking and subquestion.teacher_marking.visible_individual_score is not None:
                    total += subquestion.teacher_marking.visible_individual_score.obtained
                    has_visible_mark = True
    return total if has_visible_mark else None
