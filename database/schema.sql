CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, roll_number TEXT UNIQUE, class_name TEXT, parent_guardian_name TEXT, parent_email TEXT, is_demo INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS assessments (id INTEGER PRIMARY KEY, student_id INTEGER REFERENCES students(id), audit_id INTEGER UNIQUE REFERENCES assessment_audits(id), assessment_name TEXT, subject TEXT, assessment_date TEXT, worksheet_reported_score REAL, maximum_marks REAL, verified_status TEXT, verified_at TEXT, is_demo INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS grading_drafts (id INTEGER PRIMARY KEY, assessment_id INTEGER NOT NULL REFERENCES assessments(id), extraction_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'extracted', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS extracted_question_marks (id INTEGER PRIMARY KEY, draft_id INTEGER NOT NULL REFERENCES grading_drafts(id), section_identifier TEXT, question_identifier TEXT NOT NULL, subquestion_identifier TEXT, visible_individual_score REAL, teacher_marking_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS assessment_audits (
    id INTEGER PRIMARY KEY,
    extraction_fingerprint TEXT NOT NULL UNIQUE,
    original_extraction_json TEXT NOT NULL,
    provenance_json TEXT,
    lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN ('EXTRACTED','REVIEW_REQUIRED','TEACHER_CONFIRMED','VERIFIED','DEFERRED','BLOCKED')),
    is_demo INTEGER NOT NULL DEFAULT 0,
    demo_label TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS assessment_audit_events (
    id INTEGER PRIMARY KEY,
    assessment_audit_id INTEGER NOT NULL REFERENCES assessment_audits(id),
    event_type TEXT NOT NULL CHECK (event_type IN ('VERIFICATION','ROUTING','TEACHER_CORRECTION','SEMANTIC_SUGGESTION','PARTIAL_CREDIT_DECISION','LIFECYCLE')),
    payload_json TEXT NOT NULL,
    teacher_rationale TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_assessment_audit_events_audit ON assessment_audit_events(assessment_audit_id, id);
CREATE TABLE IF NOT EXISTS assessment_email_events (
    id INTEGER PRIMARY KEY,
    assessment_audit_id INTEGER NOT NULL REFERENCES assessment_audits(id),
    outcome TEXT NOT NULL CHECK (outcome IN ('ATTEMPTED','SENT','FAILED')),
    detail TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_assessment_email_events_audit ON assessment_email_events(assessment_audit_id, id);
