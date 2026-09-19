GradeAssist
AI-powered assessment verification, partial-credit assistance and teacher action system
1. Problem
Teachers spend significant time doing more than simply assigning marks.
After an assessment is corrected, they may still need to verify totals, identify students who need attention, understand class-level performance, identify strengths and weak areas, review partial-credit cases, prepare reports, communicate results, and maintain records over time.
Most assessment tools focus on one part of this process, usually grading or data entry. GradeAssist focuses on the workflow that happens around the assessment.
The goal is to reduce repetitive work without removing teacher control.
2. What GradeAssist Does
GradeAssist currently provides:
• Handwritten answer-sheet evidence extraction
• Question-paper extraction
• Rubric extraction
• Deterministic mark verification
• Confidence-based routing
• Human-in-the-loop teacher review
• Semantic rubric matching
• Partial-credit suggestions
• Append-only audit information
• Verified assessment lifecycle
• Teacher performance dashboard
• Class-level analytics
• Student performance history
• Excel report export
• Teacher-approved email reporting
• Local demo data for end-to-end demonstrations
3. Core Design Principle
Evidence → Confidence → Action
Evidence:
GradeAssist extracts visible information from assessment documents, including student identity, answers, visible teacher marks, written totals, question IDs, maximum marks and rubric criteria.
Confidence / Verification:
Extracted information is checked using deterministic rules. For example, individual visible marks can be summed and compared with the written worksheet total. Inconsistent cases can be routed to teacher review.
Action:
Once an assessment is genuinely verified, GradeAssist can use the result for dashboard analytics, student history, class performance, reports and teacher-approved email.
4. System Architecture
Student Answer Sheet
        ↓
Groq Vision / Qwen3.8-27B
        ↓
Structured Evidence
        ↓
Answer Sheet + Question Paper + Rubric
        ↓
Evidence Verification
        ↓
Confidence / Routing
        ↓
High confidence → Continue
Needs review → Teacher Review
        ↓
BGE Semantic Matching / Partial Credit
        ↓
VERIFIED Result
        ↓
Teacher Dashboard / Excel / Teacher-approved Email
5. Tech Stack
Frontend / Application
• Streamlit — teacher-facing web application and interactive dashboard.
Backend
• Python 3.11 — orchestration, verification, semantic matching, database operations, analytics, reports and email.
Vision / Evidence Extraction
• Groq Vision
• Model: qwen/qwen3.8-27b
• Used to extract visible evidence from answer sheets, question papers and rubrics.
Data Models
• Pydantic — structured validation and typed evidence models.
Semantic Matching
• Sentence Transformers
• BAAI/bge-small-en-v1.5
• Local CPU inference
• Cosine similarity for rubric-to-answer evidence matching.
Database
• SQLite — local persistence for assessments, students, lifecycle state, roster/contact information and audit records.
Reporting
• openpyxl — Excel report generation.
Document Processing
• PyMuPDF and supporting image/document processing utilities.
Testing
• Pytest
Caching / Evidence
• SHA-256-based local caching and persisted processed evidence where appropriate.
Email
• SMTP through the existing email service, with configuration supplied through environment variables.
6. Vision / Evidence Extraction
GradeAssist uses Groq's vision-capable model, qwen/qwen3.8-27b, for extracting visible evidence from assessment documents.
Different prompts and structured models are used for student answer sheets, question papers and rubrics.
The three sources are joined using normalized question IDs such as Q1, Q2 and Q3.
The extraction model is not asked to solve questions, grade answers, calculate marks, invent missing IDs, invent student information or rewrite student answers.
The extraction layer answers: "What evidence is visibly present in this document?" rather than "What do I think the student probably meant?
7. Deterministic Evidence Verification
AI extraction is not treated as the final authority.
GradeAssist uses Python-based deterministic verification after extraction.
For example, if visible question marks are 1, 2, 1 and 2, the backend calculates 1 + 2 + 1 + 2 = 6 and compares that value with the visible written worksheet total.
If the values do not agree, the system does not silently guess which value is correct. The case can instead be routed to teacher review.
8. Confidence and Routing
GradeAssist uses routing states such as:
• AUTO_CONTINUE
• TEACHER_REVIEW
• BLOCKED
The purpose is to prevent uncertain extraction or verification from silently becoming trusted assessment data.
9. Human-in-the-Loop Review
The teacher can confirm extracted evidence, correct extracted evidence, defer a case, accept a partial-credit suggestion, edit a final mark, or proceed with the teacher's original mark.
The original extracted evidence is preserved. Corrections are recorded separately rather than silently overwriting the original evidence.
This creates a flow of:
Original evidence → Teacher correction → Effective evidence → Final decision
10. Semantic Partial Credit
GradeAssist uses BAAI/bge-small-en-v1.5 through Sentence Transformers for semantic matching.
The model runs locally on CPU and is used to compare rubric criteria with student answer segments.
The semantic model does not directly decide the student's final mark. It provides evidence coverage that is converted into a deterministic suggested mark based on criterion weights and the printed question maximum.
Typical statuses are:
• AGREES_WITH_TEACHER
• SUGGEST_REVIEW
• INSUFFICIENT_EVIDENCE
The teacher can accept, edit or reject the suggestion.
11. Verified Assessment Lifecycle
Extracted
    ↓
Verified / Review Required
    ↓
Teacher Review
    ↓
Finalized
    ↓
VERIFIED
Only finalized verified assessments are allowed into class analytics.
The dashboard excludes draft, blocked, deferred and review-required evidence.
12. Database and Audit Trail
GradeAssist uses SQLite for local persistence.
The database stores information required for assessments, students, audit information, lifecycle states, verified results and roster/contact information.
Important assessment decisions are preserved in an append-oriented audit trail. This can include the original teacher mark, system suggestion, final decision, action, rationale, timestamp and lifecycle state.
Hidden model reasoning is not stored.
13. Teacher Dashboard
The Teacher Dashboard consumes VERIFIED assessment data.
It provides:
• Student count
• Assessments processed
• Average score
• Highest and lowest scores
• Review information
• Student performance table
• Parent/guardian information when persisted
• Assessment marks and percentages
• Class score distribution
• Student performance comparison
• Question/topic mastery
• Student-by-question/topic heatmap
• Student assessment history
The dashboard is separate from the Assessment Review workspace.
14. Assessment-to-Dashboard Flow
A newly finalized assessment should not be manually entered into the dashboard.
The intended flow is:
Answer Sheet
    ↓
Evidence Extraction
    ↓
Verification
    ↓
Teacher Review when required
    ↓
Final VERIFIED result
    ↓
Teacher Dashboard consumes the verified record
    ↓
Class analytics and student history update
This keeps the dashboard downstream of the actual assessment lifecycle.
15. Excel Export
GradeAssist provides an Excel class report using openpyxl.
The workbook contains:
• Student Summary
• Question-Topic Performance
• Class Summary
The export uses the same verified data as the dashboard.
16. Email Workflow
The email workflow is teacher-controlled.
Select verified student → Generate preview → Teacher reviews → Teacher clicks Send Email → SMTP
Preview generation never sends an email.
If SMTP is not configured, the application shows a configuration message instead of pretending that an email was sent.
Parent/guardian information is read from persisted roster data. Missing information is displayed as Not available.
17. The Agentic Part
The agentic component is not simply "we used an LLM."
The important part is the system's ability to move an assessment through a sequence of evidence-based decisions and actions:
Observe
  ↓
Extract evidence
  ↓
Verify
  ↓
Assess confidence
  ↓
Route
  ↓
Request human input when necessary
  ↓
Finalize verified result
  ↓
Update downstream systems
  ↓
Prepare next action
GradeAssist automatically handles repetitive work such as evidence extraction, visible-mark verification, uncertain-case routing, semantic rubric matching, verified analytics and report preparation.
The teacher controls consequential actions such as correcting evidence, changing a final mark, approving a partial-credit suggestion and sending an email.
This is the human-in-the-loop agentic workflow.
18. Current Demo Data
The repository contains clearly labelled local demo records for testing the Teacher Dashboard.
Demo records are marked:
LOCAL DEMO DATA
Current demo students include:
• Sachin C
• Rufina Thomas
Seed:
py -3.11 -m tools.seed_demo_records --seed-demo
Clear:
py -3.11 -m tools.seed_demo_records --clear-demo
19. Running the Application
Recommended environment: Python 3.11
Create a virtual environment:
py -3.11 -m venv .venv
Windows activation:
.venv\Scripts\activate
Install:
pip install -r requirements.txt
Run:
py -3.11 -m streamlit run app.py
Streamlit normally opens at:
http://localhost:8501
20. Running Tests
Run the complete test suite:
pytest -q
The development checkpoint has passed 117+ tests. The exact count may change as development continues.
Tests cover extraction, routing, caching, semantic matching, partial credit, document persistence, dashboard aggregation, verified-only analytics, demo data, Excel generation, email preview, SMTP behavior and regression behavior.
21. Environment Configuration
Create a local .env file based on .env.example.
Typical configuration:
GROQ_API_KEY=your_key_here
SMTP_HOST=
SMTP_PORT=587
SMTP_FROM_EMAIL=
SMTP_USERNAME=
SMTP_PASSWORD=
DEMO_SACHIN_EMAIL=
DEMO_RUFINA_EMAIL=
Do not commit .env, API keys, SMTP passwords or private credentials to GitHub.
22. Project Structure
gradeassist/
├── app.py
├── config/
├── core/
├── database/
├── services/
├── ui/
├── tools/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
Important modules include:
• core/evidence_verification.py
• core/semantic_matching.py
• core/partial_credit.py
• core/routing.py
• core/teacher_review.py
• database/audit_store.py
• services/dashboard_service.py
• services/report_export.py
• services/email_service.py
• ui/extraction_demo.py
• ui/teacher_dashboard.py
23. Important Design Decisions
1. Separate extraction from grading.
2. Use deterministic verification after AI extraction.
3. Keep answer sheets, question papers and rubrics semantically separate.
4. Allow only VERIFIED results into analytics.
5. Keep humans in the loop for uncertain or consequential decisions.
6. Preserve an audit trail.
7. Keep email sending behind explicit teacher approval.
24. Current Limitations
The MVP does not currently claim to provide:
• Full school ERP integration
• Autonomous parent communication
• Automatic parent messaging without teacher approval
• Production-scale cloud deployment
• Completely autonomous grading
• Replacement of teachers
• Guaranteed interpretation of every handwriting style
A production deployment would also require stronger authentication, authorization, secure secret management, encrypted storage, access logging and institution-level privacy controls.
25. Future Scope
Possible extensions include:
• Teacher-configurable automation thresholds
• Targeted practice based on verified weaknesses
• Longitudinal student analytics
• More detailed topic mastery
• Teacher-approved parent communication
• School/ERP integration
• Wider deployment across classes and subjects
26. Recommended Hackathon Demo
1. Start with a handwritten answer sheet.
2. Upload it in Assessment Review.
3. Show the extracted evidence.
4. Show deterministic verification.
5. Show teacher review if required.
6. Show rubric-based partial-credit assistance.
7. Finalize the assessment as VERIFIED.
8. Open Teacher Dashboard.
9. Show that the verified student appears automatically.
10. Show class analytics and student history.
11. Export the Excel report.
12. Select a verified student for email.
13. Generate the preview.
14. Explain that GradeAssist prepares the action but teacher approval is required.
15. Click Send Email when SMTP is configured.
The important demonstration is that the dashboard is not manually populated. It consumes the verified result produced by the assessment workflow.
27. One-line Product Summary
GradeAssist observes assessment evidence, verifies and routes it, involves the teacher when required, and carries the verified result forward into analytics, reporting and teacher-approved communication.
28. MVP Summary
GradeAssist is an assessment-to-action system.
The teacher continues to make the important decisions. GradeAssist handles the repetitive work around those decisions.
Evidence
   ↓
Verification
   ↓
Routing
   ↓
Teacher Review
   ↓
Verified Result
   ↓
Analytics
   ↓
Report
   ↓
Teacher-approved Action
