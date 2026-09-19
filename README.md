# GradeAssist — Segment 1

Evidence-only local worksheet extraction for teacher review. It does not grade, solve questions, or modify marks.

Segment 2 adds local OpenCV capture validation: upload a short teacher-marking video, then GradeAssist detects page changes, waits for stable frames, rejects likely blurry frames using Laplacian variance, and saves selected candidates under `data/processed/candidates/`. It does not send video frames to Ollama automatically.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
pytest -q
python validate_ollama.py C:\path\to\real-worksheet.jpg
```

The validation command prints success only after an actual Ollama response validates against the canonical schema.

## Local dashboard demo data

The dashboard continues to read only `VERIFIED` audit records. To demonstrate it locally with the existing validated Sachin and Rufina evidence, explicitly seed separate records labelled **LOCAL DEMO DATA**:

```powershell
py -3.11 -m tools.seed_demo_records --seed-demo
```

This command makes no provider or SMTP calls and never changes review-required or non-demo records. To remove only those local demo records:

```powershell
py -3.11 -m tools.seed_demo_records --clear-demo
```

Before an email demonstration, set `DEMO_SACHIN_EMAIL` and `DEMO_RUFINA_EMAIL` in the ignored `.env` to actual demonstration inboxes, and configure the existing SMTP settings. Leaving either value blank correctly shows `Email not available` and blocks sending.
