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
