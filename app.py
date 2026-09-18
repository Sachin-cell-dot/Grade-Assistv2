import streamlit as st
from database.db import initialize_database
from ui.extraction_demo import render_extraction_demo
from ui.capture_demo import render_capture_demo

st.set_page_config(page_title="GradeAssist", page_icon="GA", layout="wide")
initialize_database().close()
st.title("GradeAssist")
st.caption("Teacher workflow support: visible-mark extraction only. GradeAssist does not grade papers.")
capture_tab, extraction_tab = st.tabs(["1. Capture frames", "2. Extraction preview"])
with capture_tab:
    render_capture_demo()
with extraction_tab:
    render_extraction_demo()
