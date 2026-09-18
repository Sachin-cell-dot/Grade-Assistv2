import streamlit as st
from database.db import initialize_database
from ui.extraction_demo import render_extraction_demo

st.set_page_config(page_title="GradeAssist", page_icon="GA", layout="wide")
initialize_database().close()
st.title("GradeAssist")
st.caption("Teacher workflow support: visible-mark extraction only. GradeAssist does not grade papers.")
render_extraction_demo()
