import streamlit as st
from database.db import initialize_database
from ui.extraction_demo import render_extraction_demo
from ui.teacher_dashboard import render_teacher_dashboard

st.set_page_config(page_title="GradeAssist", page_icon="GA", layout="wide")
connection = initialize_database()
st.title("GradeAssist")
st.caption("Teacher workflow support: visible-mark extraction only. GradeAssist does not grade papers.")
screen = st.radio("Workspace", ["Assessment review", "Teacher Dashboard"], horizontal=True)
if screen == "Assessment review":
    render_extraction_demo()
else:
    render_teacher_dashboard(connection)
connection.close()
