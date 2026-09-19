"""Additive verified-only Teacher Dashboard."""
from __future__ import annotations

import sqlite3

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import get_settings
from services.dashboard_service import dashboard_summary, topic_insights, verified_student_performance
from services.email_service import report_email_preview, send_report_email, smtp_ready, student_report_body
from services.report_export import build_class_report


def _display(value: float | None) -> float | str:
    return value if value is not None else "Not available"


def render_teacher_dashboard(connection: sqlite3.Connection) -> None:
    st.header("Teacher Dashboard")
    st.caption("Verified assessments only. Review-required and draft evidence are excluded from class analytics.")
    records = verified_student_performance(connection)
    if not records:
        st.info("No verified assessments are available yet. Complete teacher confirmation before class analytics can be shown.")
        return
    classes = sorted({record.class_name for record in records if record.class_name})
    assessments = sorted({record.assessment_name for record in records})
    selected_class = st.selectbox("Class", ["All classes", *classes])
    selected_assessment = st.selectbox("Assessment", ["All assessments", *assessments])
    filtered = [record for record in records if (selected_class == "All classes" or record.class_name == selected_class) and (selected_assessment == "All assessments" or record.assessment_name == selected_assessment)]
    if any(record.is_demo for record in filtered):
        st.warning("LOCAL DEMO DATA")
    summary = dashboard_summary(filtered)
    cards = st.columns(6)
    for card, label, value in zip(cards, ["Students", "Assessments processed", "Average score", "Highest score", "Lowest score", "Needs review"], [summary.students, summary.assessments_processed, _display(summary.average_score), _display(summary.highest_score), _display(summary.lowest_score), summary.needs_review]):
        card.metric(label, value)
    st.subheader("Student performance")
    st.dataframe([
        {"Student name": record.student_name, "Class": record.class_name or "Not available", "Parent/guardian": record.parent_guardian_name or "Not available", "Parent email": record.parent_email or "Email not available", "Assessment": record.assessment_name, "Marks obtained": record.marks, "Maximum marks": record.maximum, "Percentage": record.percentage, "Review status": record.review_status, "Data source": record.demo_label or "Verified assessment"}
        for record in filtered
    ], hide_index=True, use_container_width=True)
    st.subheader("Class performance")
    chart_rows = [{"Student": record.student_name, "Percentage": record.percentage, "Assessment": record.assessment_name} for record in filtered if record.percentage is not None]
    if chart_rows:
        left, right = st.columns(2)
        left.plotly_chart(px.histogram(chart_rows, x="Percentage", nbins=10, title="Class score distribution"), use_container_width=True)
        right.plotly_chart(px.bar(chart_rows, x="Student", y="Percentage", color="Assessment", barmode="group", title="Student performance comparison"), use_container_width=True)
    topic_rows = [{"Question/topic": question.question_id, "Percentage": question.percentage, "Student": record.student_name} for record in filtered for question in record.questions if question.percentage is not None]
    if topic_rows:
        topic_chart = px.bar(topic_rows, x="Question/topic", y="Percentage", color="Student", barmode="group", title="Topic/criterion mastery")
        st.plotly_chart(topic_chart, use_container_width=True)
        students = sorted({row["Student"] for row in topic_rows})
        topics = sorted({row["Question/topic"] for row in topic_rows})
        values = [[next((row["Percentage"] for row in topic_rows if row["Student"] == student and row["Question/topic"] == topic), None) for topic in topics] for student in students]
        st.plotly_chart(go.Figure(data=go.Heatmap(z=values, x=topics, y=students, colorscale="Blues", colorbar_title="%"), layout_title_text="Class performance heatmap"), use_container_width=True)
    else:
        st.info("Question/topic performance is unavailable until verified records include question maximum evidence.")
    names = sorted({record.student_name for record in filtered})
    selected_student = st.selectbox("Student detail", names)
    student_records = [record for record in filtered if record.student_name == selected_student]
    latest = student_records[-1]
    strengths, attention = topic_insights(filtered, selected_student)
    st.subheader(f"Student detail: {selected_student}")
    st.write({"Class": latest.class_name or "Not available", "Assessment history": len(student_records), "Strengths": strengths or "Not enough repeated evidence", "Needs attention": attention or "Not enough repeated evidence"})
    history = [{"Assessment": record.assessment_name, "Date": record.assessment_date, "Percentage": record.percentage} for record in student_records if record.percentage is not None]
    if history:
        st.plotly_chart(px.line(history, x="Date", y="Percentage", markers=True, title="Assessment history"), use_container_width=True)
    report_bytes = build_class_report(filtered, summary)
    st.download_button("Export Class Report", report_bytes, file_name="gradeassist-class-report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Teacher-controlled email report")
    recipient_options = {record.student_name: record for record in filtered}
    selected_students = st.multiselect("Select student recipient(s)", list(recipient_options))
    selected_records = [recipient_options[name] for name in selected_students]
    recipient_emails = [record.parent_email for record in selected_records if record.parent_email]
    for record in selected_records:
        st.write(f"{record.student_name}: {record.parent_email or 'Email not available'}")
    bodies = []
    for record in selected_records:
        strengths, attention = topic_insights(filtered, record.student_name)
        bodies.append(student_report_body(
            student_name=record.student_name,
            assessment_name=record.assessment_name,
            marks=record.marks,
            maximum=record.maximum,
            percentage=record.percentage,
            verified_status=record.review_status,
            timestamp=record.assessment_date,
            strengths=strengths,
            needs_attention=attention,
        ))
    body = "\n\n".join(bodies)
    subject = "GradeAssist verified student report"
    if selected_records:
        preview = report_email_preview(subject, body, recipient_emails)
        with st.expander("Email preview"):
            st.write(preview.get_content())
    settings = get_settings()
    if not smtp_ready(settings):
        st.info("SMTP is not configured. Configure SMTP_HOST and SMTP_FROM_EMAIL to enable sending.")
    elif st.button("Send Email", type="primary"):
        if not selected_records:
            st.error("Select at least one student before sending.")
        elif len(recipient_emails) != len(selected_records):
            st.error("Email not available for one or more selected students. No email was sent.")
        else:
            try:
                send_report_email(settings, recipient_emails, subject, body, report_bytes)
                st.success("Email sent.")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Email could not be sent: {type(exc).__name__}.")
