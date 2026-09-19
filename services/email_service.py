"""Explicit teacher-controlled SMTP report delivery."""
from __future__ import annotations

from email.message import EmailMessage
import smtplib

from config.settings import Settings


def smtp_ready(settings: Settings) -> bool:
    return bool(settings.smtp_host and settings.smtp_from_email)


def email_send_readiness(settings: Settings, *, selected_count: int, recipient_count: int) -> tuple[bool, str]:
    """Return a teacher-readable, non-secret reason before a send can occur."""
    if selected_count == 0:
        return False, "Select at least one verified student before sending."
    if recipient_count != selected_count:
        return False, "Email not available for one or more selected students. No email will be sent."
    if not smtp_ready(settings):
        return False, "SMTP configuration required: set SMTP_HOST and SMTP_FROM_EMAIL."
    return True, "Ready for teacher approval. Sending occurs only after you click Send Email."


def report_email_preview(subject: str, body: str, recipients: list[str]) -> EmailMessage:
    message = EmailMessage()
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)
    return message


def student_report_body(
    *,
    student_name: str,
    assessment_name: str,
    marks: float | None,
    maximum: float | None,
    percentage: float | None,
    verified_status: str,
    timestamp: str | None,
    strengths: list[str],
    needs_attention: list[str],
) -> str:
    """Compose only verified facts supplied by the dashboard."""
    lines = [
        f"Student: {student_name}",
        f"Assessment: {assessment_name}",
        f"Marks: {marks if marks is not None else 'Not available'} / {maximum if maximum is not None else 'Not available'}",
        f"Percentage: {percentage if percentage is not None else 'Not available'}",
        f"Verified status: {verified_status}",
    ]
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")
    if strengths:
        lines.append("Strengths supported by repeated verified evidence: " + ", ".join(strengths))
    if needs_attention:
        lines.append("Areas needing attention supported by repeated verified evidence: " + ", ".join(needs_attention))
    return "\n".join(lines)


def send_report_email(settings: Settings, recipients: list[str], subject: str, body: str, report_bytes: bytes) -> None:
    if not smtp_ready(settings):
        raise RuntimeError("SMTP is not configured. Set SMTP_HOST and SMTP_FROM_EMAIL before sending.")
    message = report_email_preview(subject, body, recipients)
    message["From"] = settings.smtp_from_email
    message.add_attachment(report_bytes, maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="gradeassist-class-report.xlsx")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as client:
        if settings.smtp_use_tls:
            client.starttls()
        if settings.smtp_username:
            client.login(settings.smtp_username, settings.smtp_password or "")
        client.send_message(message)
