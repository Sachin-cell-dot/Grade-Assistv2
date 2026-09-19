import sqlite3
from pathlib import Path
from config.settings import get_settings


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

def initialize_database(database_path: Path | None = None) -> sqlite3.Connection:
    path = database_path or get_settings().database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
    # Idempotent additions for databases initialized before the dashboard.
    _ensure_column(connection, "students", "parent_guardian_name", "TEXT")
    _ensure_column(connection, "students", "parent_email", "TEXT")
    _ensure_column(connection, "students", "is_demo", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(connection, "assessments", "assessment_name", "TEXT")
    _ensure_column(connection, "assessments", "maximum_marks", "REAL")
    _ensure_column(connection, "assessments", "verified_status", "TEXT")
    _ensure_column(connection, "assessments", "verified_at", "TEXT")
    _ensure_column(connection, "assessments", "is_demo", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(connection, "assessment_audits", "is_demo", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(connection, "assessment_audits", "demo_label", "TEXT")
    connection.commit()
    return connection
