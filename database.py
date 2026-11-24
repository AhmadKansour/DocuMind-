
import sqlite3
import os
import psycopg
from pathlib import Path
from datetime import datetime, timezone

DATABASE_PATH = Path(__file__).resolve().parent / "documind.db"


def get_database_backend():
    """
    Return the configured database backend.

    Defaults to SQLite for local development.
    """
    import streamlit as st

    backend = st.secrets.get("database_backend", "sqlite")

    if backend not in ("sqlite", "postgres"):
        raise ValueError(
            "Invalid database_backend. "
            "Expected 'sqlite' or 'postgres'."
        )

    return backend



def get_postgres_connection():
    import streamlit as st

    config = st.secrets["postgres"]

    return psycopg.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        sslmode=config["sslmode"],
        connect_timeout=10
    )


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    if get_database_backend() == "postgres":
        # PostgreSQL tables were already created
        # in Supabase during database setup.
        return

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                page_count INTEGER NOT NULL,
                ocr_language TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_usage (
                visitor_id TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                question_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (visitor_id, usage_date)
            )
            """
        )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS global_daily_usage (
                usage_date TEXT PRIMARY KEY,
                request_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        connection.commit()


def save_document(filename, page_count, ocr_language, chunk_count):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                filename,
                page_count,
                ocr_language,
                chunk_count
            )
            VALUES (?, ?, ?, ?)
            """,
            (filename, page_count, ocr_language, chunk_count)
        )

        connection.commit()
        return cursor.lastrowid


def get_all_documents():
    with get_connection() as connection:
        cursor = connection.execute(
            """
            SELECT
                id,
                filename,
                uploaded_at,
                page_count,
                ocr_language,
                chunk_count
            FROM documents
            ORDER BY id DESC
            """
        )

        documents = cursor.fetchall()
        return documents



def prepare_document_fingerprints():

    if get_database_backend() == "postgres":
        # PostgreSQL already has the fingerprint column
        # and its UNIQUE constraint in Supabase.
        return

    with get_connection() as connection:
        columns = connection.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

        column_names = {column[1] for column in columns}

        if "fingerprint" not in column_names:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN fingerprint TEXT"
            )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_documents_fingerprint
            ON documents(fingerprint)
            """
        )


def save_document_once(
    filename,
    page_count,
    ocr_language,
    chunk_count,
    fingerprint
):

    if get_database_backend() == "postgres":
        return save_document_once_postgres(
            filename=filename,
            page_count=page_count,
            ocr_language=ocr_language,
            chunk_count=chunk_count,
            fingerprint=fingerprint
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO documents (
                filename,
                page_count,
                ocr_language,
                chunk_count,
                fingerprint
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                page_count,
                ocr_language,
                chunk_count,
                fingerprint
            )
        )

        document = connection.execute(
            "SELECT id FROM documents WHERE fingerprint = ?",
            (fingerprint,)
        ).fetchone()

        return document[0]




def save_document_once_postgres(
    filename,
    page_count,
    ocr_language,
    chunk_count,
    fingerprint
):
    """
    Save document metadata through a restricted
    PostgreSQL function.

    Return the existing ID if the document's
    fingerprint already exists.
    """

    with get_postgres_connection() as connection:
        result = connection.execute(
            """
            SELECT public.save_document_metadata(
                %s, %s, %s, %s, %s
            )
            """,
            (
                filename,
                page_count,
                ocr_language,
                chunk_count,
                fingerprint
            )
        ).fetchone()

        return result[0]



def get_daily_question_count(visitor_id):

    if get_database_backend() == "postgres":
        return get_daily_question_count_postgres(visitor_id)

    today = datetime.now(timezone.utc).date().isoformat()

    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT question_count
            FROM daily_usage
            WHERE visitor_id = ? AND usage_date = ?
            """,
            (visitor_id, today)
        ).fetchone()

        return result[0] if result else 0




def get_daily_question_count_postgres(visitor_id):
    """
    Retrieve today's question count using the
    restricted PostgreSQL function.
    """

    today = datetime.now(timezone.utc).date()

    with get_postgres_connection() as connection:
        result = connection.execute(
            """
            SELECT public.get_visitor_daily_count(
                %s, %s
            )
            """,
            (visitor_id, today)
        ).fetchone()

        return result[0]


def use_daily_question(visitor_id, daily_limit=10):
    today = datetime.now(timezone.utc).date().isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO daily_usage (
                visitor_id, usage_date, question_count
            )
            VALUES (?, ?, 0)
            """,
            (visitor_id, today)
        )

        cursor = connection.execute(
            """
            UPDATE daily_usage
            SET question_count = question_count + 1
            WHERE visitor_id = ?
              AND usage_date = ?
              AND question_count < ?
            """,
            (visitor_id, today, daily_limit)
        )

        return cursor.rowcount == 1


def use_global_daily_request(daily_limit=100):
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO global_daily_usage
                (usage_date, request_count)
            VALUES (?, 0)
            """,
            (today,)
        )

        result = connection.execute(
            """
            UPDATE global_daily_usage
            SET request_count = request_count + 1
            WHERE usage_date = ?
              AND request_count < ?
            """,
            (today, daily_limit)
        )

        connection.commit()

        return result.rowcount == 1

def use_ai_request(visitor_id, visitor_limit=10, global_limit=100):

    if get_database_backend() == "postgres":
        return use_ai_request_postgres(
            visitor_id,
            visitor_limit=visitor_limit,
            global_limit=global_limit
        )

    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()

    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        visitor_row = connection.execute(
            """
            SELECT question_count
            FROM daily_usage
            WHERE visitor_id = ? AND usage_date = ?
            """,
            (visitor_id, today)
        ).fetchone()

        global_row = connection.execute(
            """
            SELECT request_count
            FROM global_daily_usage
            WHERE usage_date = ?
            """,
            (today,)
        ).fetchone()

        visitor_count = visitor_row[0] if visitor_row else 0
        global_count = global_row[0] if global_row else 0

        if visitor_count >= visitor_limit:
            return "visitor_limit"

        if global_count >= global_limit:
            return "global_limit"

        connection.execute(
            """
            INSERT INTO daily_usage
                (visitor_id, usage_date, question_count)
            VALUES (?, ?, 1)
            ON CONFLICT(visitor_id, usage_date)
            DO UPDATE SET question_count = question_count + 1
            """,
            (visitor_id, today)
        )

        connection.execute(
            """
            INSERT INTO global_daily_usage
                (usage_date, request_count)
            VALUES (?, 1)
            ON CONFLICT(usage_date)
            DO UPDATE SET request_count = request_count + 1
            """,
            (today,)
        )

        connection.commit()
        return "allowed"

def get_global_daily_request_count():
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()

    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT request_count
            FROM global_daily_usage
            WHERE usage_date = ?
            """,
            (today,)
        ).fetchone()

        return result[0] if result else 0


def use_ai_request_postgres(
    visitor_id,
    visitor_limit=10,
    global_limit=100
):
    """
    Atomically enforce visitor and global AI quotas
    using the restricted PostgreSQL function.

    Fail closed if the database is unavailable.
    """

    try:
        with get_postgres_connection() as connection:
            result = connection.execute(
                """
                SELECT public.use_ai_request(
                    %s, %s, %s
                )
                """,
                (
                    visitor_id,
                    visitor_limit,
                    global_limit
                )
            ).fetchone()

            if result is None:
                return "database_error"

            if result[0] not in (
                "allowed",
                "visitor_limit",
                "global_limit"
            ):
                return "database_error"

            return result[0]

    except Exception as error:
        print(
            f"Database quota error: {type(error).__name__}",
            flush=True
        )

        return "database_error"


if __name__ == "__main__":
    initialize_database()
    print("DocuMind database initialized successfully.")
