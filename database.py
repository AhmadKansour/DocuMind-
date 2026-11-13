
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DATABASE_PATH = Path(__file__).resolve().parent / "documind.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
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




def get_daily_question_count(visitor_id):
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


if __name__ == "__main__":
    initialize_database()
    print("DocuMind database initialized successfully.")
