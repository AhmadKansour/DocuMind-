
import sqlite3
from pathlib import Path

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



if __name__ == "__main__":
    initialize_database()
    print("DocuMind database initialized successfully.")
