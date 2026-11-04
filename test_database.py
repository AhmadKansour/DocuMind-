
from database import get_connection

with get_connection() as connection:
    documents = connection.execute(
        """
        SELECT id, filename, page_count,
               ocr_language, chunk_count
        FROM documents
        ORDER BY id
        """
    ).fetchall()

for document in documents:
    print(document)

print("Total records:", len(documents))
