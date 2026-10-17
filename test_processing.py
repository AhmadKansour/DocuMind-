
from pdf_processor import clean_text, split_text_into_chunks

sample_text = (
    "Artificial intelligence helps businesses automate tasks. "
    "Machine learning identifies patterns in data. "
    "Natural language processing helps computers understand text."
)

cleaned = clean_text(sample_text)

chunks = split_text_into_chunks(
    cleaned,
    chunk_size=80,
    overlap=20
)

print("Cleaned text:")
print(cleaned)

print("\nTotal chunks:", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {number}:")
    print(chunk)
