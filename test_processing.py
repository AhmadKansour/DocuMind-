
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


from pdf_processor import split_text_by_sentences

english_text = (
    "Artificial intelligence helps businesses automate tasks. "
    "Machine learning identifies patterns in data. "
    "Natural language processing helps computers understand text."
)

arabic_text = (
    "الذكاء الاصطناعي يساعد الشركات على تحسين أعمالها. "
    "التعلم الآلي يساعد الحواسيب على التعلم من البيانات. "
    "هل يمكن استخدام الذكاء الاصطناعي في الشركات؟ "
    "نعم، يمكن استخدامه في مجالات عديدة."
)

print("\nENGLISH CHUNKS:")
for number, chunk in enumerate(
    split_text_by_sentences(english_text, chunk_size=100),
    start=1
):
    print(f"Chunk {number}: {chunk}")

print("\nARABIC CHUNKS:")
for number, chunk in enumerate(
    split_text_by_sentences(arabic_text, chunk_size=100),
    start=1
):
    print(f"Chunk {number}: {chunk}")

