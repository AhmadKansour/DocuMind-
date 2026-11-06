
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

from pdf_processor import split_text_into_sections

sample_text = """
Chapter 1: Fundamentals of Project Management

Project:
Temporary endeavor that produces a unique result.

Project Phases:
1. Concept: Marketing
2. Definition: Correct problem definition
3. Planning: Tasks, resources, time, logistics
4. Execution & Control: Perform, monitor, correct
5. Closeout: Review and lessons learned

PMBOK Knowledge Areas:
Integration: Coordinate and control work
Scope: Define boundaries
"""

sections = split_text_into_sections(sample_text)

for number, section in enumerate(sections, start=1):
    print(f"\nSECTION {number}")
    print(section)


    
# TEST: Extract sections from a real PDF

from pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    split_text_into_sections
)

from io import BytesIO
# Open the PDF in binary mode.

from io import BytesIO

# Open the PDF and read its contents.
with open("I3330 Summary.pdf", "rb") as pdf_file:
    pdf_bytes = pdf_file.read()

# Convert the bytes into an in-memory file.
uploaded_file = BytesIO(pdf_bytes)

# Extract the pages.
pages = extract_text_from_pdf(uploaded_file)

# Get the first page.
first_page = pages[0]

# Clean the text.
cleaned_text = clean_text(first_page["text"])

# Split into sections.
sections = split_text_into_sections(cleaned_text)

# Find the Project Phases section.
for section in sections:
    if "Project Phases" in section:
        print("\nPROJECT PHASES SECTION:")
        print(section)

from pdf_processor import is_section_heading

test_lines = [
    "Employees and Benefits",
    "Financial Performance",
    "Project Phases:",
    "1. Concept: Marketing",
    "The company employs 120 people.",
    "Customer Support"
]

print("\nHEADING DETECTION TEST:")

for line in test_lines:
    result = is_section_heading(line)
    print(f"{result}: {line}")
