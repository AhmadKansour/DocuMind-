
import pymupdf
import re





def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?؟])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]



def split_text_by_sentences(text, chunk_size=500, overlap=1):
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("Invalid chunk size or overlap")

    sentences = split_into_sentences(text)
    chunks = []
    current_sentences = []

    for sentence in sentences:
        candidate = " ".join(current_sentences + [sentence])#combines sentences into 1 string, seperated by spaces

        if current_sentences and len(candidate) > chunk_size:
            chunks.append(" ".join(current_sentences))

            current_sentences = (
                current_sentences[-overlap:] if overlap > 0 else []
            )

            while current_sentences and len(
                " ".join(current_sentences + [sentence])
            ) > chunk_size:
                current_sentences.pop(0)

        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks





def extract_text_from_pdf(uploaded_file):
    pages = []

    pdf_bytes = uploaded_file.getvalue()

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:

        metadata = document.metadata or {}
        document_title = (metadata.get("title") or "").strip()

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text",sort=True)

            pages.append({
                "page_number": page_number,
                "text": text,
                "document_title": document_title
            })

    return pages




def clean_text(text):
    text = re.sub(r"[^\S\n]+", " ", text) #replaces whitespace with a single space
    text = re.sub(r" *\n *", "\n", text)# removes spaces immediately before or after a line break
    text = re.sub(r"\n{3,}", "\n\n", text)#replaces three or more consecutive line breaks with two line breaks.

    return text.strip()



def split_text_into_chunks(text, chunk_size=250, overlap=70):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks



def is_section_heading(line):
    line = line.strip()

    if not line:
        return False

    if re.match(r"^(\d+[.)]\s+|[-•]\s+)", line):
        return False

    if line.endswith(":"):
        return True

    if line.endswith((".", "!", "?", "؟", "。")):
        return False

    words = line.split()

    if 2 <= len(words) <= 6:
        capitalized_words = sum(
            1 for word in words
            if word[0].isupper()
        )

        return capitalized_words >= max(2, len(words) - 1)

    return False



def split_text_into_sections(text):

    # Split the text into individual lines.
    lines = text.splitlines()

    # Store completed sections here.
    sections = []

    # Store the lines of the section we're currently building.
    current_section = []

    for line in lines:

        # Remove unnecessary spaces.
        line = line.strip()

        # Skip empty lines.
        if not line:
            continue

        # Check whether this line is a heading.
        is_heading = is_section_heading(line)

        # If we find a new heading, save the previous section.
        if is_heading and current_section:
            sections.append("\n".join(current_section))
            current_section = []

        # Add the line to the current section.
        current_section.append(line)

    # Save the last section after the loop finishes.
    if current_section:
        sections.append("\n".join(current_section))

    # Return the completed list.
    return sections



def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(" ", start, end)
            )

            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - overlap)

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
