
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
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": text
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
