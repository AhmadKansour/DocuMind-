
import re
import time
import unicodedata

import streamlit as st

from pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    split_text_into_chunks,
    split_text_into_sections,
    split_text_by_sentences,
    is_section_heading,
)

from semantic_search import (
    load_embedding_model,
    load_reranker,
    rerank_passages,
    create_embeddings,
    search_similar_chunks,
    find_best_sentence,
    contains_numbered_list,
)

from rag_pipeline import prepare_rag_context
from llm_provider import generate_answer


def normalize_search_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"\b1ere\b", "premiere", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def remove_duplicate_chunks(chunks):
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        key = (
            chunk["page_number"],
            " ".join(chunk["text"].split())
        )

        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks


st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_reranker():
    return load_reranker()


@st.cache_data(show_spinner=False)
def get_extracted_pages(pdf_bytes, ocr_language):

    class PDFUpload:
        def __init__(self, data):
            self.data = data

        def getvalue(self):
            return self.data

    print(
        "PDF EXTRACTION CACHE MISS: Processing PDF",
        flush=True
    )

    start = time.perf_counter()

    pages = extract_text_from_pdf(
        PDFUpload(pdf_bytes),
        ocr_language=ocr_language
    )

    duration = time.perf_counter() - start

    print(
        f"ACTUAL PDF EXTRACTION: {duration:.2f} seconds",
        flush=True
    )

    return pages


@st.cache_data(show_spinner=False)
def get_document_embeddings(_model, chunk_texts):

    print(
        "EMBEDDING CACHE MISS: Generating embeddings",
        flush=True
    )

    start = time.perf_counter()

    chunks = [
        {"text": text}
        for text in chunk_texts
    ]

    embeddings = create_embeddings(
        _model,
        chunks
    )

    duration = time.perf_counter() - start

    print(
        f"ACTUAL EMBEDDING COMPUTATION: "
        f"{duration:.2f} seconds",
        flush=True
    )

    return embeddings


st.title("DocuMind")

st.subheader("Your AI Document Assistant")

st.write(
    "Upload your PDF and ask about its content"
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

ocr_language = st.selectbox(
    "Language of scanned PDF pages",
    options=["English", "French", "Arabic"],
    index=0,
    help=(
        "Select the main language of scanned pages. "
        "PDFs with selectable text are extracted normally."
    )
)

ocr_language_codes = {
    "English": "eng",
    "French": "fra",
    "Arabic": "ara"
}

selected_ocr_language = ocr_language_codes[ocr_language]

all_chunks = []
embeddings = None


if uploaded_file is not None:

    try:
        processing_start = time.perf_counter()
        extraction_start = time.perf_counter()

        with st.spinner(
            "Reading your document and recognizing scanned pages..."
        ):
            pages = get_extracted_pages(
                uploaded_file.getvalue(),
                selected_ocr_language
            )

        extraction_duration = (
            time.perf_counter() - extraction_start
        )

        print(
            f"PDF EXTRACTION CALL: "
            f"{extraction_duration:.2f} seconds",
            flush=True
        )

        print(
            "\n========== ACTIVE PDF ==========",
            flush=True
        )

        print(
            "Filename:",
            uploaded_file.name,
            flush=True
        )

        print(
            "Pages extracted:",
            len(pages),
            flush=True
        )

        print(
            "========== END ACTIVE PDF ==========\n",
            flush=True
        )

        for page in pages:

            cleaned_text = clean_text(
                page["text"]
            )

            if not cleaned_text:
                continue

            page_chunks = split_text_into_chunks(
                cleaned_text,
                chunk_size=1000,
                overlap=200
            )

            for chunk_number, chunk_text in enumerate(
                page_chunks,
                start=1
            ):
                all_chunks.append({
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                    "section_heading": None
                })

            sections = split_text_into_sections(
                cleaned_text
            )

            page_chunks = []

            for section in sections:

                lines = section.splitlines()
                heading = None

                if lines and is_section_heading(lines[0]):
                    heading = lines[0].strip()

                if len(section) <= 1000:
                    smaller_chunks = [section]
                else:
                    smaller_chunks = split_text_by_sentences(
                        section,
                        chunk_size=500,
                        overlap=1
                    )

                for chunk in smaller_chunks:
                    page_chunks.append({
                        "text": chunk,
                        "section_heading": heading
                    })

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1
            ):
                all_chunks.append({
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk["text"],
                    "section_heading": chunk["section_heading"]
                })

        original_chunk_count = len(all_chunks)

        all_chunks = remove_duplicate_chunks(all_chunks)

        print(
            f"CHUNK DEDUPLICATION: "
            f"{original_chunk_count} original chunks -> "
            f"{len(all_chunks)} unique chunks",
            flush=True
        )

        print(
            "\nDOCUMENT CHUNK METADATA:",
            flush=True
        )

        standard_count = sum(
            1
            for chunk in all_chunks
            if chunk["section_heading"] is None
        )

        section_count = (
            len(all_chunks) - standard_count
        )

        print(
            f"STANDARD CHUNKS: {standard_count}",
            flush=True
        )

        print(
            f"SECTION CHUNKS: {section_count}",
            flush=True
        )

        unique_texts = {
            chunk["text"].strip()
            for chunk in all_chunks
        }

        print(
            "\n--- FRENCH AVERAGE DIAGNOSTIC ---",
            flush=True
        )

        for chunk in all_chunks:
            if "57.75" in chunk["text"]:
                print(
                    f"FOUND 57.75 | "
                    f"Page: {chunk['page_number']} | "
                    f"Chunk: {chunk['chunk_number']} | "
                    f"Text: {chunk['text']}",
                    flush=True
                )

        print(
            f"TOTAL CHUNKS: {len(all_chunks)}",
            flush=True
        )

        print(
            f"UNIQUE CHUNKS: {len(unique_texts)}",
            flush=True
        )

        print(
            f"EXACT DUPLICATES: "
            f"{len(all_chunks) - len(unique_texts)}",
            flush=True
        )

        processing_duration = (
            time.perf_counter() - processing_start
        )

        print(
            f"DOCUMENT PROCESSING COMPLETE: "
            f"{len(all_chunks)} chunks",
            flush=True
        )

        st.success(
            f"Successfully processed {len(pages)} pages!"
        )

        st.info(
            f"Created {len(all_chunks)} text chunks."
        )

        st.write(
            f"PDF extraction and chunking time: "
            f"{processing_duration:.2f} seconds"
        )

        if all_chunks:

            with st.spinner(
                "Preparing semantic search..."
            ):
                model_start = time.perf_counter()

                model = get_embedding_model()

                model_duration = (
                    time.perf_counter() - model_start
                )

                chunk_texts = tuple(
                    chunk["text"]
                    for chunk in all_chunks
                )

                embedding_start = time.perf_counter()

                embeddings = get_document_embeddings(
                    model,
                    chunk_texts
                )

                embedding_duration = (
                    time.perf_counter() - embedding_start
                )

            st.success(
                "Semantic search is ready!"
            )

            st.write(
                f"Model loading time: "
                f"{model_duration:.2f} seconds"
            )

            st.write(
                f"Embedding creation time: "
                f"{embedding_duration:.2f} seconds"
            )

        else:
            st.warning(
                "No readable text was extracted from this PDF."
            )

    except Exception as error:
        st.error(
            "Unable to process this PDF."
        )
        st.exception(error)


question_input = st.text_input(
    "What would you like to know about your document?"
)

ask_clicked = st.button(
    "Ask DocuMind"
)


if ask_clicked:

    print(
        f"ASK CLICKED | question={question_input!r}",
        flush=True
    )

    if uploaded_file is None:
        st.warning(
            "Please upload a PDF."
        )

    elif not question_input.strip():
        st.warning(
            "Please enter a question."
        )

    elif embeddings is None:
        st.warning(
            "Semantic search is not ready. "
            "Please check your PDF."
        )

    else:
        question_start = time.perf_counter()

        with st.spinner(
            "Searching your document..."
        ):
            model = get_embedding_model()

            search_start = time.perf_counter()

            semantic_results = search_similar_chunks(
                model,
                question_input,
                all_chunks,
                embeddings,
                top_k=30
            )

            print(
                f"SEMANTIC SEARCH TIME: "
                f"{time.perf_counter() - search_start:.2f} seconds",
                flush=True
            )

            stop_words = {
                "de", "la", "le", "les",
                "des", "du", "est", "quelle"
            }

            question_words = set(
                normalize_search_text(question_input).split()
            ) - stop_words

            keyword_results = []

            for index, chunk in enumerate(all_chunks):

                chunk_words = set(
                    normalize_search_text(chunk["text"]).split()
                )

                matching_words = question_words.intersection(
                    chunk_words
                )

                if matching_words:
                    keyword_results.append({
                        "chunk_index": index,
                        "page_number": chunk["page_number"],
                        "text": chunk["text"],
                        "section_heading": chunk["section_heading"],
                        "score": float(len(matching_words))
                    })

            keyword_results.sort(
                key=lambda result: result["score"],
                reverse=True
            )

            print(
                "\n--- TOP KEYWORD RESULTS ---",
                flush=True
            )

            for result in keyword_results[:5]:
                print(
                    f"Page {result['page_number']} | "
                    f"Score {result['score']} | "
                    f"{result['text'][:150]!r}",
                    flush=True
                )

            combined_results = []
            seen_chunks = set()

            for result in (
                keyword_results[:10] + semantic_results
            ):
                chunk_index = result["chunk_index"]

                if chunk_index not in seen_chunks:
                    combined_results.append(result)
                    seen_chunks.add(chunk_index)

            print(
                f"HYBRID RETRIEVAL: "
                f"{len(semantic_results)} semantic candidates, "
                f"{min(10, len(keyword_results))} keyword candidates, "
                f"{len(combined_results)} unique candidates",
                flush=True
            )

            load_start = time.perf_counter()

            reranker = get_reranker()

            load_time = (
                time.perf_counter() - load_start
            )

            compute_start = time.perf_counter()

            reranked_passages = rerank_passages(
                reranker,
                question_input,
                combined_results,
                top_k=5
            )

            compute_time = (
                time.perf_counter() - compute_start
            )

            print(
                f"RERANKER LOAD: {load_time:.2f}s",
                flush=True
            )

            print(
                f"RERANKER COMPUTE: {compute_time:.2f}s",
                flush=True
            )

            rag_context = prepare_rag_context(
                reranked_passages,
                max_passages=5
            )

            print(
                f"RAG CONTEXT LENGTH: "
                f"{len(rag_context)} characters",
                flush=True
            )

        if not rag_context.strip():
            st.info(
                "No relevant passages were found."
            )

        else:
            with st.spinner(
                "Generating AI answer..."
            ):
                try:
                    print(
                        "GEMINI CALL STARTED",
                        flush=True
                    )

                    gemini_start = time.perf_counter()

                    ai_answer = generate_answer(
                        question_input,
                        rag_context
                    )

                    gemini_duration = (
                        time.perf_counter() - gemini_start
                    )

                    print(
                        f"GEMINI RESPONSE TIME: "
                        f"{gemini_duration:.2f} seconds",
                        flush=True
                    )

                    question_duration = (
                        time.perf_counter() - question_start
                    )

                    print(
                        f"TOTAL QUESTION TIME: "
                        f"{question_duration:.2f} seconds",
                        flush=True
                    )

                    st.subheader(
                        "DocuMind AI Answer"
                    )

                    st.write(
                        ai_answer
                    )

                except Exception as error:
                    error_code = getattr(
                        error,
                        "code",
                        None
                    )

                    print(
                        f"Gemini API error: "
                        f"{type(error).__name__}, "
                        f"status={error_code}",
                        flush=True
                    )

                    if error_code in (503, 504):
                        st.warning(
                            "The AI service is temporarily "
                            "unavailable. Please try again later."
                        )

                    elif error_code == 429:
                        st.warning(
                            "The AI service has reached its "
                            "usage limit. Please try again later."
                        )

                    else:
                        st.error(
                            "AI answer generation failed."
                        )

                        print(
                            f"Gemini error: {error}",
                            flush=True
                        )

            with st.expander(
                "View supporting passages"
            ):
                for passage in reranked_passages:
                    st.markdown(
                        f"**Page {passage['page_number']}**"
                    )

                    st.write(
                        passage["text"]
                    )

                    st.divider()
