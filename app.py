
import re
import time
import unicodedata
import hmac
import secrets
import html
from datetime import datetime, timedelta

import streamlit as st
import extra_streamlit_components as stx
import uuid
import hashlib

from database import (
    initialize_database,
    prepare_document_fingerprints,
    save_document_once,
    get_daily_question_count,
    use_ai_request,
)

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

@st.cache_data(show_spinner=False, scope="session")
def get_processed_chunks(pages):

    print(
        "CHUNK CACHE MISS: Processing document chunks",
        flush=True
    )

    all_chunks = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        # Standard fixed-size chunks
        standard_chunks = split_text_into_chunks(
            cleaned_text,
            chunk_size=1000,
            overlap=200
        )

        for chunk_number, chunk_text in enumerate(
            standard_chunks,
            start=1
        ):
            all_chunks.append({
                "page_number": page["page_number"],
                "chunk_number": chunk_number,
                "text": chunk_text,
                "section_heading": None
            })

        # Section-aware chunks
        sections = split_text_into_sections(cleaned_text)

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

    return all_chunks, original_chunk_count


def sign_visitor_id(visitor_id):
    """
    Create a signed cookie value for a visitor ID.
    """
    signing_key = st.secrets["VISITOR_SIGNING_KEY"].encode("utf-8")

    signature = hmac.new(
        signing_key,
        visitor_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return f"{visitor_id}.{signature}"


def verify_visitor_cookie(cookie_value):
    """
    Return the visitor ID if the cookie is valid.
    Otherwise, return None.
    """
    if not isinstance(cookie_value, str):
        return None

    try:
        visitor_id, signature = cookie_value.rsplit(".", 1)

        # Only accept properly formatted UUIDs.
        parsed_id = uuid.UUID(visitor_id)

        if str(parsed_id) != visitor_id:
            return None

        expected_signature = sign_visitor_id(visitor_id)

        # Compare signatures without timing-based information leakage.
        if secrets.compare_digest(
            cookie_value,
            expected_signature
        ):
            return visitor_id

    except (ValueError, TypeError):
        return None

    return None


st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)



COOKIE_NAME = "documind_visitor_id"

cookie_manager = stx.CookieManager(
    key="documind_cookie_manager"
)

# Retrieve browser cookies.
# None means the component has not returned its result yet.
browser_cookies = cookie_manager.cookie_manager(
    method="getAll",
    key="documind_cookie_reader",
    default=None
)

# First check cookies received by Streamlit.
cookie_value = st.context.cookies.get(COOKIE_NAME)

visitor_id = verify_visitor_cookie(cookie_value)

# If necessary, check cookies returned by the browser component.
if visitor_id is None:

    if browser_cookies is None:
        st.info("Preparing your visitor session...")
        st.stop()

    cookie_value = browser_cookies.get(COOKIE_NAME)
    visitor_id = verify_visitor_cookie(cookie_value)

# Create a new identity only after cookie retrieval is complete.
if visitor_id is None:

    if "pending_visitor_id" not in st.session_state:
        st.session_state.pending_visitor_id = str(uuid.uuid4())

    pending_id = st.session_state.pending_visitor_id

    cookie_manager.set(
        COOKIE_NAME,
        sign_visitor_id(pending_id),
        key="documind_visitor_cookie",
        path="/",
        expires_at=datetime.now() + timedelta(days=365),
        same_site="lax",
    )

    st.info(
        "Your visitor session is being initialized. "
        "Please refresh the page if this message remains."
    )

    st.stop()

# A valid visitor cookie has been recovered.
st.session_state.pop("pending_visitor_id", None)



st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #F5FAFF;
        color: #17324D;
    }

    /* Main content */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #17324D;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Primary buttons */
    button[kind="primary"] {
        background-color: #0867D8;
        color: white;
        border-radius: 10px;
        border: none;
    }

    button[kind="primary"]:hover {
        background-color: #0555B5;
        color: white;
    }

    /* Input fields */
    [data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    /* Calm Blue sidebar */
[data-testid="stSidebar"] {
    background-color: #EAF4FF;
    border-right: 1px solid #D5E5F5;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0867D8;
}

/* Sidebar text */
[data-testid="stSidebar"] p {
    color: #17324D;
}

/* Sidebar upload area */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border-radius: 12px;
}


/* Question input styling */
[data-testid="stTextInput"] input {
    background-color: #FFFFFF;
    color: #17324D;
    border-radius: 10px;
    padding: 12px 14px;
}
/* Prevent Safari from zooming when focusing inputs on iPhone */
@media (max-width: 768px) {
    [data-testid="stTextInput"] input {
        font-size: 16px !important;
    }
}

/* Form submit button */
[data-testid="stFormSubmitButton"] button {
    background-color: #0867D8;
    color: #FFFFFF;
    border-radius: 10px;
    min-height: 44px;
}

[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0555B5;
    color: #FFFFFF;
}

/* Compact AI answer card */
.st-key-saved_answer_card {
    background-color: #FFFFFF;
    border: 1px solid #D5E5F5;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 4px;
}

/* Answer heading */
.st-key-saved_answer_card h3 {
    color: #17324D;
    font-size: 19px;
    margin-bottom: 12px;
}

/* Answer text */
.st-key-saved_answer_card p,
.st-key-saved_answer_card li {
    color: #17324D;
    line-height: 1.6;
}


/* Sidebar conversation-history buttons */
[data-testid="stSidebar"] .st-key-new_chat_button {
    margin-bottom: 8px;
}

/* History buttons: consistent spacing */
[data-testid="stSidebar"] [class*="st-key-history_"] {
    margin-bottom: 6px;
}

/* Unselected history buttons */
[data-testid="stSidebar"] [class*="st-key-history_"] button[kind="secondary"] {
    background-color: #FFFFFF;
    color: #17324D;
    border: 1px solid #D5E5F5;
    border-radius: 10px;
    min-height: 42px;
    text-align: left;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

/* Hover effect */
[data-testid="stSidebar"] [class*="st-key-history_"] button[kind="secondary"]:hover {
    background-color: #F5FAFF;
    border-color: #0867D8;
    color: #0867D8;
}

/* Selected history button */
[data-testid="stSidebar"] [class*="st-key-history_"] button[kind="primary"] {
    background-color: #0867D8;
    color: #FFFFFF;
    border: 1px solid #0867D8;
    border-radius: 10px;
    min-height: 42px;
    text-align: left;
}

/* Selected history button hover */
[data-testid="stSidebar"] [class*="st-key-history_"] button[kind="primary"]:hover {
    background-color: #0555B5;
    color: #FFFFFF;
}

/* History question text */
[data-testid="stSidebar"] [class*="st-key-history_"] button p {
    text-align: left;
    width: 100%;
    line-height: 1.4;
}

/* Ensure selected history question text is white */
[data-testid="stSidebar"] [class*="st-key-history_"] button[kind="primary"] p {
    color: #FFFFFF !important;
}


/* Supporting passage cards */
[class*="st-key-saved_passage_"] {
    background-color: #FFFFFF;
    border: 1px solid #D5E5F5;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* Passage headings */
[class*="st-key-saved_passage_"] strong {
    color: #0867D8;
    font-weight: 600;
}

/* Page number labels */
[class*="st-key-saved_passage_"] [data-testid="stCaptionContainer"] {
    background-color: #EAF4FF;
    border-radius: 6px;
    padding: 5px 10px;
    text-align: center;
    width: fit-content;
    margin-left: auto;
}

/* Page label text */
[class*="st-key-saved_passage_"] [data-testid="stCaptionContainer"] p {
    color: #0867D8;
    font-size: 13px;
    font-weight: 500;
}

/* Passage body text */
[class*="st-key-saved_passage_"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.65;
}

/* Ask DocuMind button - compact and right aligned */
[data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: flex-end;
}

[data-testid="stFormSubmitButton"] button {
    width: 180px !important;
    min-width: 180px;
    max-width: 100%;
    white-space: nowrap;
    border: 1px solid #0867D8 !important;
    box-shadow: none;
}
/* Smaller screens */
@media (max-width: 768px) {
    [data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        min-width: 0;
    }
}
/* Ask button focus styling */
[data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 2px solid #86B7F5;
    outline-offset: 2px;
}
/* DocuMind sidebar branding */
.documind-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}

.documind-brand-icon {
     display: flex;
    align-items: center;
    justify-content: center;
    background-color: #0867D8;
    color: #FFFFFF;
    width: 48px;
    height: 48px;
    border-radius: 12px;
    flex-shrink: 0;
}

.documind-brand-name {
    color: #17324D;
    font-size: 25px;
    font-weight: 700;
    line-height: 1.2;
}

.documind-brand-subtitle {
    color: #64748B;
    font-size: 12px;
    margin-top: 4px;
}

/* Calm Blue PDF uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #F5FAFF;
    border: 1px dashed #BFD9F7;
    border-radius: 14px;
    padding: 18px 12px;
    transition: border-color 0.2s, background-color 0.2s;
}

/* Highlight the upload area on hover */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
    background-color: #EAF4FF;
    border-color: #0867D8;
}

/* Style the upload button */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #FFFFFF;
    color: #0867D8;
    border: 1px solid #BFD9F7;
    border-radius: 9px;
}

/* Style the upload helper text */
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #64748B;
}


/* Safe Calm Blue uploader styling */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #F5FAFF;
    border: 1px dashed #BFD9F7;
    border-radius: 14px;
}

/* Upload button appearance */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    color: #0867D8;
    background-color: #FFFFFF;
    border: 1px solid #BFD9F7;
    border-radius: 9px;
}

/* New Chat button styling */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
    background-color: #0867D8;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
}

/* New Chat button hover */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #0555B5;
    color: #FFFFFF !important;
}

/* Keep button text white */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] p {
    color: #FFFFFF !important;
}


/* Match Streamlit's top header to DocuMind background */
[data-testid="stHeader"] {
    background-color: #F5FAFF;
}


/* Reduce unnecessary space above the main content */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem;
}
/* Make sidebar open and close controls clearly visible */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    color: #0867D8 !important;
    background-color: #EAF4FF !important;
    border-radius: 8px !important;
}

/* Streamlit uses Material Symbols for sidebar arrows */
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    color: #0867D8 !important;
    opacity: 1 !important;
}

/* Ensure Streamlit warning messages have readable text */
[data-testid="stAlert"][data-baseweb="notification"] {
    color: #664D03 !important;
}

/* DocuMind status and expander containers */
[data-testid="stStatusWidget"],
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    color: #17324D !important;
    border: 1px solid #D5E5F5 !important;
    border-radius: 12px !important;
}

/* Status and expander labels */
[data-testid="stStatusWidget"] summary,
[data-testid="stExpander"] summary,
[data-testid="stStatusWidget"] p,
[data-testid="stExpander"] p {
    color: #17324D !important;
}

/* Icons and arrows */
[data-testid="stStatusWidget"] svg,
[data-testid="stExpander"] svg {
    color: #0867D8 !important;
}

/* AI processing status container */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] details,
[data-testid="stStatusWidget"] summary,
[data-testid="stStatusWidget"] [data-testid="stExpanderDetails"] {
    background-color: #FFFFFF !important;
    color: #17324D !important;
}

/* Streamlit status messages may use an expander */
[data-testid="stExpander"]:has([data-testid="stStatusWidget"]) {
    background-color: #FFFFFF !important;
}

/* Status labels and icons */
[data-testid="stStatusWidget"] span,
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] svg {
    color: #17324D !important;
}

/* Fix AI processing status header */
[data-testid="stExpander"] summary {
    background-color: #FFFFFF !important;
    color: #17324D !important;
}




/* Keep status text and icons visible*/

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary svg {
    color: #17324D !important;
    opacity: 1 !important;
}


[data-testid="stAlert"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #664D03 !important;
}


/* Ask DocuMind placeholder */
input[placeholder="Ask anything about your document..."]::placeholder {
    color: #475569 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

/* Compact question heading on mobile */
@media (max-width: 768px) {
    .st-key-user_question_heading h4 {
        font-size: 18px !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word;
    }
}


    </style>
    """,
    unsafe_allow_html=True
)

initialize_database()
prepare_document_fingerprints()



if visitor_id is not None:
    questions_used = get_daily_question_count(visitor_id)
    questions_remaining = max(0, 10 - questions_used)

    st.sidebar.caption(
        f"AI questions remaining today: {questions_remaining}/10"
    )

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}


if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None


if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "active_document_key" not in st.session_state:
    st.session_state.active_document_key = None


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner= False)
def get_reranker():
    return load_reranker()


@st.cache_data(show_spinner=False, scope="session")
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


@st.cache_data(show_spinner=False, scope="session")
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


st.markdown(
    """
<div style="margin-bottom: 28px;">
    <h1 style="color: #0867D8; font-size: 42px; margin-bottom: 5px;">
         DocuMind
    </h1>
    <p style="color: #17324D; font-size: 19px; font-weight: 500;">
        Your AI Document Assistant
    </p>
    <p style="color: #64748B; font-size: 15px;">
        Upload your PDF, ask questions, and discover answers instantly.
    </p>
</div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:


    st.markdown(
        """
<div class="documind-brand">
    <div class="documind-brand-icon">
        <svg xmlns="http://www.w3.org/2000/svg"
             width="25"
             height="25"
             viewBox="0 0 24 24"
             fill="none"
             stroke="currentColor"
             stroke-width="2"
             stroke-linecap="round"
             stroke-linejoin="round"
             aria-hidden="true">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h7"/>
            <path d="M14 2v6h6"/>
            <path d="M14 2l6 6"/>
            <circle cx="16.5" cy="16.5" r="3.5"/>
            <path d="m19 19 3 3"/>
        </svg>
    </div>
    <div>
        <div class="documind-brand-name">DocuMind</div>
        <div class="documind-brand-subtitle">AI Document Assistant</div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )


    st.divider()





    st.markdown("### Documents")



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
current_document_key = None

if uploaded_file is None and st.session_state.active_document_key is not None:
    st.session_state.active_document_key = None
    st.session_state.selected_chat = None
    st.session_state.question_text = ""
    st.rerun()


if uploaded_file is not None:

    try:
        pdf_bytes = uploaded_file.getvalue()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            st.error("The PDF exceeds the 10 MB upload limit.")
            st.stop()

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
            "Pages extracted:",
            len(pages),
            flush=True
        )

        print(
            "========== END ACTIVE PDF ==========\n",
            flush=True
        )

        all_chunks, original_chunk_count = get_processed_chunks(
            pages
        )

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

        print(
    f"PDF PROCESSING TIME: {processing_duration:.2f}s",
    flush=True
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


                pdf_bytes = uploaded_file.getvalue()

                document_fingerprint = hashlib.sha256(
                    pdf_bytes + selected_ocr_language.encode("utf-8")
                ).hexdigest()
                current_document_key = document_fingerprint


                if st.session_state.active_document_key != current_document_key:
                    st.session_state.active_document_key = current_document_key
                    st.session_state.selected_chat = None
                    st.session_state.question_text = ""

                if current_document_key not in st.session_state.chat_histories:
                    st.session_state.chat_histories[current_document_key] = []

                document_id = save_document_once(
                    filename="Private document",
                    page_count=len(pages),
                    ocr_language=selected_ocr_language,
                    chunk_count=len(all_chunks),
                    fingerprint=document_fingerprint
                )

                print(
                    f"DATABASE: Document ID {document_id}",
                    flush=True
                )


            print(
            f"MODEL LOADING TIME: {model_duration:.2f}s",
            flush=True
        )




            print(
                f"EMBEDDING TIME: {embedding_duration:.2f}s",
                flush=True
            )

            st.markdown(
                f"""
            <div style="
                background-color: #FFFFFF;
                border: 1px solid #D5E5F5;
                border-radius: 14px;
                padding: 18px;
                margin: 18px 0;
            ">
                <div style="color: #17324D; font-size: 17px; font-weight: 600;">
                    📄 {html.escape(uploaded_file.name)}
                </div>
                <div style="color: #64748B; font-size: 14px; margin-top: 6px;">
                    {len(pages)} pages · Ready for questions
                </div>
                <div style="color: #16803C; font-size: 13px; margin-top: 8px;">
                    ✓ Document ready
                </div>
            </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning(
                "No readable text was extracted from this PDF."
            )


    except Exception:
        st.error(
            "Unable to process this PDF. Please check the file and try again."
    )


st.subheader(
    "Ask DocuMind",
    anchor="ask-documind"
)

# --------------------------------------------------
# SIDEBAR: NEW CHAT AND QUESTION HISTORY
# --------------------------------------------------


with st.sidebar:

    if st.button(
        "+ New Chat",
        key="new_chat_button",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.selected_chat = None
        st.session_state.question_text = ""
        st.rerun()


    st.divider()

    st.markdown("### Question History")


    if current_document_key is not None:

        # Get the history for the currently uploaded PDF
        document_history = st.session_state.chat_histories.get(
            current_document_key,
            []
        )

        if document_history:

            # Display newest questions first
            for index in range(len(document_history) - 1, -1, -1):

                entry = document_history[index]

                # Shorten long questions
                question_preview = entry["question"]

                if len(question_preview) > 45:
                    question_preview = question_preview[:42].rstrip() + "..."

                # Highlight the selected question
                is_selected = (
                    st.session_state.selected_chat
                    == (current_document_key, index)
                )

                if st.button(
                    f"{index + 1}. {question_preview}",
                    key=f"history_{current_document_key}_{index}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                    help=entry["question"]
                ):
                    st.session_state.selected_chat = (
                        current_document_key,
                        index
                    )

                    st.session_state.question_text = entry["question"]
                    st.rerun()

        else:
            st.caption("No questions asked yet.")

    else:
        st.caption("Upload a PDF to view its question history.")


# --------------------------------------------------
# QUESTION INPUT FORM
# --------------------------------------------------

with st.form("question_form", border=False):

    question_input = st.text_input(
        "Your question",
        placeholder="Ask anything about your document...",
        label_visibility="collapsed",
        key="question_text"
    )

    left_space, right_button = st.columns([3, 2])

    with right_button:
        ask_clicked = st.form_submit_button(
            "Ask DocuMind",
            type="primary",
            use_container_width=True
        )







# --------------------------------------------------
# DISPLAY A SAVED ANSWER
# --------------------------------------------------

if not ask_clicked and current_document_key is not None:

    selected_chat = st.session_state.selected_chat

    if (
        selected_chat is not None
        and selected_chat[0] == current_document_key
    ):

        selected_index = selected_chat[1]

        history = st.session_state.chat_histories.get(
            current_document_key,
            []
        )

        if 0 <= selected_index < len(history):



            selected_entry = history[selected_index]

            # Display the user's selected question as a heading
            with st.container(key="user_question_heading"):
                st.caption("YOUR QUESTION")
                st.markdown(f"#### {selected_entry['question']}")

            # Display the saved AI answer
            with st.container(key="saved_answer_card"):
                st.markdown("### DocuMind AI Answer")
                st.markdown(selected_entry["answer"])






            if selected_entry["answer_found"]:

                with st.expander(
                    "View supporting passages",
                    expanded=False
                ):

                    for passage_number, passage in enumerate(
                        selected_entry["passages"],
                        start=1
                    ):

                        with st.container(
                            key=f"saved_passage_{passage_number}",
                            border=True
                        ):

                            left_column, right_column = st.columns(
                                [4, 1],
                                vertical_alignment="center"
                            )

                            with left_column:
                                st.markdown(
                                    f"**Passage {passage_number}**"
                                )

                            with right_column:
                                st.caption(
                                    f"Page {passage['page_number']}"
                                )

                            st.write(passage["text"])



# --------------------------------------------------
# PROCESS A NEW QUESTION
# --------------------------------------------------

if ask_clicked:

    # Clear the previously selected answer
    st.session_state.selected_chat = None

    if uploaded_file is None:

        st.warning("Please upload a PDF.")

    elif not question_input.strip():

        st.warning("Please enter a question.")


    elif embeddings is None:

        st.warning(
            "Semantic search is not ready. "
            "Please check your PDF."
        )

    elif current_document_key is not None and any(
        " ".join(entry["question"].split()).casefold()
        == " ".join(question_input.split()).casefold()
        for entry in st.session_state.chat_histories.get(
            current_document_key, []
        )
    ):

        # Find the existing question in this PDF's history
        history = st.session_state.chat_histories[current_document_key]

        normalized_question = " ".join(
            question_input.split()
        ).casefold()

        for index, entry in enumerate(history):

            saved_question = " ".join(
                entry["question"].split()
            ).casefold()

            if saved_question == normalized_question:

                # Select the previously saved answer

                # Select the previously saved answer
                st.session_state.selected_chat = (
                    current_document_key,
                    index
                )

                # Display the saved answer without calling Gemini
                st.rerun()


    else:

        question_start = time.perf_counter()

        # ------------------------------------------
        # DOCUMENT RETRIEVAL
        # ------------------------------------------

        with st.status(
            "Searching your document...",
            expanded=True
) as answer_status:

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

            # Keyword search
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

            # --------------------------------------
            # COMBINE SEMANTIC AND KEYWORD RESULTS
            # --------------------------------------

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

            # --------------------------------------
            # RERANKING
            # --------------------------------------

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

            # --------------------------------------
            # PREPARE RAG CONTEXT
            # --------------------------------------

            rag_context = prepare_rag_context(
                reranked_passages,
                max_passages=5
            )

            print(
                f"RAG CONTEXT LENGTH: "
                f"{len(rag_context)} characters",
                flush=True
            )

        answer_status.update(
            label="Generating your AI answer...",
            state="running",
            expanded=True
        )


        if not rag_context.strip():

            st.info("No relevant passages were found.")

        else:
            quota_result = None

            if visitor_id is not None:
                quota_result = use_ai_request(
                    visitor_id,
                    visitor_limit=10,
                    global_limit=100
                )


            if visitor_id is None:
                answer_status.update(
                    label="Visitor session not ready",
                    state="complete",
                    expanded=False
                )
                st.warning("Preparing your visitor session. Please try again.")

            elif quota_result == "global_limit":
                answer_status.update(
                    label="Global AI request limit reached",
                    state="complete",
                    expanded=False
                )
                st.warning(
                    "DocuMind has reached its daily AI request limit. "
                    "Please come back tomorrow."
                )

            elif quota_result == "visitor_limit":
                answer_status.update(
                    label="Daily question limit reached",
                    state="complete",
                    expanded=False
                )
                st.warning(
                    "You've reached your daily limit of 10 AI questions. "
                    "Please come back tomorrow."
                )
            elif quota_result == "database_error":
                answer_status.update(
                    label="Service temporarily unavailable",
                    state="error",
                    expanded=False
                )

                st.warning(
                    "DocuMind is temporarily unavailable. "
                    "Please try again in a few moments."
                )


            elif quota_result == "allowed":
                try:

                    print(
                        "GEMINI CALL STARTED",
                        flush=True
                    )

                    gemini_start = time.perf_counter()


                    ai_answer, answer_found = generate_answer(
                        question_input,
                        rag_context
                    )
                    answer_status.update(
                        label="Answer generated successfully",
                        state="complete",
                        expanded=False
                    )

                    # Measure performance before rerunning
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

                    # ----------------------------------
                    # SAVE ANSWER TO CONVERSATION HISTORY
                    # ----------------------------------

                    if current_document_key is not None:

                        st.session_state.chat_histories[
                            current_document_key
                        ].append({
                            "question": question_input.strip(),
                            "answer": ai_answer,
                            "answer_found": answer_found,
                            "passages": [
                                passage.copy()
                                for passage in reranked_passages
                            ]
                        })


                        st.session_state.selected_chat = (
                            current_document_key,
                            len(
                                st.session_state.chat_histories[
                                    current_document_key
                                ]
                            ) - 1
                        )


                        st.rerun()

                    else:

                        with st.container(key="ai_answer_card"):
                            st.markdown("### DocuMind AI Answer")
                            st.markdown(ai_answer)

                        if answer_found:

                            with st.expander(
                                "View supporting passages",
                                expanded=False
                            ):

                                for passage in reranked_passages:

                                    st.markdown(
                                        f"**Page {passage['page_number']}**"
                                    )
                                    st.write(passage["text"])
                                    st.divider()

                # ----------------------------------
                # GEMINI ERROR HANDLING
                # ----------------------------------

                except Exception as error:
                    answer_status.update(
                        label="Unable to generate an answer",
                        state="error",
                        expanded=False
                    )

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
                           "We couldn't complete your request. "
                           "Please check your internet connection "
                           "and try again. If the problem continues, "
                           "please try again later."
                       )

                       print(
                           f"Answer processing error: "
                           f"{type(error).__name__}",
                           flush=True
                       )

            else:
                answer_status.update(
                    label="Service temporarily unavailable",
                    state="error",
                    expanded=False
                )

                st.warning(
                    "We couldn't verify your request at the moment. "
                    "Please try again later."
                )

                print(
                    "Unexpected quota result received.",
                    flush=True
                )
