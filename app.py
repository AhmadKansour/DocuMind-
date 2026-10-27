
import streamlit as st
import time

from pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    split_text_into_chunks,
    split_text_into_sections,
    split_text_by_sentences,
    is_section_heading
)

from semantic_search import (
    load_embedding_model,
    load_reranker,
    rerank_passages,
    create_embeddings,
    search_similar_chunks,
    find_best_sentence,
    contains_numbered_list
)
from rag_pipeline import prepare_rag_context

from llm_provider import generate_answer


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
def get_document_embeddings(_model, chunk_texts):
    print("EMBEDDING CACHE MISS: Generating embeddings", flush=True)

    start = time.perf_counter()

    chunks = [{"text": text} for text in chunk_texts]

    embeddings = create_embeddings(_model, chunks)

    duration = time.perf_counter() - start

    print(
        f"ACTUAL EMBEDDING COMPUTATION: {duration:.2f} seconds",
        flush=True
    )

    return embeddings



left, center, right = st.columns([2, 1, 2])

with center:
    st.title("DocuMind")

st.subheader("Your AI Document Assistant")
st.write("Upload your PDF and ask about its content")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

all_chunks = []
embeddings = None

if uploaded_file is not None:

    try:
        processing_start = time.perf_counter()

        pages = extract_text_from_pdf(uploaded_file)
        print("PDF TITLE:", repr(pages[0]["document_title"]) if pages else "")
        if pages:
            print("FIRST PAGE PREVIEW:")
            print(repr(pages[0]["text"][:400]))


        for page in pages:

            cleaned_text = clean_text(page["text"])

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
                all_chunks.append(
                    {
                        "page_number": page["page_number"],
                        "chunk_number": chunk_number,
                        "text": chunk_text,
                        "section_heading": None
                    }
                )


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
                    page_chunks.append(
                        {
                            "text": chunk,
                            "section_heading": heading
                        }
                    )

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1
            ):

                all_chunks.append(
                    {
                        "page_number": page["page_number"],
                        "chunk_number": chunk_number,
                        "text": chunk["text"],
                        "section_heading": chunk["section_heading"]
                    }
                )

        print("\nDOCUMENT CHUNK METADATA:")

        print(f"DOCUMENT PROCESSING COMPLETE: {len(all_chunks)} chunks", flush=True)
        processing_duration = time.perf_counter() - processing_start

        st.write(
            f"PDF extraction and chunking time: "
            f"{processing_duration:.2f} seconds"
        )
        st.info(f"Created {len(all_chunks)} text chunks.")

        #with st.expander("Preview extracted chunks"):

            #for chunk in all_chunks:

              #  st.write(
                  #  f"Page {chunk['page_number']} — "
                  #  f"Chunk {chunk['chunk_number']}"
               # )

               # st.text(chunk["text"])
               # st.divider()

        st.success(
            f"Successfully processed {len(pages)} pages!"
        )

        #for page in pages:

           # with st.expander(
                #f"Page {page['page_number']}"
           # ):

               # if page["text"].strip():
                    #st.text(page["text"])

                #else:
                  #  st.warning(
                       # "No readable text found on this page. "
                       # "It may be a scanned document."
                   # )

        if all_chunks:

            with st.spinner("Preparing semantic search..."):

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

            st.success("Semantic search is ready!")

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
        st.error("Unable to process this PDF.")
        st.exception(error)


question_input = st.text_input(
    "What would you like to know about your document?"
)

ask_clicked = st.button("Ask DocuMind")


if ask_clicked:
    print(
        f"ASK CLICKED | question={question_input!r}",
        flush=True
    )

    if uploaded_file is None:
        st.warning("Please upload a PDF.")

    elif not question_input.strip():
        st.warning("Please enter a question.")

    elif embeddings is None:
        st.warning(
            "Semantic search is not ready. "
            "Please check your PDF."
        )

    else:
        # Only valid questions reach this section.
        with st.spinner("Searching your document..."):

            model = get_embedding_model()


            results = search_similar_chunks(
                model,
                question_input,
                all_chunks,
                embeddings,
                top_k=30
            )

            print("\n--- SEMANTIC SEARCH RESULTS ---", flush=True)

            for index, result in enumerate(results, start=1):
                print(
                    f"\nResult {index} | Page {result['page_number']}",
                    flush=True
                )
                print(result["text"], flush=True)

            reranker = get_reranker()


            reranked_passages = rerank_passages(
                reranker,
                question_input,
                results,
                top_k=5
            )

            rag_context = prepare_rag_context(
                reranked_passages,
                max_passages=5
            )

        if not rag_context.strip():
            st.info(
                "No relevant passages were found."
            )

        else:
            with st.spinner("Generating AI answer..."):
                try:
                    print(
                        "GEMINI CALL STARTED",
                        flush=True
                    )

                    ai_answer = generate_answer(
                        question_input,
                        rag_context
                    )

                    st.subheader("DocuMind AI Answer")
                    st.write(ai_answer)

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

            with st.expander("View supporting passages"):

                for passage in reranked_passages:

                    st.markdown(
                        f"**Page {passage['page_number']}**"
                    )

                    st.write(passage["text"])

                    st.divider()
