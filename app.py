
import streamlit as st
import time 

from pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    split_text_into_chunks,
    split_text_by_sentences
)

from semantic_search import (
    load_embedding_model,
    create_embeddings,
    search_similar_chunks
)


st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)



@st.cache_resource # so streamlit reuses the loaded model 
def get_embedding_model():
    return load_embedding_model()


@st.cache_data(show_spinner=False)# streamlit can use existing embeddings rather than to calculate them again
def get_document_embeddings(_model, chunk_texts):
    chunks = [{"text": text} for text in chunk_texts]
    return create_embeddings(_model, chunks)



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
        pages = extract_text_from_pdf(uploaded_file)

        for page in pages:
            cleaned_text = clean_text(page["text"])

            if cleaned_text:
                page_chunks = split_text_by_sentences(
                    cleaned_text,
                    chunk_size=500,
                    overlap=1
                )

                for chunk_number, chunk in enumerate(
                    page_chunks,
                    start=1
                ):
                    all_chunks.append(
                        {
                            "page_number": page["page_number"],
                            "chunk_number": chunk_number,
                            "text": chunk,
                        }
                    )

        st.info(f"Created {len(all_chunks)} text chunks.")

        with st.expander("Preview extracted chunks"):
            for chunk in all_chunks:
                st.write(
                    f"Page {chunk['page_number']} — "
                    f"Chunk {chunk['chunk_number']}"
                )
                st.text(chunk["text"])
                st.divider()

        st.success(
            f"Successfully processed {len(pages)} pages!"
        )

        for page in pages:
            with st.expander(f"Page {page['page_number']}"):
                if page["text"].strip():
                    st.text(page["text"])
                else:
                    st.warning(
                        "No readable text found on this page. "
                        "It may be a scanned document."
                    )

                     
        if all_chunks:
            with st.spinner("Preparing semantic search..."):

                model_start = time.perf_counter()
                model = get_embedding_model()
                model_duration = time.perf_counter() - model_start

                chunk_texts = tuple(
                    chunk["text"] for chunk in all_chunks
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
                f"Model loading time: {model_duration:.2f} seconds"
            )

            st.write(
                f"Embedding creation time: {embedding_duration:.2f} seconds"
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
        with st.spinner("Searching your document..."):
            model = get_embedding_model()

            results = search_similar_chunks(
                model=model,
                question=question_input,
                chunks=all_chunks,
                embeddings=embeddings,
                top_k=3
            )

        st.subheader("Most relevant passages")

        for result in results:
            st.markdown(
                f"**Page {result['page_number']}** "
                f"— Similarity: {result['score']:.3f}"
            )

            st.write(result["text"])
            st.divider()
