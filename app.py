import streamlit as st
from pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    split_text_into_chunks,
    split_text_by_sentences
)



st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


left, center, right = st.columns([2, 1, 2])
with center:
    title = st.title(" DocuMind")

st.subheader(" Your AI Document Assistant")

st.write("Upload your PDF and ask about its content")


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


if uploaded_file is not None:
    try:
        pages = extract_text_from_pdf(uploaded_file)

        all_chunks = []

        for page in pages:
            cleaned_text = clean_text(page["text"])

            if cleaned_text:
                page_chunks = split_text_by_sentences(cleaned_text,chunk_size=500, overlap=1)

                for chunk_number, chunk in enumerate(page_chunks, start=1):
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

        st.success(f"Successfully processed {len(pages)} pages!")

        for page in pages:
            with st.expander(f"Page {page['page_number']}"):
                if page["text"].strip():
                    st.text(page["text"])
                else:
                    st.warning(
                        "No readable text found on this page. "
                        "It may be a scanned document."
                    )

    except Exception:
        st.error("Unable to read this PDF.")


question_input = st.text_input(" What would you like to know about your documents ")


if st.button("Ask DocuMind"):
    if uploaded_file is None:
        st.warning("Please upload a pdf ")
    elif not question_input.strip():
        st.warning("Please enter a question")
    else:
        st.info("AI functionality coming soon ")


        st.subheader("Arabic Language Test")
