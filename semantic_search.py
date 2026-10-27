
import numpy as np
import re

from pdf_processor import split_into_sentences


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"



def load_embedding_model():
    import time

    start_import = time.perf_counter()

    from sentence_transformers import SentenceTransformer

    import_duration = time.perf_counter() - start_import

    print(
        f"SentenceTransformer import time: {import_duration:.2f} seconds",
        flush=True
    )

    start_loading = time.perf_counter()

    model = SentenceTransformer(MODEL_NAME)

    loading_duration = time.perf_counter() - start_loading

    print(
        f"Model initialization time: {loading_duration:.2f} seconds",
        flush=True
    )

    return model



def load_reranker():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        RERANKER_NAME,
        device="cpu"
    )

    return model


def create_embeddings(model, chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def search_similar_chunks(
    model,
    question,
    chunks,
    embeddings,
    top_k=10
):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    similarity_scores = np.dot(
        embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(similarity_scores)[::-1]

    results = []

    for index in ranked_indices:
        chunk = chunks[index]
        text = chunk["text"].strip()

        if len(text.split()) < 4:
            continue

        results.append({
            "chunk_index": int(index),
            "page_number": chunk["page_number"],
            "text": text,
            "section_heading": chunk.get("section_heading"),
            "score": float(similarity_scores[index])
        })

        if len(results) >= top_k:
            break

    return results

def rerank_passages(reranker, question, results, top_k=5):
    if not results:
        return []

    pairs = [
        (question, result["text"])
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        batch_size=1,
        show_progress_bar=False
    )

    ranked_indices = np.argsort(scores)[::-1]

    reranked_results = []

    for index in ranked_indices[:top_k]:
        result = results[int(index)].copy()
        result["rerank_score"] = float(scores[index])
        reranked_results.append(result)

    return reranked_results




def normalize_sentence(text):
    return " ".join(text.split())


def build_contextual_candidates(results, all_chunks):
    candidates = []
    seen = set()

    for result in results:
        chunk_index = result["chunk_index"]
        chunk = all_chunks[chunk_index]

        page_number = chunk["page_number"]
        heading = chunk.get("section_heading")
        text = chunk["text"]


        context_parts = []

        if heading:
             context_parts.append(
                f"Section: {heading}"
             )

        sentences = split_into_sentences(text)

        for sentence in sentences:
            sentence = normalize_sentence(sentence)

            if len(sentence.split()) < 5:
                continue

            key = (page_number, sentence)

            if key in seen:
                continue

            seen.add(key)

            context = "\n".join(context_parts)

            if context:
                rerank_text = (
                    f"Context:\n{context}\n"
                    f"Sentence: {sentence}"
                )
            else:
                rerank_text = f"Sentence: {sentence}"

            candidates.append({
                "text": sentence,
                "page_number": page_number,
                "rerank_text": rerank_text
            })

    return candidates


def find_best_sentence(
    model,
    question,
    results,
    all_chunks=None,
    reranker=None
):
    if not results:
        return None

    if all_chunks is None or reranker is None:
        sentences = []
        page_numbers = []

        for result in results:
            for sentence in split_into_sentences(result["text"]):
                sentence = normalize_sentence(sentence)

                if len(sentence.split()) >= 5:
                    if sentence not in sentences:
                        sentences.append(sentence)
                        page_numbers.append(result["page_number"])

        if not sentences:
            return None

        question_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        sentence_embeddings = model.encode(
            sentences,
            normalize_embeddings=True
        )

        scores = np.dot(
            sentence_embeddings,
            question_embedding
        )

        best_index = int(np.argmax(scores))

        return {
            "text": sentences[best_index],
            "page_number": page_numbers[best_index]
        }

    candidates = build_contextual_candidates(
        results,
        all_chunks
    )

    if not candidates:
        return None

    pairs = [
        (question, candidate["rerank_text"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=1,
        show_progress_bar=False
    )

    ranked_indices = np.argsort(scores)[::-1]

    print("\nQUESTION:", question)
    print("\nTOP CONTEXTUAL CANDIDATES:")

    for index in ranked_indices[:5]:
        print(f"Score: {float(scores[index]):.4f}")
        print(f"Answer: {candidates[index]['text']}")
        print("-" * 50)

    best_index = int(ranked_indices[0])

    return {
        "text": candidates[best_index]["text"],
        "page_number": candidates[best_index]["page_number"]
    }


def contains_numbered_list(text):
    numbered_items = re.findall(
        r"(?m)^\s*\d+[.)]\s+",
        text
    )

    return len(numbered_items) >= 2
