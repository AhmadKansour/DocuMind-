
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def create_embeddings(model, chunks):#This creates the embedding model object and returns it.
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
    top_k=3
                     ):
    # Convert the user's question into an embedding.
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Calculate similarity between the question and each chunk.
    similarity_scores = np.dot(
        embeddings,
        question_embedding
    )

    # Rank all chunks from highest to lowest similarity.
    ranked_indices = np.argsort(similarity_scores)[::-1]

    results = []

    for index in ranked_indices:
        chunk = chunks[index]
        text = chunk["text"].strip()

        # Skip very short chunks that are unlikely to be useful.
        if len(text.split()) < 8:
            continue

        results.append(
            {
                "page_number": chunk["page_number"],
                "text": text,
                "score": float(similarity_scores[index])
            }
        )

        # Stop after collecting the requested number of results.
        if len(results) >= top_k:
            break

    return results
