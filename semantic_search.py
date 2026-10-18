
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


def search_similar_chunks(model, question, chunks, embeddings, top_k=3):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    similarity_scores = np.dot(
        embeddings,
        question_embedding
    )

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "page_number": chunks[index]["page_number"],
            "text": chunks[index]["text"],
            "score": float(similarity_scores[index])
        })

    return results
