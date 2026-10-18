from semantic_search import(
    load_embedding_model,
    create_embeddings,
    search_similar_chunks
)


chunks = [
    {
        "page_number": 1,
        "text": "Artificial intelligence helps automate repetitive tasks."
    },
    {
        "page_number": 2,
        "text": "The company increased its revenue by 18%."
    },
    {
        "page_number": 3,
        "text": "Employees receive 25 days of annual leave."
    },
    {
        "page_number": 4,
        "text": "ارتفعت إيرادات الشركة بنسبة ١٨٪ خلال العام."
    }
]


print("Loading embedding model...")
model = load_embedding_model()

print("Creating embeddings...")
embeddings = create_embeddings(model, chunks)

question = "How much did the company's revenue increase?"

print("\nQuestion:", question)

results = search_similar_chunks(
    model,
    question,
    chunks,
    embeddings,
    top_k=3
)

for result in results:
    print("\nPage:", result["page_number"])
    print("Score:", round(result["score"], 3))
    print("Text:", result["text"])
