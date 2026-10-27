
def prepare_rag_context(results, max_passages=5):
    context_parts = []
    seen = set()

    for result in results:
        text = result["text"].strip()
        page_number = result["page_number"]

        if not text:
            continue

        key = (page_number, text)

        if key in seen:
            continue

        seen.add(key)

        context_parts.append(
            f"[Source: Page {page_number}]\n{text}"
        )

        if len(context_parts) >= max_passages:
            break

    return "\n\n---\n\n".join(context_parts)
