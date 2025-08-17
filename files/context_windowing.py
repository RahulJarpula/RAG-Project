# context_windowing.py

from langchain.schema import Document

def build_chunk_parent_mapping(chunks_with_metadata):
    """
    Create a mapping from chunk_id to parent section.
    Expects metadata with keys: 'chunk_id', 'parent_id', 'parent_text'
    """
    mapping = {}
    for chunk in chunks_with_metadata:
        mapping[chunk['chunk_id']] = {
            'parent_id': chunk['parent_id'],
            'parent_text': chunk['parent_text']
        }
    return mapping


def retrieve_parents(retrieved_chunks, chunk_to_parent_map):
    """
    Replace retrieved summary chunks with their full parent documents.
    Avoids duplicates.
    """
    seen_parents = set()
    expanded_docs = []

    for doc in retrieved_chunks:
        chunk_id = doc.metadata.get('chunk_id')
        if chunk_id in chunk_to_parent_map:
            parent_info = chunk_to_parent_map[chunk_id]
            parent_id = parent_info['parent_id']
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                expanded_docs.append(Document(
                    page_content=parent_info['parent_text'],
                    metadata={'parent_id': parent_id}
                ))

    #print("Retrieved chunk_id:", chunk_id)
    #print("Available chunk_map keys (sample):", list(chunk_map.keys())[:5])
    return expanded_docs


def truncate_chunks_to_fit(chunks, max_token_budget=4096, model_max=8192,
                           prompt_tokens=500, response_tokens=1500):
    """
    Selects as many chunks as possible to fit within the remaining LLM token budget.
    """
    token_limit = model_max - prompt_tokens - response_tokens
    selected = []
    current_total = 0

    for chunk in chunks:
        tokens = len(chunk.page_content.split())  # approx token count
        if current_total + tokens > token_limit:
            break
        selected.append(chunk)
        current_total += tokens

    return selected
