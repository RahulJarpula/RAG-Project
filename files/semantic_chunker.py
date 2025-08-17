# semantic_chunker.py

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def semantic_chunking(text: str, threshold: float = 0.75):
    """
    Splits a document into semantically coherent chunks using embedding similarity.
    Sentences are grouped together until similarity with previous drops below the threshold.
    """
    if not text.strip():
        return []

    # Naively split into sentence-like chunks
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) == 0:
        return []

    model = HuggingFaceEmbeddings()
    embeddings = model.embed_documents(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i]], [embeddings[i - 1]]
        )[0][0]

        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(". ".join(current_chunk).strip())
            current_chunk = [sentences[i]]

    # Add last chunk
    if current_chunk:
        chunks.append(". ".join(current_chunk).strip())

    return chunks
