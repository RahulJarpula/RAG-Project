# adaptive_splitter.py
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_document_aware_splitter(chunk_size=500, chunk_overlap=100):
    """
    Returns a RecursiveCharacterTextSplitter that uses document-aware separators
    like double newlines, newlines, sentence breaks, etc.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def get_splitter_by_complexity(query_complexity: str):
    """
    Returns different splitters based on the estimated query complexity level.
    - high → large context, wider chunks
    - medium → default
    - low → smaller chunks
    """
    if query_complexity == "high":
        return get_document_aware_splitter(chunk_size=800, chunk_overlap=100)
    elif query_complexity == "medium":
        return get_document_aware_splitter(chunk_size=500, chunk_overlap=50)
    else:  # low
        return get_document_aware_splitter(chunk_size=300, chunk_overlap=0)


def adaptive_split_documents(docs: list[Document], query_complexity: str = "medium") -> list[Document]:
    """
    Splits a list of Documents using an adaptive splitter based on query complexity.
    Returns a flattened list of smaller Documents.
    """
    splitter = get_splitter_by_complexity(query_complexity)
    split_docs = splitter.split_documents(docs)
    return split_docs
