# retriever_setup.py

from langchain_community.vectorstores import FAISS
from typing import List, Any
from files.hybrid_retriever import HybridRetriever
from files.build_vectorstore import load_documents
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
import pickle
import json
import os

CHUNK_MAP_PATH = "vectorstore/chunk_parent_map.json"
VECTORSTORE_PATH = "vectorstore/faiss_index.pkl"

def load_vectorstore(path=VECTORSTORE_PATH):
    """
    Loads the FAISS vectorstore from disk.
    """
    with open(path, "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

def load_chunk_parent_map(path=CHUNK_MAP_PATH):
    """
    Loads the chunk-to-parent mapping from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Parent mapping file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def get_retriever_and_chunk_map():
    """
    Returns both retriever and parent chunk mapping for downstream usage.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chunk_map = load_chunk_parent_map()
    return retriever, chunk_map


class HybridLangChainRetriever(BaseRetriever):
    hybrid_retriever: Any

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.hybrid_retriever.hybrid_search(query, top_k=5)
        return [
            Document(page_content=doc, metadata={"score": score, "source": doc_id})
            for doc_id, doc, score in results
        ]
    
def get_configurable_retriever(k=4,use_hybrid=False):
    if not use_hybrid:
        with open("vectorstore/faiss_index.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    else:
        #from files.build_vectorstore import load_documents
        docs = load_documents(data_dir="data")
        corpus_texts = [doc.page_content for doc in docs]
        corpus_ids = [f"doc_{i}" for i in range(len(docs))]

        hybrid = HybridRetriever(corpus_texts, corpus_ids)

        # ✅ construct the retriever safely via Pydantic hook
        return HybridLangChainRetriever.model_construct(hybrid_retriever=hybrid)

def load_hybrid_retriever(k=5):
    """
    Returns the full HybridLangChainRetriever instance using stored documents.
    """
    docs = load_documents(data_dir="data")
    corpus_texts = [doc.page_content for doc in docs]
    corpus_ids = [f"doc_{i}" for i in range(len(docs))]

    hybrid = HybridRetriever(corpus_texts, corpus_ids)
    return HybridLangChainRetriever.model_construct(hybrid_retriever=hybrid)

