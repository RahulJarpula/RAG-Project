# upgraded_vectorstore_builder.py

import os
import json
import pickle
from uuid import uuid4

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from files.adaptive_splitter import get_document_aware_splitter
from files.semantic_chunker import semantic_chunking

def load_documents(data_dir="data"):
    loaders = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            loaders.append(TextLoader(file_path))
        elif filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(file_path))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def split_doc_into_sections(document_text):
    return [section.strip() for section in document_text.split("\n\n") if section.strip()]

def process_document(doc_text, doc_id, use_semantic=False):
    sections = split_doc_into_sections(doc_text)

    all_chunks = []
    chunk_parent_map = {}

    for i, section_text in enumerate(sections):
        parent_id = f"{doc_id}_section_{i}"
        splitter = get_document_aware_splitter(chunk_size=200)

        if use_semantic:
            sub_chunks = semantic_chunking(section_text)
        else:
            sub_chunks = splitter.split_text(section_text)

        for chunk in sub_chunks:
            chunk_id = str(uuid4())
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "parent_text": section_text
                    }
                )
            )
            chunk_parent_map[chunk_id] = {
                "parent_id": parent_id,
                "parent_text": section_text
            }

    return all_chunks, chunk_parent_map

def process_and_build_vectorstore(docs, use_semantic=False, save_path="vectorstore/faiss_index.pkl",
                                   map_path="vectorstore/chunk_parent_map.json"):
    all_chunks = []
    master_chunk_map = {}

    for i, doc in enumerate(docs):
        chunks, chunk_map = process_document(doc.page_content, doc_id=f"doc_{i}", use_semantic=use_semantic)
        all_chunks.extend(chunks)
        master_chunk_map.update(chunk_map)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embedding_model)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorstore, f)

    with open(map_path, "w") as f:
        json.dump(master_chunk_map, f)

    print(f"✅ Vectorstore saved to: {save_path}")
    print(f"✅ Parent mapping saved to: {map_path}")


if __name__ == "__main__":
    #from files.build_vectorstore import load_documents, process_and_build_vectorstore

    #print("📥 Loading documents...")
    docs = load_documents("data")  # Assumes your data is in /data/
    #print(f"📄 Loaded {len(docs)} documents")

    #print("🔗 Building vectorstore and chunk map...")
    process_and_build_vectorstore(docs, use_semantic=True)  # or False

