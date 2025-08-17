import os
import pickle

from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. Load all documents from /data
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

# 2. Split text into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    return splitter.split_documents(docs)

# 3. Embed and build vectorstore
def build_vectorstore(docs, save_path="vectorstore/faiss_index.pkl"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # or your Watsonx embedding
    vectorstore = FAISS.from_documents(docs, embedding_model)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(f"✅ Vectorstore saved to: {save_path}")

if __name__ == "__main__":
    print("📥 Loading documents...")
    docs = load_documents()
    print(f"📄 Loaded {len(docs)} documents")

    print("✂️ Splitting into chunks...")
    split_docs = split_documents(docs)
    print(f"🧩 Created {len(split_docs)} chunks")

    print("🔗 Building embeddings and FAISS index...")
    build_vectorstore(split_docs)
