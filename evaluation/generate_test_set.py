import os
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from files.watsonx_llm import get_llm
from files.retriever_setup import load_hybrid_retriever  # Feature 1
from files.adaptive_splitter import adaptive_split_documents  # Feature 3
from files.semantic_chunker import semantic_chunking  # Feature 3

from langchain.schema import Document

# -------------- CONFIG -----------------------
CHUNK_SAMPLE_SIZE = 5
QUESTIONS_PER_CHUNK = 5
OUTPUT_PATH = "evaluation/test_set.json"

# -------------- Load Retriever ----------------
retriever = load_hybrid_retriever()

# -------------- Get Sample Chunks -------------
sample_docs = retriever.get_relevant_documents("sample")
sample_docs = sample_docs[:CHUNK_SAMPLE_SIZE]  # trim if needed
sample_chunks = []

# Redo chunking using Adaptive + Semantic chunking (Feature 3)
for doc in sample_docs:
    adaptive_chunks = adaptive_split_documents([doc])
    for chunk in adaptive_chunks:
        semantically_chunked = semantic_chunking(chunk.page_content)
        for small_chunk in semantically_chunked:
            sample_chunks.append(small_chunk)
            if len(sample_chunks) >= CHUNK_SAMPLE_SIZE:
                break
        if len(sample_chunks) >= CHUNK_SAMPLE_SIZE:
            break

# -------------- Setup LLM Chain ----------------
llm = get_llm()

qa_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Based ONLY on the text below, generate {n} numbered Q&A pairs.

Use this format only:
Question 1: <...>
Answer 1: <...>
...

Text:
\"\"\"{context}\"\"\"
"""
)

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# -------------- Generate QA Pairs ----------------
all_examples = []
for i, chunk in enumerate(sample_chunks):
    print(f">>> Generating QA pairs from chunk {i+1}/{CHUNK_SAMPLE_SIZE}...")
    result = qa_chain.run({"context": chunk, "n": QUESTIONS_PER_CHUNK})

    lines = result.strip().split("\n")
    questions, answers = [], []
    for line in lines:
        if line.lower().startswith("question"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                questions.append(parts[1].strip())
        elif line.lower().startswith("answer"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                answers.append(parts[1].strip())


    for q, a in zip(questions, answers):
        if q.strip() and a.strip():  # ✅ only log valid entries
            all_examples.append({
                "question": q,
                "ground_truth": a,
                "source_chunk": chunk
            })

# -------------- Save QA Pairs --------------------
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_examples, f, indent=2)

print(f"\n✅ Saved {len(all_examples)} QA pairs to {OUTPUT_PATH}")
