# run_rag_on_testset.py
# Step 3: Run full RAG pipeline on each test question and record outputs

import json
from files.rag_chain import load_rag_chain  # Your full RAG pipeline loader

INPUT_TESTSET = "evaluation/test_set.json"
OUTPUT_PATH = "evaluation/rag_outputs.json"

# Load test set
test_examples = json.load(open(INPUT_TESTSET))

# Load RAG pipeline
rag_chain = load_rag_chain()

all_results = []

for i, row in enumerate(test_examples):
    question = row["question"]
    gold_answer = row["ground_truth"]
    source_chunk = row["source_chunk"]  # Not used here but useful for comparison/debugging

    print(f"\n>>> [{i+1}/{len(test_examples)}] Question: {question}")

    try:
        result = rag_chain({"query": question})

        # result must include 'answer' and 'source_documents'
        generated_answer = result.get("answer", "")
        retrieved_docs = result.get("source_documents", [])
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        all_results.append({
            "question": question,
            "answer": generated_answer,
            "contexts": retrieved_contexts,
            "ground_truth": gold_answer
        })

    except Exception as e:
        print(f"❌ Error for question {i}: {e}")

# Save outputs
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Saved {len(all_results)} results to {OUTPUT_PATH}")
