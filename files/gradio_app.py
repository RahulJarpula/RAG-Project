# gradio_app.py

import gradio as gr
from files.rag_chain import classify_and_reformulate
from files.retriever_setup import load_chunk_parent_map, get_configurable_retriever
from files.context_windowing import retrieve_parents, truncate_chunks_to_fit
from files.watsonx_llm import get_llm

llm = get_llm()
chunk_map = load_chunk_parent_map()

# response style instructions per intent
instruction_map = {
    "summarization": "Summarize the content in 3–5 lines.",
    "comparison": "Provide a side-by-side comparison of concepts.",
    "procedure": "List all the steps in numbered form.",
    "list_generation": "Provide a bullet list.",
    "fact_lookup": "Give a direct and concise answer.",
    "exploration": "Write an exploratory paragraph explaining the concept."
}

def ask_question(query, use_hybrid):
    # Step 1: classify + rewrite
    intent, rewritten_queries = classify_and_reformulate(query)

    # Step 2: choose retriever based on intent
    if intent == "summarization":
        retriever = get_configurable_retriever(k=8, use_hybrid=use_hybrid)
    elif intent == "fact_lookup":
        retriever = get_configurable_retriever(k=2, use_hybrid=use_hybrid)
    else:
        retriever = get_configurable_retriever(k=4, use_hybrid=use_hybrid)

    # Step 3: retrieve across rewritten queries
    all_chunks = []
    for q in rewritten_queries:
        retrieved = retriever.invoke(q)
        all_chunks.extend(retrieved)

    # Step 4: parent doc expansion + truncation
    parent_docs = retrieve_parents(all_chunks, chunk_map)
    final_docs = truncate_chunks_to_fit(parent_docs)

    # Step 5: combine docs and form prompt
    combined = "\n\n".join(doc.page_content for doc in final_docs)
    task_instruction = instruction_map.get(intent, "Explore the content and respond appropriately.")

    prompt = f"""{task_instruction}

    {combined}

    Answer: {query}
    Be concise. Do not repeat content.
    """


    # Step 6: generate response
    return llm.invoke(prompt)


# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 RAG Chatbot with Query Reformulation + Contextual Windowing")
    query = gr.Textbox(label="Enter your question", placeholder="e.g. Compare LangChain and Haystack")
    toggle = gr.Checkbox(label="Use Hybrid Retrieval (BM25 + Dense)", value=True)
    output = gr.Markdown()
    submit_btn = gr.Button("Ask")

    submit_btn.click(fn=ask_question, inputs=[query, toggle], outputs=output)

# Launch
if __name__ == "__main__":
    demo.launch()
