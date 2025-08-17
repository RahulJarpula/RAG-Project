from langchain.chains import RetrievalQA
from files.watsonx_llm import get_llm
from files.retriever_setup import get_configurable_retriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from files.retriever_setup import load_chunk_parent_map
from files.context_windowing import retrieve_parents, truncate_chunks_to_fit
from files.watsonx_llm import get_llm
from langchain.chains import RetrievalQA
from files.retriever_setup import get_configurable_retriever 
import numpy as np
import re

model = SentenceTransformer('all-MiniLM-L6-v2')
llm = get_llm()

def get_embedding(text: str):
    return model.encode([text])[0]

def load_rag_chain(use_hybrid=False):
    retriever = get_configurable_retriever(k=4,use_hybrid=use_hybrid)
    chunk_map = load_chunk_parent_map()  # Load parent mapping

    def run(query: str):
        retrieved_chunks = retriever.get_relevant_documents(query)
        #retrieved_chunks = retriever.invoke(query)
        parent_docs = retrieve_parents(retrieved_chunks, chunk_map)
        final_docs = truncate_chunks_to_fit(parent_docs)

        # LLM chain, manually injecting docs
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,  # Not needed
            return_source_documents=True  # optional
        )
        return qa_chain.run(input_documents=final_docs, question=query)
        # result = qa_chain.invoke({"input_documents": final_docs, "question": query})
        # return {
        #     "answer": result["result"],  # RetrievalQA stores final answer in 'result'
        #     "source_documents": result["source_documents"]
        # }
    
    return run

intent_prompt = PromptTemplate.from_template("""
Classify the user's intent into one of the following categories:

- fact_lookup
- comparison
- procedure
- list_generation
- summarization
- exploration

Respond ONLY with one of the above.

Query: "{query}"
Intent:
""")


intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# PROMPT: Query Reformulation
reformulation_prompt = PromptTemplate.from_template("""
You are a search query optimizer.

Your job is to generate 3 alternate versions of the given user query that are:
- On the exact same topic
- Focused only on what the user asked
- Retrieval-friendly (longer, more explicit)
- Do NOT introduce new subtopics or drift to adjacent domains

Original Query: "{query}"

Rewritten Queries:
1.
2.
3.
""")


reformulation_chain = LLMChain(llm=llm, prompt=reformulation_prompt)
#embedder = OpenAIEmbeddings()

def classify_and_reformulate(query: str):
    # 1. Intent classification
    intent_output = intent_chain.invoke({"query": query})
    #intent_text = intent_output["text"]
    intent_text = intent_output["text"].strip().lower()
    intent = re.search(r"\b(fact_lookup|comparison|procedure|list_generation|summarization|exploration)\b", intent_text)
    intent = intent.group(1) if intent else "exploration"


    # 2. Reformulation – only for complex intents
    if intent in ["comparison", "summarization", "procedure", "exploration"]:
        reform_output = reformulation_chain.invoke({"query": query})
        reform_lines = reform_output["text"].strip().split("\n")

        reformulated = []
        for line in reform_lines:
            if line.strip().startswith(("1.", "2.", "3.")):
                reformulated.append(line[2:].strip())
                if len(reformulated) == 3:
                    break
        
        # Similarity filtering
        orig_vector = get_embedding(query)
        filtered_rewrites = []
        for r in reformulated:
            vec = get_embedding(r)
            sim = cosine_similarity([orig_vector], [vec])[0][0]
            if sim >= 0.75:
                filtered_rewrites.append(r)

        if not filtered_rewrites:
            filtered_rewrites = reformulated[:1]  # fallback
    else:
        # No reformulation needed
        filtered_rewrites = [query]

    print("Current Query:", query)
    print("Prompt being sent:", filtered_rewrites)

    return intent, filtered_rewrites
