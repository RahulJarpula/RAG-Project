# files/hybrid_retriever.py

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, corpus_texts, corpus_ids):
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids

        # Load models
        self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # # Precompute embeddings and BM25 index
        # self.corpus_embeddings = self.bi_encoder.encode(self.corpus_texts, convert_to_tensor=True)
        # tokenized_corpus = [doc.split() for doc in corpus_texts]
        # self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query, top_k=5):
        # Precompute embeddings and BM25 index
        self.corpus_embeddings = self.bi_encoder.encode(self.corpus_texts, convert_to_tensor=True)
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        tokenized_corpus = [doc.split() for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # bm25;
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_idxs = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_docs = [(i, self.corpus_texts[i], bm25_scores[i]) for i in bm25_top_idxs]

        #query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        cos_scores = cos_sim(query_embedding, self.corpus_embeddings)[0]
        dense_top_idxs = np.argsort(-cos_scores.cpu())[:top_k]
        dense_docs = [(i, self.corpus_texts[i], cos_scores[i].item()) for i in dense_top_idxs]

        combined_ids = {i for i, _, _ in bm25_docs + dense_docs}
        candidates = [(i, self.corpus_texts[i]) for i in combined_ids]

        inputs = [[query, doc] for _, doc in candidates]
        cross_scores = self.cross_encoder.predict(inputs)
        reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)

        return [(self.corpus_ids[i], doc, score) for ((i, doc), score) in reranked[:top_k]]
