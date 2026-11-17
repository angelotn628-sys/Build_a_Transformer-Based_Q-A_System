"""
Transformer-based Retrieval using Sentence Embeddings
"""
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        print(f"🤖 Loading transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        print(f"✅ Model loaded: {model_name}")
        
    def build_index(self, corpus_texts: List[str]):
        """Build semantic index by encoding corpus texts."""
        print("🧠 Building semantic index...")
        self.corpus = corpus_texts
        
        # YOUR CODE HERE: Encode corpus using sentence transformer
        self.corpus_embeddings = self.model.encode(self.corpus)
        
        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents using semantic similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")

        # YOUR CODE HERE: Encode query texts using the transformer model
        query_embeddings = self.model.encode(query_texts)         

        # YOUR CODE HERE: Calculate similarities and retrieve top-k documents
        scores = cosine_similarity(query_embeddings, self.corpus_embeddings) 

        results = {}
        for q in range(len(query_texts)):
            top_idx = np.argsort(scores[q])[::-1][:k]
            results[q] = [int(i) for i in top_idx]

        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
