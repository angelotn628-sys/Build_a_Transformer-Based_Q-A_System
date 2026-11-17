"""
Evaluation Metrics for Information Retrieval
"""
from typing import Dict, List
import numpy as np

class IRMetrics:
    """Information Retrieval evaluation metrics."""

    @staticmethod
    def _relevant(qrels_q: Dict[int, int]) -> set:
        """Return the set of doc_ids with relevance > 0."""
        return {doc_id for doc_id, rel in qrels_q.items() if rel > 0}

    @staticmethod
    def recall_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Recall@k: fraction of relevant documents found in top-k.

        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation
        """
        recall_scores = []
        for q_id, docs in results.items():
            if q_id not in qrels:
                continue
            relevant = IRMetrics._relevant(qrels[q_id])
            if not relevant:
                continue
            top_k = set(docs[:k])
            recall_scores.append(len(top_k & relevant) / len(relevant))
        return float(np.mean(recall_scores)) if recall_scores else 0.0

    @staticmethod
    def precision_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Precision@k: fraction of top-k that are relevant.
        """
        precision_scores = []
        for q_id, docs in results.items():
            if q_id not in qrels:
                continue
            relevant = IRMetrics._relevant(qrels[q_id])
            top_k = docs[:k]
            hits = sum(1 for doc_id in top_k if doc_id in relevant)
            precision_scores.append(hits / k)
        return float(np.mean(precision_scores)) if precision_scores else 0.0

    @staticmethod
    def mrr(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        """
        reciprocal_ranks = []
        for q_id, docs in results.items():
            if q_id not in qrels:
                continue
            relevant = IRMetrics._relevant(qrels[q_id])
            rr = 0.0
            for rank, doc_id in enumerate(docs, start=1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_retrieval(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> Dict[str, float]:
        """
        Comprehensive evaluation with standard IR metrics.

        Returns:
            Dictionary with metric names and values
        """
        return {
            'Recall@1': IRMetrics.recall_at_k(results, qrels, 1),
            'Recall@5': IRMetrics.recall_at_k(results, qrels, 5),
            'Recall@10': IRMetrics.recall_at_k(results, qrels, 10),
            'Precision@5': IRMetrics.precision_at_k(results, qrels, 5),
            'MRR': IRMetrics.mrr(results, qrels),
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation metrics."""
        print(f"\n📊 {title}")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        print("=" * 40)