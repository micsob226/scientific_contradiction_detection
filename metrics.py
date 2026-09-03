"""Ranking-quality metrics shared by the evaluate_* scripts.

Each function takes a list of *relevance grades* for the retrieved documents,
in ranked order (grade 0 = not relevant, higher = more relevant).
"""
import math


def dcg(relevances):
    """Discounted Cumulative Gain: reward relevant hits, discount lower ranks."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_relevances, ground_truth_relevances, k):
    """nDCG@k: DCG of the top k hits divided by the best possible DCG."""
    actual = dcg(retrieved_relevances[:k])
    ideal = dcg(sorted(ground_truth_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def average_precision(retrieved_relevances, total_relevant):
    """Mean of the precision values measured at each relevant hit."""
    if total_relevant == 0:
        return 0.0
    relevant_count = 0
    precision_sum = 0.0
    for i, rel in enumerate(retrieved_relevances):
        if rel > 0:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / total_relevant


def recall_at_k(retrieved_relevances, total_relevant, k):
    """Fraction of all relevant documents that appear in the top k hits."""
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in retrieved_relevances[:k] if r > 0) / total_relevant
