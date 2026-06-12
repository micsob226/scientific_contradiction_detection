import math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data import load_jsonl


def dcg(relevances):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(retrieved_relevances, ground_truth_relevances, k):
    actual = dcg(retrieved_relevances[:k])
    ideal = dcg(sorted(ground_truth_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0

def main():
    corpus = load_jsonl("data/corpus.jsonl")
    monitor_claims = load_jsonl("data/claims_dev_monitor.jsonl")

    model = SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        device="cuda",
        trust_remote_code=True,
    )

    texts = [entry["title"] + " " + " ".join(entry["abstract"]) for entry in corpus]
    doc_ids = [entry["doc_id"] for entry in corpus]

    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    QUERY_PREFIX = "query: "
    ndcgs = []

    for claim in monitor_claims:
        if not claim.get("cited_doc_ids"):
            continue

        query_text = QUERY_PREFIX + claim["claim"]
        query_emb = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)

        D, I = index.search(query_emb, k=10)
        retrieved_doc_ids = [doc_ids[idx] for idx in I[0]]

        truth_lookup = {}
        for doc_id in claim["cited_doc_ids"]:
            relevance = 2 if str(doc_id) in claim.get("evidence", {}) else 1
            truth_lookup[doc_id] = relevance

        retrieved_rels = [truth_lookup.get(did, 0) for did in retrieved_doc_ids]
        all_rels = list(truth_lookup.values())

        ndcgs.append(ndcg_at_k(retrieved_rels, all_rels, 10))

    avg_ndcg = sum(ndcgs) / len(ndcgs)

    print(f"____RESULTS____")
    print(f"Arctic-embed-l-v2.0 nDCG@10 on dev_monitor: {avg_ndcg:.4f}")
    print(f"BGE-Large (Baseline)        on dev_monitor: 0.7046")
    print(f"BGE-Large (Fine-tuned)      on dev_monitor: 0.7663")
    print(f"Δ vs BGE-Baseline: {avg_ndcg - 0.7046:+.4f}")
    print(f"Δ vs BGE-FT:       {avg_ndcg - 0.7663:+.4f}")


if __name__ == "__main__":
    main()