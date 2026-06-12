from data import load_jsonl
from retrieval import search_bm25, search_dense, search_reranked, search_tfidf, search_specter, search_specter_reranked, search_bge, search_bge_reranked
import json
import math


def dcg(relevances):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(retrieved_relevances, ground_truth_relevances, k):
    actual = dcg(retrieved_relevances[:k])
    ideal = dcg(sorted(ground_truth_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0

def average_precision(retrieved_relevances, total_relevant):
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
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in retrieved_relevances[:k] if r > 0) / total_relevant

claims_train = load_jsonl("data/claims_train.jsonl")

eval_set = []
for claim in claims_train:
    if not claim["cited_doc_ids"]:
        continue
    ranked_truth = []
    for doc_id in claim["cited_doc_ids"]:
        relevance = 2 if str(doc_id) in claim["evidence"] else 1
        ranked_truth.append({"doc_id": doc_id, "relevance": relevance})
    ranked_truth.sort(key=lambda x: x["relevance"], reverse=True)
    eval_set.append({"query": claim["claim"], "ground_truth": ranked_truth})

with open("data/eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print(f"Evaluating on {len(eval_set)} claims...")

K = 10

search_fns = {
    "TF-IDF":           search_tfidf,
    "BM25":             search_bm25,
    "FAISS":            search_dense,
    "FAISS+Reranked":   search_reranked,
    "SPECTER":          search_specter,
    "SPECTER+Reranked": search_specter_reranked,
    "BGE":              search_bge,
    "BGE+Reranked":     search_bge_reranked,
}

results = {name: {"ndcg": [], "ap": [], "recall": []} for name in search_fns}

for i, item in enumerate(eval_set):
    query = item["query"]
    ground_truth = item["ground_truth"]
    truth_lookup = {entry["doc_id"]: entry["relevance"] for entry in ground_truth}
    all_relevances = [entry["relevance"] for entry in ground_truth]
    total_relevant = len(ground_truth)

    for name, fn in search_fns.items():
        retrieved = fn(query, n=K)
        rels = [truth_lookup.get(r["doc_id"], 0) for r in retrieved]
        results[name]["ndcg"].append(ndcg_at_k(rels, all_relevances, K))
        results[name]["ap"].append(average_precision(rels, total_relevant))
        results[name]["recall"].append(recall_at_k(rels, total_relevant, K))

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(eval_set)} done")


print(f"\n{'Method':<20} {'nDCG@'+str(K):>10} {'MAP':>10} {'Recall@'+str(K):>12}")
print("-" * 55)
for name in search_fns:
    avg_ndcg   = sum(results[name]["ndcg"])   / len(results[name]["ndcg"])
    avg_ap     = sum(results[name]["ap"])     / len(results[name]["ap"])
    avg_recall = sum(results[name]["recall"]) / len(results[name]["recall"])
    print(f"{name:<20} {avg_ndcg:>10.3f} {avg_ap:>10.3f} {avg_recall:>12.3f}")