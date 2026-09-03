from data import load_jsonl
from retrieval import (
    search_bm25, search_minilm, search_minilm_reranked, search_tfidf,
    search_specter, search_specter_reranked,
    search_bge, search_bge_reranked,
    search_minilm_ft, search_minilm_ft_reranked,
    search_bge_ft, search_bge_ft_reranked,
)
import json

from metrics import ndcg_at_k, average_precision, recall_at_k

claims_dev_test = load_jsonl("data/claims_dev_test.jsonl")
eval_set = []
for claim in claims_dev_test:
    if not claim["cited_doc_ids"]:
        continue
    ranked_truth = []
    for doc_id in claim["cited_doc_ids"]:
        relevance = 2 if str(doc_id) in claim["evidence"] else 1
        ranked_truth.append({"doc_id": doc_id, "relevance": relevance})
    ranked_truth.sort(key=lambda x: x["relevance"], reverse=True)
    eval_set.append({"query": claim["claim"], "ground_truth": ranked_truth})

with open("results/eval_set_dev_test.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print(f"Evaluating on {len(eval_set)} claims (held-out dev_test)...")
K = 10
search_fns = {
    "TF-IDF":             search_tfidf,
    "BM25":               search_bm25,
    "MiniLM":             search_minilm,
    "MiniLM+Reranked":    search_minilm_reranked,
    "MiniLM-FT":          search_minilm_ft,
    "MiniLM-FT+Reranked": search_minilm_ft_reranked,
    "SPECTER":            search_specter,
    "SPECTER+Reranked":   search_specter_reranked,
    "BGE":                search_bge,
    "BGE+Reranked":       search_bge_reranked,
    "BGE-FT":             search_bge_ft,
    "BGE-FT+Reranked":    search_bge_ft_reranked,
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

print(f"\n{'Method':<22} {'nDCG@'+str(K):>10} {'MAP':>10} {'Recall@'+str(K):>12}")
print("-" * 57)
for name in search_fns:
    avg_ndcg   = sum(results[name]["ndcg"])   / len(results[name]["ndcg"])
    avg_ap     = sum(results[name]["ap"])     / len(results[name]["ap"])
    avg_recall = sum(results[name]["recall"]) / len(results[name]["recall"])
    print(f"{name:<22} {avg_ndcg:>10.3f} {avg_ap:>10.3f} {avg_recall:>12.3f}")