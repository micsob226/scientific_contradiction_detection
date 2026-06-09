from data import load_jsonl
from retrieval import search_bm25, search_dense, search_reranked, search_tfidf, search_specter, search_specter_reranked
import json
import math


def dcg(relevances):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_relevances, ground_truth_relevances, k):
    actual = dcg(retrieved_relevances[:k])
    ideal = dcg(sorted(ground_truth_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


claims_train = load_jsonl("data/claims_train.jsonl")

# Eval-Set direkt aus claims_train bauen, Claims ohne cited_doc_ids überspringen
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

with open("results/eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print(f"Evaluating on {len(eval_set)} claims...")

ndcg_tfidf = []
ndcg_bm25 = []
ndcg_dense = []
ndcg_reranked = []
ndcg_specter = []
ndcg_specter_reranked = []

for i, item in enumerate(eval_set):
    query = item["query"]
    ground_truth = item["ground_truth"]

    truth_lookup = {entry["doc_id"]: entry["relevance"] for entry in ground_truth}
    all_relevances = [entry["relevance"] for entry in ground_truth]

    tfidf_results = search_tfidf(query, n=5)
    tfidf_rels = [truth_lookup.get(r["doc_id"], 0) for r in tfidf_results]
    ndcg_tfidf.append(ndcg_at_k(tfidf_rels, all_relevances, 5))

    bm25_results = search_bm25(query, n=5)
    bm25_rels = [truth_lookup.get(r["doc_id"], 0) for r in bm25_results]
    ndcg_bm25.append(ndcg_at_k(bm25_rels, all_relevances, 5))

    dense_results = search_dense(query, n=5)
    dense_rels = [truth_lookup.get(r["doc_id"], 0) for r in dense_results]
    ndcg_dense.append(ndcg_at_k(dense_rels, all_relevances, 5))

    reranked_results = search_reranked(query, n=5)
    reranked_rels = [truth_lookup.get(r["doc_id"], 0) for r in reranked_results]
    ndcg_reranked.append(ndcg_at_k(reranked_rels, all_relevances, 5))

    specter_results = search_specter(query, n=5)
    specter_rels = [truth_lookup.get(r["doc_id"], 0) for r in specter_results]
    ndcg_specter.append(ndcg_at_k(specter_rels, all_relevances, 5))

    specter_reranked_results = search_specter_reranked(query, n=5)
    specter_reranked_rels = [truth_lookup.get(r["doc_id"], 0) for r in specter_reranked_results]
    ndcg_specter_reranked.append(ndcg_at_k(specter_reranked_rels, all_relevances, 5))

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(eval_set)} done...")


avg_tfidf = sum(ndcg_tfidf) / len(ndcg_tfidf)
avg_bm25 = sum(ndcg_bm25) / len(ndcg_bm25)
avg_dense = sum(ndcg_dense) / len(ndcg_dense)
avg_reranked = sum(ndcg_reranked) / len(ndcg_reranked)
avg_specter = sum(ndcg_specter) / len(ndcg_specter)
avg_specter_reranked = sum(ndcg_specter_reranked) / len(ndcg_specter_reranked)


print(f"Reranked avg nDCG@5: {avg_reranked:.3f}")
print(f"BM25     avg nDCG@5: {avg_bm25:.3f}")
print(f"TF-IDF   avg nDCG@5: {avg_tfidf:.3f}")
print(f"FAISS    avg nDCG@5: {avg_dense:.3f}")
print(f"SPECTER          avg nDCG@5: {avg_specter:.3f}")
print(f"SPECTER+Reranked avg nDCG@5: {avg_specter_reranked:.3f}")
