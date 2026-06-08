from data import load_jsonl
from retrieval import search_bm25, search_dense, search_reranked, search_tfidf
import json
import math


def dcg(relevances):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_relevances, ground_truth_relevances, k):
    actual = dcg(retrieved_relevances[:k])
    ideal = dcg(sorted(ground_truth_relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


queries = [
    "A strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage.",
    "ART substantially reduces infectiveness of HIV-positive people.",
    "Citrullinated proteins externalized in neutrophil extracellular traps act indirectly to disrupt the inflammatory cycle.",
    "Klf4 is not required for proper myeloid cell differentiation.",
    "MICAL redox enzymes regulate actin dynamics.",
    "Mice that lack Interferon-\u03b3 or its receptor are highly susceptible to experimental autoimmune myocarditis.",
    "Physical activity level has no association with the difference in maximal oxygen consumption between black and white youth.",
    "Physical activity level is associated with the difference in maximal oxygen consumption between black and white youth.", 
    "Primary cervical cancer screening with HPV detection has lower longitudinal sensitivity than conventional cytology to detect cervical intraepithelial neoplasia grade 2.", 
    "Primary cervical cytology screening with HPV detection has higher longitudinal sensitivity to detect severe cervical intraepithelial neoplasia than conventional cytology."
    ]

claims_train = load_jsonl("data/claims_train.jsonl")

claim_to_info = {}
for claim in claims_train:
    claim_to_info[claim["claim"]] = {
        "evidence": claim["evidence"],
        "cited_doc_ids": claim["cited_doc_ids"]
    }

eval_set = []
for query in queries:
    info = claim_to_info[query]

    ranked_truth = []
    for doc_id in info["cited_doc_ids"]:
        if str(doc_id) in info["evidence"]:
            relevance = 2
        else:
            relevance = 1
        ranked_truth.append({"doc_id": doc_id, "relevance": relevance})
    ranked_truth.sort(key=lambda x: x["relevance"], reverse=True)
    eval_set.append({"query": query, "ground_truth": ranked_truth})

with open("results/eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)


ndcg_bm25 = []
ndcg_dense = []
ndcg_reranked = []
ndcg_tfidf = []

for item in eval_set:
    query = item["query"]
    ground_truth = item["ground_truth"]

    # Lookup: doc_id → relevance
    truth_lookup = {}
    for entry in ground_truth:
        truth_lookup[entry["doc_id"]] = entry["relevance"]

    # Alle ground-truth Relevanzen (für IDCG)
    all_relevances = [entry["relevance"] for entry in ground_truth]

    # BM25 laufen lassen, Relevanzen extrahieren, nDCG
    bm25_results = search_bm25(query, n=5)
    bm25_rels = [truth_lookup.get(r["doc_id"], 0) for r in bm25_results]
    score_bm25 = ndcg_at_k(bm25_rels, all_relevances, 5)
    ndcg_bm25.append(score_bm25)

    # FAISS dasselbe
    dense_results = search_dense(query, n=5)
    dense_rels = [truth_lookup.get(r["doc_id"], 0) for r in dense_results]
    score_dense = ndcg_at_k(dense_rels, all_relevances, 5)
    ndcg_dense.append(score_dense)

    # Reranked dasselbe
    reranked_results = search_reranked(query, n=5)
    reranked_rels = [truth_lookup.get(r["doc_id"], 0) for r in reranked_results]
    score_reranked = ndcg_at_k(reranked_rels, all_relevances, 5)
    ndcg_reranked.append(score_reranked)

    # TF-IDF dasselbe
    tfidf_results = search_tfidf(query, n=5)
    tfidf_rels = [truth_lookup.get(r["doc_id"], 0) for r in tfidf_results]
    score_tfidf = ndcg_at_k(tfidf_rels, all_relevances, 5)
    ndcg_tfidf.append(score_tfidf)


# Mittelwerte
avg_bm25 = sum(ndcg_bm25) / len(ndcg_bm25)
avg_dense = sum(ndcg_dense) / len(ndcg_dense)
avg_reranked = sum(ndcg_reranked) / len(ndcg_reranked)
avg_tfidf = sum(ndcg_tfidf) / len(ndcg_tfidf)

print(f"BM25  avg nDCG@5: {avg_bm25:.3f}")
print(f"FAISS avg nDCG@5: {avg_dense:.3f}")
print(f"Reranked avg nDCG@5: {avg_reranked:.3f}")
print(f"TF-IDF avg nDCG@5: {avg_tfidf:.3f}")