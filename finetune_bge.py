from data import load_jsonl
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    corpus = load_jsonl("data/corpus.jsonl")
    corpus_lookup = {entry["doc_id"]: entry for entry in corpus}

    claims_train = load_jsonl("data/claims_train.jsonl")
    monitor_claims = load_jsonl("data/claims_dev_monitor.jsonl")

    train_examples = []
    for claim in claims_train:
        if not claim.get("cited_doc_ids"):
            continue
        claim_text = BGE_QUERY_PREFIX + claim["claim"]
        for doc_id in claim["cited_doc_ids"]:
            if doc_id not in corpus_lookup:
                continue
            doc = corpus_lookup[doc_id]
            doc_text = doc["title"] + " " + " ".join(doc["abstract"])
            train_examples.append(InputExample(texts=[claim_text, doc_text]))

    print(f"Training pairs: {len(train_examples)} aus {len(claims_train)} Claims")

    queries = {}
    relevant_docs = {}
    for claim in monitor_claims:
        if not claim.get("cited_doc_ids"):
            continue
        qid = str(claim["id"])
        queries[qid] = BGE_QUERY_PREFIX + claim["claim"]
        relevant_docs[qid] = {str(d) for d in claim["cited_doc_ids"]}

    corpus_dict = {
        str(entry["doc_id"]): entry["title"] + " " + " ".join(entry["abstract"])
        for entry in corpus
    }

    print(f"Evaluator: {len(queries)} Queries against {len(corpus_dict)} Docs")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="dev_monitor",
        show_progress_bar=False,
        batch_size=64,
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[10],
        map_at_k=[10],
    )

    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    model.max_seq_length = 512

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_epochs = 5
    steps_per_epoch = len(train_dataloader)
    warmup_steps = int(steps_per_epoch * num_epochs * 0.1)

    output_path = "results/bge_finetuned"

    print("\n____Baseline (pretrained BGE-Large)____")
    baseline_scores = evaluator(model)
    print(f"Baseline all Metrics: {baseline_scores}")
    ndcg_key = next(k for k in baseline_scores if "ndcg@10" in k)
    baseline_ndcg = baseline_scores[ndcg_key]
    print(f"Baseline nDCG@10: {baseline_ndcg:.4f}")

    print(f"\n Training: {num_epochs} Epochs × {steps_per_epoch} Steps")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=steps_per_epoch,
        output_path=output_path,
        save_best_model=True,
        optimizer_params={"lr": 1e-5},
        weight_decay=0.01,
        show_progress_bar=True,
    )

    best_model = SentenceTransformer(output_path, device="cuda")
    final_scores = evaluator(best_model)
    final_ndcg = final_scores[ndcg_key]
    print(f"\n____Summary____")
    print(f"Baseline nDCG@10:        {baseline_ndcg:.4f}")
    print(f"Best fine-tuned nDCG@10: {final_ndcg:.4f}")
    print(f"Δ:                       {final_ndcg - baseline_ndcg:+.4f}")
    print(f"\nAlle finalen Metriken: {final_scores}")


if __name__ == "__main__":
    main()