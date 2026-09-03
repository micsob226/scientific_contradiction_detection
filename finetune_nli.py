"""Fine-tune a 3-class NLI classifier on the SciFact training claims.

Input pair : (abstract text, claim)
Label      : 0 = CONTRADICT, 1 = SUPPORT, 2 = NEI  (matches the base model's head)

Runs in a few minutes on one GPU. The best model is saved to
results/nli_finetuned/. To use it, set MODEL_NAME in nli.py to
"results/nli_finetuned" and run evaluate_nli.py again.

We start from "cross-encoder/nli-roberta-base", which already knows generic
NLI, and adapt it to SciFact. Starting from a plain "roberta-base" instead
(a cold 3-way head) was tried and did clearly worse.
"""
import os

# This server has several GPUs. sentence-transformers would try to use all of
# them (DataParallel), which currently crashes the CrossEncoder loss. Pin the
# run to a single GPU. Override from the shell if GPU 0 is busy, e.g.
#     CUDA_VISIBLE_DEVICES=2 python finetune_nli.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from data import load_jsonl
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from torch.utils.data import DataLoader

BASE_MODEL = "cross-encoder/nli-roberta-base"

VERDICT_TO_LABEL = {"CONTRADICT": 0, "SUPPORT": 1, "NEI": 2}  # matches the base model's head


def abstract_text(doc):
    return doc["title"] + " " + " ".join(doc["abstract"])


def gold_verdict(claim):
    for snippets in claim["evidence"].values():
        for snippet in snippets:
            return snippet["label"]
    return "NEI"


def build_pairs(claims, doc_by_id):
    """One (abstract, claim) example per cited document."""
    pairs = []
    for claim in claims:
        label = VERDICT_TO_LABEL[gold_verdict(claim)]
        for doc_id in claim["cited_doc_ids"]:
            if doc_id not in doc_by_id:
                continue
            text = abstract_text(doc_by_id[doc_id])
            pairs.append(InputExample(texts=[text, claim["claim"]], label=label))
    return pairs


def main():
    corpus = load_jsonl("data/corpus.jsonl")
    doc_by_id = {}
    for doc in corpus:
        doc_by_id[doc["doc_id"]] = doc

    train_claims = load_jsonl("data/claims_train.jsonl")
    dev_claims = load_jsonl("data/claims_dev_monitor.jsonl")

    train_pairs = build_pairs(train_claims, doc_by_id)
    dev_pairs = build_pairs(dev_claims, doc_by_id)
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Dev pairs:      {len(dev_pairs)}")

    model = CrossEncoder(BASE_MODEL, num_labels=3, device="cuda")
    model.max_length = 512

    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_pairs, name="dev_monitor")

    num_epochs = 5
    steps_per_epoch = len(train_dataloader)
    warmup_steps = int(steps_per_epoch * num_epochs * 0.1)
    output_path = "results/nli_finetuned"

    print(f"\nTraining {num_epochs} epochs x {steps_per_epoch} steps -> {output_path}")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=steps_per_epoch,
        output_path=output_path,
        save_best_model=True,
        optimizer_params={"lr": 2e-5},
    )

    print(f"\nDone. Best model saved to {output_path}")
    print('To use it: set MODEL_NAME = "results/nli_finetuned" in nli.py, then run evaluate_nli.py')


if __name__ == "__main__":
    main()
