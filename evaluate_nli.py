"""End-to-end evaluation of the contradiction detector on the dev claims.

For every dev claim we make two predictions:

  pipeline : retrieve the top abstract with BGE-FT, then run the NLI model.
             Measures the whole system (retrieval mistakes included).

  nli only : run the NLI model on the *gold* cited abstract.
             Measures just the NLI model, with retrieval taken out of the way.

For each we print accuracy, a confusion matrix, and per-class
precision / recall / F1 plus the macro-average.
"""
from data import load_jsonl
from retrieval import search_bge_ft
from nli import classify

VERDICTS = ["SUPPORT", "CONTRADICT", "NEI"]


def abstract_text(doc):
    return doc["title"] + " " + " ".join(doc["abstract"])


def gold_verdict(claim):
    """The true verdict for a claim.

    SciFact stores evidence as {doc_id: [{"sentences": [...], "label": ...}]}.
    If there are no evidence entries, the claim cannot be verified -> NEI.
    """
    for snippets in claim["evidence"].values():
        for snippet in snippets:
            return snippet["label"]   # "SUPPORT" or "CONTRADICT"
    return "NEI"


def empty_confusion():
    """A table with confusion[gold][predicted] = 0 for every combination."""
    table = {}
    for gold in VERDICTS:
        table[gold] = {}
        for predicted in VERDICTS:
            table[gold][predicted] = 0
    return table


def report(title, confusion):
    total = 0
    correct = 0
    for gold in VERDICTS:
        for predicted in VERDICTS:
            total += confusion[gold][predicted]
            if gold == predicted:
                correct += confusion[gold][predicted]

    print(f"\n===== {title} =====")
    print(f"Accuracy: {correct / total:.3f}  ({correct}/{total})\n")

    print(f"{'gold \\ pred':<14}" + "".join(f"{p:>12}" for p in VERDICTS))
    for gold in VERDICTS:
        cells = "".join(f"{confusion[gold][p]:>12}" for p in VERDICTS)
        print(f"{gold:<14}{cells}")

    print(f"\n{'class':<14}{'precision':>12}{'recall':>12}{'f1':>12}")
    f1_sum = 0.0
    for c in VERDICTS:
        true_pos = confusion[c][c]
        predicted_as_c = sum(confusion[g][c] for g in VERDICTS)
        actually_c = sum(confusion[c][p] for p in VERDICTS)

        precision = true_pos / predicted_as_c if predicted_as_c > 0 else 0.0
        recall = true_pos / actually_c if actually_c > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_sum += f1
        print(f"{c:<14}{precision:>12.3f}{recall:>12.3f}{f1:>12.3f}")

    print(f"\nMacro-F1: {f1_sum / len(VERDICTS):.3f}")


def main():
    dev_claims = load_jsonl("data/claims_dev.jsonl")
    corpus = load_jsonl("data/corpus.jsonl")

    doc_by_id = {}
    for doc in corpus:
        doc_by_id[doc["doc_id"]] = doc

    print(f"Evaluating NLI on {len(dev_claims)} dev claims ...")

    pipeline_confusion = empty_confusion()
    nli_only_confusion = empty_confusion()

    for i, claim in enumerate(dev_claims):
        gold = gold_verdict(claim)

        # pipeline: the retriever chooses the evidence
        top_hit = search_bge_ft(claim["claim"], n=1)[0]
        pipeline_pred = classify(claim["claim"], abstract_text(top_hit))
        pipeline_confusion[gold][pipeline_pred] += 1

        # nli only: use the gold cited document as the evidence
        cited_ids = claim["cited_doc_ids"]
        if cited_ids and cited_ids[0] in doc_by_id:
            cited_doc = doc_by_id[cited_ids[0]]
            nli_only_pred = classify(claim["claim"], abstract_text(cited_doc))
            nli_only_confusion[gold][nli_only_pred] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dev_claims)} done")

    report("pipeline (retrieved evidence)", pipeline_confusion)
    report("nli only (gold evidence)", nli_only_confusion)


if __name__ == "__main__":
    main()
