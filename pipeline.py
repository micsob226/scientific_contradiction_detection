"""End-to-end claim checker: a claim goes in, a verdict comes out.

    python pipeline.py "Statins reduce all-cause mortality in healthy adults."

Two steps:
  1. retrieve the most relevant abstract from the corpus  (BGE-FT retriever)
  2. classify the claim against that abstract             (NLI model from nli.py)

Importing this loads the retrieval models and the NLI model, so the first
run takes a minute.
"""
import sys

from retrieval import search_bge_ft
from nli import classify

VERDICT_TEXT = {
    "SUPPORT": "SUPPORTED by the evidence",
    "CONTRADICT": "CONTRADICTED by the evidence",
    "NEI": "NOT ENOUGH INFO in the evidence",
}


def abstract_text(doc):
    return doc["title"] + " " + " ".join(doc["abstract"])


def check_claim(claim, n_evidence=3):
    """Return the verdict for a claim plus the evidence it was based on."""
    hits = search_bge_ft(claim, n=n_evidence)
    top = hits[0]
    verdict = classify(claim, abstract_text(top))
    return {
        "claim": claim,
        "verdict": verdict,
        "evidence": top,
        "other_hits": hits[1:],
    }


def main():
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
    else:
        claim = input("Enter a claim: ").strip()

    result = check_claim(claim)
    top = result["evidence"]
    snippet = " ".join(top["abstract"])[:300]

    print()
    print(f"CLAIM:   {result['claim']}")
    print(f"VERDICT: {result['verdict']}  ({VERDICT_TEXT[result['verdict']]})")
    print()
    print(f"EVIDENCE (doc {top['doc_id']}):")
    print(f"  {top['title']}")
    print(f"  {snippet}...")

    if result["other_hits"]:
        print()
        print("Other retrieved abstracts:")
        for doc in result["other_hits"]:
            print(f"  - {doc['title']} (doc {doc['doc_id']})")


if __name__ == "__main__":
    main()
