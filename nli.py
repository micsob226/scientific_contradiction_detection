"""NLI: does an abstract SUPPORT, CONTRADICT, or give NOT ENOUGH INFO for a claim?

An NLI model takes a (premise, hypothesis) pair and predicts one of
"entailment", "neutral", "contradiction". We read that off and translate:

    entailment    -> SUPPORT
    neutral       -> NEI          (not enough info)
    contradiction -> CONTRADICT

MODEL_NAME can be either a pretrained NLI model from the Hugging Face Hub
(a zero-shot baseline) or a local folder produced by finetune_nli.py.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# "results/nli_finetuned"  -> our model fine-tuned on SciFact (best, needs finetune_nli.py first)
# "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"  -> zero-shot baseline
# "roberta-large-mnli"  -> smaller zero-shot alternative
MODEL_NAME = "results/nli_finetuned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading NLI model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# finetune_nli.py trains the head in this position order (label id 0, 1, 2).
# A model we fine-tuned ourselves may report its labels only as
# "LABEL_0" / "LABEL_1" / "LABEL_2", so we fall back to this list.
FINETUNED_ORDER = ["CONTRADICT", "SUPPORT", "NEI"]


def nli_label_to_verdict(name):
    """Translate the model's label name into our verdict vocabulary."""
    name = name.lower()
    if "entail" in name:
        return "SUPPORT"
    if "contradic" in name:
        return "CONTRADICT"
    if "neutral" in name:
        return "NEI"
    if name.startswith("label_"):
        return FINETUNED_ORDER[int(name.split("_")[1])]
    raise ValueError(f"Unexpected NLI label from the model: {name!r}")


def classify(claim, abstract):
    """Return "SUPPORT", "CONTRADICT" or "NEI" for one claim / abstract pair."""
    # premise = the evidence text, hypothesis = the claim we are checking
    inputs = tokenizer(
        abstract,
        claim,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    scores = torch.softmax(output.logits[0], dim=-1)
    best_id = int(torch.argmax(scores))
    label_name = model.config.id2label[best_id]
    return nli_label_to_verdict(label_name)


if __name__ == "__main__":
    # tiny sanity check
    demo_abstract = (
        "In a randomized trial, statin therapy reduced all-cause mortality "
        "in adults without prior cardiovascular disease."
    )
    print(classify("Statins reduce all-cause mortality in healthy adults.", demo_abstract))
    print(classify("Statins increase all-cause mortality in healthy adults.", demo_abstract))
    print(classify("Statins are manufactured in Ireland.", demo_abstract))
