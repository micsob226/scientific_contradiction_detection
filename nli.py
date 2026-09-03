"""Zero-shot NLI: does an abstract SUPPORT, CONTRADICT, or give NOT ENOUGH INFO
for a claim?

We use a model that was pretrained on natural-language-inference (NLI) data.
Given a (premise, hypothesis) pair it predicts one of "entailment",
"neutral", "contradiction". We read that off and translate:

    entailment    -> SUPPORT
    neutral       -> NEI          (not enough info)
    contradiction -> CONTRADICT

No training happens here. finetune_nli.py trains a model on SciFact itself;
to use that model instead, point MODEL_NAME at "results/nli_finetuned".
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# RoBERTa-large trained on SNLI + MNLI + FEVER + ANLI. The FEVER/ANLI parts
# make it decent at fact-checking style claims.
# Simpler alternative: "roberta-large-mnli".
MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading NLI model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


def nli_label_to_verdict(name):
    """Translate the model's label name into our verdict vocabulary."""
    name = name.lower()
    if "entail" in name:
        return "SUPPORT"
    if "contradic" in name:
        return "CONTRADICT"
    if "neutral" in name:
        return "NEI"
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
