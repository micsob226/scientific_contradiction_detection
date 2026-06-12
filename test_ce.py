from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([("query", "doc 1"), ("query", "doc 2")])

pairs = [
    ("Does smoking cause lung cancer?", "Lung cancer is strongly linked to cigarette smoking based on epidemiological evidence."),
    ("Does smoking cause lung cancer?", "Mitochondria produce ATP via oxidative phosphorylation.")
]
scores = model.predict(pairs)
print(scores)