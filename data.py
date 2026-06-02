import json


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


corpus = load_jsonl("data/corpus.jsonl")
claims_train = load_jsonl("data/claims_train.jsonl")
claims_dev = load_jsonl("data/claims_dev.jsonl")
claims_test = load_jsonl("data/claims_test.jsonl")

print(len(corpus), len(claims_train), len(claims_dev), len(claims_test))