import json
import random
from pathlib import Path


def main():
    random.seed(42)

    dev_path = Path("data/claims_dev.jsonl")
    with open(dev_path) as f:
        dev_claims = [json.loads(line) for line in f]

    random.shuffle(dev_claims)

    split_idx = len(dev_claims) // 2
    monitor = dev_claims[:split_idx]
    test = dev_claims[split_idx:]

    with open("data/claims_dev_monitor.jsonl", "w") as f:
        for claim in monitor:
            f.write(json.dumps(claim) + "\n")

    with open("data/claims_dev_test.jsonl", "w") as f:
        for claim in test:
            f.write(json.dumps(claim) + "\n")

    print(f"Total dev claims: {len(dev_claims)}")
    print(f"  monitor: {len(monitor)} → data/claims_dev_monitor.jsonl")
    print(f"  test:    {len(test)} → data/claims_dev_test.jsonl")


if __name__ == "__main__":
    main()