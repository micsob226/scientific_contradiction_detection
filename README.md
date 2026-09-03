# Scientific Claim Contradiction Detection

End-to-end NLP pipeline that retrieves relevant evidence from a scientific literature corpus for a given claim and classifies it as SUPPORTS / CONTRADICTS / NOT ENOUGH INFO. Based on the SciFact dataset.


## Setup

```bash
uv sync
```

Place SciFact data in `data/`:
- `data/corpus.jsonl`
- `data/claims_train.jsonl`
- `data/claims_dev.jsonl`
- `data/claims_test.jsonl`


## Status

- [x] TF-IDF and BM25 retrieval (Snowball stemming), indices persisted to disk
- [x] Dense retrieval with FAISS (MiniLM, SPECTER, BGE-Large)
- [x] Fine-tuning of MiniLM and BGE-Large on SciFact claim/evidence pairs (`finetune_minilm.py`, `finetune_bge.py`)
- [x] Cross-encoder reranking (mxbai-rerank-large)
- [x] Evaluation on held-out claims (nDCG@10, MAP, Recall@10) — see `evaluate_claims_dev_test.py`
- [~] NLI classification (SUPPORT / CONTRADICT / NEI): zero-shot baseline done (`nli.py`, `evaluate_nli.py`); fine-tuning script ready (`finetune_nli.py`, not yet run)
- [ ] Explanation generation
- [ ] Out-of-domain evaluation
- [ ] Demo (Gradio — `gradio` not yet installed)

Best retrieval result: fine-tuned BGE-Large, nDCG@10 = 0.830 on the held-out dev-test split (`results/eval_dev_test.log`).
Zero-shot NLI: 0.543 accuracy / macro-F1 0.489 on dev (`results/eval_nli.log`).

## Structure

```
.
├── data/                    # SciFact raw data + train/dev/test splits
├── results/
│   ├── indices/              # FAISS indices + embeddings (generated, gitignored)
│   ├── *_finetuned/           # Fine-tuned model weights (generated, gitignored)
│   ├── eval_set*.json        # Ground-truth relevance judgments built by the evaluate scripts
│   └── eval_dev_test.log     # Evaluation output
├── data.py                  # JSONL loading
├── metrics.py               # nDCG / MAP / Recall, shared by the evaluate scripts
├── embeddings_*.py           # Build FAISS index for one model (MiniLM/SPECTER/BGE/fine-tuned variants)
├── finetune_minilm.py        # Fine-tune an embedding model on SciFact claim/evidence pairs
├── finetune_bge.py           #   (finetune_bge.py produced the best model)
├── retrieval.py              # search_* functions: TF-IDF, BM25, dense (+ optional reranking) per model
├── nli.py                    # classify(claim, abstract) -> SUPPORT / CONTRADICT / NEI (zero-shot NLI)
├── finetune_nli.py           # Fine-tune a 3-class NLI classifier on SciFact (the overnight run)
├── evaluate_claims_train.py  # Evaluate all retrievers on the training claims
├── evaluate_claims_dev_test.py # Evaluate all retrievers on the held-out dev-test split
├── evaluate_nli.py           # End-to-end: retrieve + classify, scored against the dev labels
├── make_plots.py             # Generate comparison plots from evaluation results
├── pyproject.toml
│
│   # --- kept for the write-up, not part of the main pipeline ---
├── evaluate.py               # early eval prototype (nDCG@5 on 10 fixed claims)
├── finetune_arctic.py        # side experiment: fine-tune Snowflake Arctic-embed
└── evaluate_arctic.py        # side experiment: evaluate Snowflake Arctic-embed
```

## Usage

```bash
# 1. Build the retrieval indices (run once per model)
python embeddings.py            # MiniLM
python embeddings_bge.py        # BGE-Large

# 2. (Optional) fine-tune an embedding model on SciFact
python finetune_bge.py
python embeddings_bge_ft.py     # rebuild the index with the fine-tuned model

# 3. Evaluate
python evaluate_claims_dev_test.py
```