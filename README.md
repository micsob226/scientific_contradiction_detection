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

- [x] BM25 retrieval with Snowball stemming, index persisted as pickle
- [x] Mini evaluation set (10 queries from SciFact claims)
- [ ] Dense retrieval with FAISS + embedding model
- [ ] Cross-encoder reranking
- [ ] NLI classification (DeBERTa)
- [ ] Explanation generation (Mistral 7B + LoRA)
- [ ] Out-of-domain evaluation
- [ ] Demo (Gradio)

## Structure

```
.
├── data/              # SciFact raw data
├── results/           # Indices, eval outputs
├── retrieval.py       # BM25 + (later) FAISS
├── data.py            # JSONL loading, preprocessing
└── pyproject.toml
```