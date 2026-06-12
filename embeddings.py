import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data import load_jsonl
from pathlib import Path

corpus = load_jsonl("data/corpus.jsonl")
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [entry["title"] + " " + " ".join(entry["abstract"]) for entry in corpus]
doc_ids  = [entry["doc_id"] for entry in corpus]

embedding = model.encode(texts)
print(embedding.shape)

results_dir = Path("results/indices")
results_dir.mkdir(exist_ok=True)

np.save(results_dir / "corpus_doc_ids.npy", doc_ids)
np.save(results_dir / "corpus_embeddings.npy", embedding)

faiss.normalize_L2(embedding)
d = embedding.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embedding)
faiss.write_index(index, str(results_dir / "faiss_index.bin"))