import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data import load_jsonl
from pathlib import Path

corpus = load_jsonl("data/corpus.jsonl")
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

texts = [entry["title"] + " " + " ".join(entry["abstract"]) for entry in corpus]
doc_ids = [entry["doc_id"] for entry in corpus]

embedding = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
print(embedding.shape)

results_dir = Path("results/indices")
results_dir.mkdir(parents=True, exist_ok=True)

np.save(results_dir / "corpus_bge_doc_ids.npy", doc_ids)
np.save(results_dir / "corpus_bge_embeddings.npy", embedding)

faiss.normalize_L2(embedding)
d = embedding.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embedding)
faiss.write_index(index, str(results_dir / "faiss_bge_index.bin"))