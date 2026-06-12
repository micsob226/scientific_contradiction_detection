import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data import load_jsonl
from pathlib import Path

corpus = load_jsonl("data/corpus.jsonl")
model = SentenceTransformer("allenai/specter", device="cuda")

texts = [entry["title"] + " [SEP] " + " ".join(entry["abstract"]) for entry in corpus]
doc_ids = [entry["doc_id"] for entry in corpus]

print(f"Encoding {len(texts)} documents...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
print(f"Shape: {embeddings.shape}")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

np.save(results_dir / "corpus_specter_doc_ids.npy", doc_ids)

embeddings = embeddings.astype(np.float32)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(results_dir / "faiss_specter_index.bin"))