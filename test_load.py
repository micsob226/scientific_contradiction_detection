import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from data import load_jsonl
from retrieval import search_reranked

index = faiss.read_index("results/faiss_index.bin")
#print(index.ntotal)

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "Does smoking cause lung cancer?"
embedding = model.encode(text)
embedding = embedding.reshape(1, -1)
faiss.normalize_L2(embedding)
#print(embedding.shape)

D, I = index.search(embedding, k=5)
#print("Scores:", D)
#print("Positions:", I)

doc_ids = np.load("results/corpus_doc_ids.npy")
hit_ids = doc_ids[I[0]]
#print("Doc IDs:", hit_ids)

corpus = load_jsonl("data/corpus.jsonl")

id_to_entry = {}
for entry in corpus:
    doc_id = entry["doc_id"]
    id_to_entry[doc_id] = entry

for did in hit_ids:
    entry = id_to_entry[did]
    #print(entry["title"])

results = search_reranked("Does smoking cause lung cancer?")
for r in results:
    print(r["title"])

