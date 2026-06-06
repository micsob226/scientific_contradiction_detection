# ========== IMPORTS ==========
import re
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer
from pathlib import Path
from sentence_transformers import SentenceTransformer

from data import corpus


# ========== SHARED SETUP ==========
# Lookup-Dict für beide Suchen
id_to_entry = {}
for entry in corpus:
    id_to_entry[entry["doc_id"]] = entry


# ========== BM25 SETUP ==========
stemmer = SnowballStemmer("english")

def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# BM25-Index aufbauen
tokenized_corpus = []
for entry in corpus:
    title = entry["title"]
    abstract = " ".join(entry["abstract"])
    full_text = title + " " + abstract
    tokens = preprocess(full_text)
    tokenized_corpus.append(tokens)

bm25 = BM25Okapi(tokenized_corpus)

# BM25 auf Disk speichern
Path("results").mkdir(exist_ok=True)
with open("results/bm25_index.pkl", "wb") as file:
    pickle.dump(bm25, file)


# ========== DENSE SETUP ==========
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("results/faiss_index.bin")
doc_ids = np.load("results/corpus_doc_ids.npy")


# ========== SEARCH FUNCTIONS ==========
def search_bm25(query: str, n: int = 5):
    tokenized_query = preprocess(query)
    return bm25.get_top_n(tokenized_query, corpus, n=n)

def search_dense(query: str, n: int = 5):
    embedding = model.encode(query)
    embedding = embedding.reshape(1, -1)
    faiss.normalize_L2(embedding)

    D, I = index.search(embedding, k=n)
    hit_ids = doc_ids[I[0]]

    return [id_to_entry[did] for did in hit_ids]