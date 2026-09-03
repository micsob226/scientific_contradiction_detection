import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import load_jsonl

corpus = load_jsonl("data/corpus.jsonl")


id_to_entry = {}
for entry in corpus:
    id_to_entry[entry["doc_id"]] = entry


# TF-IDF
tfidf_texts = [entry["title"] + " " + " ".join(entry["abstract"]) for entry in corpus]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_texts)


# BM25
stemmer = SnowballStemmer("english")

def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# BM25-Index
tokenized_corpus = []
for entry in corpus:
    title = entry["title"]
    abstract = " ".join(entry["abstract"])
    full_text = title + " " + abstract
    tokens = preprocess(full_text)
    tokenized_corpus.append(tokens)

bm25 = BM25Okapi(tokenized_corpus)


# DENSE
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
index = faiss.read_index("results/indices/faiss_index.bin")
doc_ids = np.load("results/indices/corpus_doc_ids.npy")

cross_encoder = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")


# SPECTER
specter_model = SentenceTransformer("allenai/specter", device="cuda")
specter_index = faiss.read_index("results/indices/faiss_specter_index.bin")
specter_doc_ids = np.load("results/indices/corpus_specter_doc_ids.npy")


# BGE
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
bge_index = faiss.read_index("results/indices/faiss_bge_index.bin")
bge_doc_ids = np.load("results/indices/corpus_bge_doc_ids.npy")

# BGE Fine-tuned
bge_ft_model = SentenceTransformer("results/bge_finetuned", device="cuda")
bge_ft_index = faiss.read_index("results/indices/faiss_bge_ft_index.bin")
bge_ft_doc_ids = np.load("results/indices/corpus_bge_ft_doc_ids.npy")

# MiniLM Fine-tuned
minilm_ft_model = SentenceTransformer("results/minilm_finetuned", device="cuda")
minilm_ft_index = faiss.read_index("results/indices/faiss_minilm_ft_index.bin")
minilm_ft_doc_ids = np.load("results/indices/corpus_minilm_ft_doc_ids.npy")


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def rerank(query: str, candidates: list, n: int) -> list:
    """Re-score candidate docs with the cross-encoder, return the top n."""
    pairs = [(query, c["title"] + " " + " ".join(c["abstract"])) for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _score, doc in ranked[:n]]


def search_tfidf(query: str, n: int = 5):
    query_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:n]
    return [corpus[i] for i in top_indices]

def search_bm25(query: str, n: int = 5):
    tokenized_query = preprocess(query)
    return bm25.get_top_n(tokenized_query, corpus, n=n)

def search_minilm(query: str, n: int = 5):
    embedding = model.encode(query)
    embedding = embedding.reshape(1, -1)
    faiss.normalize_L2(embedding)
    _scores, positions = index.search(embedding, k=n)
    hit_ids = doc_ids[positions[0]]
    return [id_to_entry[did] for did in hit_ids]

def search_specter(query: str, n: int = 5):
    embedding = specter_model.encode(query, convert_to_numpy=True)
    embedding = embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(embedding)
    _scores, positions = specter_index.search(embedding, k=n)
    hit_ids = specter_doc_ids[positions[0]]
    return [id_to_entry[did] for did in hit_ids]

def search_specter_reranked(query: str, n: int = 5, candidate_pool: int = 50):
    candidates = search_specter(query, n=candidate_pool)
    return rerank(query, candidates, n)

def search_minilm_reranked(query: str, n: int = 5, candidate_pool: int = 50):
    candidates = search_minilm(query, n=candidate_pool)
    return rerank(query, candidates, n)

def search_bge(query: str, n: int = 5):
    query_with_prefix = BGE_QUERY_PREFIX + query
    embedding = bge_model.encode([query_with_prefix], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embedding)
    _scores, positions = bge_index.search(embedding, k=n)
    hit_ids = bge_doc_ids[positions[0]]
    return [id_to_entry[did] for did in hit_ids]

def search_bge_reranked(query: str, n: int = 5, candidate_pool: int = 50):
    candidates = search_bge(query, n=candidate_pool)
    return rerank(query, candidates, n)

def search_bge_ft(query: str, n: int = 5):
    query_with_prefix = BGE_QUERY_PREFIX + query
    embedding = bge_ft_model.encode([query_with_prefix], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embedding)
    _scores, positions = bge_ft_index.search(embedding, k=n)
    hit_ids = bge_ft_doc_ids[positions[0]]
    return [id_to_entry[did] for did in hit_ids]

def search_bge_ft_reranked(query: str, n: int = 5, candidate_pool: int = 50):
    candidates = search_bge_ft(query, n=candidate_pool)
    return rerank(query, candidates, n)

def search_minilm_ft(query: str, n: int = 5):
    embedding = minilm_ft_model.encode(query)
    embedding = embedding.reshape(1, -1)
    faiss.normalize_L2(embedding)
    _scores, positions = minilm_ft_index.search(embedding, k=n)
    hit_ids = minilm_ft_doc_ids[positions[0]]
    return [id_to_entry[did] for did in hit_ids]

def search_minilm_ft_reranked(query: str, n: int = 5, candidate_pool: int = 50):
    candidates = search_minilm_ft(query, n=candidate_pool)
    return rerank(query, candidates, n)