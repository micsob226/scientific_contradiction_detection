import re
import pickle
from rank_bm25 import BM25Okapi
from data import corpus
from nltk.stem import SnowballStemmer
from pathlib import Path

stemmer = SnowballStemmer("english")

def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def search(query: str, n: int = 5):
    tokenized_query = preprocess(query)
    results = bm25.get_top_n(tokenized_query, corpus, n=n)
    return results


tokenized_corpus = []

for entry in corpus:
    title = entry["title"]
    abstract = " ".join(entry["abstract"])
    full_text = title + " " + abstract
    tokens = preprocess(full_text)
    tokenized_corpus.append(tokens)


bm25 = BM25Okapi(tokenized_corpus)

Path("results").mkdir(exist_ok=True)
with open("results/bm25_index.pkl", "wb") as file:
    pickle.dump(bm25, file)

queries = [
    "A strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage.",
    "ART substantially reduces infectiveness of HIV-positive people.",
    "Citrullinated proteins externalized in neutrophil extracellular traps act indirectly to disrupt the inflammatory cycle.",
    "Klf4 is not required for proper myeloid cell differentiation.",
    "MICAL redox enzymes regulate actin dynamics.",
    "Mice that lack Interferon-\u03b3 or its receptor are highly susceptible to experimental autoimmune myocarditis.",
    "Physical activity level has no association with the difference in maximal oxygen consumption between black and white youth.",
    "Physical activity level is associated with the difference in maximal oxygen consumption between black and white youth.", 
    "Primary cervical cancer screening with HPV detection has lower longitudinal sensitivity than conventional cytology to detect cervical intraepithelial neoplasia grade 2.", 
    "Primary cervical cytology screening with HPV detection has higher longitudinal sensitivity to detect severe cervical intraepithelial neoplasia than conventional cytology."
    ]

for q in queries:
    print("\nQUERY:", q)
    results = search(q)

    for r in results:
        print(f"- ({r['doc_id']}) {r['title']}")
