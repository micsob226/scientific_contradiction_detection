from data import load_jsonl
import json


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

claims_train = load_jsonl("data/claims_train.jsonl")

claim_to_truth = {}
for claim in claims_train:
    claim_to_truth[claim["claim"]] = claim["cited_doc_ids"]

eval_set = []
for query in queries:
    ground_truth = claim_to_truth[query]
    eval_set.append({"query": query, "ground_truth": ground_truth})

with open("results/eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)