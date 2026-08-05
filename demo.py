from retrieval import search_bge_ft

CLAIMS = [
    "Antioxidant supplements for cancer prevention",

    "The availability of safe places to study is effective at decreasing homelessness.",

    "Statins reduce all-cause mortality in healthy adults without prior cardiovascular disease.",
]

for claim in CLAIMS:
    print("-" * 20)
    print(f"CLAIM: {claim}")
    print("-" * 20)

    results = search_bge_ft(claim, n=7)
    for i, r in enumerate(results, 1):
        abstract_snippet = " ".join(r["abstract"])[:200] + "..."
        print(f"\n  {i}. {r['title']}")
        print(f"     doc_id: {r['doc_id']}")
        print(f"     {abstract_snippet}")
    print()