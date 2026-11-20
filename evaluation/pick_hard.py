import csv, json, sys
csv.field_size_limit(sys.maxsize)
rows = list(csv.DictReader(open("routing_accuracy.csv", newline="")))
hard = [r for r in rows if r["gold"] == "hard" and r["decision"] == "hard-route"]
hard.sort(key=lambda r: float(r["margin"]), reverse=True)
print("Top hard cases by margin:")
for r in hard[:5]:
    print(" ", r["case_id"], r["product"], "margin", r["margin"], "hard", r["hard"], "easy", r["easy"])
pick = next((r for r in hard if r["case_id"] == sys.argv[1]), hard[0]) if len(sys.argv) > 1 else hard[0]
body = {"model": "auto", "max_tokens": 300,
        "messages": [{"role": "user", "content": pick["prompt"]}]}
json.dump(body, open("hard_claim.json", "w"))
print("Wrote hard_claim.json from", pick["case_id"], "margin", pick["margin"])
