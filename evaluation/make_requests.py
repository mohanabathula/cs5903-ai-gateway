import csv, json, sys
csv.field_size_limit(sys.maxsize)
rows = list(csv.DictReader(open("routing_accuracy.csv", newline="")))
want = {"easy": "easy-route", "hard": "hard-route"}
for tier, dec in want.items():
    r = next(x for x in rows if x["gold"] == tier and x["decision"] == dec)
    body = {"model": "auto", "max_tokens": 300,
            "messages": [{"role": "user", "content": r["prompt"]}]}
    json.dump(body, open(f"{tier}_claim.json", "w"))
    print(tier, r["case_id"], r["decision"], r["selected_model"], r["margin"])
