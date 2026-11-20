#!/usr/bin/env python3
"""Build the `easy` / `hard` candidate lists for the router from CALIBRATION cases only.

  python build_examples.py dataset.zip --split-file split.json --k 20 --max-chars 2000

Writes examples_fragment.yaml. In config.yaml, replace the `easy:` and `hard:` blocks
under routing.signals.complexity[difficulty] with the two blocks in that file (the
`description:` line stays). Prompts use the same format as routing_accuracy.py, so run
that script with the SAME --max-chars and --docs to evaluate on the held-out test cases.
Hard (Qwen) examples = hard + medium cases by default (--hard-tiers). gold.jsonl is used only to pick examples by tier.
"""
import argparse, json, pathlib, random, subprocess, tempfile, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--split-file", required=True)
ap.add_argument("--k", type=int, default=20, help="examples per tier, balanced across products")
ap.add_argument("--max-chars", type=int, default=2000)
ap.add_argument("--docs", default="claim_form,discharge_summary,hospital_bill,policy_certificate")
ap.add_argument("--hard-tiers", default="hard,medium", help="gold tiers used for the hard (Qwen) examples")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--out", default="examples_fragment.yaml")
a = ap.parse_args()

tmp = pathlib.Path(tempfile.mkdtemp())
zipfile.ZipFile(a.zip).extractall(tmp)
root = next(tmp.glob("**/public"))
gold = {j["case_id"]: j for j in map(json.loads, (root.parent / "gold.jsonl").read_text().splitlines())}
split = json.load(open(a.split_file))

def text(p):  # same extractor logic as routing_accuracy.py
    try:
        return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages).strip()

def prompt_for(cid):
    d = root / "cases" / cid / "documents"
    parts = [f"## {n}\n{text(d / (n + '.pdf'))}" for n in a.docs.split(",") if (d / (n + ".pdf")).exists()]
    p = "Settle this health insurance claim from the documents below.\n\n" + "\n\n".join(parts)
    return p[: a.max_chars] if a.max_chars else p

rnd = random.Random(a.seed)
out = []
prods = sorted({g["product_id"] for g in gold.values()})
for tier, srcs in (("easy", ["easy"]), ("hard", a.hard_tiers.split(","))):
    chosen = []
    per = max(1, a.k // (len(prods) * len(srcs)))
    for src in srcs:
        for prod in prods:
            pool = sorted(c for c, g in gold.items()
                          if split[c] == "calib" and g["difficulty"] == src and g["product_id"] == prod)
            chosen += rnd.sample(pool, per)
    out.append(f"        {tier}:\n          candidates:")
    for c in chosen:
        out.append(f"            # {c} ({gold[c]['difficulty']})")
        out.append(f"            - {json.dumps(prompt_for(c), ensure_ascii=False)}")
open(a.out, "w").write("\n".join(out) + "\n")
print(f"wrote {a.out}: {a.k} easy + {a.k} hard examples, <= {a.max_chars} chars each, calibration cases only")
