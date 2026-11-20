#!/usr/bin/env python3
"""How well does the router's easy/hard signal match the dataset's gold difficulty?

Runs on netx3 (needs the `vllm-sr` CLI, VLLM_SR_PORT_OFFSET set, stack running).
Only previews routing: no model is called, so Qwen/gemma response problems don't matter.
gold.jsonl is used for scoring only; it is never part of the prompt.

  python routing_accuracy.py dataset.zip --per-tier 60          # 180 cases, balanced
  python routing_accuracy.py dataset.zip --per-tier 60 --max-chars 3000
  python routing_accuracy.py --from-csv routing_accuracy.csv    # rescore saved results
"""
import argparse, collections, csv, json, pathlib, random, subprocess, sys, tempfile, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("zip", nargs="?")
ap.add_argument("--per-tier", type=int, default=60)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--docs", default="claim_form,discharge_summary,hospital_bill,policy_certificate")
ap.add_argument("--max-chars", type=int, default=0, help="truncate prompt (0 = full text)")
ap.add_argument("--out", default="routing_accuracy.csv")
ap.add_argument("--split-file", help="split.json from the classification split")
ap.add_argument("--subset", default="test", choices=["calib", "test"])
ap.add_argument("--from-csv")
a = ap.parse_args()

def auc(pos, neg):  # P(score_pos > score_neg), ties count half
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, rsum = 0, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rsum += avg * sum(l for _, l in allv[i:j])
        i = j
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)

def report(rows):
    rows = [r for r in rows if r["pred"] in ("easy", "medium", "hard")]
    toq = lambda r: r["pred"] != "easy"
    tiers = ("easy", "medium", "hard")
    print(f"\nn={len(rows)} scored\n")
    print("share routed to Qwen (router label medium or hard), by gold tier:")
    for t in tiers:
        v = [r for r in rows if r["gold"] == t]
        if v:
            print(f"  {t:7s} n={len(v):3d}  {sum(toq(r) for r in v)/len(v):.0%}")
    for p in sorted({r["product"] for r in rows}):
        line = []
        for t in tiers:
            v = [r for r in rows if r["gold"] == t and r["product"] == p]
            if v:
                line.append(f"{t}={sum(toq(x) for x in v)/len(v):.0%}")
        print(f"    {p:24s}", "  ".join(line))
    m = lambda t: [float(r["margin"]) for r in rows if r["gold"] == t]
    print("\nranking quality of the margin score (AUC; 0.5 = no signal, 1.0 = perfect):")
    print(f"  hard vs easy   : {auc(m('hard'), m('easy')):.3f}")
    print(f"  hard vs rest   : {auc(m('hard'), m('easy') + m('medium')):.3f}")
    print("\nmean margin by tier:", {t: round(sum(m(t)) / len(m(t)), 3) for t in tiers if m(t)})
    # accuracy for hard-vs-easy with the router's own cut, and with the best possible cut
    he = [r for r in rows if r["gold"] in ("easy", "hard")]
    if he:
        acc = sum(toq(r) == (r["gold"] == "hard") for r in he) / len(he)
        best = max(((sum((float(r["margin"]) > th) == (r["gold"] == "hard") for r in he) / len(he), th)
                    for th in sorted({float(r["margin"]) for r in he})), key=lambda x: x[0])
        print(f"\neasy-vs-hard accuracy: current config {acc:.0%}; best possible single cut {best[0]:.0%} at margin>{best[1]:.3f}")
    # policy view: easy -> gemma, medium + hard -> Qwen
    need = [r["gold"] != "easy" for r in rows]
    qwen = [toq(r) for r in rows]
    n_need, n_easy = sum(need), len(rows) - sum(need)
    miss = sum(n and not q for n, q in zip(need, qwen))      # needed Qwen, sent to gemma
    waste = sum((not n) and q for n, q in zip(need, qwen))   # easy, sent to Qwen
    mg = lambda flag: [float(r["margin"]) for r, n in zip(rows, need) if n == flag]
    print("\n--- routing policy: easy -> gemma, medium+hard -> Qwen ---")
    print(f"AUC (needs Qwen vs easy): {auc(mg(True), mg(False)):.3f}")
    print(f"routing accuracy: {sum(n == q for n, q in zip(need, qwen)) / len(rows):.0%}")
    print(f"missed (medium/hard sent to gemma): {miss}/{n_need} = {miss / max(n_need, 1):.0%}")
    print(f"wasted (easy sent to Qwen):         {waste}/{n_easy} = {waste / max(n_easy, 1):.0%}")
    print(f"share of ALL traffic sent to gemma: {sum(not q for q in qwen) / len(rows):.0%}")
    cuts = sorted({float(r["margin"]) for r in rows})
    bt = max(cuts, key=lambda t: sum((float(r["margin"]) > t) == n for r, n in zip(rows, need)))
    print("\n--- gold tier vs router label ---")
    print(f"{'gold':8s}{'n':>5s}{'->easy':>9s}{'->medium':>10s}{'->hard':>9s}")
    for t in tiers:
        v = [r for r in rows if r["gold"] == t]
        print(f"{t:8s}{len(v):5d}" + "".join(f"{sum(r['pred'] == p for r in v):{w}d}" for p, w in (("easy", 9), ("medium", 10), ("hard", 9))))
    print(f"best single cut for this policy: margin>{bt:.3f} -> "
          f"{sum((float(r['margin']) > bt) == n for r, n in zip(rows, need)) / len(rows):.0%}")

if a.from_csv:
    report(list(csv.DictReader(open(a.from_csv))))
    sys.exit()

tmp = pathlib.Path(tempfile.mkdtemp())
zf = zipfile.ZipFile(a.zip); zf.extractall(tmp)
root = next(tmp.glob("**/public"))
gold = {j["case_id"]: j for j in map(json.loads, (root.parent / "gold.jsonl").read_text().splitlines())}
SPLIT = json.load(open(a.split_file)) if a.split_file else {}
rnd = random.Random(a.seed)
ids = []
for t in ("easy", "medium", "hard"):
    pool = sorted(c for c, g in gold.items() if g["difficulty"] == t
                  and (not a.split_file or SPLIT[c] == a.subset))
    ids += rnd.sample(pool, min(a.per_tier, len(pool)))
rnd.shuffle(ids)

def text(p):
    try:
        return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:  # no poppler / no root: fall back to pypdf (pip install pypdf)
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages).strip()

FIELDS = ["case_id", "product", "gold", "pred", "decision", "selected_model", "margin", "easy", "hard", "chars", "prompt"]
fh = open(a.out, "w", newline="")
wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader()
rows = []
for i, cid in enumerate(ids, 1):
    d = root / "cases" / cid / "documents"
    parts = [f"## {n}\n{text(d / (n + '.pdf'))}" for n in a.docs.split(",") if (d / (n + ".pdf")).exists()]
    prompt = "Settle this health insurance claim from the documents below.\n\n" + "\n\n".join(parts)
    if a.max_chars:
        prompt = prompt[: a.max_chars]
    o = subprocess.run(["vllm-sr", "route", "preview", "--json", "--prompt", prompt], capture_output=True, text=True)
    try:
        j = json.loads(o.stdout)
        sv = j.get("signal_values", {})
        ms = str(j["decision_result"]["matched_signals"])
        pred = "hard" if "difficulty:hard" in ms else "medium" if "difficulty:medium" in ms else "easy"
        rows.append(dict(case_id=cid, product=gold[cid]["product_id"], gold=gold[cid]["difficulty"], pred=pred,
                         decision=j.get("routing_decision"), selected_model=j.get("selected_model"),
                         margin=round(sv.get("complexity:difficulty:text_margin", 0), 4),
                         easy=round(sv.get("complexity:difficulty:text_easy_score", 0), 4),
                         hard=round(sv.get("complexity:difficulty:text_hard_score", 0), 4), chars=len(prompt), prompt=prompt))
    except Exception:
        rows.append(dict(case_id=cid, product=gold[cid]["product_id"], gold=gold[cid]["difficulty"], pred="ERROR",
                         decision="", selected_model="", margin=0, easy=0, hard=0, chars=len(prompt), prompt=prompt))
    row = rows[-1]
    row.setdefault("decision", ""); row.setdefault("selected_model", "")
    wr.writerow({k: row.get(k, "") for k in FIELDS}); fh.flush()
    if i % 20 == 0:
        print(f"{i}/{len(ids)}", file=sys.stderr)
fh.close()
print("errors:", sum(r["pred"] == "ERROR" for r in rows), " saved:", a.out)
report(rows)
