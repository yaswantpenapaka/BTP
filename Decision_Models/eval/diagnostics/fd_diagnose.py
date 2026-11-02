"""Offline diagnosis of Fast Decisions results. No API calls. Reads saved rows only.
Writes eval/diagnostics/fd_report.md and CSVs. Run from Decision_Models:
    python eval\\diagnostics\\fd_diagnose.py
"""
import csv, difflib, itertools, json, math, re, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from eval.datasets import load_items, DATASETS_DIR  # noqa

MODELS = ["jev", "luna"]
RAW = ROOT / "eval" / "outputs" / "raw"
items = load_items("fastdecisions")
IT = {i.item_id: i for i in items}
R = {m: {} for m in MODELS}
for m in MODELS:
    for line in open(RAW / f"{m}__fastdecisions.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); R[m][r["item_id"]] = r
assert all(len(R[m]) == 255 for m in MODELS)

OUT = []
def P(s=""): OUT.append(s)
def table(header, rows):
    P("| " + " | ".join(header) + " |"); P("|" + "---|" * len(header))
    for r in rows: P("| " + " | ".join(str(x) for x in r) + " |")
    P()
def pct(a, b): return f"{100*a/b:.1f}%" if b else "n/a"
def f2(x): return "n/a" if x is None else f"{x:.3f}"

# ------------------------------------------------------------ structure
# head -> list of questions, per item
def heads_of(item):
    d = defaultdict(list)
    for q in item.questions: d[q.head].append(q)
    return d
def qres(m, item, q):
    return R[m][item.item_id]["questions"][q.name]
def head_pred(m, item, qs):
    """predicted label list (set) for a head, from saved question choices"""
    if qs[0].kind == "label_yesno":
        return [q.label for q in qs if qres(m, item, q)["choice"] == "yes"]
    return [qres(m, item, qs[0])["choice"]]
def head_gold(item, qs):
    if qs[0].kind == "label_yesno": return [q.label for q in qs if q.gold == "yes"]
    return [qs[0].gold]
def head_correct(m, item, qs):
    return set(head_pred(m, item, qs)) == set(head_gold(item, qs))

P("# Fast Decisions diagnosis (offline, saved rows only)")
P()
# sanity
errs = {m: sum(bool(r["error"]) for r in R[m].values()) for m in MODELS}
nonec = {m: sum(1 for i in items for q in i.questions if qres(m, i, q)["choice"] is None) for m in MODELS}
P(f"Rows: {len(items)} items, {sum(len(i.questions) for i in items)} questions. Rows with error: {errs}. Questions with no choice: {nonec}.")
rec_ok = {m: sum(R[m][i.item_id]["correct"] == int(all(head_correct(m, i, qs) for qs in heads_of(i).values())) for i in items) for m in MODELS}
P(f"Recomputed item-exact-match equals saved `correct` for: {rec_ok} of 255 items.")
P()

# ------------------------------------------------------------ 1
P("## 1. Accuracy: item vs head vs question")
def stats(m, sel=lambda i: True):
    its = [i for i in items if sel(i)]
    item_c = sum(R[m][i.item_id]["correct"] == 1 for i in its)
    hd = [(head_correct(m, i, qs), qs[0].kind) for i in its for qs in heads_of(i).values()]
    qq = [(qres(m, i, q)["correct"], q.kind) for i in its for q in i.questions]
    return dict(n=len(its), item=item_c, heads=len(hd), head_c=sum(h for h, _ in hd),
                q=len(qq), q_c=sum(c == 1 for c, _ in qq), hd=hd, qq=qq)
rows = []
for m in MODELS:
    s = stats(m)
    sl = [c for c, k in s["qq"] if k == "single_label"]; ml = [c for c, k in s["qq"] if k == "label_yesno"]
    hs = [h for h, k in s["hd"] if k == "single_label"]; hm = [h for h, k in s["hd"] if k == "label_yesno"]
    rows.append([m, pct(s["item"], s["n"]) + f" ({s['item']}/{s['n']})",
                 pct(s["head_c"], s["heads"]) + f" ({s['head_c']}/{s['heads']})",
                 pct(s["q_c"], s["q"]) + f" ({s['q_c']}/{s['q']})",
                 pct(sum(c == 1 for c in sl), len(sl)) + f" (n={len(sl)})",
                 pct(sum(c == 1 for c in ml), len(ml)) + f" (n={len(ml)})",
                 pct(sum(hs), len(hs)) + f" (n={len(hs)})", pct(sum(hm), len(hm)) + f" (n={len(hm)})"])
table(["model", "ITEM exact (all heads)", "HEAD exact (card-style)", "per-question", "q: single-label", "q: multi-label per-label yes/no", "heads: single", "heads: multi-label (set exact)"], rows)
both = sum(R["jev"][i.item_id]["correct"] == 1 and R["luna"][i.item_id]["correct"] == 1 for i in items)
neither = sum(R["jev"][i.item_id]["correct"] == 0 and R["luna"][i.item_id]["correct"] == 0 for i in items)
P(f"Items: both right {both}, neither right {neither}, jev-only {sum(R['jev'][i.item_id]['correct']==1 and R['luna'][i.item_id]['correct']==0 for i in items)}, luna-only {sum(R['jev'][i.item_id]['correct']==0 and R['luna'][i.item_id]['correct']==1 for i in items)}")
P()
P("### Per domain (item exact / head exact / per-question)")
doms = sorted({i.meta["domain"] for i in items})
rows, csvrows = [], []
for d in doms:
    sel = lambda i, d=d: i.meta["domain"] == d
    row = [d, stats("jev", sel)["n"], sum(len(i.questions) for i in items if sel(i)), len(heads_of(next(i for i in items if sel(i))))]
    for m in MODELS:
        s = stats(m, sel)
        row += [pct(s["item"], s["n"]), pct(s["head_c"], s["heads"]), pct(s["q_c"], s["q"])]
        csvrows.append([d, m, s["n"], s["item"], s["heads"], s["head_c"], s["q"], s["q_c"]])
    rows.append(row)
table(["domain", "items", "questions", "heads", "jev item", "jev head", "jev q", "luna item", "luna head", "luna q"], rows)
with open(HERE / "fd_per_domain.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["domain", "model", "items", "item_correct", "heads", "head_correct", "questions", "q_correct"]); w.writerows(csvrows)

# per head task
P("### Per head (task), with majority-class baseline on the same test items")
head_keys = sorted({(i.meta["domain"], h) for i in items for h in heads_of(i)})
rows = []
for d, h in head_keys:
    its = [i for i in items if i.meta["domain"] == d and h in heads_of(i)]
    golds = [tuple(sorted(head_gold(i, heads_of(i)[h]))) for i in its]
    maj, mc = Counter(golds).most_common(1)[0]
    kind = "multi" if heads_of(its[0])[h][0].kind == "label_yesno" else "single"
    nl = len(its[0].questions) and len([q for q in heads_of(its[0])[h]]) if kind == "multi" else len(heads_of(its[0])[h][0].options)
    row = [d, h, kind, nl, len(its), f"{100*mc/len(its):.0f}% ({'/'.join(maj) or 'empty'})"]
    for m in MODELS:
        row.append(pct(sum(head_correct(m, i, heads_of(i)[h]) for i in its), len(its)))
    rows.append(row)
table(["domain", "head", "kind", "#labels", "n", "majority baseline", "jev", "luna"], rows)
# majority-baseline item-level
maj = {}
for d, h in head_keys:
    its = [i for i in items if i.meta["domain"] == d and h in heads_of(i)]
    maj[(d, h)] = Counter(tuple(sorted(head_gold(i, heads_of(i)[h]))) for i in its).most_common(1)[0][0]
mb = sum(all(tuple(sorted(head_gold(i, qs))) == maj[(i.meta["domain"], h)] for h, qs in heads_of(i).items()) for i in items)
P(f"Item-level exact match of an always-majority predictor (oracle priors from test gold): {pct(mb, 255)}")
P()

# ------------------------------------------------------------ 2
P("## 2. Questions / heads per item and the all-heads penalty")
dist = Counter(len(i.questions) for i in items); hdist = Counter(len(heads_of(i)) for i in items)
P(f"Questions per item distribution: {dict(sorted(dist.items()))}")
P(f"Heads per item distribution: {dict(sorted(hdist.items()))}")
P()
rows = []
for m in MODELS:
    # independence: per-(domain,head) head accuracy; product over heads of item
    hacc = {}
    for d, h in head_keys:
        its = [i for i in items if i.meta["domain"] == d and h in heads_of(i)]
        hacc[(d, h)] = sum(head_correct(m, i, heads_of(i)[h]) for i in its) / len(its)
    exp_head = sum(math.prod(hacc[(i.meta["domain"], h)] for h in heads_of(i)) for i in items) / 255
    qacc = {}
    for d, h in head_keys:
        its = [i for i in items if i.meta["domain"] == d and h in heads_of(i)]
        qs = [qres(m, i, q)["correct"] == 1 for i in its for q in heads_of(i)[h]]
        qacc[(d, h)] = sum(qs) / len(qs)
    exp_q = sum(math.prod(qacc[(i.meta["domain"], q.head)] for q in i.questions) for i in items) / 255
    act = sum(R[m][i.item_id]["correct"] == 1 for i in items) / 255
    gq = stats(m); gqacc = gq["q_c"] / gq["q"]
    exp_glob = sum(gqacc ** len(i.questions) for i in items) / 255
    rows.append([m, f"{100*act:.1f}%", f"{100*exp_head:.1f}%", f"{100*exp_q:.1f}%", f"{100*exp_glob:.1f}%"])
table(["model", "actual item exact", "expected if heads independent (product of per-head acc)", "expected if questions independent (per-(domain,head) q acc)", "global q-acc ^ #q"], rows)
rows = []
for nh in sorted(hdist):
    its = [i for i in items if len(heads_of(i)) == nh]
    row = [f"{nh} head(s)", len(its)]
    for m in MODELS:
        row.append(pct(sum(R[m][i.item_id]["correct"] == 1 for i in its), len(its)))
    for m in MODELS:
        hh = [head_correct(m, i, qs) for i in its for qs in heads_of(i).values()]
        row.append(pct(sum(hh), len(hh)))
    rows.append(row)
table(["#heads", "items", "jev item", "luna item", "jev per-head", "luna per-head"], rows)
rows = []
for lo, hi in [(1, 1), (2, 3), (4, 8), (9, 20)]:
    its = [i for i in items if lo <= len(i.questions) <= hi]
    if not its: continue
    row = [f"{lo}-{hi} q" if lo != hi else f"{lo} q", len(its)]
    for m in MODELS: row.append(pct(sum(R[m][i.item_id]["correct"] == 1 for i in its), len(its)))
    rows.append(row)
table(["#questions", "items", "jev item", "luna item"], rows)

# ------------------------------------------------------------ 3
P("## 3. Multi-label (per-label yes/no)")
ml_items = [i for i in items if any(qs[0].kind == "label_yesno" for qs in heads_of(i).values())]
ml_heads = [(d, h) for d, h in head_keys if heads_of(next(i for i in items if i.meta["domain"] == d and h in heads_of(i)))[h][0].kind == "label_yesno"]
P(f"Items containing a multi-label head: {len(ml_items)}; multi-label heads: {sum(1 for i in items for qs in heads_of(i).values() if qs[0].kind=='label_yesno')}; per-label yes/no questions: {sum(1 for i in items for q in i.questions if q.kind=='label_yesno')}")
rows = []
for m in MODELS:
    for tag, sel in [("ALL", None)] + [(h, h) for _, h in ml_heads]:
        tp = fp = fn = tn = 0
        for i in items:
            for q in i.questions:
                if q.kind != "label_yesno" or (sel and q.head != sel): continue
                c = qres(m, i, q)["choice"]
                if q.gold == "yes" and c == "yes": tp += 1
                elif q.gold == "no" and c == "yes": fp += 1
                elif q.gold == "yes": fn += 1
                else: tn += 1
        n = tp + fp + fn + tn
        pr = tp / (tp + fp) if tp + fp else None; rc = tp / (tp + fn) if tp + fn else None
        f1 = 2 * pr * rc / (pr + rc) if pr and rc else None
        rows.append([m, tag, n, tp, fp, fn, tn, f2(pr), f2(rc), f2(f1), pct(tp + fn, n), pct(tp + fp, n), pct(tp + tn, n)])
table(["model", "head", "n", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "gold=yes rate", "pred=yes rate", "q acc"], rows)
P("Set comparison per multi-label head (predicted label set vs gold set):")
rows = []
for m in MODELS:
    for tag in ["ALL"] + [h for _, h in ml_heads]:
        eq = d1 = d1fp = d1fn = d2 = d3 = empty = jac = n = 0; gsz = Counter(); psz = Counter()
        for i in items:
            for h, qs in heads_of(i).items():
                if qs[0].kind != "label_yesno" or (tag != "ALL" and h != tag): continue
                g, p = set(head_gold(i, qs)), set(head_pred(m, i, qs)); n += 1
                sd = len(g ^ p); gsz[len(g)] += 1; psz[len(p)] += 1
                eq += sd == 0; d1 += sd == 1; d1fp += sd == 1 and len(p) > len(g); d1fn += sd == 1 and len(p) < len(g)
                d2 += sd == 2; d3 += sd >= 3; empty += len(p) == 0
                jac += len(g & p) / len(g | p) if g | p else 1
        rows.append([m, tag, n, pct(eq, n), pct(d1, n) + f" (+1 extra {d1fp}, 1 missing {d1fn})", pct(d2, n), pct(d3, n), pct(empty, n), f"{jac/n:.3f}",
                     "mean gold size %.2f / pred size %.2f" % (sum(k * v for k, v in gsz.items()) / n, sum(k * v for k, v in psz.items()) / n)])
table(["model", "head", "n heads", "set equal", "differs by 1", "by 2", "by >=3", "empty pred", "mean Jaccard", "sizes"], rows)
P("Gold set size distribution (multi-label heads): " + str(dict(Counter(len(head_gold(i, qs)) for i in items for qs in heads_of(i).values() if qs[0].kind == "label_yesno"))))
P()
P("Threshold sweep on P(yes) for multi-label head exact match (offline; probs saved). 0.5 = what was scored. 'top1+' = labels with p>=0.5, or the argmax label if none:")
rows = []
for m in MODELS:
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, "top1+"]:
        ok = n = 0
        for i in items:
            for h, qs in heads_of(i).items():
                if qs[0].kind != "label_yesno": continue
                py = {q.label: qres(m, i, q)["probs"]["yes"] for q in qs}
                if th == "top1+":
                    p = {l for l, v in py.items() if v >= 0.5} or {max(py, key=py.get)}
                else:
                    p = {l for l, v in py.items() if v >= th}
                ok += p == set(head_gold(i, qs)); n += 1
        rows.append([m, th, pct(ok, n)])
table(["model", "threshold", "multi-label head exact"], rows)
# effect on item acc of top1+ 
for m in MODELS:
    cnt = 0
    for i in items:
        ok = True
        for h, qs in heads_of(i).items():
            if qs[0].kind == "label_yesno":
                py = {q.label: qres(m, i, q)["probs"]["yes"] for q in qs}
                p = {l for l, v in py.items() if v >= 0.5} or {max(py, key=py.get)}
                ok &= p == set(head_gold(i, qs))
            else:
                ok &= head_correct(m, i, qs)
        cnt += ok
    P(f"- {m}: item exact with top1+ rule for multi-label heads: {pct(cnt,255)} (was {pct(sum(R[m][i.item_id]['correct']==1 for i in items),255)})")
P()

# ------------------------------------------------------------ 4
P("## 4. Single-label heads: where they fail")
sl_keys = [k for k in head_keys if k not in ml_heads]
hrows = []
for d, h in sl_keys:
    its = [i for i in items if i.meta["domain"] == d and h in heads_of(i)]
    a = {m: sum(head_correct(m, i, heads_of(i)[h]) for i in its) / len(its) for m in MODELS}
    hrows.append(((a["jev"] + a["luna"]) / 2, d, h, len(its), a))
hrows.sort()
P("Worst single-label heads by mean accuracy of the two models:")
table(["domain", "head", "n", "jev", "luna"], [[d, h, n, f"{100*a['jev']:.0f}%", f"{100*a['luna']:.0f}%"] for _, d, h, n, a in hrows[:10]])
P("Single-label question summary:")
sl_q = [(i, q) for i in items for q in i.questions if q.kind == "single_label"]
jw = [(i, q) for i, q in sl_q if qres("jev", i, q)["correct"] == 0]
lw = [(i, q) for i, q in sl_q if qres("luna", i, q)["correct"] == 0]
bw = [(i, q) for i, q in jw if qres("luna", i, q)["correct"] == 0]
same = [(i, q) for i, q in bw if qres("jev", i, q)["choice"] == qres("luna", i, q)["choice"]]
P(f"- {len(sl_q)} single-label questions; jev wrong {len(jw)}, luna wrong {len(lw)}, both wrong {len(bw)}, of which SAME wrong label {len(same)} ({pct(len(same), len(bw))}), different wrong labels {len(bw)-len(same)}")
conf = [(i, q) for i, q in same if min(qres("jev", i, q)["confidence"], qres("luna", i, q)["confidence"]) >= 0.8]
P(f"- both wrong with the same label AND both confidence >= 0.8 (gold-noise / ambiguity suspects): {len(conf)}")
P()
for _, d, h, n, a in hrows[:6]:
    cnt = {m: Counter() for m in MODELS}
    for i in items:
        if i.meta["domain"] != d or h not in heads_of(i): continue
        q = heads_of(i)[h][0]
        for m in MODELS:
            c = qres(m, i, q)["choice"]
            if c != q.gold: cnt[m][(q.gold, c)] += 1
    P(f"**{d}.{h}** top errors (gold -> pred: count): " + "; ".join(f"{m}: " + ", ".join(f"{g}->{p}:{k}" for (g, p), k in cnt[m].most_common(5)) for m in MODELS))
P()
# examples
P("### 15 misclassified examples from the worst single-label heads")
worst = {(d, h) for _, d, h, _, _ in hrows[:8]}
pool = [(i, q) for i, q in sl_q if (i.meta["domain"], q.head) in worst]
def cat(i, q):
    j, l = qres("jev", i, q), qres("luna", i, q)
    if j["correct"] == 0 and l["correct"] == 0: return "both wrong, SAME" if j["choice"] == l["choice"] else "both wrong, DIFFERENT"
    if j["correct"] == 0: return "only Jev wrong"
    if l["correct"] == 0: return "only Luna wrong"
    return None
byc = defaultdict(list)
for i, q in sorted(pool, key=lambda t: t[0].item_id):
    c = cat(i, q)
    if c: byc[c].append((i, q))
pick = byc["both wrong, SAME"][:7] + byc["both wrong, DIFFERENT"][:4] + byc["only Jev wrong"][:2] + byc["only Luna wrong"][:2]
ex_csv = []
for k, (i, q) in enumerate(pick, 1):
    j, l = qres("jev", i, q), qres("luna", i, q)
    top = lambda r: ", ".join(f"{a}={b:.2f}" for a, b in sorted(r["probs"].items(), key=lambda t: -t[1])[:3])
    txt = i.state["input"].replace("\n", " ")
    P(f"{k}. [{cat(i, q)}] `{i.item_id}` head `{q.head}`")
    P(f"   - input: {txt[:380]}{'...' if len(txt) > 380 else ''}")
    P(f"   - candidates: {', '.join(q.option_ids)}")
    P(f"   - gold: **{q.gold}** | Jev: {j['choice']} ({top(j)}) | Luna: {l['choice']} ({top(l)})")
    ex_csv.append([i.item_id, q.head, cat(i, q), q.gold, j["choice"], l["choice"], txt])
with open(HERE / "fd_examples.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["item_id", "head", "category", "gold", "jev", "luna", "input"]); w.writerows(ex_csv)
P()

# ------------------------------------------------------------ 5
P("## 5. Adapter vs dataset card")
rawrows = {}
for f in sorted((DATASETS_DIR / "FastDecisions" / "raw").glob("*.jsonl")):
    rawrows[f.stem] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
test_raw = {json.loads(l)["item_id"]: json.loads(l) for l in open(DATASETS_DIR / "FastDecisions" / "test" / "test.jsonl", encoding="utf-8")}
# test rows vs raw
mism = 0
for iid, r in test_raw.items():
    dom, idx = r["domain"], int(iid.rsplit("_", 1)[1])
    rr = rawrows[dom][idx]
    mism += not (rr["input"] == r["input"] and rr["output"] == r["output"])
P(f"- test rows vs original raw file line (item_id index): {mism} mismatches of {len(test_raw)}.")
checks = Counter()
for iid, r in test_raw.items():
    for hd in r["output"]["classifications"]:
        labs, tl = hd["labels"], hd["true_label"]
        checks["heads"] += 1
        checks["gold_not_in_labels"] += any(t not in labs for t in tl)
        checks["dup_candidate_labels"] += len(set(labs)) != len(labs)
        checks["labels_not_stripped_or_nonlower"] += any(l != l.strip() or l != l.lower() for l in labs + tl)
        checks["single_with_len_ne_1"] += (not hd.get("multi_label")) and len(tl) != 1
        checks["multi_with_0_gold"] += bool(hd.get("multi_label")) and len(tl) == 0
        checks["multi_with_1_gold"] += bool(hd.get("multi_label")) and len(tl) == 1
        checks["multi_with_2plus_gold"] += bool(hd.get("multi_label")) and len(tl) >= 2
        checks["gold_listed_duplicates"] += len(set(tl)) != len(tl)
        checks["heads_in_row_same_task_name"] += 0
    names = [hd["task"] for hd in r["output"]["classifications"]]
    checks["items_with_duplicate_task_names"] += len(set(names)) != len(names)
P("- data integrity: " + ", ".join(f"{k}={v}" for k, v in checks.items()))
# adapter structure
bad_opts = bad_name = 0
for i in items:
    heads = test_raw[i.item_id]["output"]["classifications"]
    for h, qs in heads_of(i).items():
        hd = next(x for x in heads if x["task"] == h)
        if qs[0].kind == "single_label":
            bad_opts += qs[0].option_ids != hd["labels"]; bad_name += qs[0].gold != hd["true_label"][0]
        else:
            bad_opts += [q.label for q in qs] != hd["labels"]
            bad_name += set(q.label for q in qs if q.gold == "yes") != set(hd["true_label"])
P(f"- adapter options/labels differ from card labels (order or content): {bad_opts}; adapter gold != card true_label: {bad_name}")
P("- Option keys: single-label options are sent with description == the label string itself (no extra description); yes/no heads use labels 'yes'/'no' as-is.")
P(f"- Rows whose chosen option is outside the option set (would be dropped): {sum('not in options' in (R[m][i]['error']) for m in MODELS for i in R[m])}; rows with notes about missing/extra option keys: "
  f"{sum(bool(qq['notes']) for m in MODELS for r in R[m].values() for qq in r['questions'].values())}")
# alt scoring
def norm(s): return re.sub(r"[\s_\-]+", "", s.lower())
alt = defaultdict(dict)
for m in MODELS:
    base = c_ci = c_set = c_list = c_norm = 0
    for i in items:
        raw_h = test_raw[i.item_id]["output"]["classifications"]
        ok_strict = ok_ci = ok_list = ok_norm = True
        for hd in raw_h:
            qs = heads_of(i)[hd["task"]]
            pred = head_pred(m, i, qs); gold = hd["true_label"]
            ok_strict &= set(pred) == set(gold)
            ok_ci &= {p.lower().strip() for p in pred} == {g.lower().strip() for g in gold}
            ok_norm &= {norm(p) for p in pred} == {norm(g) for g in gold}
            # list compare: gold order vs candidate-label order
            ok_list &= pred == gold
        base += ok_strict; c_ci += ok_ci; c_norm += ok_norm; c_list += ok_list
    alt[m] = dict(strict_set=base, case_insens=c_ci, normalized=c_norm, ordered_list=c_list)
table(["model", "set-exact (used)", "case-insensitive", "normalized (strip _ - space)", "ordered-list compare (pred order = candidate order)"],
      [[m, pct(a["strict_set"], 255), pct(a["case_insens"], 255), pct(a["normalized"], 255), pct(a["ordered_list"], 255)] for m, a in alt.items()])
# gold order vs candidate order for multi gold
ordd = sum(1 for r in test_raw.values() for hd in r["output"]["classifications"] if hd.get("multi_label") and len(hd["true_label"]) > 1 and hd["true_label"] != [l for l in hd["labels"] if l in hd["true_label"]])
P(f"- multi-label gold lists NOT in candidate-label order (so ordered-list compare would be wrong, set compare is right): {ordd}")
P()

# ------------------------------------------------------------ 6
P("## 6. Duplicates and ambiguity")
allrows = [(dom, k, r) for dom, rs in rawrows.items() for k, r in enumerate(rs)]
def gold_sig(r): return json.dumps({h["task"]: sorted(h["true_label"]) for h in r["output"]["classifications"]}, sort_keys=True)
byin = defaultdict(list)
for dom, k, r in allrows: byin[(dom, re.sub(r"\s+", " ", r["input"].strip().lower()))].append((k, gold_sig(r)))
dup = {k: v for k, v in byin.items() if len(v) > 1}
P(f"- full dev set (1,700 rows): identical (normalized) inputs within a domain: {len(dup)} groups; with DIFFERENT gold: {sum(len({g for _, g in v}) > 1 for v in dup.values())}")
tin = Counter(re.sub(r"\s+", " ", r['input'].strip().lower()) for r in test_raw.values())
P(f"- test split: duplicate inputs {sum(v > 1 for v in tin.values())}")
# near duplicates by word jaccard within domain
nd = 0; nd_diff = 0
for dom, rs in rawrows.items():
    sets = [set(re.findall(r"\w+", r["input"].lower())) for r in rs]
    for a, b in itertools.combinations(range(len(rs)), 2):
        j = len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
        if j >= 0.8:
            nd += 1; nd_diff += gold_sig(rs[a]) != gold_sig(rs[b])
P(f"- near-duplicate pairs (word Jaccard >= 0.8) within a domain in the 1,700 dev rows: {nd}; with different gold: {nd_diff}")
P("- Candidate label sets with overlapping meaning (labels sharing a word, or containing 'other'), per single-label head:")
inv = {}
for dom, rs in rawrows.items():
    for r in rs:
        for hd in r["output"]["classifications"]: inv[(dom, hd["task"])] = hd["labels"]
rows = []
for (dom, h), labs in sorted(inv.items()):
    if set(labs) == {"yes", "no"}: continue
    pairs = [f"{a}~{b}" for a, b in itertools.combinations(labs, 2) if set(a.split("_")) & set(b.split("_"))]
    rows.append([dom, h, len(labs), "yes" if "other" in labs else "no", ", ".join(pairs) or "-"])
table(["domain", "head", "#labels", "has 'other'", "labels sharing a word"], rows)
# gold=other performance
for m in MODELS:
    o = [(i, q) for i, q in sl_q if q.gold == "other"]
    nono = [(i, q) for i, q in sl_q if q.gold != "other" and "other" in q.option_ids]
    P(f"- {m}: accuracy when gold is 'other' = {pct(sum(qres(m,i,q)['correct']==1 for i,q in o), len(o))} (n={len(o)}); non-'other' gold in heads that have an 'other' option: predicted 'other' wrongly in {pct(sum(qres(m,i,q)['choice']=='other' for i,q in nono), len(nono))} of {len(nono)}")
# ambiguity: confident agreement wrong
P(f"- Among single-label questions both models got wrong ({len(bw)}), they gave the same label in {len(same)}: strongly suggests label ambiguity/gold noise on a part of the data rather than random model error. Per-domain counts of same-wrong: " + str(dict(Counter(i.meta['domain'] for i, q in same).most_common(8))))
P()

# ------------------------------------------------------------ 7
P("## 7. Request actually sent (reconstructed with the saved adapter + client body builders; nothing is sent)")
from eval.models import JevPlugin, LunaPlugin
tmpd = Path(tempfile.mkdtemp())
jp, lp = JevPlugin("dummy", tmpd), LunaPlugin("dummy", tmpd)
def short(o, n=260):
    s = json.dumps(o, ensure_ascii=False, indent=1)
    return s
samples = [next(i for i in items if i.meta["domain"] == "support_intent"), next(i for i in items if i.meta["domain"] == "email_triage"), next(i for i in items if i.meta["domain"] == "screen_tags")]
for i in samples:
    P(f"### {i.item_id} ({len(i.questions)} questions)")
    jb = jp.build_body(i); lb = lp.build_body(i)
    for body in (jb, lb):
        # abridge long text
        def cut(x):
            if isinstance(x, str): return x if len(x) < 160 else x[:160] + "…[+%d chars]" % (len(x) - 160)
            if isinstance(x, dict): return {k: cut(v) for k, v in x.items()}
            if isinstance(x, list): return [cut(v) for v in x]
            return x
        c = cut(body)
        if len(json.dumps(c)) > 2500:  # shorten big option lists for display
            P(f"(body truncated for display; {len(i.questions)} questions)")
        P("```json\n" + json.dumps(c, ensure_ascii=False)[:2400] + "\n```")
P(f"Mean input_tokens per item: jev {sum(r['input_tokens'] or 0 for r in R['jev'].values())/255:.0f}, luna {sum(r['input_tokens'] or 0 for r in R['luna'].values())/255:.0f}")
P()
(HERE / "fd_report.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n".join(OUT))
