r"""Deterministic, group-aware, stratified 70/15/15 splitter for BTP datasets.
Rerun:  .venv\Scripts\python scripts\split_datasets.py
Algorithm: items are grouped (items in one group never cross splits); groups are bucketed
by stratum; strata and groups are shuffled with random.Random(42); each group is assigned
to the split with the largest deficit vs. its 70/15/15 target, computed both within the
stratum and globally (vs. items seen so far), so every stratum is represented ~proportionally in test."""
import json, hashlib, random, csv, collections, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42; FR = {"train": .70, "validation": .15, "test": .15}; SPLITS = list(FR)

def h(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def assign(items, stratum_of, group_of):
    groups = collections.OrderedDict()
    for it in items: groups.setdefault(group_of(it), []).append(it)
    strata = collections.defaultdict(list)
    for g, mem in groups.items():
        strata[collections.Counter(stratum_of(m) for m in mem).most_common(1)[0][0]].append(g)
    rng = random.Random(SEED); keys = sorted(strata); rng.shuffle(keys)
    N = len(items); glob = dict.fromkeys(SPLITS, 0); out = {}; seen = 0
    for k in keys:
        gs = sorted(strata[k]); rng.shuffle(gs)
        gs.sort(key=lambda g: -len(groups[g]))  # big groups first (stable -> still seeded)
        n = sum(len(groups[g]) for g in gs); loc = dict.fromkeys(SPLITS, 0)
        for g in gs:
            sz = len(groups[g])
            def score(s):
                return (FR[s]*n - loc[s]) + (FR[s]*(seen + sz) - glob[s])
            s = max(SPLITS, key=score)
            loc[s] += sz; glob[s] += sz; seen += sz
            for m in groups[g]: out[m["item_id"]] = s
    return out

def write(name, items, split, meta):
    d = os.path.join(ROOT, name)
    for s in SPLITS:
        os.makedirs(os.path.join(d, s), exist_ok=True)
        with open(os.path.join(d, s, f"{s}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                if split[it["item_id"]] == s: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item_id", "split", "stratum", "group"])
        for it in items: w.writerow([it["item_id"], split[it["item_id"]], meta["stratum"](it), meta["group"](it)])
    cnt = collections.Counter(split.values())
    bys = collections.defaultdict(collections.Counter)
    for it in items: bys[meta["stratum"](it)][split[it["item_id"]]] += 1
    rows = "\n".join(f"| {k} | {v['train']} | {v['validation']} | {v['test']} |" for k, v in sorted(bys.items()))
    ng = len(set(meta["group"](it) for it in items))
    readme = f"""# {name} split

- Source: {meta['url']}
- Version / revision: {meta['rev']}
- License: {meta['license']}
- Total items: {len(items)}  (groups: {ng})
- Split (seed {SEED}, 70/15/15): train {cnt['train']}, validation {cnt['validation']}, test {cnt['test']}
- item_id: {meta['idnote']}
- Stratification key: {meta['strat_note']}
- Grouping (never split across sets): {meta['group_note']}
- Notes: {meta.get('notes','')}
- Files: train/train.jsonl, validation/validation.jsonl, test/test.jsonl (all original fields + item_id); split_manifest.csv
- Regenerate: `.venv\\Scripts\\python scripts\\split_datasets.py`

| stratum | train | validation | test |
|---|---|---|---|
{rows}
"""
    open(os.path.join(d, "SPLIT_README.md"), "w", encoding="utf-8").write(readme)
    print(name, len(items), dict(cnt))

def opendecisions():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "OpenDecisions/raw/opendecisions.jsonl"), encoding="utf-8")]
    for it in items: it["item_id"] = it["id"]
    # union groups: native `group` field, plus identical state text
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    for it in items:
        union(it["id"], "S:" + h(json.dumps(it["state"], sort_keys=True, ensure_ascii=False)))
        if it.get("group"): union(it["id"], "G:" + str(it["group"]))
    gid = lambda it: find(it["id"])
    strat = lambda it: f"{it['family']}|{it['question']['type']}"
    write("OpenDecisions", items, assign(items, strat, gid), dict(
        stratum=strat, group=gid, url="https://huggingface.co/datasets/jgeuter/OpenDecisions",
        rev="ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe", license="CC BY-SA 4.0",
        idnote="native `id` field (unique)",
        strat_note="family x question type (choice/noul/score); tier, use_case and topic follow proportionally",
        group_note="native `group` field (paired/metamorphic variants: fairness, negation pairs, perturbations, option order) unioned with identical `state` text",
        notes="UNUSED SPLIT: per user decision (2026-10-23) OpenDecisions is not split; all 2,352 items in raw/opendecisions.jsonl are the evaluation set. Source is itself the published validation set."))

def fastdecisions():
    raw = os.path.join(ROOT, "FastDecisions/raw"); items = []
    for fn in sorted(f for f in os.listdir(raw) if f.endswith(".jsonl")):
        dom = fn[:-6]
        for i, l in enumerate(open(os.path.join(raw, fn), encoding="utf-8")):
            it = json.loads(l); it["domain"] = dom; it["item_id"] = f"fastdecisions_{dom}_{i:04d}"; items.append(it)
    def primary(it):
        c = it["output"]["classifications"][0]; return "+".join(sorted(c["true_label"])) or "none"
    strat = lambda it: f"{it['domain']}|{primary(it)}"
    gid = lambda it: it["domain"] + ":" + h(it["input"])
    write("FastDecisions", items, assign(items, strat, gid), dict(
        stratum=strat, group=gid, url="https://huggingface.co/datasets/fastino/fast-decisions",
        rev="35afb79a0f611a1db44380e855ac2a821b0a4da1", license="Apache-2.0",
        idnote="fastdecisions_<domain>_<0-based line index in raw/<domain>.jsonl> (no native ids); `domain` field added",
        strat_note="domain x true label of the first classification head (keeps all 17 domains and their label mix in test)",
        group_note="identical input text within a domain",
        notes="Only the public DEVELOPMENT split (100/domain, 1,700 rows) is available; the official 300/domain test split is held out by Fastino. Card asks not to report dev-split scores as the benchmark."))

def jevbench_org():
    src = os.path.join(ROOT, "JevBench/raw/JevBench-org-hf")
    items = [json.loads(l) for l in open(os.path.join(src, "data/questions.jsonl"), encoding="utf-8")]
    for it in items: it["item_id"] = f"jevbench_{it['case_id']}_{it['question_key']}"
    strat = lambda it: it["domain"]
    gid = lambda it: it["case_id"]
    write("JevBench/org", items, assign(items, strat, gid), dict(
        stratum=strat, group=gid, url="https://huggingface.co/datasets/JevBench/jevbench (code: https://github.com/JevBench/jevbench)",
        rev="cfd3325a6ff0752cda83f799c507ce81c4d0c58d", license="Apache-2.0",
        idnote="jevbench_<case_id>_<question_key> (pair is unique)",
        strat_note="domain (12 domains); each case's questions are kept together, so type mix (noul/choice/score) follows the case",
        group_note="case_id: all questions of a case (same state text) are in one split",
        notes="Unit = one question from data/questions.jsonl (2,240). rewrites.jsonl is NOT split: it has no case_id and its `source` matches question-instruction TEMPLATES shared by 3-20 cases (684 of 1,697 match no question text verbatim), so rewrites cannot be assigned to a case without merging cases. gold.jsonl duplicates the `gold` field already present in questions. 302 questions have gold 'unknown'. Original raw files untouched in JevBench/raw/JevBench-org-hf. The old fstandhartinger files in JevBench/raw are not split."))

if __name__ == "__main__":
    opendecisions(); fastdecisions(); jevbench_org()



