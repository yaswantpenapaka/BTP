"""Addendum: domain-macro head accuracy (card-comparable) + bootstrap CI (resampling items). Offline."""
import csv, random, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(H/"fd_per_domain.csv", encoding="utf-8")))
out = ["## Addendum: card-comparable numbers"]
for m in ("jev", "luna"):
    r = [x for x in rows if x["model"] == m]
    macro_head = st.mean(int(x["head_correct"])/int(x["heads"]) for x in r)
    macro_item = st.mean(int(x["item_correct"])/int(x["items"]) for x in r)
    micro_head = sum(int(x["head_correct"]) for x in r)/sum(int(x["heads"]) for x in r)
    out.append(f"- {m}: domain-macro HEAD exact = {100*macro_head:.1f}%, micro HEAD exact = {100*micro_head:.1f}%, domain-macro ITEM exact = {100*macro_item:.1f}%")
    # bootstrap over items per domain (stratified), macro head acc
    import json
    random.seed(0)
    per = {}
    for line in open(H.parent/"outputs"/"raw"/f"{m}__fastdecisions.jsonl", encoding="utf-8"):
        d = json.loads(line); per.setdefault(d["item_id"].split("_",1)[1].rsplit("_",1)[0], []).append(d)
    def head_hits(d):
        q = d["questions"]; heads = {}
        for name, v in q.items():
            heads.setdefault(name.split("_")[0], []).append(v)
        hit = 0
        for hs in heads.values():
            hit += all(v["correct"] == 1 for v in hs)
        return hit, len(heads)
    base = {k: [head_hits(d) for d in v] for k, v in per.items()}
    bs = []
    for _ in range(2000):
        accs = []
        for k, v in base.items():
            s = [random.choice(v) for _ in v]
            accs.append(sum(a for a, _ in s)/sum(b for _, b in s))
        bs.append(st.mean(accs))
    bs.sort(); out.append(f"  bootstrap 95% CI of domain-macro head exact: [{100*bs[50]:.1f}, {100*bs[1949]:.1f}]")
open(H/"fd_report.md","a",encoding="utf-8").write("\n"+"\n".join(out)+"\n"); print("\n".join(out))
