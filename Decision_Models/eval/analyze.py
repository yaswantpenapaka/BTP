"""Offline analysis of saved runs. No API calls.

    python -m eval.analyze                       # all models/datasets found in eval/outputs/raw
    python -m eval.analyze --models jev luna --datasets jevbench

Writes to eval/outputs/analysis/:
  <dataset>/merged.csv              one row per item, a block of columns per model
  <dataset>/matrix_<metric>.csv     models as first row and first column, diagonal '—'
  accuracy.csv                      per model per dataset
  summary.csv                       every pair x metric x dataset, long format

Decision units: a JevBench / OpenDecisions item is one unit. A Fast Decisions
item has one unit per question (one per single-label head, one yes/no per
candidate label of a multi-label head). Agreement, kappa, correlations and
distances are computed over units both models answered; accuracy and McNemar
over items (exact match) both models answered that have gold.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path

from eval import metrics as M
from eval.datasets import DATASET_ORDER
from eval.models import MODEL_ORDER
from eval.records import read_rows
from eval.run import OUT_DIR

DIAG = "—"
METRICS = ["agreement", "kappa", "spearman_intra", "pearson_intra", "spearman_cross", "pearson_cross",
           "tvd", "jsd", "kl", "mcnemar_b", "mcnemar_c", "mcnemar_p", "mcnemar"]


def units(row: dict) -> dict[str, tuple[str, list[str], list[float], float]]:
    """qname -> (choice, option order, prob vector, confidence) for answered questions."""
    out = {}
    if row.get("error") and not row.get("questions"):
        return out
    for name, q in (row.get("questions") or {}).items():
        if q.get("choice") is None or not q.get("probs"):
            continue
        opts = list(q["probs"].keys())
        out[name] = (q["choice"], opts, [q["probs"][o] for o in opts], q["confidence"])
    return out


def pair_metrics(ri: dict[str, dict], rj: dict[str, dict]) -> dict:
    ids = sorted(set(ri) & set(rj))
    ch_i, ch_j, conf_i, conf_j = [], [], [], []
    tv, js, kld, sp, pe = [], [], [], [], []
    for iid in ids:
        ui, uj = units(ri[iid]), units(rj[iid])
        for name in sorted(set(ui) & set(uj)):
            ci, oi, pi, fi = ui[name]
            cj, oj, pj, fj = uj[name]
            if oi != oj:
                raise ValueError(f"{iid}/{name}: option order differs between models")
            ch_i.append(ci); ch_j.append(cj); conf_i.append(fi); conf_j.append(fj)
            tv.append(M.tvd(pi, pj)); js.append(M.jsd(pi, pj)); kld.append(M.kl(pi, pj))
            if len(oi) >= 3:
                sp.append(M.spearman(pi, pj)); pe.append(M.pearson(pi, pj))
    cor_i, cor_j = [], []
    for iid in ids:
        a, b = ri[iid].get("correct"), rj[iid].get("correct")
        if a is not None and b is not None:
            cor_i.append(a); cor_j.append(b)
    mc = M.mcnemar(cor_i, cor_j)
    sp_ok, pe_ok = [x for x in sp if x is not None], [x for x in pe if x is not None]
    return {
        "n_units": len(ch_i), "n_units_k3": len(sp), "n_intra_defined": len(sp_ok), "n_gold_items": len(cor_i),
        "agreement": M.agreement(ch_i, ch_j), "kappa": M.cohen_kappa(ch_i, ch_j),
        "spearman_intra": M.mean(sp_ok), "pearson_intra": M.mean(pe_ok),
        "spearman_cross": M.spearman(conf_i, conf_j), "pearson_cross": M.pearson(conf_i, conf_j),
        "tvd": M.mean(tv), "jsd": M.mean(js), "kl": M.mean(kld),
        "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p"],
        "mcnemar": f"b={mc['b']}; c={mc['c']}; p={_fmt(mc['p'])} ({mc['test']})",
    }


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def write_matrix(path: Path, models: list[str], cells: dict):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Mi \\ Mj"] + models)
        for mi in models:
            w.writerow([mi] + [DIAG if mi == mj else _fmt(cells.get((mi, mj))) for mj in models])


def merged_table(path: Path, models: list[str], data: dict[str, dict[str, dict]]):
    ids = []
    for m in models:
        for iid in data[m]:
            if iid not in ids:
                ids.append(iid)
    cols = ["item_id", "gold"]
    for m in models:
        cols += [f"{m}_chosen", f"{m}_correct", f"{m}_confidence", f"{m}_probs", f"{m}_error"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for iid in ids:
            gold = next((data[m][iid].get("gold") for m in models if iid in data[m]), None)
            line = [iid, _js(gold)]
            for m in models:
                r = data[m].get(iid)
                if r is None:
                    line += ["", "", "", "", "not run"]
                else:
                    line += [_js(r.get("chosen")), _fmt(r.get("correct")), _fmt(r.get("confidence")),
                             _js(r.get("probs")), r.get("error") or ""]
            w.writerow(line)


def _js(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def analyze(out_dir: Path, models: list[str] | None = None, datasets: list[str] | None = None) -> dict:
    raw, dest = out_dir / "raw", out_dir / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    acc_rows, summary = [], []
    result = {}
    for d in datasets or DATASET_ORDER:
        avail = [m for m in (models or MODEL_ORDER) if (raw / f"{m}__{d}.jsonl").exists()]
        if not avail:
            continue
        data = {m: read_rows(raw / f"{m}__{d}.jsonl") for m in avail}
        ddir = dest / d
        ddir.mkdir(exist_ok=True)
        merged_table(ddir / "merged.csv", avail, data)
        for m in avail:
            rows = list(data[m].values())
            graded = [r["correct"] for r in rows if r.get("correct") is not None]
            acc_rows.append({"dataset": d, "model": m, "rows": len(rows),
                             "errors": sum(bool(r.get("error")) for r in rows),
                             "refusals": sum(bool(r.get("refusal")) for r in rows),
                             "gold_items_answered": len(graded),
                             "accuracy": _fmt(sum(graded) / len(graded)) if graded else ""})
        cells = {}
        for mi, mj in permutations(avail, 2):
            pm = pair_metrics(data[mi], data[mj])
            cells[(mi, mj)] = pm
            for k, v in pm.items():
                summary.append({"dataset": d, "Mi": mi, "Mj": mj, "metric": k, "value": _fmt(v)})
        for metric in METRICS:
            write_matrix(ddir / f"matrix_{metric}.csv", avail, {k: v[metric] for k, v in cells.items()})
        result[d] = cells
    for name, rows, fields in (("accuracy.csv", acc_rows, None), ("summary.csv", summary, None)):
        with open(dest / name, "w", newline="", encoding="utf-8-sig") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER))
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    res = analyze(a.out, a.models, a.datasets)
    print(f"Analysed {', '.join(res) or 'nothing'} -> {a.out / 'analysis'}")


if __name__ == "__main__":
    main()
