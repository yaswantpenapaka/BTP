"""Paired comparison of saved jev-1.13.0 and gpt-6-luna results. No API calls.

Reads the item rows and probability files already written by run.py.
Does not modify those files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, wilcoxon
from sklearn.metrics import confusion_matrix

from metrics import brier_score, expected_calibration_error
from records import ITEM_COLUMNS
from tasks import JEV_MODEL, OPENAI_MODEL, TASKS

ROOT = Path(__file__).resolve().parent
DATASETS = ("sst2", "rotten_tomatoes", "emotion", "financial_phrasebank")
ALPHA = 0.05
Z95 = 1.96
N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
LOG_LOSS_FLOOR = 1e-12
PROB_MATCH_ATOL = 1e-6
BOOTSTRAP_CHUNK = 500

SUMMARY_COLUMNS = [
    "dataset",
    "n",
    "both_right",
    "only_jev",
    "only_luna",
    "both_wrong",
    "agreement",
    "shared_mistakes",
    "both_wrong_different",
    "acc_jev",
    "acc_luna",
    "delta",
    "paired_se",
    "unpaired_se",
    "ci95_low",
    "ci95_high",
    "mcnemar_p",
    "margin",
    "tost_p_lower",
    "tost_p_upper",
    "tost_p",
    "tost_equivalent",
    "ci90_low",
    "ci90_high",
    "ece_jev",
    "ece_luna",
    "brier_jev",
    "brier_luna",
    "logloss_jev",
    "logloss_luna",
    "brier_diff",
    "brier_ci95_low",
    "brier_ci95_high",
    "brier_wilcoxon_p",
    "brier_wilcoxon_n",
    "logloss_diff",
    "logloss_ci95_low",
    "logloss_ci95_high",
    "logloss_wilcoxon_p",
    "logloss_wilcoxon_n",
    "ece_diff",
    "ece_ci95_low",
    "ece_ci95_high",
    "n_missing_prob",
    "mcnemar_p_holm",
    "brier_wilcoxon_p_holm",
    "mcnemar_holm_significant",
    "brier_holm_significant",
    "top_shared_mistake",
    "alpha",
    "resamples",
    "seed",
]


class DataError(Exception):
    """The saved rows cannot be paired. No statistics were written."""


@dataclass
class Paired:
    dataset: str
    frame: pd.DataFrame
    dist_jev: list[dict[str, float] | None]
    dist_luna: list[dict[str, float] | None]
    labels: tuple[str, ...]
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_item_rows_until_blank(path: Path) -> list[dict[str, str]]:
    """Item rows only. Stop at the blank line that starts the score block."""
    if not path.exists():
        raise DataError(f"Missing {path}.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line or not header_line.strip():
            raise DataError(f"{path} has no header.")
        lines = [header_line]
        for line in handle:
            if not line.strip():
                break
            lines.append(line)
    reader = csv.DictReader(lines)
    fieldnames = [name.strip() for name in (reader.fieldnames or []) if name is not None]
    if fieldnames != ITEM_COLUMNS:
        raise DataError(f"{path} header is {fieldnames}, expected {ITEM_COLUMNS}.")
    rows: list[dict[str, str]] = []
    for raw in reader:
        item_id = (raw.get("item_id") or "").strip()
        if not item_id:
            raise DataError(f"{path} has an item row with a blank item_id.")
        rows.append({column: (raw.get(column) or "").strip() for column in ITEM_COLUMNS})
    if not rows:
        raise DataError(f"{path} has no item rows.")
    return rows


def read_probability_jsonl(path: Path) -> dict[tuple[str, str], dict[str, float] | None]:
    """One distribution per (item_id, model). Duplicates are an error.

    A missing, empty, non-numeric, or non-finite distribution is stored as None
    so the caller can report that item instead of dropping it.
    """
    if not path.exists():
        raise DataError(f"Missing {path}.")
    found: dict[tuple[str, str], dict[str, float] | None] = {}
    counts: dict[tuple[str, str], int] = {}
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                problems.append(f"line {line_number} is blank")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path} line {line_number} is not JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise DataError(f"{path} line {line_number} is not an object.")
            item_id = str(obj.get("item_id", "")).strip()
            model = str(obj.get("model", "")).strip()
            if not item_id or not model:
                raise DataError(f"{path} line {line_number} is missing item_id or model.")
            key = (item_id, model)
            counts[key] = counts.get(key, 0) + 1
            found[key] = _parse_distribution(item_id, model, obj.get("probabilities"), problems)
    duplicates = [
        f"item {item_id} {model} appears {count} times"
        for (item_id, model), count in sorted(counts.items(), key=_prob_sort_key)
        if count != 1
    ]
    if duplicates:
        listed = "\n".join(f"- {line}" for line in duplicates)
        raise DataError(f"{path} has duplicate probability rows:\n{listed}")
    blank = [message for message in problems if message.startswith("line ")]
    if blank:
        listed = "\n".join(f"- {line}" for line in blank)
        raise DataError(f"{path} has blank lines inside the probability file:\n{listed}")
    return found


def _prob_sort_key(item: tuple[tuple[str, str], int]) -> tuple:
    item_id = item[0][0]
    return (int(item_id) if item_id.isdigit() else item_id, item[0][1])


def _parse_distribution(item_id: str, model: str, raw: object, problems: list[str]) -> dict[str, float] | None:
    if not isinstance(raw, dict) or not raw:
        problems.append(f"item {item_id} {model}: probability distribution is missing")
        return None
    parsed: dict[str, float] = {}
    bad = False
    for label, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"item {item_id} {model}: probability for {label} is {value!r}")
            bad = True
            continue
        number = float(value)
        if not math.isfinite(number):
            problems.append(f"item {item_id} {model}: probability for {label} is not finite")
            bad = True
            continue
        parsed[str(label)] = number
    if bad:
        return None
    return parsed


def load_paired(csv_path: Path, jsonl_path: Path, dataset: str) -> Paired:
    if dataset not in TASKS:
        raise DataError(f"Unknown dataset {dataset!r}.")
    labels = TASKS[dataset].prompt_labels
    rows = read_item_rows_until_blank(csv_path)
    problems: list[str] = []
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        item_id = row["item_id"]
        if not item_id.isdigit():
            problems.append(f"item_id {item_id!r} is not an integer")
            continue
        model = row["model"]
        slot = grouped.setdefault(item_id, {})
        if model in slot:
            problems.append(f"item {item_id} has more than one {model} row")
            continue
        if model not in (JEV_MODEL, OPENAI_MODEL):
            problems.append(f"item {item_id} has unexpected model {model!r}")
            continue
        derived, flag_problem = _derived_correct(row)
        if flag_problem:
            problems.append(f"item {item_id} {model}: {flag_problem}")
        row["_correct"] = derived
        slot[model] = row

    for item_id in sorted(grouped, key=int):
        slot = grouped[item_id]
        if JEV_MODEL not in slot:
            problems.append(f"item {item_id} is missing {JEV_MODEL}")
        if OPENAI_MODEL not in slot:
            problems.append(f"item {item_id} is missing {OPENAI_MODEL}")
        if JEV_MODEL in slot and OPENAI_MODEL in slot and slot[JEV_MODEL]["gold"] != slot[OPENAI_MODEL]["gold"]:
            problems.append(
                f"item {item_id} gold mismatch: {JEV_MODEL}={slot[JEV_MODEL]['gold']}, "
                f"{OPENAI_MODEL}={slot[OPENAI_MODEL]['gold']}"
            )
        if JEV_MODEL in slot and slot[JEV_MODEL]["gold"] not in labels:
            problems.append(f"item {item_id} gold {slot[JEV_MODEL]['gold']!r} is not one of {list(labels)}")
    if problems:
        listed = "\n".join(f"- {line}" for line in problems)
        raise DataError(f"{dataset} pairing failed:\n{listed}")

    probabilities = read_probability_jsonl(jsonl_path)
    expected = {(item_id, model) for item_id in grouped for model in (JEV_MODEL, OPENAI_MODEL)}
    extra = sorted(set(probabilities) - expected, key=lambda key: (int(key[0]), key[1]))
    if extra:
        listed = "\n".join(f"- item {item_id} {model}" for item_id, model in extra)
        raise DataError(f"{dataset} probability file has rows that are not in the CSV:\n{listed}")

    records = []
    dist_jev: list[dict[str, float] | None] = []
    dist_luna: list[dict[str, float] | None] = []
    missing: list[str] = []
    notes: list[str] = []
    for item_id in sorted(grouped, key=int):
        jev = grouped[item_id][JEV_MODEL]
        luna = grouped[item_id][OPENAI_MODEL]
        jev_dist = _usable_distribution(item_id, JEV_MODEL, probabilities.get((item_id, JEV_MODEL)), labels, missing)
        luna_dist = _usable_distribution(item_id, OPENAI_MODEL, probabilities.get((item_id, OPENAI_MODEL)), labels, missing)
        jev_prob = _optional_float(item_id, JEV_MODEL, "prob_pred", jev["prob_pred"], missing)
        luna_prob = _optional_float(item_id, OPENAI_MODEL, "prob_pred", luna["prob_pred"], missing)
        jev_gold_p = _optional_float(item_id, JEV_MODEL, "prob_gold", jev["prob_gold"], missing)
        luna_gold_p = _optional_float(item_id, OPENAI_MODEL, "prob_gold", luna["prob_gold"], missing)
        _note_match(item_id, JEV_MODEL, jev["pred"], jev_prob, jev_dist, "prob_pred", notes)
        _note_match(item_id, OPENAI_MODEL, luna["pred"], luna_prob, luna_dist, "prob_pred", notes)
        _note_match(item_id, JEV_MODEL, jev["gold"], jev_gold_p, jev_dist, "prob_gold", notes)
        _note_match(item_id, OPENAI_MODEL, luna["gold"], luna_gold_p, luna_dist, "prob_gold", notes)
        records.append(
            {
                "item_id": item_id,
                "gold": jev["gold"],
                "pred_jev": jev["pred"],
                "pred_luna": luna["pred"],
                "correct_jev": int(jev["_correct"]),
                "correct_luna": int(luna["_correct"]),
                "prob_pred_jev": jev_prob,
                "prob_pred_luna": luna_prob,
            }
        )
        dist_jev.append(jev_dist)
        dist_luna.append(luna_dist)
    frame = pd.DataFrame.from_records(records)
    return Paired(dataset, frame, dist_jev, dist_luna, labels, missing, notes)


def _derived_correct(row: dict[str, str]) -> tuple[int, str | None]:
    pred = row["pred"]
    derived = 1 if pred and pred == row["gold"] else 0
    if row["correct"] not in {"0", "1"}:
        return derived, f"correct column is {row['correct']!r}"
    if int(row["correct"]) != derived:
        return derived, f"correct column is {row['correct']} but pred/gold say {derived}"
    return derived, None


def _optional_float(item_id: str, model: str, column: str, text: str, missing: list[str]) -> float:
    if text == "":
        missing.append(f"item {item_id} {model}: {column} is blank")
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        missing.append(f"item {item_id} {model}: {column} is {text!r}")
        return float("nan")
    if not math.isfinite(value):
        missing.append(f"item {item_id} {model}: {column} is not finite")
        return float("nan")
    if value < 0.0 or value > 1.0:
        missing.append(f"item {item_id} {model}: {column}={value} is outside [0, 1]")
        return float("nan")
    return value


def _usable_distribution(
    item_id: str,
    model: str,
    dist: dict[str, float] | None,
    labels: tuple[str, ...],
    missing: list[str],
) -> dict[str, float] | None:
    if dist is None:
        missing.append(f"item {item_id} {model}: full probability distribution is missing")
        return None
    absent = [label for label in labels if label not in dist]
    if absent:
        missing.append(f"item {item_id} {model}: distribution is missing {absent}")
        return None
    bad = [label for label in labels if not math.isfinite(dist[label]) or dist[label] < 0.0 or dist[label] > 1.0]
    if bad:
        missing.append(f"item {item_id} {model}: distribution has a non-probability on {bad}")
        return None
    return {label: float(dist[label]) for label in labels}


def _note_match(
    item_id: str,
    model: str,
    label: str,
    csv_value: float,
    dist: dict[str, float] | None,
    column: str,
    notes: list[str],
) -> None:
    if not label or dist is None or label not in dist or not math.isfinite(csv_value):
        return
    if abs(csv_value - dist[label]) > PROB_MATCH_ATOL:
        notes.append(
            f"item {item_id} {model}: CSV {column}={csv_value:.12g} but JSONL P({label})={dist[label]:.12g}"
        )


def outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    """2x2 of correctness, plus agreement and shared mistakes. a+b+c+d = n."""
    jev_right = frame["correct_jev"].to_numpy() == 1
    luna_right = frame["correct_luna"].to_numpy() == 1
    same = (frame["pred_jev"].to_numpy() == frame["pred_luna"].to_numpy()) & (frame["pred_jev"].to_numpy() != "")
    both_wrong = ~jev_right & ~luna_right
    counts = {
        "n": int(len(frame)),
        "both_right": int((jev_right & luna_right).sum()),
        "only_jev": int((jev_right & ~luna_right).sum()),
        "only_luna": int((~jev_right & luna_right).sum()),
        "both_wrong": int(both_wrong.sum()),
        "agreement": int(same.sum()),
        "shared_mistakes": int((same & both_wrong).sum()),
        "both_wrong_different": int((both_wrong & ~same).sum()),
    }
    if counts["both_right"] + counts["only_jev"] + counts["only_luna"] + counts["both_wrong"] != counts["n"]:
        raise DataError(f"2x2 counts do not add to n: {counts}")
    if counts["shared_mistakes"] + counts["both_wrong_different"] != counts["both_wrong"]:
        raise DataError(f"both-wrong split does not add up: {counts}")
    if counts["agreement"] != counts["both_right"] + counts["shared_mistakes"]:
        raise DataError(f"agreement is not both-right plus shared mistakes: {counts}")
    return counts


def paired_se(differences: np.ndarray) -> float:
    """Sample standard deviation of the paired differences, divisor n-1, over sqrt(n)."""
    n = int(differences.size)
    if n < 2:
        return float("nan")
    return float(np.std(differences, ddof=1) / math.sqrt(n))


def unpaired_se(acc_jev: float, acc_luna: float, n: int) -> float:
    """SE of the difference of two independent proportions. Comparison only."""
    if n < 1:
        return float("nan")
    variance = acc_jev * (1.0 - acc_jev) / n + acc_luna * (1.0 - acc_luna) / n
    return float(math.sqrt(variance))


def ci95(delta: float, se: float) -> tuple[float, float]:
    if not math.isfinite(se):
        return float("nan"), float("nan")
    return delta - Z95 * se, delta + Z95 * se


def mcnemar_p(only_jev: int, only_luna: int) -> float:
    """Two-sided exact McNemar: Binomial(only_jev + only_luna, 0.5) on the Jev-only count."""
    discordant = only_jev + only_luna
    if discordant == 0:
        return 1.0
    return float(binomtest(only_jev, discordant, p=0.5, alternative="two-sided").pvalue)


def tost(delta: float, se: float, margin: float, alpha: float = ALPHA) -> dict[str, float | bool]:
    """Two one-sided z tests. Equivalent when the (1-2α) paired CI is inside (-margin, +margin)."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    z_ci = float(norm.ppf(1.0 - alpha))
    if se < 0:
        raise ValueError("se must be non-negative")
    if se == 0:
        p_lower = 0.0 if delta > -margin else (0.5 if delta == -margin else 1.0)
        p_upper = 0.0 if delta < margin else (0.5 if delta == margin else 1.0)
        low = high = delta
    else:
        p_lower = float(norm.sf((delta + margin) / se))
        p_upper = float(norm.cdf((delta - margin) / se))
        low = delta - z_ci * se
        high = delta + z_ci * se
    equivalent = bool(low > -margin and high < margin)
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "ci90_low": low,
        "ci90_high": high,
        "equivalent": equivalent,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values, returned in the original order and capped at 1."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        raw = (m - rank) * p_values[index]
        running = max(running, raw)
        adjusted[index] = min(1.0, running)
    return adjusted


def per_item_brier(dist: dict[str, float], gold: str, labels: tuple[str, ...]) -> float:
    err = 0.0
    for label in labels:
        y = 1.0 if label == gold else 0.0
        err += (dist[label] - y) ** 2
    return err


def per_item_logloss(dist: dict[str, float], gold: str) -> float:
    probability = min(1.0, max(float(dist[gold]), LOG_LOSS_FLOOR))
    return -math.log(probability)


def ece_batch(confidences: np.ndarray, correct: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """ECE for many paired resamples. Bin edges match metrics.expected_calibration_error."""
    sampled_c = confidences[indices]
    sampled_h = correct[indices]
    m, n = indices.shape
    bins = np.clip(np.searchsorted(BIN_EDGES, sampled_c, side="right") - 1, 0, N_BINS - 1)
    ece = np.zeros(m, dtype=float)
    for bin_id in range(N_BINS):
        mask = bins == bin_id
        count = mask.sum(axis=1).astype(float)
        nonzero = count > 0
        acc_sum = (sampled_h * mask).sum(axis=1)
        conf_sum = (sampled_c * mask).sum(axis=1)
        bin_acc = np.zeros(m, dtype=float)
        bin_conf = np.zeros(m, dtype=float)
        bin_acc[nonzero] = acc_sum[nonzero] / count[nonzero]
        bin_conf[nonzero] = conf_sum[nonzero] / count[nonzero]
        ece += np.where(nonzero, (count / n) * np.abs(bin_acc - bin_conf), 0.0)
    return ece


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float, float]:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = int(values.size)
    means = np.empty(n_resamples, dtype=float)
    cursor = 0
    while cursor < n_resamples:
        width = min(BOOTSTRAP_CHUNK, n_resamples - cursor)
        indices = rng.integers(0, n, size=(width, n))
        means[cursor : cursor + width] = values[indices].mean(axis=1)
        cursor += width
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return observed, float(low), float(high)


def bootstrap_ece_diff(
    conf_jev: np.ndarray,
    hit_jev: np.ndarray,
    conf_luna: np.ndarray,
    hit_luna: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    point_jev = expected_calibration_error(conf_jev.tolist(), hit_jev.astype(int).tolist(), n_bins=N_BINS)
    point_luna = expected_calibration_error(conf_luna.tolist(), hit_luna.astype(int).tolist(), n_bins=N_BINS)
    observed = float(point_jev - point_luna)
    rng = np.random.default_rng(seed)
    n = int(conf_jev.size)
    diffs = np.empty(n_resamples, dtype=float)
    cursor = 0
    while cursor < n_resamples:
        width = min(BOOTSTRAP_CHUNK, n_resamples - cursor)
        indices = rng.integers(0, n, size=(width, n))
        diffs[cursor : cursor + width] = ece_batch(conf_jev, hit_jev, indices) - ece_batch(conf_luna, hit_luna, indices)
        cursor += width
    low, high = np.quantile(diffs, [0.025, 0.975], method="linear")
    return observed, float(low), float(high)


def wilcoxon_pvalue(differences: np.ndarray) -> tuple[float, int]:
    nonzero = differences[np.isfinite(differences) & (differences != 0.0)]
    n_used = int(nonzero.size)
    if n_used == 0:
        return 1.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.pvalue), n_used


def confusion_table(golds: list[str], preds: list[str], labels: tuple[str, ...]) -> tuple[list[str], np.ndarray]:
    shown = list(labels)
    extras: list[str] = []
    cleaned: list[str] = []
    for pred in preds:
        label = pred if pred else "(none)"
        cleaned.append(label)
        if label not in shown and label not in extras:
            extras.append(label)
    shown.extend(extras)
    matrix = confusion_matrix(golds, cleaned, labels=shown)
    return shown, matrix


def pattern_counts(frame: pd.DataFrame, kind: str) -> list[tuple]:
    both_wrong = (frame["correct_jev"] == 0) & (frame["correct_luna"] == 0)
    same = (frame["pred_jev"] == frame["pred_luna"]) & (frame["pred_jev"] != "")
    if kind == "shared":
        sub = frame.loc[both_wrong & same, ["gold", "pred_jev"]]
        if sub.empty:
            return []
        grouped = sub.groupby(["gold", "pred_jev"], sort=False).size().reset_index(name="n_items")
        grouped = grouped.sort_values(["n_items", "gold", "pred_jev"], ascending=[False, True, True])
        return [
            (str(row.gold), str(row.pred_jev), int(row.n_items))
            for row in grouped.itertuples(index=False)
        ]
    sub = frame.loc[both_wrong & ~same, ["gold", "pred_jev", "pred_luna"]]
    if sub.empty:
        return []
    grouped = sub.groupby(["gold", "pred_jev", "pred_luna"], sort=False).size().reset_index(name="n_items")
    grouped = grouped.sort_values(
        ["n_items", "gold", "pred_jev", "pred_luna"],
        ascending=[False, True, True, True],
    )
    return [
        (str(row.gold), str(row.pred_jev) or "(none)", str(row.pred_luna) or "(none)", int(row.n_items))
        for row in grouped.itertuples(index=False)
    ]


def analyze_paired(paired: Paired, margin: float, n_resamples: int, seed: int) -> dict:
    frame = paired.frame
    counts = outcome_counts(frame)
    correct_jev = frame["correct_jev"].to_numpy(dtype=float)
    correct_luna = frame["correct_luna"].to_numpy(dtype=float)
    differences = correct_jev - correct_luna
    acc_jev = float(correct_jev.mean())
    acc_luna = float(correct_luna.mean())
    delta = acc_jev - acc_luna
    se = paired_se(differences)
    low95, high95 = ci95(delta, se)
    equivalence = tost(delta, se, margin, ALPHA)
    labels = paired.labels
    brier_jev_items, brier_luna_items, log_jev_items, log_luna_items = _item_scores(paired)
    calibration_ok = not paired.missing and all(
        math.isfinite(value)
        for value in (
            *frame["prob_pred_jev"].tolist(),
            *frame["prob_pred_luna"].tolist(),
            *brier_jev_items.tolist(),
            *brier_luna_items.tolist(),
            *log_jev_items.tolist(),
            *log_luna_items.tolist(),
        )
    )
    result = {
        "dataset": paired.dataset,
        "labels": labels,
        "missing": list(paired.missing),
        "notes": list(paired.notes),
        "margin": margin,
        "alpha": ALPHA,
        "resamples": n_resamples,
        "seed": seed,
        **counts,
        "acc_jev": acc_jev,
        "acc_luna": acc_luna,
        "delta": delta,
        "paired_se": se,
        "unpaired_se": unpaired_se(acc_jev, acc_luna, counts["n"]),
        "ci95_low": low95,
        "ci95_high": high95,
        "mcnemar_p": mcnemar_p(counts["only_jev"], counts["only_luna"]),
        "tost_p_lower": equivalence["p_lower"],
        "tost_p_upper": equivalence["p_upper"],
        "tost_p": equivalence["p_tost"],
        "tost_equivalent": "yes" if equivalence["equivalent"] else "no",
        "ci90_low": equivalence["ci90_low"],
        "ci90_high": equivalence["ci90_high"],
    }
    if not calibration_ok:
        nan = float("nan")
        result.update(
            {
                "ece_jev": nan,
                "ece_luna": nan,
                "brier_jev": nan,
                "brier_luna": nan,
                "logloss_jev": nan,
                "logloss_luna": nan,
                "brier_diff": nan,
                "brier_ci95_low": nan,
                "brier_ci95_high": nan,
                "brier_wilcoxon_p": nan,
                "brier_wilcoxon_n": 0,
                "logloss_diff": nan,
                "logloss_ci95_low": nan,
                "logloss_ci95_high": nan,
                "logloss_wilcoxon_p": nan,
                "logloss_wilcoxon_n": 0,
                "ece_diff": nan,
                "ece_ci95_low": nan,
                "ece_ci95_high": nan,
                "n_missing_prob": len(paired.missing),
            }
        )
    else:
        golds = frame["gold"].tolist()
        conf_jev = frame["prob_pred_jev"].to_numpy(dtype=float)
        conf_luna = frame["prob_pred_luna"].to_numpy(dtype=float)
        hit_jev = correct_jev
        hit_luna = correct_luna
        brier_diff = brier_jev_items - brier_luna_items
        log_diff = log_jev_items - log_luna_items
        brier_mean, brier_low, brier_high = bootstrap_mean_ci(brier_diff, n_resamples, seed)
        log_mean, log_low, log_high = bootstrap_mean_ci(log_diff, n_resamples, seed)
        ece_diff, ece_low, ece_high = bootstrap_ece_diff(conf_jev, hit_jev, conf_luna, hit_luna, n_resamples, seed)
        brier_p, brier_n = wilcoxon_pvalue(brier_diff)
        log_p, log_n = wilcoxon_pvalue(log_diff)
        result.update(
            {
                "ece_jev": expected_calibration_error(conf_jev.tolist(), hit_jev.astype(int).tolist()),
                "ece_luna": expected_calibration_error(conf_luna.tolist(), hit_luna.astype(int).tolist()),
                "brier_jev": brier_score(paired.dist_jev, golds, list(labels)),
                "brier_luna": brier_score(paired.dist_luna, golds, list(labels)),
                "logloss_jev": float(np.mean(log_jev_items)),
                "logloss_luna": float(np.mean(log_luna_items)),
                "brier_diff": brier_mean,
                "brier_ci95_low": brier_low,
                "brier_ci95_high": brier_high,
                "brier_wilcoxon_p": brier_p,
                "brier_wilcoxon_n": brier_n,
                "logloss_diff": log_mean,
                "logloss_ci95_low": log_low,
                "logloss_ci95_high": log_high,
                "logloss_wilcoxon_p": log_p,
                "logloss_wilcoxon_n": log_n,
                "ece_diff": ece_diff,
                "ece_ci95_low": ece_low,
                "ece_ci95_high": ece_high,
                "n_missing_prob": 0,
            }
        )
    shared = pattern_counts(frame, "shared")
    different = pattern_counts(frame, "different")
    result["shared_patterns"] = shared
    result["different_patterns"] = different
    result["top_shared_mistake"] = f"{shared[0][0]}->{shared[0][1]}:{shared[0][2]}" if shared else ""
    result["confusion_jev"] = confusion_table(frame["gold"].tolist(), frame["pred_jev"].tolist(), labels)
    result["confusion_luna"] = confusion_table(frame["gold"].tolist(), frame["pred_luna"].tolist(), labels)
    return result


def _item_scores(paired: Paired) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(paired.frame)
    brier_jev = np.full(n, np.nan)
    brier_luna = np.full(n, np.nan)
    log_jev = np.full(n, np.nan)
    log_luna = np.full(n, np.nan)
    golds = paired.frame["gold"].tolist()
    for index, gold in enumerate(golds):
        if paired.dist_jev[index] is not None and gold in paired.dist_jev[index]:
            brier_jev[index] = per_item_brier(paired.dist_jev[index], gold, paired.labels)
            log_jev[index] = per_item_logloss(paired.dist_jev[index], gold)
        if paired.dist_luna[index] is not None and gold in paired.dist_luna[index]:
            brier_luna[index] = per_item_brier(paired.dist_luna[index], gold, paired.labels)
            log_luna[index] = per_item_logloss(paired.dist_luna[index], gold)
    return brier_jev, brier_luna, log_jev, log_luna


def apply_holm(results: list[dict]) -> None:
    _attach_holm(results, "mcnemar_p", "mcnemar_p_holm", "mcnemar_holm_significant")
    _attach_holm(results, "brier_wilcoxon_p", "brier_wilcoxon_p_holm", "brier_holm_significant")


def _attach_holm(results: list[dict], source: str, adjusted_key: str, flag_key: str) -> None:
    raw = [float(result[source]) for result in results]
    if any(math.isnan(value) for value in raw):
        for result in results:
            result[adjusted_key] = float("nan")
            result[flag_key] = "not_computed"
        return
    adjusted = holm_bonferroni(raw)
    for result, value in zip(results, adjusted):
        result[adjusted_key] = value
        result[flag_key] = "yes" if value < ALPHA else "no"


def render_report(result: dict) -> str:
    a = result["both_right"]
    b = result["only_jev"]
    c = result["only_luna"]
    d = result["both_wrong"]
    lines = [
        f"Paired comparison of {JEV_MODEL} and {OPENAI_MODEL}",
        f"dataset: {result['dataset']}",
        "No API calls. The source CSV and probability file were not modified.",
        "",
        "Definitions",
        "delta = accuracy(jev) - accuracy(luna)",
        "d_i = correct_jev - correct_luna, each value in {-1, 0, 1}",
        "paired SE = sample sd(d, divisor n-1) / sqrt(n)",
        "95% CI = delta ± 1.96 * paired SE",
        "unpaired SE = sqrt(p_jev(1-p_jev)/n + p_luna(1-p_luna)/n), reported for comparison and not used in the tests",
        "McNemar = two-sided exact binomial test that the Jev-only count is Binomial(b+c, 0.5)",
        f"TOST margin = ±{result['margin']}, alpha = {result['alpha']}",
        "equivalent when the 90% paired CI lies strictly inside (-margin, +margin)",
        "Brier_i = sum over labels of (p - one_hot_gold)^2",
        "log-loss_i = -log(p_gold), with p_gold floored at 1e-12. Lower is better.",
        "ECE uses 10 equal-width bins of the predicted-label probability. The last bin includes 1. Lower is better.",
        (
            f"Bootstrap: {result['resamples']} paired resamples of items, percentile 2.5 and 97.5, "
            f"seed {result['seed']} restarted for the Brier, log-loss, and ECE intervals"
        ),
        "Wilcoxon: two-sided signed-rank test on the non-zero paired differences",
        "",
        f"n = {result['n']}",
        "2x2 of correctness. a = both right, b = only Jev, c = only Luna, d = both wrong.",
        f"                  luna right   luna wrong",
        f"jev right         {a:8d}     {b:8d}",
        f"jev wrong         {c:8d}     {d:8d}",
        f"a + b + c + d = {a + b + c + d}",
        f"same label (agreement) = {result['agreement']}",
        f"same label and wrong (shared mistakes) = {result['shared_mistakes']}",
        f"both wrong, different labels = {result['both_wrong_different']}",
        "",
        "Missing or invalid probabilities",
        _bullet_block(result["missing"], "none"),
        "CSV cells checked against the JSONL probability of the same label",
        _bullet_block(result["notes"], "no mismatches above 1e-6"),
        "",
        "Accuracy",
        f"Jev     {_fmt(result['acc_jev'], 6)}",
        f"Luna    {_fmt(result['acc_luna'], 6)}",
        f"delta   {_fmt(result['delta'], 6)}",
        f"paired SE   {_fmt(result['paired_se'], 6)}",
        f"unpaired SE {_fmt(result['unpaired_se'], 6)}",
        f"95% paired CI [{_fmt(result['ci95_low'], 6)}, {_fmt(result['ci95_high'], 6)}]",
        _accuracy_verdict(result),
        "",
        f"Exact McNemar p = {_fmt_p(result['mcnemar_p'])}",
        _mcnemar_verdict(result),
        f"Holm-adjusted McNemar p = {_fmt_p(result['mcnemar_p_holm'])}",
        _holm_verdict(result["mcnemar_holm_significant"], "McNemar"),
        "",
        f"TOST at margin ±{result['margin']}",
        f"p_lower = {_fmt_p(result['tost_p_lower'])}",
        f"p_upper = {_fmt_p(result['tost_p_upper'])}",
        f"p_TOST  = {_fmt_p(result['tost_p'])}",
        f"90% paired CI [{_fmt(result['ci90_low'], 6)}, {_fmt(result['ci90_high'], 6)}]",
        _tost_verdict(result),
        "",
        "Calibration. Lower ECE, Brier, and log-loss are better.",
        f"ECE     Jev {_fmt(result['ece_jev'], 6)}   Luna {_fmt(result['ece_luna'], 6)}",
        f"Brier   Jev {_fmt(result['brier_jev'], 6)}   Luna {_fmt(result['brier_luna'], 6)}",
        f"log-loss Jev {_fmt(result['logloss_jev'], 6)}   Luna {_fmt(result['logloss_luna'], 6)}",
        "",
        "Per-item Brier difference (Jev - Luna)",
        (
            f"mean {_fmt(result['brier_diff'], 6)}   "
            f"95% bootstrap CI [{_fmt(result['brier_ci95_low'], 6)}, {_fmt(result['brier_ci95_high'], 6)}]"
        ),
        (
            f"Wilcoxon p = {_fmt_p(result['brier_wilcoxon_p'])} "
            f"on {result['brier_wilcoxon_n']} non-zero differences"
        ),
        _score_verdict("Brier", result["brier_diff"], result["brier_ci95_low"], result["brier_ci95_high"], result["brier_wilcoxon_p"]),
        f"Holm-adjusted Brier Wilcoxon p = {_fmt_p(result['brier_wilcoxon_p_holm'])}",
        _holm_verdict(result["brier_holm_significant"], "Brier Wilcoxon"),
        "",
        "Per-item log-loss difference (Jev - Luna)",
        (
            f"mean {_fmt(result['logloss_diff'], 6)}   "
            f"95% bootstrap CI [{_fmt(result['logloss_ci95_low'], 6)}, {_fmt(result['logloss_ci95_high'], 6)}]"
        ),
        (
            f"Wilcoxon p = {_fmt_p(result['logloss_wilcoxon_p'])} "
            f"on {result['logloss_wilcoxon_n']} non-zero differences"
        ),
        _score_verdict(
            "log-loss",
            result["logloss_diff"],
            result["logloss_ci95_low"],
            result["logloss_ci95_high"],
            result["logloss_wilcoxon_p"],
        ),
        "",
        "ECE difference (Jev - Luna)",
        (
            f"observed {_fmt(result['ece_diff'], 6)}   "
            f"95% bootstrap CI [{_fmt(result['ece_ci95_low'], 6)}, {_fmt(result['ece_ci95_high'], 6)}]"
        ),
        _ece_verdict(result),
        "",
        f"Confusion matrix, {JEV_MODEL}. Rows are gold, columns are the prediction.",
        _format_matrix(*result["confusion_jev"]),
        "",
        f"Confusion matrix, {OPENAI_MODEL}. Rows are gold, columns are the prediction.",
        _format_matrix(*result["confusion_luna"]),
        "",
        "Most common shared wrong labels (gold -> the label both models predicted)",
        _format_shared(result["shared_patterns"]),
        "",
        "Most common both-wrong items with different labels (gold | Jev | Luna)",
        _format_different(result["different_patterns"]),
        "",
    ]
    return "\n".join(lines)


def _bullet_block(items: list[str], empty: str) -> str:
    if not items:
        return empty
    shown = items[:30]
    lines = [f"- {item}" for item in shown]
    if len(items) > len(shown):
        lines.append(f"- ... {len(items) - len(shown)} more")
    return "\n".join(lines)


def _accuracy_verdict(result: dict) -> str:
    low, high = result["ci95_low"], result["ci95_high"]
    if not math.isfinite(low) or not math.isfinite(high):
        return "Verdict: The paired accuracy CI was not computed."
    if high < 0:
        return "Verdict: The 95% paired CI is entirely below 0, so gpt-6-luna is ahead on accuracy."
    if low > 0:
        return "Verdict: The 95% paired CI is entirely above 0, so Jev is ahead on accuracy."
    return "Verdict: The 95% paired CI includes 0, so this normal approximation does not detect an accuracy gap."


def _mcnemar_verdict(result: dict) -> str:
    p = result["mcnemar_p"]
    if p < ALPHA and result["only_jev"] > result["only_luna"]:
        return f"Verdict: The discordant items favor Jev (exact McNemar p = {_fmt_p(p)})."
    if p < ALPHA and result["only_luna"] > result["only_jev"]:
        return f"Verdict: The discordant items favor gpt-6-luna (exact McNemar p = {_fmt_p(p)})."
    return f"Verdict: The discordant items do not show a significant difference (exact McNemar p = {_fmt_p(p)})."


def _tost_verdict(result: dict) -> str:
    margin = result["margin"]
    if result["tost_equivalent"] == "yes":
        return f"Verdict: The accuracy gap is statistically inside ±{margin:g} at alpha {ALPHA}."
    return f"Verdict: Equivalence within ±{margin:g} is not established at alpha {ALPHA}."


def _score_verdict(name: str, mean: float, low: float, high: float, p_value: float) -> str:
    if not all(math.isfinite(value) for value in (mean, low, high, p_value)):
        return f"Verdict: The paired {name} comparison was not computed because a probability is missing or invalid."
    direction = "Jev is lower" if mean < 0 else ("gpt-6-luna is lower" if mean > 0 else "the means are equal")
    ci = "the 95% bootstrap CI excludes 0" if high < 0 or low > 0 else "the 95% bootstrap CI includes 0"
    rank = "the Wilcoxon test is significant" if p_value < ALPHA else "the Wilcoxon test is not significant"
    return (
        f"Verdict: Mean {name} difference (Jev - Luna) is {_fmt(mean, 6)}; {direction}. "
        f"{ci.capitalize()}, and {rank}."
    )


def _ece_verdict(result: dict) -> str:
    mean, low, high = result["ece_diff"], result["ece_ci95_low"], result["ece_ci95_high"]
    if not all(math.isfinite(value) for value in (mean, low, high)):
        return "Verdict: The ECE comparison was not computed because a probability is missing or invalid."
    if high < 0:
        who = "Jev has the lower ECE"
    elif low > 0:
        who = "gpt-6-luna has the lower ECE"
    else:
        who = "the CI includes 0"
    excludes = "The CI excludes 0" if high < 0 or low > 0 else "The CI includes 0"
    return f"Verdict: ECE difference (Jev - Luna) is {_fmt(mean, 6)}. {excludes}; {who}."


def _holm_verdict(flag: str, name: str) -> str:
    if flag == "yes":
        return f"Verdict: This {name} comparison survives Holm-Bonferroni at alpha {ALPHA} across the four datasets."
    if flag == "no":
        return f"Verdict: This {name} comparison does not survive Holm-Bonferroni at alpha {ALPHA} across the four datasets."
    return f"Verdict: Holm-Bonferroni for {name} was not computed."


def _format_matrix(labels: list[str], matrix: np.ndarray) -> str:
    cells = [str(int(value)) for value in matrix.ravel()] + list(labels) + ["gold\\pred"]
    width = max(len(cell) for cell in cells)
    header = " ".join(f"{label:>{width}}" for label in labels)
    lines = [f"{'gold\\pred':>{width}} {header}"]
    for label, row in zip(labels, matrix):
        body = " ".join(f"{int(value):>{width}}" for value in row)
        lines.append(f"{label:>{width}} {body}")
    return "\n".join(lines)


def _format_shared(patterns: list[tuple]) -> str:
    if not patterns:
        return "none"
    return "\n".join(f"{gold} -> {pred}: {count}" for gold, pred, count in patterns)


def _format_different(patterns: list[tuple]) -> str:
    if not patterns:
        return "none"
    shown = patterns[:10]
    lines = [f"{gold} | {jev} | {luna}: {count}" for gold, jev, luna, count in shown]
    if len(patterns) > len(shown):
        lines.append(f"... {len(patterns) - len(shown)} more patterns")
    return "\n".join(lines)


def _fmt(value: float, digits: int) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_p(value: float) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "nan"
    number = float(value)
    if number < 1e-4:
        return f"{number:.2e}"
    return f"{number:.4f}"


def summary_frame(results: list[dict]) -> pd.DataFrame:
    rows = [{column: result[column] for column in SUMMARY_COLUMNS} for result in results]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_outputs(results: list[dict], outputs: Path) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    summary_path = outputs / "analysis_summary.csv"
    frame = summary_frame(results)
    _write_retry(summary_path, lambda: frame.to_csv(summary_path, index=False, float_format="%.12g", na_rep="nan"))
    for result in results:
        path = outputs / f"analysis_{result['dataset']}.txt"
        text = render_report(result)
        _write_retry(path, lambda path=path, text=text: path.write_text(text, encoding="utf-8", newline="\n"))


def _write_retry(path: Path, action) -> None:
    last: OSError | None = None
    for attempt in range(1, 16):
        try:
            action()
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if not isinstance(exc, PermissionError) and winerror not in (5, 32, 33):
                raise
            last = exc
            print(
                f"Cannot write {path}. Close it if it is open in Excel or locked by OneDrive. "
                f"Retry {attempt}/15."
            )
            if attempt == 15:
                break
            time.sleep(2)
    raise SystemExit(f"Gave up writing {path}. Close the file and rerun.") from last


def print_summary(results: list[dict]) -> None:
    print("Accuracy. delta = acc(jev-1.13.0) - acc(gpt-6-luna). The 95% CI is the paired interval.")
    header = (
        f"{'dataset':<22} {'n':>5} {'acc_jev':>8} {'acc_luna':>8} {'delta':>9} "
        f"{'ci95_low':>9} {'ci95_high':>9} {'only_jev':>8} {'only_luna':>9} "
        f"{'mcnemar_p':>10} {'TOST':>5} {'agree':>7}"
    )
    print(header)
    for result in results:
        agree = result["agreement"] / result["n"] if result["n"] else float("nan")
        print(
            f"{result['dataset']:<22} {result['n']:5d} {_fmt(result['acc_jev'], 4):>8} "
            f"{_fmt(result['acc_luna'], 4):>8} {_fmt(result['delta'], 4):>9} "
            f"{_fmt(result['ci95_low'], 4):>9} {_fmt(result['ci95_high'], 4):>9} "
            f"{result['only_jev']:8d} {result['only_luna']:9d} "
            f"{_fmt_p(result['mcnemar_p']):>10} {result['tost_equivalent']:>5} {_fmt(agree, 4):>7}"
        )
    print()
    print("Calibration. Differences are Jev minus Luna. Negative means Jev is better.")
    header = (
        f"{'dataset':<22} {'ece_jev':>8} {'ece_luna':>8} {'ece_diff':>9} "
        f"{'ece_low':>9} {'ece_high':>9} {'brier_diff':>11} {'brier_p':>10} "
        f"{'logloss_diff':>12} {'logloss_p':>10}"
    )
    print(header)
    for result in results:
        print(
            f"{result['dataset']:<22} {_fmt(result['ece_jev'], 4):>8} {_fmt(result['ece_luna'], 4):>8} "
            f"{_fmt(result['ece_diff'], 4):>9} {_fmt(result['ece_ci95_low'], 4):>9} "
            f"{_fmt(result['ece_ci95_high'], 4):>9} {_fmt(result['brier_diff'], 4):>11} "
            f"{_fmt_p(result['brier_wilcoxon_p']):>10} {_fmt(result['logloss_diff'], 4):>12} "
            f"{_fmt_p(result['logloss_wilcoxon_p']):>10}"
        )
    print()
    print("Holm-Bonferroni across the four datasets, alpha = 0.05.")
    print(f"{'dataset':<22} {'mcnemar_p':>10} {'p_holm':>10} {'survives':>9} {'brier_p':>10} {'p_holm':>10} {'survives':>9}")
    for result in results:
        print(
            f"{result['dataset']:<22} {_fmt_p(result['mcnemar_p']):>10} "
            f"{_fmt_p(result['mcnemar_p_holm']):>10} {result['mcnemar_holm_significant']:>9} "
            f"{_fmt_p(result['brier_wilcoxon_p']):>10} {_fmt_p(result['brier_wilcoxon_p_holm']):>10} "
            f"{result['brier_holm_significant']:>9}"
        )
    for result in results:
        if result["missing"]:
            print(f"{result['dataset']}: {len(result['missing'])} missing or invalid probability values. Calibration was not computed.")
        elif result["notes"]:
            print(f"{result['dataset']}: {len(result['notes'])} CSV/JSONL probability mismatches. See the text report.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired statistics for saved Jev and gpt-6-luna results. No API calls.")
    parser.add_argument("--margin", type=float, default=0.02, help="TOST equivalence margin on the accuracy gap. Default 0.02.")
    parser.add_argument("--resamples", type=int, default=10_000, help="Paired bootstrap resamples. Default 10000.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed. Default 0.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs", help="Directory with the result CSVs and JSONL files.")
    args = parser.parse_args(argv)
    if args.margin <= 0:
        raise SystemExit("--margin must be positive. No files were read.")
    if args.resamples < 1:
        raise SystemExit("--resamples must be at least 1. No files were read.")
    loaded: list[Paired] = []
    failures: list[str] = []
    for dataset in DATASETS:
        print(f"Reading {dataset}.")
        try:
            loaded.append(
                load_paired(args.outputs / f"{dataset}.csv", args.outputs / f"{dataset}_probs.jsonl", dataset)
            )
        except DataError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("\n\n".join(failures))
    results = []
    for paired in loaded:
        print(f"Computing {paired.dataset} ({len(paired.frame)} items, {args.resamples} bootstrap resamples).")
        results.append(analyze_paired(paired, args.margin, args.resamples, args.seed))
    apply_holm(results)
    write_outputs(results, args.outputs)
    print()
    print_summary(results)
    print()
    print(f"Wrote {args.outputs / 'analysis_summary.csv'}")
    for result in results:
        print(f"Wrote {args.outputs / ('analysis_' + result['dataset'] + '.txt')}")


if __name__ == "__main__":
    main()
