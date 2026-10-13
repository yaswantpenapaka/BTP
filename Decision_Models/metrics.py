"""Accuracy, macro-F1, 10-bin ECE, and Brier score.

Errors and refusals count as wrong for accuracy and macro-F1. They are left
out of ECE and Brier when no full probability distribution came back.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


ERROR_LABEL = "__error__"


def accuracy(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return float("nan")
    hits = sum(1 for gold, pred in zip(golds, preds) if pred and pred == gold)
    return hits / len(golds)


def macro_f1(golds: list[str], preds: list[str], labels: list[str]) -> float:
    """Empty predictions use a sentinel that is outside `labels`, so they are misses."""
    if not golds:
        return float("nan")
    y_pred = [pred if pred else ERROR_LABEL for pred in preds]
    return float(
        f1_score(
            golds,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def expected_calibration_error(confidences: list[float], correct: list[int], n_bins: int = 10) -> float:
    """ECE on the probability of the predicted label. Bins are [0, 0.1), ..., [0.9, 1]."""
    if not confidences:
        return float("nan")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if np.any(conf < 0) or np.any(conf > 1):
        raise ValueError("confidence must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(hit[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (count / total) * abs(bin_acc - bin_conf)
    return ece


def brier_score(distributions: list[dict[str, float]], golds: list[str], labels: list[str]) -> float:
    """Mean squared error of the full option distribution against a one-hot gold label."""
    if len(distributions) != len(golds):
        raise ValueError("distributions and golds must have the same length")
    if not distributions:
        return float("nan")
    total = 0.0
    for dist, gold in zip(distributions, golds):
        if gold not in labels:
            raise ValueError(f"gold label {gold!r} is not in {labels}")
        err = 0.0
        for label in labels:
            p = float(dist[label])
            y = 1.0 if label == gold else 0.0
            err += (p - y) ** 2
        total += err
    return total / len(distributions)


def _percentile_50(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), 50))


def score_models(
    rows: list[dict],
    probs_by_key: dict[tuple[str, str], dict[str, float]],
    labels: list[str],
    model_order: tuple[str, ...],
) -> dict[str, dict[str, float | int | str | None]]:
    """Per-model metrics plus exact_agreement across the first two models.

    `rows` are item dicts with keys item_id, model, gold, pred, correct,
    prob_pred, latency_ms, input_tokens, cost_usd, error.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out: dict[str, dict[str, float | int | str | None]] = {}
    for model in model_order:
        group = by_model.get(model, [])
        golds = [r["gold"] for r in group]
        preds = [r["pred"] for r in group]
        ece_conf: list[float] = []
        ece_hit: list[int] = []
        brier_dists: list[dict[str, float]] = []
        brier_golds: list[str] = []
        excluded = 0
        tokens_missing = 0
        token_sum = 0
        have_tokens = False
        latencies: list[float] = []
        cost_sum = 0.0
        refusals = 0
        for row in group:
            if row.get("error") or not row.get("pred"):
                refusals += 1
            latency = row.get("latency_ms")
            if latency is not None and latency != "":
                latencies.append(float(latency))
            tokens = row.get("input_tokens")
            if tokens is None or tokens == "":
                tokens_missing += 1
            else:
                have_tokens = True
                token_sum += int(tokens)
            cost = row.get("cost_usd")
            if cost is not None and cost != "":
                cost_sum += float(cost)
            dist = probs_by_key.get((str(row["item_id"]), model))
            prob_pred = row.get("prob_pred")
            full = dist is not None and all(label in dist for label in labels)
            if not full:
                # No full distribution: leave the item out of both ECE and Brier.
                excluded += 1
                continue
            narrowed = {label: float(dist[label]) for label in labels}
            brier_dists.append(narrowed)
            brier_golds.append(row["gold"])
            # ECE is on the probability of the predicted label, so a refusal
            # with a distribution still counts for Brier and not for ECE.
            if prob_pred is None or prob_pred == "":
                continue
            ece_conf.append(float(prob_pred))
            ece_hit.append(1 if row["pred"] == row["gold"] else 0)
        out[model] = {
            "n": len(group),
            "accuracy": accuracy(golds, preds),
            "macro_f1": macro_f1(golds, preds, labels),
            "ece_10bin": expected_calibration_error(ece_conf, ece_hit, n_bins=10),
            "brier": brier_score(brier_dists, brier_golds, labels),
            "refusal_rate": (refusals / len(group)) if group else float("nan"),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
            "p50_latency_ms": _percentile_50(latencies),
            "total_input_tokens": token_sum if have_tokens else None,
            "total_cost_usd": cost_sum,
            "excluded_no_probs": excluded,
            "input_tokens_missing": tokens_missing,
        }

    agreement = _exact_agreement(by_model, model_order)
    if model_order:
        out[model_order[0]]["exact_agreement"] = agreement
    return out


def _exact_agreement(by_model: dict[str, list[dict]], model_order: tuple[str, ...]) -> float:
    if len(model_order) < 2:
        return float("nan")
    left = {str(r["item_id"]): r for r in by_model.get(model_order[0], [])}
    right = {str(r["item_id"]): r for r in by_model.get(model_order[1], [])}
    shared = sorted(set(left) & set(right))
    if not shared:
        return float("nan")
    hits = 0
    for item_id in shared:
        a = left[item_id]["pred"]
        b = right[item_id]["pred"]
        if a and b and a == b:
            hits += 1
    return hits / len(shared)
