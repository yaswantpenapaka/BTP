"""Pairwise comparison metrics. Pure Python, no network, no randomness."""

from __future__ import annotations

import math
from collections import Counter

KL_EPS = 1e-6  # added to every probability before KL, then renormalised


def agreement(a: list, b: list) -> float | None:
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None  # undefined: one category only
    return (po - pe) / (1 - pe)


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None  # constant vector: undefined
    return sxy / math.sqrt(sxx * syy)


def rank(x: list[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(rank(x), rank(y))


def tvd(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kl(p: list[float], q: list[float], eps: float = KL_EPS) -> float:
    """KL(P||Q) in bits, after adding eps to every entry of both and renormalising."""
    p = _smooth(p, eps)
    q = _smooth(q, eps)
    return sum(a * math.log2(a / b) for a, b in zip(p, q) if a > 0)


def jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence, base 2 (in [0, 1]). No smoothing needed."""
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def _kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _smooth(p, eps):
    s = [x + eps for x in p]
    t = sum(s)
    return [x / t for x in s]


def mcnemar(ci: list[int], cj: list[int]) -> dict:
    """b = Mi right & Mj wrong, c = Mi wrong & Mj right.
    Exact two-sided binomial test if b + c < 25, else chi-square with
    continuity correction (1 df)."""
    b = sum(1 for x, y in zip(ci, cj) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ci, cj) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": len(ci), "test": "none", "stat": None, "p": 1.0}
    if n < 25:
        k = min(b, c)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return {"b": b, "c": c, "n": len(ci), "test": "exact", "stat": None, "p": min(1.0, p)}
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # chi-square 1 df survival function
    return {"b": b, "c": c, "n": len(ci), "test": "chi2_cc", "stat": stat, "p": p}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
