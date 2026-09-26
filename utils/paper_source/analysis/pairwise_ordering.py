"""Exact tie-aware pairwise audit, without constructing stock pairs.

Implements the pairwise forecast-ordering definition on a finite common monthly sample:
P = Pr(A == B), F = E[(B - A) Z / 2 | A != B].  A, B and Z may be zero.
F is None when there are no changed pairs.  All pairs have equal weight.

The fast O(n log n) path recovers integer sign-product sums from SciPy's
tie-aware Kendall tau-b, then checks integer recovery, range and parity.
An exact integer Fenwick counter is the O(n log n) fallback.  Both paths use
O(n) space.  Reported tau-a values always divide by ALL n(n-1)/2 pairs.

Reuse the shallow-return calculation across target depths with:
    anchor = prepare_anchor(shallow, returns)
    result = pairwise_statistics_from_anchor(anchor, deep)

Inputs must be aligned in the same stock order. Float64 values are not rounded,
rank-transformed or perturbed to break ties. No observations are silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import kendalltau


# The conservative error budget must remain far below half an integer.  Larger
# arrays are supported by the integer fallback rather than unreliable rounding.
_RECOVERY_MAX_ERROR = 0.125
_FLOAT_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class PreparedScores:
    values: np.ndarray
    tied_pairs: int

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class PairSignCounts:
    total_pairs: int
    x_tied_pairs: int
    y_tied_pairs: int
    both_tied_pairs: int
    untied_both_pairs: int
    signed_count: int
    concordant_pairs: int
    discordant_pairs: int


@dataclass(frozen=True)
class PreparedAnchor:
    shallow: PreparedScores
    returns: PreparedScores
    shallow_return: PairSignCounts


def _tie_pairs_from_counts(counts: np.ndarray) -> int:
    counts = np.asarray(counts, dtype=np.int64)
    return int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))


def prepare_scores(values) -> PreparedScores:
    """Validate and own a read-only float64 copy, so cached counts stay valid."""
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("Scores must be a one-dimensional array with n >= 2.")
    if not np.isfinite(array).all():
        raise ValueError("Scores must be finite; supply an explicit common sample.")
    _, counts = np.unique(array, return_counts=True)
    array.setflags(write=False)
    return PreparedScores(array, _tie_pairs_from_counts(counts))


def _prepared(values) -> PreparedScores:
    return values if isinstance(values, PreparedScores) else prepare_scores(values)


def _joint_tied_pairs(x: PreparedScores, y: PreparedScores, total: int) -> int:
    if x.tied_pairs == 0 or y.tied_pairs == 0:
        return 0
    if x.tied_pairs == total:
        return y.tied_pairs
    if y.tied_pairs == total:
        return x.tied_pairs
    order = np.lexsort((y.values, x.values))
    xs, ys = x.values[order], y.values[order]
    same = (xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1])
    boundaries = np.concatenate(([0], np.flatnonzero(~same) + 1, [x.n]))
    return _tie_pairs_from_counts(np.diff(boundaries))


def _integer_signed_count(x: np.ndarray, y: np.ndarray) -> int:
    """Count concordant minus discordant pairs exactly using Python integers."""
    _, y_codes = np.unique(y, return_inverse=True)
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    ranks = y_codes[order] + 1
    boundaries = np.concatenate(
        ([0], np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1, [len(x)])
    )
    n_ranks = int(y_codes.max()) + 1
    tree = [0] * (n_ranks + 1)
    equal_seen = [0] * (n_ranks + 1)
    seen = 0
    signed = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # Query before inserting this entire x-tied block: within-block pairs
        # have sign(x_i-x_j) == 0 and must never enter the signed count.
        for raw_rank in ranks[start:stop]:
            rank = int(raw_rank)
            index, less = rank - 1, 0
            while index:
                less += tree[index]
                index -= index & -index
            signed += 2 * less + equal_seen[rank] - seen
        for raw_rank in ranks[start:stop]:
            rank = int(raw_rank)
            equal_seen[rank] += 1
            index = rank
            while index <= n_ranks:
                tree[index] += 1
                index += index & -index
        seen += int(stop - start)
    return signed


def _valid_signed_count(signed: int, untied_both: int) -> bool:
    return abs(signed) <= untied_both and (untied_both + signed) % 2 == 0


def count_sign_products(x, y) -> PairSignCounts:
    """Exact sum sign(x_i-x_j)*sign(y_i-y_j), with full tie accounting."""
    x, y = _prepared(x), _prepared(y)
    if x.n != y.n:
        raise ValueError("Arrays must describe the same common stock sample.")
    total = x.n * (x.n - 1) // 2
    both_tied = _joint_tied_pairs(x, y, total)
    untied_both = total - x.tied_pairs - y.tied_pairs + both_tied
    if not 0 <= untied_both <= total:
        raise ArithmeticError("Inconsistent tie counts.")

    if x.tied_pairs == total or y.tied_pairs == total:
        signed = 0
    elif x.n == 2:
        # SciPy's asymptotic p-value formula divides by n-2, although the one
        # pair's sign-product statistic is perfectly well-defined.
        signed = _integer_signed_count(x.values, y.values)
    else:
        error_budget = 64 * _FLOAT_EPS * total
        signed = None
        if error_budget < _RECOVERY_MAX_ERROR:
            tau_b = float(kendalltau(
                x.values, y.values, variant="b", method="asymptotic",
                nan_policy="raise",
            ).statistic)
            raw_count = (
                tau_b * math.sqrt(total - x.tied_pairs)
                * math.sqrt(total - y.tied_pairs)
            )
            if math.isfinite(raw_count):
                candidate = int(round(raw_count))
                tolerance = max(1e-12, error_budget)
                if (abs(raw_count - candidate) <= tolerance
                        and _valid_signed_count(candidate, untied_both)):
                    signed = candidate
        if signed is None:
            signed = _integer_signed_count(x.values, y.values)

    if not _valid_signed_count(signed, untied_both):
        raise ArithmeticError("Sign count violates integer parity or range.")
    concordant = (untied_both + signed) // 2
    discordant = (untied_both - signed) // 2
    if concordant + discordant != untied_both:
        raise ArithmeticError("Concordant/discordant counts do not reconcile.")
    return PairSignCounts(
        total, x.tied_pairs, y.tied_pairs, both_tied, untied_both,
        signed, concordant, discordant,
    )


def prepare_anchor(shallow, returns) -> PreparedAnchor:
    """Cache the common sample and shallow-return signed count for one month."""
    shallow, returns = _prepared(shallow), _prepared(returns)
    return PreparedAnchor(shallow, returns, count_sign_products(shallow, returns))


def pairwise_statistics_from_anchor(anchor: PreparedAnchor, deep) -> dict:
    deep = _prepared(deep)
    if deep.n != anchor.shallow.n:
        raise ValueError("Deep scores must use the anchor's same common sample.")
    sd = count_sign_products(anchor.shallow, deep)
    dr = count_sign_products(deep, anchor.returns)
    sr = anchor.shallow_return
    total = sd.total_pairs
    unchanged = sd.concordant_pairs + sd.both_tied_pairs
    changed = total - unchanged
    released_ties = sd.x_tied_pairs - sd.both_tied_pairs
    introduced_ties = sd.y_tied_pairs - sd.both_tied_pairs
    one_tied = released_ties + introduced_ties
    if changed != sd.discordant_pairs + one_tied:
        raise ArithmeticError("Changed-pair partition does not reconcile.")
    numerator_twice = dr.signed_count - sr.signed_count
    if abs(numerator_twice) > 2 * changed:
        raise ArithmeticError("Correction numerator exceeds changed-pair range.")
    if changed == 0 and numerator_twice != 0:
        raise ArithmeticError("No changed pairs but a nonzero correction.")

    return {
        "n_stocks": anchor.shallow.n,
        "total_pairs": total,
        "unchanged_pairs": unchanged,
        "changed_pairs": changed,
        "strict_concordant_forecast_pairs": sd.concordant_pairs,
        "strict_discordant_forecast_pairs": sd.discordant_pairs,
        "shallow_tied_pairs": sd.x_tied_pairs,
        "deep_tied_pairs": sd.y_tied_pairs,
        "both_forecasts_tied_pairs": sd.both_tied_pairs,
        "released_tie_pairs": released_ties,
        "introduced_tie_pairs": introduced_ties,
        "one_forecast_tied_pairs": one_tied,
        "return_tied_pairs": anchor.returns.tied_pairs,
        "signed_count_shallow_deep": sd.signed_count,
        "signed_count_shallow_return": sr.signed_count,
        "signed_count_deep_return": dr.signed_count,
        "correction_numerator_twice": numerator_twice,
        "P": unchanged / total,
        "F": numerator_twice / (2 * changed) if changed else None,
        "tau_a_shallow_return": sr.signed_count / total,
        "tau_a_deep_return": dr.signed_count / total,
        "total_contribution": numerator_twice / (2 * total),
    }


def pairwise_statistics(shallow, deep, returns) -> dict:
    """Reference-compatible entry point; callers must provide the common sample."""
    return pairwise_statistics_from_anchor(prepare_anchor(shallow, returns), deep)
