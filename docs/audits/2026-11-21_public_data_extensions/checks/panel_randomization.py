"""Exact sharp-null tests for four players rematched in three rounds (synthetic data)."""

from collections import Counter
from fractions import Fraction
from itertools import product
import json

P = Fraction(5, 11)
# Within each round the two players with the same label share their coin.
PAIRS = (0, 0, 1, 1, 2, 3, 2, 3, 4, 5, 5, 4)
# Hundredths of the preceding benchmark's assignment-invariant response.
OUTCOMES = tuple(15 + 3 * i + 10 * t for t in range(1, 4) for i in range(1, 5))


def statistic(bits, lag):
    # Positive constant factors common to all assignments cancel in the tail ordering.
    if lag == 0:
        return sum(y * (6 if w else -5) for w, y in zip(bits, OUTCOMES, strict=True))
    return sum(
        OUTCOMES[i] * (1 if bits[i - 4] else -1) * (6 if bits[i - 4] else 5) * (6 if bits[i] else 5)
        for i in range(4, 12)
    )


def ht(bits, lag):
    terms = []
    for i in range(4 * lag, len(bits)):
        prob = P if bits[i] else 1 - P
        if lag:
            prob *= P if bits[i - 4] else 1 - P
        terms.append(OUTCOMES[i] * (2 * bits[i - 4 * lag] - 1) / (100 * 2**lag * prob))
    return sum(terms) / len(terms)


def distribution(lag, paired):
    n = 6 if paired else 12
    mass = Counter()
    for assignment in product((0, 1), repeat=n):
        bits = tuple(assignment[i] for i in PAIRS) if paired else assignment
        score = statistic(bits, lag)
        if paired:
            multiplier = (
                Fraction(11, 30 * 100 * 12) if lag == 0 else Fraction(121, 2 * 900 * 100 * 8)
            )
            if ht(bits, lag) != score * multiplier:
                raise ValueError("Integer ranking disagrees with the rational HT statistic")
        mass[abs(score)] += 5 ** sum(assignment) * 6 ** (n - sum(assignment))
    if sum(mass.values()) != 11**n:
        raise ValueError("Incomplete assignment distribution")
    return mass


def upper_tails(mass):
    count = 0
    out = {}
    for score in sorted(mass, reverse=True):
        count += mass[score]
        out[score] = Fraction(count, sum(mass.values()))
    return out


def run():
    tests = []
    for lag in (0, 1):
        truth = distribution(lag, True)
        for name, mass in (("pair_round", truth), ("individual_round", distribution(lag, False))):
            p = upper_tails(mass)
            size = sum(Fraction(v, 11**6) for s, v in truth.items() if p[s] <= Fraction(1, 20))
            if name == "pair_round" and size > Fraction(1, 20):
                raise ValueError("Exact randomization size exceeds the nominal level")
            tests.append(
                dict(
                    lag=lag,
                    reference=name,
                    size=str(size),
                    size_decimal=float(size),
                    distinct_absolute_statistics=len(mass),
                    assignments=64 if name == "pair_round" else 4096,
                )
            )
    return dict(
        pair_coins=6,
        treatment_probability=str(P),
        alpha="1/20",
        outcomes_hundredths=OUTCOMES,
        two_sided="absolute HT; inclusive ties",
        tests=tests,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
