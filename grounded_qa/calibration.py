from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    answer_coverage: float
    answer_precision: float
    false_answer_rate: float
    false_refusal_rate: float
    refusal_recall: float


def sweep_thresholds(probabilities: list[float], answerable: list[bool], thresholds: int = 101) -> list[ThresholdPoint]:
    if len(probabilities) != len(answerable):
        raise ValueError("probabilities and labels must have the same length")
    points = []
    grid = [index / max(thresholds - 1, 1) for index in range(thresholds)]
    # BFloat16 sigmoid values can round to exactly 1.0; this is the required
    # always-refuse fallback when no finite in-range threshold is safe.
    for threshold in [*grid, math.nextafter(1.0, math.inf)]:
        answered = [probability >= threshold for probability in probabilities]
        positives = sum(answerable)
        negatives = len(answerable) - positives
        true_answers = sum(answered[i] and answerable[i] for i in range(len(answerable)))
        false_refusals = sum(not answered[i] and answerable[i] for i in range(len(answerable)))
        false_answers = sum(answered[i] and not answerable[i] for i in range(len(answerable)))
        answered_count = sum(answered)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                answer_coverage=true_answers / max(positives, 1),
                answer_precision=true_answers / max(answered_count, 1),
                false_answer_rate=false_answers / max(negatives, 1),
                false_refusal_rate=false_refusals / max(positives, 1),
                refusal_recall=(negatives - false_answers) / max(negatives, 1),
            )
        )
    return points


def choose_threshold(points: list[ThresholdPoint], max_false_answer_rate: float = 0.02) -> ThresholdPoint:
    valid = [point for point in points if point.false_answer_rate <= max_false_answer_rate]
    if not valid:
        raise ValueError("No threshold meets the false-answer constraint")
    return max(valid, key=lambda point: point.answer_coverage)
