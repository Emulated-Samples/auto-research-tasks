"""Gauge-invariant behavioral metrics for additive grouped featurizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import Prediction
from .zoo import Split


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular O(n^3) Hungarian assignment without a modeling dependency."""
    values = np.asarray(cost, dtype=np.float64)
    rows, columns = values.shape
    size = max(rows, columns)
    pad_value = float(values.max() + 1.0) if values.size else 1.0
    padded = np.full((size, size), pad_value, dtype=np.float64)
    padded[:rows, :columns] = values
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    assigned_row = np.zeros(size + 1, dtype=np.int64)
    previous_column = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        assigned_row[0] = row
        column = 0
        reduced = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = assigned_row[column]
            delta = np.inf
            next_column = -1
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                current = padded[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if current < reduced[candidate]:
                    reduced[candidate] = current
                    previous_column[candidate] = column
                if reduced[candidate] < delta:
                    delta = reduced[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[assigned_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    reduced[candidate] -= delta
            column = next_column
            if assigned_row[column] == 0:
                break
        while column:
            prior = previous_column[column]
            assigned_row[column] = assigned_row[prior]
            column = prior
    row_to_column = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        row_to_column[assigned_row[column] - 1] = column - 1
    row_indices = np.arange(rows, dtype=np.int64)
    column_indices = row_to_column[:rows]
    keep = column_indices < columns
    return row_indices[keep], column_indices[keep]


@dataclass(frozen=True)
class Match:
    truth_to_feature: np.ndarray
    presence_threshold: np.ndarray


@dataclass(frozen=True)
class FactorMetrics:
    reconstruction_r2: float
    contribution_r2: float
    adjusted_average_precision: float
    mean_false_positive_rate: float
    mean_false_negative_rate: float
    support_youden: float
    fragmentation: float
    merging: float
    assignment_coherence: float
    geometry_score: float
    counterfactual_r2: float
    description_bits: float
    active_features: float


def variance_r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    values = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(estimate, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered * centered))
    if denominator <= 1e-15:
        return 0.0
    return float(1.0 - np.sum((values - predictions) ** 2) / denominator)


def _tie_group_ends(sorted_scores: np.ndarray) -> np.ndarray:
    """Indices where a run of equal scores ends -- the only realizable cut points.

    Everything downstream ranks by score, and a score is the only thing a method
    actually asserts. Two rows with the same score make no claim about their
    relative order, so any statistic that reads them one at a time is reading the
    tiebreak of a sort, not the method's output.
    """
    if sorted_scores.size == 0:
        return np.empty(0, dtype=np.int64)
    changed = np.ones(sorted_scores.size, dtype=bool)
    changed[:-1] = np.diff(sorted_scores) != 0.0
    return np.flatnonzero(changed)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision evaluated at distinct score thresholds.

    Tied scores are resolved as a group rather than one row at a time. Scoring
    within a tie group makes the result depend on the order rows happen to arrive
    in: a constant presence predictor over ``labels=[True, False]`` scored 1.0
    when the positive row came first and 0.5 when it came second, so an entirely
    uninformative method swung from perfect discovery to none on row order alone.
    Grouping makes this permutation-invariant, and prices a constant predictor at
    prevalence -- which is exactly the information it carries.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = _tie_group_ends(sorted_scores)
    true_positive = np.cumsum(sorted_labels)[ends]
    false_positive = np.cumsum(~sorted_labels)[ends]
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positives
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous_recall) * precision))


def adjusted_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    prevalence = float(np.mean(labels))
    ap = average_precision(labels, scores)
    return float(np.clip((ap - prevalence) / max(1e-12, 1.0 - prevalence), 0.0, 1.0))


def _pair_quality(
    split: Split, prediction: Prediction, truth_index: int, feature_index: int
) -> float:
    recovery = max(
        0.0,
        variance_r2(
            split.contributions[:, truth_index, :], prediction.contributions[:, feature_index, :]
        ),
    )
    discovery = adjusted_average_precision(
        split.active[:, truth_index], prediction.presence[:, feature_index]
    )
    return 0.72 * min(1.0, recovery) + 0.28 * discovery


def match_factors(split: Split, prediction: Prediction) -> Match:
    m = split.contributions.shape[1]
    g = prediction.n_features
    quality = np.empty((m, g), dtype=np.float64)
    for truth_index in range(m):
        for feature_index in range(g):
            quality[truth_index, feature_index] = _pair_quality(
                split, prediction, truth_index, feature_index
            )
    rows, columns = linear_sum_assignment(-quality)
    truth_to_feature = np.full(m, -1, dtype=np.int64)
    truth_to_feature[rows] = columns
    thresholds = np.full(m, np.inf, dtype=np.float64)
    for truth_index, feature_index in enumerate(truth_to_feature):
        if feature_index < 0:
            continue
        thresholds[truth_index] = _presence_threshold(
            split.active[:, truth_index],
            prediction.presence[:, feature_index],
        )
    return Match(
        truth_to_feature=truth_to_feature,
        presence_threshold=thresholds,
    )


def _presence_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose a deterministic balanced-accuracy cutoff on matching data.

    Matching is already a private, disjoint calibration split. Learning the
    cutoff there and freezing it for scoring tests whether a method's presence
    scale generalizes. This avoids using the hidden score prevalence as an
    oracle routing budget.
    """
    truth = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0:
        return np.inf
    if negatives == 0:
        return -np.inf
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_truth = truth[order]
    # Only cut where a tie group ends. Cutting inside one optimizes a prefix that
    # the returned threshold cannot reproduce: for scores [0, 2, 2, 1] the old
    # search selected the first of the two rows tied at 2.0 and then returned
    # 2.0, which admits both -- balanced accuracy 0.50, where the realizable cut
    # at 0.5 scores 0.75. The optimizer and the classifier it returned were
    # different classifiers.
    ends = _tie_group_ends(sorted_values)
    balanced = 0.5 * (
        np.cumsum(sorted_truth)[ends] / positives
        + 1.0
        - np.cumsum(~sorted_truth)[ends] / negatives
    )
    best = int(ends[int(np.argmax(balanced))])
    if best == len(values) - 1:
        return float(np.nextafter(sorted_values[best], -np.inf))
    return float(0.5 * (sorted_values[best] + sorted_values[best + 1]))


def _rates_for_match(split: Split, prediction: Prediction, match: Match) -> tuple[np.ndarray, ...]:
    m = split.active.shape[1]
    recoveries = np.zeros(m, dtype=np.float64)
    discoveries = np.zeros(m, dtype=np.float64)
    false_positive = np.ones(m, dtype=np.float64)
    false_negative = np.ones(m, dtype=np.float64)
    for truth_index, feature_index in enumerate(match.truth_to_feature):
        if feature_index < 0:
            continue
        recoveries[truth_index] = np.clip(
            variance_r2(
                split.contributions[:, truth_index, :],
                prediction.contributions[:, feature_index, :],
            ),
            0.0,
            1.0,
        )
        labels = split.active[:, truth_index]
        scores = prediction.presence[:, feature_index]
        discoveries[truth_index] = adjusted_average_precision(labels, scores)
        # The threshold was selected once on the disjoint matching split. It
        # is deliberately not recalibrated to the hidden score prevalence.
        predicted = scores >= match.presence_threshold[truth_index]
        false_positive[truth_index] = float(np.mean(predicted[~labels])) if np.any(~labels) else 0.0
        false_negative[truth_index] = float(np.mean(~predicted[labels])) if np.any(labels) else 0.0
    return recoveries, discoveries, false_positive, false_negative


def _stable_rank(values: np.ndarray) -> float:
    if values.shape[0] < 2 or not np.any(values):
        return 0.0
    singular = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        return 0.0
    return float(np.sum(singular * singular) / (singular[0] * singular[0]))


def _rate_bits(values: np.ndarray, distortion: float) -> float:
    if values.shape[0] < 2:
        return 0.0
    covariance = np.cov(values, rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(0.5 * np.sum(np.log2(1.0 + np.maximum(eigenvalues, 0.0) / distortion)))


def description_bits(x: np.ndarray, prediction: Prediction) -> tuple[float, float]:
    """Structured-support MDL estimate in bits per row.

    Supports are charged by their empirical pattern entropy, so a directional
    method receives full credit for predictable co-firing.  This avoids the
    biased ``log choose`` comparison that would price every scalar support as
    independent while granting blocks structured indices for free.
    """
    norms = np.linalg.norm(prediction.contributions, axis=2)
    row_scale = np.linalg.norm(x, axis=1, keepdims=True)
    support = norms > np.maximum(1e-8, 1e-5 * row_scale)
    packed = np.packbits(support, axis=1)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    support_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    support_entropy += float((counts.size - 1) / (2.0 * len(support) * np.log(2.0)))

    variance = float(np.mean(np.var(x.astype(np.float64), axis=0)))
    distortion = max(1e-8, 0.10 * variance)
    code_bits = 0.0
    dictionary_bits = 0.0
    n, _, d = prediction.contributions.shape
    for feature in range(prediction.n_features):
        rows = support[:, feature]
        if not np.any(rows):
            continue
        values = prediction.contributions[rows, feature, :]
        firing_rate = float(np.mean(rows))
        code_bits += firing_rate * _rate_bits(values, distortion)
        rank = min(d, max(1, round(_stable_rank(values))))
        dictionary_bits += 0.5 * rank * (d - rank) * np.log2(max(n, 2)) / n
    residual = x.astype(np.float64) - prediction.reconstruction
    residual_bits = _rate_bits(residual, distortion)
    return support_entropy + code_bits + residual_bits + dictionary_bits, float(
        support.sum(axis=1).mean()
    )


def fragmentation_score(split: Split, prediction: Prediction) -> float:
    """How much a second learned object competes with the best for each factor."""
    m = split.contributions.shape[1]
    values: list[float] = []
    for truth_index in range(m):
        qualities = np.array(
            [
                _pair_quality(split, prediction, truth_index, feature)
                for feature in range(prediction.n_features)
            ]
        )
        if qualities.size < 2 or qualities.max() <= 1e-12:
            values.append(0.0)
            continue
        best = np.partition(qualities, -2)[-2:]
        values.append(float(np.min(best) / np.max(best)))
    return float(np.mean(values))


def merging_score(split: Split, prediction: Prediction) -> float:
    """How much one learned object competes for two distinct true factors."""
    values: list[float] = []
    for feature in range(prediction.n_features):
        qualities = np.array(
            [
                _pair_quality(split, prediction, truth, feature)
                for truth in range(split.contributions.shape[1])
            ]
        )
        if qualities.size < 2 or qualities.max() <= 1e-6:
            continue
        best = np.partition(qualities, -2)[-2:]
        values.append(float(np.min(best) / np.max(best)))
    return float(np.mean(values)) if values else 0.0


def _knn_overlap(truth: np.ndarray, estimate: np.ndarray) -> float:
    rows = len(truth)
    if rows < 6:
        return 0.0
    if rows > 160:
        indices = np.linspace(0, rows - 1, 160, dtype=np.int64)
        truth = truth[indices]
        estimate = estimate[indices]
        rows = len(indices)
    truth_distance = np.sum((truth[:, None] - truth[None, :]) ** 2, axis=2)
    estimate_distance = np.sum((estimate[:, None] - estimate[None, :]) ** 2, axis=2)
    np.fill_diagonal(truth_distance, np.inf)
    np.fill_diagonal(estimate_distance, np.inf)
    neighbors = min(8, rows - 1)
    truth_knn = np.argpartition(truth_distance, neighbors - 1, axis=1)[:, :neighbors]
    estimate_knn = np.argpartition(estimate_distance, neighbors - 1, axis=1)[:, :neighbors]
    overlap = np.mean(
        [
            len(set(left.tolist()) & set(right.tolist())) / neighbors
            for left, right in zip(truth_knn, estimate_knn, strict=True)
        ]
    )
    chance = neighbors / (rows - 1)
    return float(np.clip((overlap - chance) / max(1.0 - chance, 1e-12), 0.0, 1.0))


def geometry_alignment(split: Split, prediction: Prediction, match: Match) -> float:
    """Gauge-free local-neighborhood recovery over each complete factor image."""
    scores: list[float] = []
    for truth_index, feature_index in enumerate(match.truth_to_feature):
        if feature_index < 0:
            scores.append(0.0)
            continue
        rows = split.active[:, truth_index]
        scores.append(
            _knn_overlap(
                split.contributions[rows, truth_index],
                prediction.contributions[rows, feature_index],
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def counterfactual_recovery(split: Split, prediction: Prediction, match: Match) -> float:
    """Localize paired one-factor changes without moving any other object.

    Leakage is defined over the submitted decomposition, not merely the subset
    of learned features that happened to win a truth-factor assignment.  An
    earlier implementation iterated over ``match.truth_to_feature`` and was
    therefore blind to every unmatched learned feature.  A submission could
    recover the matched factors cleanly, put arbitrary row-varying residual in
    one extra feature, and receive the same counterfactual score as if that
    residual object had stayed fixed.  That contradicted the public requirement
    that *unrelated learned objects* remain stable and rewarded exactly the
    overcomplete residual channel the causal test is meant to expose.
    """
    truth_energy = 0.0
    predicted_energy = 0.0
    alignment = 0.0
    leakage_energy = 0.0
    for identifier in np.unique(split.pair_id[split.pair_id >= 0]):
        rows = np.flatnonzero(split.pair_id == identifier)
        if len(rows) != 2:
            raise RuntimeError(f"counterfactual pair {identifier} does not have two rows")
        left, right = rows
        truth_index = int(split.changed_factor[left])
        if truth_index < 0 or split.changed_factor[right] != truth_index:
            raise RuntimeError(f"counterfactual pair {identifier} has invalid change metadata")
        feature_index = int(match.truth_to_feature[truth_index])
        truth_delta = (
            split.contributions[right, truth_index] - split.contributions[left, truth_index]
        )
        truth_energy += float(truth_delta @ truth_delta)
        if feature_index < 0:
            continue
        predicted_delta = (
            prediction.contributions[right, feature_index]
            - prediction.contributions[left, feature_index]
        )
        predicted_energy += float(predicted_delta @ predicted_delta)
        alignment += float(truth_delta @ predicted_delta)
        for other_feature in range(prediction.n_features):
            if other_feature == feature_index:
                continue
            leak = (
                prediction.contributions[right, other_feature]
                - prediction.contributions[left, other_feature]
            )
            leakage_energy += float(leak @ leak)
    if truth_energy <= 1e-12 or predicted_energy <= 1e-12:
        return 0.0
    cosine = max(0.0, alignment / np.sqrt(truth_energy * predicted_energy))
    scale_ratio = np.sqrt(predicted_energy / truth_energy)
    scale_match = min(scale_ratio, 1.0 / max(scale_ratio, 1e-12))
    leakage_penalty = 1.0 / (1.0 + 0.25 * leakage_energy / truth_energy)
    return float(np.clip(cosine * scale_match * leakage_penalty, 0.0, 1.0))


def support_youden(false_positive: np.ndarray, false_negative: np.ndarray) -> float:
    """Presence-decision informedness, one factor at a time then averaged.

    A one-sided rate is not on its own evidence of anything: a method that
    rarely fires has an excellent false-positive rate and a terrible
    false-negative rate, and the reverse for a method that fires constantly.
    Directional PCA in fact attains a *better* false-positive rate than the
    learned reference on most suites while missing half of all active factors.
    Youden's J is the standard threshold-free-of-prevalence summary of that
    tradeoff and is the combination that separates the two.
    """
    informedness = 1.0 - np.asarray(false_positive) - np.asarray(false_negative)
    return float(np.clip(informedness, 0.0, 1.0).mean())


def assignment_coherence(fragmentation: float, merging: float) -> float:
    """One learned object for one true factor, scored in both directions at once.

    Fragmentation and merging are the two ways a one-to-one correspondence can
    fail, and neither is evidence of anything alone. The one-object collapse
    scores a *perfect* 0.000 fragmentation on every suite -- with a single
    learned object there is no second object to compete for a factor, so the
    metric is vacuous exactly where it should be damning -- while a method that
    shatters every factor into private detectors scores a perfect merging. Only
    the joint quantity says "one learned object recovers one true factor", which
    is what the task actually asks for.
    """
    return float(np.clip(1.0 - fragmentation - merging, 0.0, 1.0))


def score_prediction(split: Split, prediction: Prediction, match: Match) -> FactorMetrics:
    recoveries, discoveries, fpr, fnr = _rates_for_match(split, prediction, match)
    bits, active_features = description_bits(split.x, prediction)
    fragmentation = fragmentation_score(split, prediction)
    merging = merging_score(split, prediction)
    return FactorMetrics(
        reconstruction_r2=float(np.clip(variance_r2(split.x, prediction.reconstruction), 0.0, 1.0)),
        contribution_r2=float(recoveries.mean()),
        adjusted_average_precision=float(discoveries.mean()),
        mean_false_positive_rate=float(fpr.mean()),
        mean_false_negative_rate=float(fnr.mean()),
        support_youden=support_youden(fpr, fnr),
        fragmentation=fragmentation,
        merging=merging,
        assignment_coherence=assignment_coherence(fragmentation, merging),
        geometry_score=geometry_alignment(split, prediction, match),
        counterfactual_r2=counterfactual_recovery(split, prediction, match),
        description_bits=bits,
        active_features=active_features,
    )
