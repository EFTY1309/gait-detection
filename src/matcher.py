"""
matcher.py
----------
Cosine-similarity nearest-neighbour matching of a query gait feature vector
against the enrolled gallery.

For each enrolled person, we compute cosine similarity between the query
vector and every stored feature vector, then take the average. The person
with the highest average similarity wins — if that similarity exceeds
the threshold, we return their ID; otherwise we return "Unknown".
"""

from typing import Dict, List, Tuple

import numpy as np

Gallery = Dict[str, List[np.ndarray]]

# Minimum similarity score to accept a match at all.
# With the current robust gait feature layout, same-person cosine similarity is
# typically very high, so this threshold can remain safely elevated.
DEFAULT_THRESHOLD = 0.85

# When 2+ people are enrolled the best score must beat runner-up by this margin.
# For small galleries with visually similar walkers, observed score gaps can be
# very small (~0.006-0.01), so keep the default conservative but not overly strict.
DEFAULT_MARGIN = 0.005

# Gallery standardization becomes useful only once the gallery has enough
# diversity to estimate per-feature spread. Below this size it is too unstable.
MIN_VECTORS_FOR_STANDARDIZATION = 4


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.
    Returns a value in [-1, 1]. Closer to 1 → more similar.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _score_gallery(
    query_feature: np.ndarray,
    gallery: Gallery,
    *,
    standardized: bool,
) -> List[Tuple[str, float]]:
    """Return average similarity per enrolled identity."""
    person_ids: List[str] = []
    vectors: List[np.ndarray] = []
    counts: List[int] = []

    for person_id, person_vectors in gallery.items():
        if not person_vectors:
            continue
        person_ids.append(person_id)
        counts.append(len(person_vectors))
        vectors.extend(person_vectors)

    if not vectors:
        return []

    query = query_feature
    gallery_vectors = vectors

    if standardized and len(vectors) >= MIN_VECTORS_FOR_STANDARDIZATION:
        stacked = np.stack(vectors, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        query = (query_feature - mean) / std
        gallery_vectors = [(vector - mean) / std for vector in vectors]

    scores: List[Tuple[str, float]] = []
    cursor = 0
    for person_id, count in zip(person_ids, counts):
        sims = [
            cosine_similarity(query, gallery_vectors[cursor + idx])
            for idx in range(count)
        ]
        scores.append((person_id, float(np.mean(sims))))
        cursor += count

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def match(
    query_feature: np.ndarray,
    gallery: Gallery,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> Tuple[str, float]:
    """
    Find the best-matching person in the gallery for a given query feature.

    Parameters
    ----------
    query_feature : np.ndarray shape (N,) — the extracted gait feature
    gallery       : dict  {person_id: [feature_vec, ...]}
    threshold     : minimum similarity score to accept (below → "Unknown")
    margin        : with 2+ people enrolled, the best score must exceed
                    the second-best by at least this amount (below → "Unknown")

    Returns
    -------
    (identity, score)
      identity : person_id string, or "Unknown"
      score    : best cosine similarity score (0–1)
    """
    if not gallery:
        return "Unknown", 0.0

    raw_scores = _score_gallery(query_feature, gallery, standardized=False)
    if not raw_scores:
        return "Unknown", 0.0

    ranked_scores = _score_gallery(query_feature, gallery, standardized=True)
    if not ranked_scores:
        ranked_scores = raw_scores

    raw_by_person = dict(raw_scores)
    best_id = ranked_scores[0][0]
    best_score = raw_by_person[best_id]

    # 1. Absolute threshold: must be similar enough at all
    if best_score < threshold:
        return "Unknown", best_score

    # 2. Margin check: with multiple enrolled people, the winner must clearly
    #    beat the runner-up.  A thin margin means the query doesn't
    #    strongly belong to any enrolled class → Unknown.
    if len(ranked_scores) >= 2:
        second_score = ranked_scores[1][1]
        if (ranked_scores[0][1] - second_score) < margin:
            return "Unknown", best_score

    return best_id, best_score


def match_all(
    query_feature: np.ndarray,
    gallery: Gallery,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Tuple[str, float]]:
    """
    Return a ranked list of (person_id, score) for all gallery entries,
    sorted by descending similarity. Useful for CMC-curve evaluation.
    """
    scores: List[Tuple[str, float]] = []
    for person_id, vectors in gallery.items():
        if not vectors:
            continue
        sims = [cosine_similarity(query_feature, v) for v in vectors]
        avg_sim = float(np.mean(sims))
        scores.append((person_id, avg_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
