from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def normalize_orientation(
    positions: Sequence[Tuple[float, float]],
    field_width: float = 68.0,
) -> np.ndarray:
    """Normalize team direction by optionally flipping Y to a shared orientation."""
    points = np.array(positions, dtype=np.float32)

    if len(points) == 0:
        return points

    normalized = points.copy()

    if float(np.median(normalized[:, 1])) > (field_width / 2.0):
        normalized[:, 1] = field_width - normalized[:, 1]

    return normalized


def remove_goalkeeper_candidate(
    positions: np.ndarray,
    enabled: bool = True,
) -> np.ndarray:
    """Heuristically remove one extreme-depth player as goalkeeper candidate."""
    if not enabled or len(positions) <= 10:
        return positions

    y_values = positions[:, 1]
    low_idx = int(np.argmin(y_values))
    high_idx = int(np.argmax(y_values))

    low_span = float(np.median(y_values) - y_values[low_idx])
    high_span = float(y_values[high_idx] - np.median(y_values))

    remove_idx = low_idx if low_span > high_span else high_idx

    return np.delete(positions, remove_idx, axis=0)


def cluster_player_lines(
    positions: np.ndarray,
    distance_threshold: float = 9.0,
    min_cluster_size: int = 2,
) -> List[Dict]:
    """Cluster players into tactical lines using 1D agglomerative clustering on Y."""
    if len(positions) == 0:
        return []

    y_only = positions[:, 1].reshape(-1, 1)

    if len(positions) == 1:
        return [{"indices": np.array([0]), "count": 1, "mean_y": float(y_only[0, 0])}]

    y_spread = float(np.percentile(y_only, 75) - np.percentile(y_only, 25))
    adaptive_threshold = max(5.0, min(14.0, 0.28 * y_spread + 5.0))
    threshold = float(distance_threshold) if distance_threshold is not None else adaptive_threshold

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
    )
    labels = clustering.fit_predict(y_only)

    clusters: List[Dict] = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < min_cluster_size:
            continue

        cluster_points = positions[indices]

        clusters.append(
            {
                "indices": indices,
                "count": int(len(indices)),
                "mean_y": float(np.mean(cluster_points[:, 1])),
                "points": cluster_points,
            }
        )

    if not clusters:
        order = np.argsort(positions[:, 1])
        split = np.array_split(order, 3)
        for split_indices in split:
            if len(split_indices) == 0:
                continue
            cluster_points = positions[split_indices]
            clusters.append(
                {
                    "indices": split_indices,
                    "count": int(len(split_indices)),
                    "mean_y": float(np.mean(cluster_points[:, 1])),
                    "points": cluster_points,
                }
            )

    clusters.sort(key=lambda cluster: cluster["mean_y"])

    return clusters


def line_counts_from_clusters(clusters: Sequence[Dict]) -> List[int]:
    """Convert sorted cluster metadata into formation counts per line."""
    return [int(cluster["count"]) for cluster in clusters]


def build_structure_graph(
    positions: Sequence[Tuple[float, float]],
    distance_threshold: float = 9.0,
) -> Dict:
    """Build graph nodes/edges using within-line and between-line connectivity."""
    points = np.array(positions, dtype=np.float32)

    if len(points) == 0:
        return {"nodes": [], "edges": [], "lines": []}

    clusters = cluster_player_lines(points, distance_threshold=distance_threshold)

    nodes = [{"id": idx, "position": (float(p[0]), float(p[1]))} for idx, p in enumerate(points)]
    edges = set()
    lines: List[Dict] = []

    prev_line_sorted = None

    for line_idx, cluster in enumerate(clusters):
        indices = list(cluster["indices"])
        sorted_indices = sorted(indices, key=lambda i: points[i, 0])

        for i in range(len(sorted_indices) - 1):
            edge = tuple(sorted((sorted_indices[i], sorted_indices[i + 1])))
            edges.add(edge)

        if prev_line_sorted is not None:
            for i, curr_idx in enumerate(sorted_indices):
                nearest_prev = prev_line_sorted[min(i, len(prev_line_sorted) - 1)]
                edge = tuple(sorted((curr_idx, nearest_prev)))
                edges.add(edge)

        line_points = points[sorted_indices]
        lines.append(
            {
                "line_index": line_idx,
                "player_ids": sorted_indices,
                "mean_y": float(np.mean(line_points[:, 1])),
                "count": int(len(sorted_indices)),
            }
        )

        prev_line_sorted = sorted_indices

    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in sorted(edges)],
        "lines": lines,
    }
