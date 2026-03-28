from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .clustering_utils import (
    build_structure_graph,
    cluster_player_lines,
    line_counts_from_clusters,
    normalize_orientation,
    remove_goalkeeper_candidate,
)
from .formation_templates import (
    VALID_FORMATIONS,
    load_formations_from_csv,
    parse_formation,
    map_to_closest_valid_formation,
    to_formation_string,
)


class FormationDetector:
    """Detects and smooths tactical formations for each team from top-down player coordinates."""
    def __init__(
        self,
        history_size: int = 50,
        distance_threshold: Optional[float] = None,
        min_players: int = 7,
        ignore_goalkeeper: bool = True,
        field_width: float = 68.0,
        field_length: float = 105.0,
        transition_penalty: float = 0.08,
        valid_formations: Optional[Sequence[str]] = None,
        formations_csv_path: Optional[str] = None,
    ):
        self.history_size = history_size
        self.distance_threshold = distance_threshold
        self.min_players = min_players
        self.ignore_goalkeeper = ignore_goalkeeper
        self.field_width = field_width
        self.field_length = field_length
        self.transition_penalty = transition_penalty
        if valid_formations is not None:
            self.valid_formations = list(valid_formations)
        else:
            default_csv_path = Path(__file__).resolve().parent.parent / "Formations.csv"
            csv_path = formations_csv_path or str(default_csv_path)
            self.valid_formations = load_formations_from_csv(csv_path, fallback=VALID_FORMATIONS)

        self.formation_history: Dict[int, deque] = {
            1: deque(maxlen=history_size),
            2: deque(maxlen=history_size),
        }
        self.last_output_formations: Dict[int, str] = {1: "Unknown", 2: "Unknown"}

        self.latest_team_graph: Dict[int, Dict] = defaultdict(dict)

    def _extract_team_positions(
        self,
        object_tracks: Dict,
        frame_num: int,
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Extract per-team transformed coordinates from tracking output for one frame."""
        team_positions = {1: [], 2: []}

        players_frame = object_tracks.get("Players", [])
        if frame_num >= len(players_frame):
            return team_positions

        for _, player in players_frame[frame_num].items():
            team_id = player.get("team")
            if team_id not in (1, 2):
                continue

            transformed = player.get("position_transformed")
            if transformed is None:
                continue

            pos = np.array(transformed, dtype=np.float32).reshape(-1)
            if len(pos) < 2 or np.any(np.isnan(pos[:2])):
                continue

            team_positions[team_id].append((float(pos[0]), float(pos[1])))

        return team_positions

    def _normalize_points_for_matching(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        normalized = points.copy().astype(np.float32)
        normalized[:, 0] /= max(self.field_length, 1.0)
        normalized[:, 1] /= max(self.field_width, 1.0)
        return normalized

    def _template_points_for_formation(self, formation: str) -> np.ndarray:
        counts = parse_formation(formation)
        if len(counts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        y_values = np.linspace(0.18, 0.82, num=len(counts), dtype=np.float32)
        template_points = []

        for y_value, count in zip(y_values, counts):
            x_values = np.linspace(0.15, 0.85, num=max(count, 1), dtype=np.float32)
            for x_value in x_values:
                template_points.append((x_value, float(y_value)))

        return np.array(template_points, dtype=np.float32)

    def _symmetric_chamfer_distance(self, a_points: np.ndarray, b_points: np.ndarray) -> float:
        if len(a_points) == 0 or len(b_points) == 0:
            return float("inf")

        distances = np.linalg.norm(a_points[:, None, :] - b_points[None, :, :], axis=2)
        a_to_b = float(np.mean(np.min(distances, axis=1)))
        b_to_a = float(np.mean(np.min(distances, axis=0)))

        return 0.5 * (a_to_b + b_to_a)

    def _score_formation_candidate(
        self,
        norm_points: np.ndarray,
        raw_line_counts: Sequence[int],
        candidate: str,
        previous_formation: str,
    ) -> float:
        template_points = self._template_points_for_formation(candidate)
        chamfer = self._symmetric_chamfer_distance(norm_points, template_points)

        target_counts = parse_formation(candidate)
        line_mismatch = sum(abs(a - b) for a, b in zip(list(raw_line_counts) + [0] * 6, list(target_counts) + [0] * 6))
        line_mismatch *= 0.02

        transition = self.transition_penalty if previous_formation not in ("Unknown", candidate) else 0.0

        return chamfer + line_mismatch + transition

    def detect_formation(
        self,
        team_positions: Sequence[Tuple[float, float]],
        previous_formation: str = "Unknown",
    ) -> Tuple[str, float]:
        """Estimate formation string and confidence from one team snapshot."""
        if len(team_positions) < self.min_players:
            return "Unknown", 0.0

        points = normalize_orientation(team_positions, self.field_width)
        points = remove_goalkeeper_candidate(points, enabled=self.ignore_goalkeeper)

        if len(points) < self.min_players:
            return "Unknown", 0.0

        clusters = cluster_player_lines(
            points,
            distance_threshold=self.distance_threshold,
            min_cluster_size=2,
        )

        line_counts = line_counts_from_clusters(clusters)

        if not line_counts:
            return "Unknown", 0.0

        norm_points = self._normalize_points_for_matching(points)

        best_candidate = None
        best_score = float("inf")

        for candidate in self.valid_formations:
            score = self._score_formation_candidate(
                norm_points,
                line_counts,
                candidate,
                previous_formation,
            )

            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            confidence = float(np.clip(1.0 - (best_score / 0.35), 0.0, 1.0))
            return best_candidate, confidence

        raw_formation = to_formation_string(line_counts)

        if raw_formation in self.valid_formations:
            return raw_formation, 0.6

        mapped = map_to_closest_valid_formation(line_counts, self.valid_formations)
        return mapped, 0.45

    def _smoothed_formation(self, team_id: int) -> str:
        """Return confidence-weighted temporal mode of recent detections for one team."""
        history = self.formation_history[team_id]

        if not history:
            return "Unknown"

        weighted_scores = defaultdict(float)

        for idx, item in enumerate(history):
            formation = item["formation"]
            confidence = float(item["confidence"])

            if formation == "Unknown":
                continue

            recency_weight = 0.65 + 0.35 * ((idx + 1) / len(history))
            weighted_scores[formation] += confidence * recency_weight

        if not weighted_scores:
            return "Unknown"

        return max(weighted_scores.items(), key=lambda x: x[1])[0]

    def get_team_structure_graph(
        self,
        player_positions: Sequence[Tuple[float, float]],
    ) -> Dict:
        """Build a line-aware graph structure for UI drawing support."""
        points = normalize_orientation(player_positions, self.field_width)
        points = remove_goalkeeper_candidate(points, enabled=self.ignore_goalkeeper)

        if len(points) == 0:
            return {"nodes": [], "edges": [], "lines": []}

        return build_structure_graph(
            points.tolist(),
            distance_threshold=self.distance_threshold,
        )

    def update(self, object_tracks: Dict, frame_num: int) -> Dict[str, str]:
        """Update both teams for the current frame and return smoothed formations."""
        team_positions = self._extract_team_positions(object_tracks, frame_num)

        team_formations = {}

        for team_id in (1, 2):
            positions = team_positions[team_id]

            prev = self.last_output_formations.get(team_id, "Unknown")
            formation, confidence = self.detect_formation(positions, previous_formation=prev)

            self.formation_history[team_id].append(
                {
                    "formation": formation,
                    "confidence": confidence,
                }
            )

            team_formations[team_id] = self._smoothed_formation(team_id)
            self.last_output_formations[team_id] = team_formations[team_id]
            self.latest_team_graph[team_id] = self.get_team_structure_graph(positions)

        return {
            "team1_formation": team_formations[1],
            "team2_formation": team_formations[2],
        }

    def draw_overlay(
        self,
        frame,
        formations: Dict[str, str],
        panel_origin: Tuple[int, int] = (30, 120),
    ):
        """Draw compact formation labels onto a video frame."""
        x, y = panel_origin

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 380, y + 90), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        cv2.putText(
            frame,
            f"Team 1 Formation: {formations.get('team1_formation', 'Unknown')}",
            (x + 12, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Team 2 Formation: {formations.get('team2_formation', 'Unknown')}",
            (x + 12, y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return frame
