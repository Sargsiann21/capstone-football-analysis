import csv
import os
import re
from typing import List, Optional, Sequence, Tuple

VALID_FORMATIONS: List[str] = [
    "4-3-3",
    "4-4-2",
    "4-2-3-1",
    "3-5-2",
    "3-4-3",
    "5-3-2",
    "4-1-4-1",
]


def normalize_formation_name(raw_value: str) -> Optional[str]:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    value_no_variant = re.sub(r"\([^)]*\)", "", value).strip()

    numbers = re.findall(r"\d+", value_no_variant)

    if len(numbers) < 3:
        return None

    return "-".join(numbers)


def load_formations_from_csv(
    csv_path: str,
    fallback: Sequence[str] = VALID_FORMATIONS,
) -> List[str]:
    if not csv_path or not os.path.exists(csv_path):
        return list(fallback)

    ordered_unique = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            candidates = [
                row.get("Formation", ""),
                row.get("Formation Base", ""),
            ]

            for candidate in candidates:
                normalized = normalize_formation_name(candidate)
                if normalized is None or normalized in seen:
                    continue

                seen.add(normalized)
                ordered_unique.append(normalized)

    return ordered_unique if ordered_unique else list(fallback)


def parse_formation(formation: str) -> Tuple[int, ...]:
    """Parse formation string into tuple form, e.g. '4-3-3' -> (4, 3, 3)."""
    return tuple(int(part) for part in formation.split("-"))


def to_formation_string(line_counts: Sequence[int]) -> str:
    """Format line counts as canonical dash-separated formation string."""
    return "-".join(str(int(count)) for count in line_counts)


def _shape_distance(source: Sequence[int], target: Sequence[int]) -> int:
    source_list = list(source)
    target_list = list(target)

    max_len = max(len(source_list), len(target_list))

    source_list.extend([0] * (max_len - len(source_list)))
    target_list.extend([0] * (max_len - len(target_list)))

    structural = sum(abs(a - b) for a, b in zip(source_list, target_list))
    line_penalty = 2 * abs(len(source) - len(target))
    player_penalty = 2 * abs(sum(source) - sum(target))

    return structural + line_penalty + player_penalty


def map_to_closest_valid_formation(
    line_counts: Sequence[int],
    valid_formations: Sequence[str] = VALID_FORMATIONS,
) -> str:
    """Map raw clustered line counts to nearest allowed tactical template."""
    if not line_counts:
        return "Unknown"

    best_formation = None
    best_distance = float("inf")

    for formation in valid_formations:
        parsed = parse_formation(formation)
        distance = _shape_distance(line_counts, parsed)

        if distance < best_distance:
            best_distance = distance
            best_formation = formation

    return best_formation or "Unknown"
