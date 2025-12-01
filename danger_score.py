"""Shared danger_score computation utilities."""
from typing import Optional


def _bucket(value: Optional[float], thresholds: list[float]) -> int:
    """Return score 0..len(thresholds) based on thresholds, treating None as max."""
    if value is None:
        return len(thresholds)
    for idx, th in enumerate(thresholds):
        if value < th:
            return idx
    return len(thresholds)


def compute_danger_score(
    width_m: Optional[float] = None,
    slope_deg: Optional[float] = None,
    curvature: Optional[float] = None,
    visibility: Optional[float] = None,
    ground_condition: Optional[int] = None,
) -> float:
    """
    暫定ヒューリスティックな danger_score を返す。
    値は 1.0〜5.0 にクリップすること。
    - 幅が狭いほど高スコア（危険）
    - 勾配が急なほど高スコア
    - カーブ（curvature）が大きいほど高スコア
    - visibility が低いほど高スコア
    - ground_condition が悪いほど高スコア
    """

    width_score = _bucket(width_m, [3.5, 3.0, 2.5, 2.0])
    slope_score = _bucket(slope_deg, [5, 8, 12, 16])
    curvature_score = _bucket(curvature, [0.1, 0.2, 0.3, 0.4])

    if visibility is None:
        visibility_score = 4
    else:
        visibility_score = _bucket(1.0 - visibility, [0.2, 0.4, 0.6, 0.8])

    if ground_condition is None:
        ground_condition = 3
    ground_score = float(ground_condition)

    raw_total = (
        width_score * 1.0
        + slope_score * 1.2
        + curvature_score * 1.2
        + visibility_score * 1.0
        + ground_score * 0.8
    )

    danger = 1.0 + raw_total / 5.0
    danger = max(1.0, min(5.0, danger))
    return round(danger, 2)
