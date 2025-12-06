from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from main import CollectorPoint, CollectorMeta, CollectorSensorSummary


def _point_distance_m(p1: CollectorPoint, p2: CollectorPoint) -> float:
    """Rudimentary planar distance using lat/lon in meters."""
    lat_factor = 111_000
    mean_lat_rad = math.radians((p1.lat + p2.lat) / 2)
    lon_factor = 111_000 * math.cos(mean_lat_rad)
    d_lat = (p2.lat - p1.lat) * lat_factor
    d_lon = (p2.lon - p1.lon) * lon_factor
    return math.sqrt(d_lat**2 + d_lon**2)


def _vector_m(p_from: CollectorPoint, p_to: CollectorPoint) -> tuple[float, float]:
    """Return planar vector (dx, dy) in meters from p_from to p_to."""
    lat_factor = 111_000
    mean_lat_rad = math.radians((p_from.lat + p_to.lat) / 2)
    lon_factor = 111_000 * math.cos(mean_lat_rad)
    dx = (p_to.lon - p_from.lon) * lon_factor
    dy = (p_to.lat - p_from.lat) * lat_factor
    return dx, dy


def _compute_track_angles(points: List[CollectorPoint]) -> List[float]:
    angles: List[float] = []
    for i in range(1, len(points) - 1):
        # Use (prev -> current) and (current -> next) vectors so a straight line yields ~0°
        # instead of ~180° when the directions are reversed.
        v1 = _vector_m(points[i - 1], points[i])
        v2 = _vector_m(points[i], points[i + 1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle_deg = math.degrees(math.acos(cos_theta))
        angles.append(angle_deg)
    return angles


def _compute_length(points: List[CollectorPoint]) -> float:
    return sum(_point_distance_m(points[i], points[i + 1]) for i in range(len(points) - 1))


def _compute_curve_counts(angles: List[float]) -> tuple[int, int, float]:
    curve_count_30 = sum(1 for a in angles if a >= 30)
    curve_count_80 = sum(1 for a in angles if a >= 80)
    max_angle = max(angles) if angles else 0.0
    return curve_count_30, curve_count_80, max_angle


def compute_danger_score_v2(
    points: List[CollectorPoint],
    source: str,
    meta: CollectorMeta,
    sensor_summary: Optional[CollectorSensorSummary],
) -> float:
    if len(points) < 2:
        return 1.0

    length_m = _compute_length(points)
    angles = _compute_track_angles(points)
    curve_count_30, curve_count_80, max_angle = _compute_curve_counts(angles)

    base = 1.0

    length_score = 0.0
    if length_m > 500:
        length_score = 0.5
    elif length_m > 200:
        length_score = 0.2

    curve_score_30 = min(curve_count_30 * 0.1, 1.0)
    curve_score_80 = min(curve_count_80 * 0.2, 1.0)
    max_angle_score = (max_angle / 180.0) * 0.5

    if source == "gps":
        source_weight = 0.3
    elif source == "manual":
        source_weight = 0.2
    elif source == "satellite":
        source_weight = 0.1
    else:
        source_weight = 0.0

    sensor_rms = 0.0
    sensor_max = 0.0
    sensor_pitch = 0.0
    worst_rms = False
    worst_max = False
    worst_pitch = False
    if sensor_summary is not None and sensor_summary.mode == "vehicle":
        raw_vertical_rms = sensor_summary.vertical_rms
        vertical_rms = raw_vertical_rms or 0.0
        worst_rms = raw_vertical_rms is not None and raw_vertical_rms >= 1.2
        if 1.2 <= vertical_rms:
            sensor_rms = 0.6
        elif 0.8 <= vertical_rms < 1.2:
            sensor_rms = 0.3

        vertical_max = sensor_summary.vertical_max
        worst_max = vertical_max is not None and vertical_max >= 2.5
        if vertical_max is not None and vertical_max >= 2.5:
            sensor_max = 0.4

        raw_pitch = sensor_summary.pitch_mean_deg
        abs_pitch = abs(raw_pitch or 0.0)
        worst_pitch = raw_pitch is not None and abs(raw_pitch) >= 6.0
        if 6.0 <= abs_pitch:
            sensor_pitch = 0.6
        elif 3.0 <= abs_pitch < 6.0:
            sensor_pitch = 0.3

    score = (
        base
        + length_score
        + curve_score_30
        + curve_score_80
        + max_angle_score
        + source_weight
        + sensor_rms
        + sensor_max
        + sensor_pitch
    )

    if sensor_summary is not None and sensor_summary.mode == "vehicle":
        if worst_rms and worst_max and worst_pitch:
            # ensure we actually hit the upper cap for very rough vehicle segments
            score = max(score, 5.0)

    score = max(1.0, min(5.0, score))
    return round(score, 2)
