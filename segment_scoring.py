from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from shapely.geometry import LineString

from geodesic_utils import (
    cumulative_distances,
    initial_bearing_deg,
    polyline_length,
    resample_polyline,
)


@dataclass
class LocalGeomSample:
    frac: float
    x: float
    y: float
    slope_deg: float
    curvature: float


@dataclass
class SegmentResult:
    start_frac: float
    end_frac: float
    intensity_max: float
    intensity_mean: float

def _total_length_m(coords: Sequence[Tuple[float, float]]) -> float:
    return polyline_length([{"lon": x, "lat": y} for x, y in coords])


def sample_link_geometry(geom: LineString, step_m: float) -> List[Tuple[float, float]]:
    """
    Return a list of (lon, lat) points along the line geometry at roughly fixed spacing.
    """

    if geom is None or geom.is_empty:
        return []

    coords: List[Tuple[float, float]] = list(geom.coords)
    if len(coords) == 1:
        return coords

    total_length = _total_length_m(coords)
    if total_length <= 0:
        return [coords[0], coords[-1]]

    step_m = max(step_m, 0.1)
    count = max(2, int(math.ceil(total_length / step_m)) + 1)

    points = resample_polyline(
        [{"lat": y, "lon": x} for x, y in coords],
        count,
    )
    return [(p["lon"], p["lat"]) for p in points]


def compute_local_geom_samples(
    points: Sequence[Tuple[float, float]],
    window: int = 2,
) -> List[LocalGeomSample]:
    if len(points) < 2:
        return []

    point_dicts = [{"lon": x, "lat": y} for x, y in points]
    cumdist = cumulative_distances(point_dicts)
    total_length = cumdist[-1] if cumdist else 0.0
    if total_length <= 0:
        return []

    samples: List[LocalGeomSample] = []
    n = len(points)
    window = max(1, int(window))
    for i, (x, y) in enumerate(points):
        frac = cumdist[i] / total_length if total_length > 0 else 0.0
        prev_idx = max(0, i - window)
        next_idx = min(n - 1, i + window)
        if prev_idx == i or next_idx == i or prev_idx == next_idx:
            curvature = 0.0
        else:
            prev = point_dicts[prev_idx]
            curr = point_dicts[i]
            nxt = point_dicts[next_idx]
            bearing1 = _bearing_deg(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
            bearing2 = _bearing_deg(curr["lat"], curr["lon"], nxt["lat"], nxt["lon"])
            delta = _wrap_angle_deg(bearing2 - bearing1)
            curvature = math.radians(delta)
        samples.append(
            LocalGeomSample(
                frac=frac,
                x=x,
                y=y,
                slope_deg=0.0,
                curvature=curvature,
            )
        )

    return samples


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing in degrees normalized to [-180, 180)."""
    bearing = initial_bearing_deg(lat1, lon1, lat2, lon2)
    return ((bearing + 180) % 360) - 180


def _wrap_angle_deg(angle: float) -> float:
    return ((angle + 180) % 360) - 180


def compute_local_intensities(
    samples: List[LocalGeomSample],
    slope_scale: float = 10.0,
    curvature_scale: float = math.radians(45.0),
) -> List[float]:
    intensities: List[float] = []
    for sample in samples:
        slope_int = min(1.0, abs(sample.slope_deg) / slope_scale) if slope_scale > 0 else 0.0
        curv_int = min(1.0, abs(sample.curvature) / curvature_scale) if curvature_scale > 0 else 0.0
        intensity = 0.3 * slope_int + 0.7 * curv_int
        intensities.append(max(0.0, min(1.0, intensity)))
    return intensities


def smooth_intensities(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    if window <= 1:
        return values
    smoothed: List[float] = []
    radius = max(0, window // 2)
    n = len(values)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        window_vals = values[start:end]
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed


def segment_link_by_intensity(
    fracs: List[float],
    intensities: List[float],
    threshold: float = 0.4,
    min_length_frac: float = 0.02,
) -> List[SegmentResult]:
    if not fracs or not intensities or len(fracs) != len(intensities):
        return []

    segments: List[SegmentResult] = []
    start_idx = None
    current_values: List[float] = []

    for idx, (frac, val) in enumerate(zip(fracs, intensities)):
        if val >= threshold:
            if start_idx is None:
                start_idx = idx
                current_values = [val]
            else:
                current_values.append(val)
        else:
            if start_idx is not None:
                segments.append((start_idx, idx - 1, current_values))
                start_idx = None
                current_values = []

    if start_idx is not None:
        segments.append((start_idx, len(fracs) - 1, current_values))

    results: List[SegmentResult] = []
    for start, end, vals in segments:
        start_frac = max(0.0, min(1.0, fracs[start]))
        end_frac = max(0.0, min(1.0, fracs[end]))
        if end_frac < start_frac:
            start_frac, end_frac = end_frac, start_frac
        if end_frac - start_frac < min_length_frac:
            continue
        intensity_max = max(vals) if vals else 0.0
        intensity_mean = sum(vals) / len(vals) if vals else 0.0
        results.append(
            SegmentResult(
                start_frac=start_frac,
                end_frac=end_frac,
                intensity_max=intensity_max,
                intensity_mean=intensity_mean,
            )
        )

    return results


def _interpolate_point_at_distance(
    coords: Sequence[Tuple[float, float]],
    target_distance: float,
    cumulative: Sequence[float],
) -> Tuple[float, float]:
    if not coords:
        return (0.0, 0.0)

    if target_distance <= 0:
        return coords[0]
    if target_distance >= cumulative[-1]:
        return coords[-1]

    for i in range(1, len(coords)):
        if cumulative[i] >= target_distance:
            prev = coords[i - 1]
            curr = coords[i]
            seg_length = cumulative[i] - cumulative[i - 1]
            if seg_length <= 0:
                return curr
            ratio = (target_distance - cumulative[i - 1]) / seg_length
            x = prev[0] + (curr[0] - prev[0]) * ratio
            y = prev[1] + (curr[1] - prev[1]) * ratio
            return (x, y)
    return coords[-1]


def build_segment_geometries(
    link_geom: LineString,
    segments: List[SegmentResult],
) -> List[Tuple[SegmentResult, LineString]]:
    if link_geom is None or link_geom.is_empty:
        return []

    coords: List[Tuple[float, float]] = list(link_geom.coords)
    if len(coords) < 2:
        return []

    cumulative = cumulative_distances([{"lon": x, "lat": y} for x, y in coords])
    total_length = cumulative[-1] if cumulative else 0.0
    if total_length <= 0:
        return []

    results: List[Tuple[SegmentResult, LineString]] = []
    for seg in segments:
        start_d = total_length * seg.start_frac
        end_d = total_length * seg.end_frac
        if end_d <= start_d:
            continue

        start_point = _interpolate_point_at_distance(coords, start_d, cumulative)
        end_point = _interpolate_point_at_distance(coords, end_d, cumulative)

        segment_coords: List[Tuple[float, float]] = [start_point]
        for i in range(1, len(coords)):
            if cumulative[i] <= start_d or cumulative[i - 1] >= end_d:
                continue
            if cumulative[i] >= end_d:
                break
            segment_coords.append(coords[i])
        segment_coords.append(end_point)

        line = LineString(segment_coords)
        results.append((seg, line))

    return results


__all__ = [
    "LocalGeomSample",
    "SegmentResult",
    "sample_link_geometry",
    "compute_local_geom_samples",
    "compute_local_intensities",
    "smooth_intensities",
    "segment_link_by_intensity",
    "build_segment_geometries",
]
