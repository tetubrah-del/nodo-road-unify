"""Geodesic utilities for accurate distance and heading calculations.

Provides helpers backed by GeographicLib with a fallback to a spherical
approximation when the dependency is unavailable.
"""
import math
from typing import List, Sequence, Tuple

try:
    from geographiclib.geodesic import Geodesic
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    class _FallbackGeodesic:
        @staticmethod
        def Inverse(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            d_lat = lat2_rad - lat1_rad
            d_lon = math.radians(lon2 - lon1)
            a = (
                math.sin(d_lat / 2) ** 2
                + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6378137.0 * c
            # initial bearing
            y = math.sin(d_lon) * math.cos(lat2_rad)
            x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
                lat2_rad
            ) * math.cos(d_lon)
            initial_bearing = math.degrees(math.atan2(y, x))
            return {"s12": distance, "azi1": initial_bearing}

        @staticmethod
        def Direct(lat: float, lon: float, azimuth: float, distance: float) -> dict:
            # Simple equirectangular projection based approximation.
            r_earth = 6378137.0
            bearing_rad = math.radians(azimuth)
            lat_rad = math.radians(lat)
            delta = distance / r_earth
            new_lat = math.asin(
                math.sin(lat_rad) * math.cos(delta)
                + math.cos(lat_rad) * math.sin(delta) * math.cos(bearing_rad)
            )
            new_lon = lon + math.degrees(
                math.atan2(
                    math.sin(bearing_rad) * math.sin(delta) * math.cos(lat_rad),
                    math.cos(delta) - math.sin(lat_rad) * math.sin(new_lat),
                )
            )
            return {"lat2": math.degrees(new_lat), "lon2": new_lon}

    class Geodesic:  # type: ignore
        WGS84 = _FallbackGeodesic()


Point = dict


def geodesic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return g["s12"]


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    bearing = g.get("azi1", 0.0)
    return ((bearing + 540) % 360) - 180  # normalize to [-180, 180)


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    g = Geodesic.WGS84.Direct(lat, lon, bearing_deg, distance_m)
    return g.get("lat2", lat), g.get("lon2", lon)


def cumulative_distances(points: Sequence[Point]) -> List[float]:
    if not points:
        return []
    dists = [0.0]
    for i in range(1, len(points)):
        dists.append(dists[-1] + geodesic_distance_m(points[i - 1]["lat"], points[i - 1]["lon"], points[i]["lat"], points[i]["lon"]))
    return dists


def polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += geodesic_distance_m(
            points[i]["lat"], points[i]["lon"], points[i + 1]["lat"], points[i + 1]["lon"]
        )
    return total


def resample_polyline(points: Sequence[Point], k: int) -> List[Point]:
    if k <= 0 or not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(k)]

    cumdist = cumulative_distances(points)
    total_length = cumdist[-1]
    if total_length == 0:
        return [points[0] for _ in range(k)]

    step = total_length / (k - 1) if k > 1 else total_length
    targets = [step * i for i in range(k)]

    resampled: List[Point] = []
    seg_index = 0
    for t in targets:
        while seg_index < len(cumdist) - 1 and cumdist[seg_index + 1] < t:
            seg_index += 1
        next_idx = min(seg_index + 1, len(points) - 1)
        prev_idx = seg_index
        prev_d = cumdist[prev_idx]
        next_d = cumdist[next_idx]
        if next_d - prev_d == 0:
            resampled.append(points[prev_idx])
            continue
        ratio = (t - prev_d) / (next_d - prev_d)
        prev_p = points[prev_idx]
        next_p = points[next_idx]
        lat = prev_p["lat"] + (next_p["lat"] - prev_p["lat"]) * ratio
        lon = prev_p["lon"] + (next_p["lon"] - prev_p["lon"]) * ratio
        resampled.append({"lat": lat, "lon": lon})

    return resampled


def average_speed_mps(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    timestamps = [p.get("timestamp_ms") for p in points]
    if any(ts is None for ts in timestamps):
        return 0.0
    duration_ms = points[-1].get("timestamp_ms") - points[0].get("timestamp_ms")
    if duration_ms is None or duration_ms <= 0:
        return 0.0
    total_distance = polyline_length(points)
    return total_distance / (duration_ms / 1000.0)


__all__ = [
    "Point",
    "Geodesic",
    "geodesic_distance_m",
    "initial_bearing_deg",
    "destination_point",
    "cumulative_distances",
    "polyline_length",
    "resample_polyline",
    "average_speed_mps",
]
