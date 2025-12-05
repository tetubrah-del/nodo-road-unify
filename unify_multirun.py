import json
import logging
import math
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

from psycopg2.extras import RealDictCursor
from psycopg2.extensions import connection as PGConnection

from geodesic_utils import (
    Geodesic,
    average_speed_mps,
    cumulative_distances,
    geodesic_distance_m,
    initial_bearing_deg,
    polyline_length,
    resample_polyline,
)
from pydantic import BaseModel
from hmm_map_matching import emission_probability, hmm_viterbi, transition_probability


Point = dict
CollectorPoint = Point


@dataclass
class Run:
    link_id: int
    points: List[Point]
    metadata: Optional[dict] = None


class MultirunParams(BaseModel):
    use_hmm: bool = False
    hmm_debug: bool = False

    use_quality_filter: bool = True
    quality_min: float = 0.3
    outlier_sigma: float = 2.0
    reference_method: Literal["best_quality", "medoid"] = "best_quality"


@dataclass
class RunQualityStats:
    run_id: int
    length_m: float
    hmm_score: Optional[float] = None
    sensor_quality: Optional[float] = None
    quality: Optional[float] = None


def _distance_m(p1: Point, p2: Point) -> float:
    return geodesic_distance_m(p1["lat"], p1["lon"], p2["lat"], p2["lon"])


def _nearest_distance_to_centerline(point: Point, centerline: Sequence[Point]) -> float:
    if not centerline:
        return math.inf
    return min(_distance_m(point, c) for c in centerline)


def _parse_linestring_wkt(wkt: str) -> List[Point]:
    try:
        inner = wkt[wkt.index("(") + 1 : wkt.rindex(")")]
    except ValueError:
        return []

    points: List[Point] = []
    for part in inner.split(","):
        tokens = part.strip().split()
        if len(tokens) != 2:
            continue
        lon, lat = map(float, tokens)
        points.append({"lat": lat, "lon": lon})

    return points


def _polyline_length(points: Sequence[Point]) -> float:
    return polyline_length(points)


def _cumulative_distances(points: Sequence[Point]) -> List[float]:
    return cumulative_distances(points)


def _average_speed_mps(points: Sequence[Point]) -> Optional[float]:
    speed = average_speed_mps(points)
    return speed if speed > 0 else None


def _sensor_weight(metadata: Optional[dict]) -> float:
    if not metadata:
        return 1.0

    summary = metadata.get("sensor_summary") if isinstance(metadata, dict) else None
    if not isinstance(summary, dict):
        return 1.0

    mode = summary.get("mode")
    vertical_rms = summary.get("vertical_rms")

    if mode == "vehicle":
        if isinstance(vertical_rms, (int, float)) and vertical_rms >= 1.2:
            return 1.1
        return 1.0

    return 0.7


def _gps_accuracy_weight(run: Run) -> float:
    hdops = [p.get("hdop") for p in run.points if isinstance(p.get("hdop"), (int, float))]
    accuracies = [
        p.get("accuracy")
        for p in run.points
        if isinstance(p.get("accuracy"), (int, float)) and p.get("accuracy") >= 0
    ]

    value: Optional[float] = None
    if hdops:
        value = median(hdops)
    elif accuracies:
        value = median(accuracies)
    elif isinstance(run.metadata, dict):
        meta_hdop = run.metadata.get("hdop")
        if isinstance(meta_hdop, (int, float)):
            value = meta_hdop

    if value is None:
        return 1.0

    return max(0.1, 1.0 / (1.0 + value))


def _run_weight(run: Run) -> float:
    w_base = 1.0
    w_gps = _gps_accuracy_weight(run)
    speed = _average_speed_mps(run.points)
    if speed is None:
        w_speed = 1.0
    elif speed < 1.5:
        w_speed = 0.6
    elif speed < 6:
        w_speed = 1.0
    else:
        w_speed = 1.1

    w_sensor = _sensor_weight(run.metadata)

    return w_base * w_gps * w_speed * w_sensor


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def compute_run_quality(
    stats: RunQualityStats, length_median: Optional[float], params: MultirunParams
) -> float:
    """Compute a normalized [0, 1] quality score for a single run."""

    hmm_score_norm = (
        _clamp(float(stats.hmm_score)) if stats.hmm_score is not None else 0.5
    )

    if length_median and length_median > 0:
        ratio = stats.length_m / length_median
        diff = abs(ratio - 1.0)
        length_consistency = max(0.0, 1.0 - diff / 0.5)
    else:
        length_consistency = 0.5

    sensor_quality_norm = (
        _clamp(float(stats.sensor_quality)) if stats.sensor_quality is not None else 0.5
    )

    w_hmm = 0.6
    w_len = 0.3
    w_sensor = 0.1

    quality = w_hmm * hmm_score_norm + w_len * length_consistency + w_sensor * sensor_quality_norm
    stats.quality = _clamp(quality)
    return stats.quality


def build_run_quality_stats(
    runs: List[Run],
    hmm_scores: Optional[dict[int, float]] = None,
    sensor_summaries: Optional[dict[int, dict]] = None,
) -> List[RunQualityStats]:
    stats_list: List[RunQualityStats] = []
    hmm_scores = hmm_scores or {}
    sensor_summaries = sensor_summaries or {}

    for run in runs:
        sensor_summary = None
        if isinstance(run.metadata, dict):
            sensor_summary = run.metadata.get("sensor_summary")
        if sensor_summaries and run.link_id in sensor_summaries:
            sensor_summary = sensor_summaries.get(run.link_id)

        sensor_quality = None
        if isinstance(sensor_summary, dict):
            candidate = sensor_summary.get("quality")
            if isinstance(candidate, (int, float)):
                sensor_quality = float(candidate)

        stats_list.append(
            RunQualityStats(
                run_id=run.link_id,
                length_m=_polyline_length(run.points),
                hmm_score=hmm_scores.get(run.link_id),
                sensor_quality=sensor_quality,
            )
        )

    return stats_list


def compute_and_filter_runs_by_quality(
    stats_list: List[RunQualityStats],
    params: MultirunParams,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[RunQualityStats], List[RunQualityStats]]:
    if not stats_list:
        return [], []

    length_median = median([s.length_m for s in stats_list if s.length_m is not None])

    for stats in stats_list:
        if stats.quality is None:
            compute_run_quality(stats, length_median, params)

    if not params.use_quality_filter:
        return stats_list, []

    qualities = [s.quality for s in stats_list if s.quality is not None]
    mean_q = mean(qualities) if qualities else 0.0
    std_q = pstdev(qualities) if len(qualities) > 1 else 0.0

    kept: List[RunQualityStats] = []
    removed: List[RunQualityStats] = []

    for stats in stats_list:
        quality = stats.quality if stats.quality is not None else 0.0
        is_low = quality < params.quality_min
        is_outlier = quality < mean_q - params.outlier_sigma * std_q
        if is_low or is_outlier:
            removed.append(stats)
        else:
            kept.append(stats)

    if not kept:
        # Keep at least the best run to avoid empty results
        best = max(stats_list, key=lambda s: s.quality or 0.0)
        kept.append(best)
        removed = [s for s in stats_list if s is not best]

    if logger:
        logger.debug(
            "Quality filtering: mean=%.3f std=%.3f kept=%d removed=%d",
            mean_q,
            std_q,
            len(kept),
            len(removed),
        )

    return kept, removed


def select_reference_run(
    stats_list: List[RunQualityStats], params: MultirunParams
) -> Optional[RunQualityStats]:
    if not stats_list:
        return None

    if params.reference_method == "best_quality":
        return max(stats_list, key=lambda s: s.quality if s.quality is not None else -math.inf)

    # Fallback to best quality for unimplemented methods
    return max(stats_list, key=lambda s: s.quality if s.quality is not None else -math.inf)


def normalize_direction(reference_points: Sequence[Point], candidate_points: Sequence[Point]) -> List[Point]:
    if not reference_points or not candidate_points:
        return list(candidate_points)

    limit = min(len(reference_points), len(candidate_points))
    forward_dists = [
        geodesic_distance_m(
            reference_points[i]["lat"],
            reference_points[i]["lon"],
            candidate_points[i]["lat"],
            candidate_points[i]["lon"],
        )
        for i in range(limit)
    ]
    reversed_candidate = list(reversed(candidate_points))
    reverse_dists = [
        geodesic_distance_m(
            reference_points[i]["lat"],
            reference_points[i]["lon"],
            reversed_candidate[i]["lat"],
            reversed_candidate[i]["lon"],
        )
        for i in range(limit)
    ]

    forward_mean = sum(forward_dists) / limit if limit else math.inf
    reverse_mean = sum(reverse_dists) / limit if limit else math.inf

    if reverse_mean < forward_mean:
        return reversed_candidate

    return list(candidate_points)


def _clip_to_length(points: Sequence[Point], target_length_m: float) -> List[Point]:
    if not points:
        return []

    cumdist = _cumulative_distances(points)
    total_length = cumdist[-1] if cumdist else 0.0

    if total_length <= target_length_m:
        return list(points)

    clipped: List[Point] = [points[0]]
    for idx in range(1, len(points)):
        prev_d = cumdist[idx - 1]
        curr_d = cumdist[idx]
        prev_p = points[idx - 1]
        curr_p = points[idx]

        if curr_d < target_length_m:
            clipped.append(curr_p)
            continue

        segment_len = curr_d - prev_d
        if segment_len <= 0:
            clipped.append(curr_p)
            continue

        ratio = (target_length_m - prev_d) / segment_len
        ratio = max(0.0, min(1.0, ratio))
        lat = prev_p["lat"] + (curr_p["lat"] - prev_p["lat"]) * ratio
        lon = prev_p["lon"] + (curr_p["lon"] - prev_p["lon"]) * ratio
        clipped.append({"lat": lat, "lon": lon})
        return clipped

    return list(points)


def load_runs_from_db(conn: PGConnection, link_ids: Iterable[int]) -> List[Run]:
    ids = list(link_ids)
    if not ids:
        return []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT link_id, ST_AsText(geom) AS wkt, metadata
            FROM road_links
            WHERE link_id = ANY(%s)
            ORDER BY link_id
            """,
            (ids,),
        )
        rows = cur.fetchall()

    runs: List[Run] = []
    for row in rows:
        wkt = row.get("wkt")
        link_id = row.get("link_id")
        if wkt is None or link_id is None:
            continue
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        points = _parse_linestring_wkt(wkt)
        runs.append(Run(link_id=link_id, points=points, metadata=metadata))

    return runs


def _drop_outlier_runs_by_endpoints(runs: List[Run], threshold_m: float = 25.0) -> List[Run]:
    if len(runs) <= 2:
        return runs

    starts = [r.points[0] for r in runs if r.points]
    ends = [r.points[-1] for r in runs if r.points]

    if not starts or not ends:
        return runs

    start_med = {"lat": median(p["lat"] for p in starts), "lon": median(p["lon"] for p in starts)}
    end_med = {"lat": median(p["lat"] for p in ends), "lon": median(p["lon"] for p in ends)}

    filtered: List[Run] = []
    for run in runs:
        if not run.points:
            continue
        start_dist = _distance_m(run.points[0], start_med)
        end_dist = _distance_m(run.points[-1], end_med)
        if start_dist <= threshold_m and end_dist <= threshold_m:
            filtered.append(run)

    return filtered if filtered else runs


def resample_polyline(points: Sequence[Point], k: int) -> List[Point]:
    if k <= 0 or not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(k)]

    cumdist = _cumulative_distances(points)
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


def _nearest_centerline_index(point: Point, centerline: Sequence[Point]) -> int:
    best_idx = 0
    best_dist = math.inf
    for idx, c in enumerate(centerline):
        dist = _distance_m(point, c)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _heading_at_index(centerline: Sequence[Point], idx: int) -> float:
    if not centerline:
        return 0.0
    if idx <= 0:
        target = min(1, len(centerline) - 1)
        return initial_bearing_deg(
            centerline[0]["lat"],
            centerline[0]["lon"],
            centerline[target]["lat"],
            centerline[target]["lon"],
        )
    if idx >= len(centerline) - 1:
        prev = max(len(centerline) - 2, 0)
        return initial_bearing_deg(
            centerline[prev]["lat"],
            centerline[prev]["lon"],
            centerline[-1]["lat"],
            centerline[-1]["lon"],
        )

    return initial_bearing_deg(
        centerline[idx - 1]["lat"],
        centerline[idx - 1]["lon"],
        centerline[idx + 1]["lat"],
        centerline[idx + 1]["lon"],
    )


def _signed_offset(point: Point, centerline: Sequence[Point], idx: int) -> float:
    anchor = centerline[idx]
    heading = _heading_at_index(centerline, idx)
    to_point_bearing = initial_bearing_deg(
        anchor["lat"], anchor["lon"], point["lat"], point["lon"]
    )
    angle_diff = math.radians(to_point_bearing - heading)
    distance = _distance_m(anchor, point)
    return distance * math.sin(angle_diff)


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _smooth_profile(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return list(values)
    smoothed: List[Optional[float]] = []
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        window_vals = [v for v in values[start:end] if v is not None]
        if not window_vals:
            smoothed.append(None)
            continue
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed


def estimate_width_profile(
    centerline: Sequence[Point],
    runs: Sequence[Run],
    search_radius_m: float = 12.0,
    percentile: float = 95.0,
    smooth_window: int = 5,
) -> Optional[dict]:
    if not centerline or not runs:
        return None

    buckets: List[List[float]] = [[] for _ in centerline]
    for run in runs:
        for pt in run.points:
            idx = _nearest_centerline_index(pt, centerline)
            anchor = centerline[idx]
            if _distance_m(pt, anchor) > search_radius_m:
                continue
            buckets[idx].append(_signed_offset(pt, centerline, idx))

    left = [_percentile([v for v in bucket if v < 0], percentile) for bucket in buckets]
    right = [_percentile([v for v in bucket if v > 0], percentile) for bucket in buckets]

    left_smoothed = _smooth_profile(left, smooth_window)
    right_smoothed = _smooth_profile(right, smooth_window)

    width: List[Optional[float]] = []
    for l, r in zip(left_smoothed, right_smoothed):
        if l is None and r is None:
            width.append(None)
        elif l is None:
            width.append(r)
        elif r is None:
            width.append(-l)
        else:
            width.append(r - l)

    points = []
    for idx, (l, r, w) in enumerate(zip(left_smoothed, right_smoothed, width)):
        if w is None:
            continue
        points.append({"index": idx, "width_m": w, "left": l or 0.0, "right": r or 0.0})

    if not points:
        return None

    return {"points": points, "method": "boundary_percentile95"}


def fuse_resampled(
    runs: Sequence[Sequence[Point]],
    drop_outlier_fraction: float = 0.25,
    weights: Optional[Sequence[float]] = None,
) -> List[Point]:
    if not runs:
        return []

    lengths = {len(r) for r in runs}
    if len(lengths) != 1:
        raise ValueError("All resampled runs must have the same length")

    if weights is None:
        weights = [1.0 for _ in runs]
    if len(weights) != len(runs):
        raise ValueError("Weights length must match runs length")

    fused: List[Point] = []
    point_count = lengths.pop()

    for idx in range(point_count):
        points_at_idx = []
        for run_idx, run in enumerate(runs):
            if idx < len(run):
                points_at_idx.append((run[idx], weights[run_idx]))

        if not points_at_idx:
            continue

        median_point = {
            "lat": median(p[0]["lat"] for p in points_at_idx),
            "lon": median(p[0]["lon"] for p in points_at_idx),
        }

        distances = [(_distance_m(p, median_point), p, w) for p, w in points_at_idx]
        distances.sort(key=lambda x: x[0])

        keep_count = max(1, int(len(distances) * (1 - drop_outlier_fraction)))
        kept = distances[:keep_count]

        weight_sum = sum(w for _, _, w in kept)
        if weight_sum == 0:
            weight_sum = len(kept)
            fused.append(
                {
                    "lat": sum(p["lat"] for _, p, _ in kept) / len(kept),
                    "lon": sum(p["lon"] for _, p, _ in kept) / len(kept),
                }
            )
            continue

        avg_lat = sum(w * p["lat"] for _, p, w in kept) / weight_sum
        avg_lon = sum(w * p["lon"] for _, p, w in kept) / weight_sum
        fused.append({"lat": avg_lat, "lon": avg_lon})

    return fused


def smooth_polyline(points: Sequence[Point], iterations: int = 2) -> List[Point]:
    if len(points) < 3:
        return list(points)

    smoothed = list(points)
    for _ in range(iterations):
        if len(smoothed) < 3:
            break
        new_points = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            avg_lat = (
                smoothed[i - 1]["lat"] + smoothed[i]["lat"] + smoothed[i + 1]["lat"]
            ) / 3
            avg_lon = (
                smoothed[i - 1]["lon"] + smoothed[i]["lon"] + smoothed[i + 1]["lon"]
            ) / 3
            new_points.append({"lat": avg_lat, "lon": avg_lon})
        new_points.append(smoothed[-1])
        smoothed = new_points
    return smoothed


def write_unified_to_db(
    conn: PGConnection,
    points: Sequence[Point],
    link_ids: Sequence[int],
    resample_points: int,
    weights: Optional[dict] = None,
    width_profile: Optional[dict] = None,
    hmm: Optional[dict] = None,
    removed_runs: Optional[List[dict]] = None,
) -> int:
    coords = ", ".join(f"{p['lon']} {p['lat']}" for p in points)
    wkt = f"LINESTRING({coords})"
    metadata = {
        "method": "multirun_centerline_v2.0",
        "unified_from": list(link_ids),
        "runs": list(link_ids),
        "num_runs": len(link_ids),
        "resample_points": resample_points,
        "geodesic": True,
        "direction_normalized": True,
        "weights": weights or {},
    }
    if width_profile:
        metadata["width_profile"] = width_profile
    if hmm:
        metadata["hmm"] = hmm
    if removed_runs:
        metadata["removed_runs"] = removed_runs

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO road_links_unified (
                geom,
                source,
                metadata
            )
            VALUES (
                ST_GeomFromText(%s, 4326),
                'unified',
                %s::jsonb
            )
            RETURNING link_id
            """,
            (wkt, json.dumps(metadata, ensure_ascii=False)),
        )
        new_id = cur.fetchone()[0]

    return new_id


def _mean_pairwise_endpoint_distance(runs: Sequence[Run]) -> float:
    if len(runs) < 2:
        return 0.0

    distances = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if not runs[i].points or not runs[j].points:
                continue
            start_d = _distance_m(runs[i].points[0], runs[j].points[0])
            end_d = _distance_m(runs[i].points[-1], runs[j].points[-1])
            distances.append((start_d + end_d) / 2)
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _link_distance_score(points: Sequence[Point], centerline: Sequence[Point]) -> float:
    if not points or not centerline:
        return math.inf
    distances = [_nearest_distance_to_centerline(pt, centerline) for pt in points]
    if not distances:
        return math.inf
    return sum(distances) / len(distances)


def map_match_runs_with_hmm(
    runs: List[List[CollectorPoint]],
    candidate_links: List[Tuple[int, List[CollectorPoint]]],
    *,
    max_link_distance_m: float = 50.0,
) -> dict:
    """
    Use a simple HMM to map-match multiple runs against candidate centerlines.

    Returns a summary containing the matched link id and lightweight quality
    metrics. If no reasonable candidate exists, matched_link_id will be None.
    """

    summary = {
        "matched_link_id": None,
        "log_likelihood": None,
        "avg_emission_prob": None,
        "matched_ratio": None,
    }

    if not runs or not candidate_links:
        return summary

    best: Tuple[float, Optional[int], dict] = (-math.inf, None, summary)

    for link_id, centerline in candidate_links:
        if not centerline:
            continue

        total_points = 0
        matched_points = 0
        emission_sum = 0.0
        log_likelihood = 0.0

        avg_dist = _link_distance_score([pt for run in runs for pt in run], centerline)
        if avg_dist > max_link_distance_m:
            continue

        for run in runs:
            if not run:
                continue
            total_points += len(run)
            path = hmm_viterbi(run, centerline, sigma=8.0, delta_d=1.0)
            if not path:
                continue
            matched_points += len(path)
            for obs, state_idx in zip(run, path):
                prob = emission_probability(obs, centerline[state_idx], sigma=8.0)
                emission_sum += prob
                log_likelihood += math.log(prob + 1e-12)
            for prev, curr in zip(path, path[1:]):
                log_likelihood += math.log(transition_probability(prev, curr, delta_d=1.0) + 1e-12)

        if total_points == 0 or matched_points == 0:
            continue

        avg_emission = emission_sum / matched_points
        matched_ratio = matched_points / total_points if total_points else None

        if log_likelihood > best[0]:
            best = (
                log_likelihood,
                link_id,
                {
                    "matched_link_id": link_id,
                    "log_likelihood": float(log_likelihood),
                    "avg_emission_prob": float(avg_emission),
                    "matched_ratio": float(matched_ratio) if matched_ratio is not None else None,
                },
            )

    return best[2] if best[1] is not None else summary


def unify_runs(
    link_ids: Sequence[int],
    conn: Optional[PGConnection] = None,
    resample_points: int = 100,
    estimate_width: bool = True,
    use_hmm: bool = False,
    hmm_debug: bool = False,
    params: Optional[MultirunParams] = None,
) -> dict:
    if len(link_ids) < 2:
        raise ValueError("At least two link_ids are required for unification")
    if resample_points < 10:
        raise ValueError("resample_points must be at least 10")
    if conn is None:
        raise ValueError("A database connection is required")

    logger = logging.getLogger(__name__)
    params = params or MultirunParams(use_hmm=use_hmm, hmm_debug=hmm_debug)
    use_hmm = params.use_hmm
    hmm_debug = params.hmm_debug

    runs = load_runs_from_db(conn, link_ids)
    if len(runs) < 2:
        raise ValueError("Not enough runs found for the provided link_ids")

    runs = _drop_outlier_runs_by_endpoints(runs)
    stats_list = build_run_quality_stats(runs)
    kept_stats, removed_stats = compute_and_filter_runs_by_quality(
        stats_list, params, logger=logger
    )

    kept_ids = {s.run_id for s in kept_stats}
    runs = [r for r in runs if r.link_id in kept_ids]

    lengths = [_polyline_length(r.points) for r in runs]
    if not lengths:
        raise ValueError("Runs are empty")

    min_len = min(lengths)
    max_len = max(lengths)
    if max_len == 0 or (max_len - min_len) / max_len > 0.4:
        raise ValueError("Length difference between runs is too large")

    reference_stats = None
    if params.use_quality_filter:
        reference_stats = select_reference_run(kept_stats, params)
    else:
        reference_stats = next((s for s in stats_list if s.run_id == runs[0].link_id), None)
    reference_run = next(
        (r for r in runs if reference_stats and r.link_id == reference_stats.run_id),
        runs[0],
    )

    reference_points = reference_run.points
    normalized_runs: List[Run] = [reference_run]
    for run in runs:
        if run.link_id == reference_run.link_id:
            continue
        aligned_points = normalize_direction(reference_points, run.points)
        normalized_runs.append(
            Run(link_id=run.link_id, points=aligned_points, metadata=run.metadata)
        )

    if _mean_pairwise_endpoint_distance(normalized_runs) > 30.0:
        raise ValueError("Runs appear to be too far apart to unify")

    target_length = min_len
    clipped = [
        Run(
            link_id=r.link_id,
            points=_clip_to_length(r.points, target_length),
            metadata=r.metadata,
        )
        for r in normalized_runs
    ]

    weights = [_run_weight(r) for r in clipped]
    weight_map = {run.link_id: weight for run, weight in zip(clipped, weights)}

    resampled = [resample_polyline(r.points, resample_points) for r in clipped]
    fused = fuse_resampled(resampled, weights=weights)
    smoothed = smooth_polyline(fused)

    width_profile = estimate_width_profile(smoothed, clipped) if estimate_width else None

    hmm_summary = {"enabled": bool(use_hmm)}
    if use_hmm:
        candidate_links = [(r.link_id, r.points) for r in clipped]
        hmm_result = map_match_runs_with_hmm(
            [r.points for r in clipped], candidate_links
        )
        hmm_summary.update(hmm_result)

    if len(smoothed) < 10:
        raise ValueError("Unified centerline is too short")

    used_ids = [r.link_id for r in clipped]
    new_link_id = write_unified_to_db(
        conn,
        smoothed,
        used_ids,
        resample_points,
        weights=weight_map,
        width_profile=width_profile,
        hmm=hmm_summary if (use_hmm or hmm_debug) else None,
        removed_runs=[
            {"run_id": stats.run_id, "quality": stats.quality}
            for stats in removed_stats
        ],
    )
    return {"unified_link_id": new_link_id, "hmm": hmm_summary}
