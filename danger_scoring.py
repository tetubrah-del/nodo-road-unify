from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class DangerScoreParams(BaseModel):
    """Configurable weighting and normalization for danger score v3."""

    # Weighting for each component (can be tuned later)
    w_roughness: float = 0.4
    w_curvature: float = 0.25
    w_slope: float = 0.2
    w_width: float = 0.1
    w_quality_penalty: float = 0.05

    # Normalization scales (so we can map raw metrics into 0–1)
    roughness_scale: float = 0.5  # e.g. vertical_rms ~ 0–0.5 g
    curvature_scale: float = 0.01  # tweak as needed
    slope_scale: float = 15.0  # degrees
    min_width_m: float = 2.0
    max_width_m: float = 5.0


class ReliabilityScoreParams(BaseModel):
    """Weighting and normalization for reliability scoring."""

    # weights for each component, should sum to ~1.0 but doesn’t have to
    w_hmm: float = 0.4
    w_alignment: float = 0.3
    w_run_count: float = 0.2
    w_sensor_mode: float = 0.1

    # Expected ranges / scales for normalization
    max_alignment_cost: float = 1.0  # after normalization from DTW
    max_runs_for_full_confidence: int = 5


@dataclass
class DangerScoreComponents:
    roughness: float
    curvature: float
    slope: float
    width: float
    quality_penalty: float


def _extract_roughness(sensor_summary: Optional[dict]) -> Optional[float]:
    if not isinstance(sensor_summary, dict):
        return None

    candidate = sensor_summary.get("vertical_rms")
    if isinstance(candidate, (int, float)):
        return float(candidate)

    candidate = sensor_summary.get("vertical_max")
    if isinstance(candidate, (int, float)):
        return float(candidate)

    return None


def _extract_quality_penalty(quality_info: Optional[dict]) -> float:
    if not isinstance(quality_info, dict):
        return 0.0

    # Prefer explicit penalty if present
    penalty = quality_info.get("penalty")
    if isinstance(penalty, (int, float)):
        return float(penalty)

    for key in ("avg_quality", "quality", "score"):
        candidate = quality_info.get(key)
        if isinstance(candidate, (int, float)):
            return max(0.0, 1.0 - float(candidate))

    return 0.0


def compute_danger_components(
    *,
    curvature: Optional[float],
    slope_deg: Optional[float],
    width_m: Optional[float],
    sensor_summary: Optional[dict],
    quality_info: Optional[dict],
    params: DangerScoreParams,
) -> Tuple[float, DangerScoreComponents]:
    """Compute the danger score and normalized components.

    This keeps the logic centralized so both the standalone scorer and the
    DB recompute helper can use the same normalization.
    """

    roughness = _extract_roughness(sensor_summary) or 0.0
    curv = abs(curvature or 0.0)
    slope = abs(slope_deg or 0.0)
    width = width_m
    quality_penalty = _extract_quality_penalty(quality_info)

    rough_norm = min(1.0, roughness / params.roughness_scale)
    curv_norm = min(1.0, curv / params.curvature_scale)
    slope_norm = min(1.0, slope / params.slope_scale)

    if width is None:
        width_norm = 0.5
    else:
        clamped = max(params.min_width_m, min(width, params.max_width_m))
        width_norm = 1.0 - (clamped - params.min_width_m) / (
            params.max_width_m - params.min_width_m
        )

    quality_norm = max(0.0, min(1.0, quality_penalty))

    intensity = (
        params.w_roughness * rough_norm
        + params.w_curvature * curv_norm
        + params.w_slope * slope_norm
        + params.w_width * width_norm
        + params.w_quality_penalty * quality_norm
    )

    intensity = max(0.0, min(1.0, intensity))
    danger_score = 1.0 + intensity * 4.0

    components = DangerScoreComponents(
        roughness=rough_norm,
        curvature=curv_norm,
        slope=slope_norm,
        width=width_norm,
        quality_penalty=quality_norm,
    )
    return danger_score, components


def compute_danger_score_v3(
    *,
    curvature: Optional[float],
    slope_deg: Optional[float],
    width_m: Optional[float],
    sensor_summary: Optional[dict],
    quality_info: Optional[dict],
    params: DangerScoreParams,
    ) -> float:
    """Compute a v3 danger score in the range [1.0, 5.0]."""

    score, _components = compute_danger_components(
        curvature=curvature,
        slope_deg=slope_deg,
        width_m=width_m,
        sensor_summary=sensor_summary,
        quality_info=quality_info,
        params=params,
    )
    return score


def compute_reliability_components(
    *,
    hmm_info: Optional[Dict[str, Any]],
    alignment_stats: Optional[Dict[str, Any]],
    run_count: int,
    sensor_mode_stats: Optional[Dict[str, int]],
    params: ReliabilityScoreParams,
) -> Tuple[float, Dict[str, float]]:
    """Compute the reliability score and component contributions."""

    hmm_score = 0.5
    if isinstance(hmm_info, dict):
        match_ratio = hmm_info.get("match_ratio") or hmm_info.get("matched_ratio")
        if match_ratio is not None:
            hmm_score = float(match_ratio)
    hmm_score = max(0.0, min(1.0, hmm_score))

    alignment_score = 0.5
    if isinstance(alignment_stats, dict):
        cost = alignment_stats.get("mean_cost")
        if cost is None:
            cost = alignment_stats.get("max_cost")
        if cost is not None:
            norm = min(1.0, float(cost) / params.max_alignment_cost)
            alignment_score = 1.0 - norm

    capped = min(max(run_count, 0), params.max_runs_for_full_confidence)
    run_count_score = capped / params.max_runs_for_full_confidence

    sensor_score = 0.5
    if isinstance(sensor_mode_stats, dict):
        vehicle = sensor_mode_stats.get("vehicle", 0)
        gps_only = sensor_mode_stats.get("gps_only", 0)
        total = vehicle + gps_only
        if total > 0:
            vehicle_ratio = vehicle / total
            sensor_score = 0.5 + 0.5 * vehicle_ratio

    reliability = (
        params.w_hmm * hmm_score
        + params.w_alignment * alignment_score
        + params.w_run_count * run_count_score
        + params.w_sensor_mode * sensor_score
    )
    reliability = max(0.0, min(1.0, reliability))

    components = {
        "hmm": hmm_score,
        "alignment": alignment_score,
        "run_count": run_count_score,
        "sensor_mode": sensor_score,
    }

    return reliability, components


def compute_reliability_score(
    *,
    hmm_info: Optional[Dict[str, Any]],
    alignment_stats: Optional[Dict[str, Any]],
    run_count: int,
    sensor_mode_stats: Optional[Dict[str, int]],
    params: ReliabilityScoreParams,
) -> float:
    """Compute a reliability score in the range [0.0, 1.0]."""

    reliability, _components = compute_reliability_components(
        hmm_info=hmm_info,
        alignment_stats=alignment_stats,
        run_count=run_count,
        sensor_mode_stats=sensor_mode_stats,
        params=params,
    )
    return reliability
