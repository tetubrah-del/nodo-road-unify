from __future__ import annotations

import json
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

    # v4 additions
    w_sensor_roughness: float = 0.3
    w_sensor_pitch: float = 0.2
    w_sensor_peak: float = 0.2
    w_reliability: float = 0.3

    # Normalization scales (so we can map raw metrics into 0–1)
    roughness_scale: float = 0.5  # e.g. vertical_rms ~ 0–0.5 g
    curvature_scale: float = 0.01  # tweak as needed
    slope_scale: float = 15.0  # degrees
    min_width_m: float = 2.0
    max_width_m: float = 5.0

    # v4 normalization scales
    sensor_roughness_scale: float = 1.0
    sensor_pitch_scale: float = 5.0
    sensor_peak_scale: float = 1.0
    reliability_scale: float = 1.0


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


def _safe_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


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


def get_sensor_metrics_from_metadata(metadata: dict) -> Dict[str, Optional[float]]:
    """Extract sensor metrics safely from metadata."""

    md = _safe_dict(metadata)
    sensor_summary = _safe_dict(md.get("sensor_summary"))

    return {
        "jerk_rms": _coerce_float(sensor_summary.get("jerk_rms")),
        "jerk_max_abs": _coerce_float(sensor_summary.get("jerk_max_abs")),
        "pitch_smooth_std": _coerce_float(sensor_summary.get("pitch_smooth_std")),
        "pitch_smooth_range": _coerce_float(sensor_summary.get("pitch_smooth_range")),
        "danger_rt_mean": _coerce_float(sensor_summary.get("danger_rt_mean")),
        "danger_rt_max": _coerce_float(sensor_summary.get("danger_rt_max")),
    }


def get_reliability_score_from_metadata(metadata: dict) -> Optional[float]:
    """Return reliability score if present in metadata."""

    md = _safe_dict(metadata)
    reliability = md.get("reliability")
    if isinstance(reliability, (int, float)):
        return float(reliability)
    if isinstance(reliability, dict):
        candidate = reliability.get("score")
        if isinstance(candidate, (int, float)):
            return float(candidate)

    candidate = md.get("reliability_score")
    if isinstance(candidate, (int, float)):
        return float(candidate)

    return None


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


def _extract_reliability_inputs(metadata: dict) -> Tuple[
    Dict[str, Any], Optional[Dict[str, Any]], int, Optional[Dict[str, int]]
]:
    md = _safe_dict(metadata)
    multirun_summary = _safe_dict(md.get("multirun_summary")) or _safe_dict(
        md.get("multirun")
    )

    alignment_stats = None
    if multirun_summary:
        alignment_stats = {
            "mean_cost": multirun_summary.get("alignment_cost_mean")
            or multirun_summary.get("mean_cost"),
            "max_cost": multirun_summary.get("alignment_cost_max")
            or multirun_summary.get("max_cost"),
        }

    run_count = 0
    if multirun_summary:
        run_count = multirun_summary.get("run_count") or 0
    if not run_count:
        runs_field = md.get("runs") if isinstance(md, dict) else []
        if isinstance(runs_field, list):
            run_count = len(runs_field)
        if not run_count:
            run_count = int(md.get("num_runs") or 0)

    sensor_mode_stats = None
    if multirun_summary and isinstance(multirun_summary.get("sensor_modes"), dict):
        sensor_mode_stats = multirun_summary.get("sensor_modes")

    hmm_info = _safe_dict(md.get("hmm")) or _safe_dict(md.get("hmm_summary"))

    return hmm_info, alignment_stats, run_count, sensor_mode_stats


def compute_danger_score_v4(
    geom_features: Dict[str, Any],
    metadata: Dict[str, Any],
    params: DangerScoreParams,
    reliability_params: Optional[ReliabilityScoreParams] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute a v4 danger score combining geometry, sensors, and reliability."""

    geom = geom_features or {}
    md = _safe_dict(metadata)
    sensor_summary = _safe_dict(md.get("sensor_summary"))
    quality_info = _safe_dict(md.get("quality")) or _safe_dict(md.get("quality_info"))

    base_score, base_components = compute_danger_components(
        curvature=geom.get("curvature"),
        slope_deg=geom.get("slope_deg") or geom.get("slope"),
        width_m=geom.get("width_m") or geom.get("width"),
        sensor_summary=sensor_summary,
        quality_info=quality_info,
        params=params,
    )

    base_intensity = (
        params.w_roughness * base_components.roughness
        + params.w_curvature * base_components.curvature
        + params.w_slope * base_components.slope
        + params.w_width * base_components.width
        + params.w_quality_penalty * base_components.quality_penalty
    )

    sensor_metrics = get_sensor_metrics_from_metadata(md)

    sensor_rough = min(
        1.0, abs(sensor_metrics.get("jerk_rms") or 0.0) / params.sensor_roughness_scale
    )
    pitch_metric = sensor_metrics.get("pitch_smooth_std")
    if pitch_metric is None:
        pitch_metric = sensor_metrics.get("pitch_smooth_range")
    sensor_pitch = min(1.0, abs(pitch_metric or 0.0) / params.sensor_pitch_scale)

    sensor_peak = 0.0
    if sensor_metrics.get("jerk_max_abs") is not None:
        sensor_peak = min(
            1.0, abs(sensor_metrics.get("jerk_max_abs") or 0.0) / params.sensor_peak_scale
        )
    elif sensor_metrics.get("danger_rt_max") is not None:
        sensor_peak = min(1.0, (sensor_metrics.get("danger_rt_max") or 0.0) / 5.0)

    reliability_value = get_reliability_score_from_metadata(md)
    reliability_components_result: Optional[Dict[str, float]] = None
    computed_reliability = False
    if reliability_value is None and reliability_params is not None:
        hmm_info, alignment_stats, run_count, sensor_mode_stats = _extract_reliability_inputs(md)
        reliability_value, reliability_components_result = compute_reliability_components(
            hmm_info=hmm_info,
            alignment_stats=alignment_stats,
            run_count=run_count,
            sensor_mode_stats=sensor_mode_stats,
            params=reliability_params,
        )
        computed_reliability = True

    if reliability_value is None:
        reliability_risk = 0.0
    else:
        reliability_risk = max(
            0.0, 1.0 - min(1.0, reliability_value / params.reliability_scale)
        )

    sensor_intensity = (
        params.w_sensor_roughness * sensor_rough
        + params.w_sensor_pitch * sensor_pitch
        + params.w_sensor_peak * sensor_peak
        + params.w_reliability * reliability_risk
    )

    total_weight = (
        params.w_roughness
        + params.w_curvature
        + params.w_slope
        + params.w_width
        + params.w_quality_penalty
        + params.w_sensor_roughness
        + params.w_sensor_pitch
        + params.w_sensor_peak
        + params.w_reliability
    )

    combined = (base_intensity + sensor_intensity) / max(total_weight, 1e-6)
    combined = max(0.0, min(1.0, combined))

    score = 1.0 + combined * 4.0

    breakdown = {
        "version": 4,
        "v3_score": base_score,
        "combined_intensity": combined,
        "geom": {
            "roughness": base_components.roughness,
            "curvature": base_components.curvature,
            "slope": base_components.slope,
            "width": base_components.width,
            "quality_penalty": base_components.quality_penalty,
        },
        "sensor": {
            "roughness": sensor_rough,
            "pitch": sensor_pitch,
            "peak": sensor_peak,
            "raw": sensor_metrics,
        },
        "reliability": {
            "risk": reliability_risk,
            "raw": reliability_value,
            "computed": computed_reliability,
            "components": reliability_components_result,
        },
    }

    return score, breakdown
