from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
