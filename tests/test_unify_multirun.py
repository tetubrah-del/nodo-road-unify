import pytest

from unify_multirun import (
    fuse_resampled,
    map_match_runs_with_hmm,
    normalize_direction,
    resample_polyline,
    smooth_polyline,
)


def test_resample_polyline_returns_requested_count():
    line = [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.001}]

    resampled = resample_polyline(line, 5)

    assert len(resampled) == 5
    assert pytest.approx(resampled[-1]["lon"], rel=1e-6) == 0.001
    assert pytest.approx(resampled[0]["lon"], rel=1e-6) == 0.0


def test_fuse_resampled_drops_endpoint_outlier():
    run_a = [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 1.0}]
    run_b = [{"lat": 0.0001, "lon": 0.0001}, {"lat": 0.0001, "lon": 1.0001}]
    run_outlier = [{"lat": 1.0, "lon": 1.0}, {"lat": 1.0, "lon": 2.0}]

    fused = fuse_resampled([run_a, run_b, run_outlier], drop_outlier_fraction=0.34)

    assert len(fused) == 2
    assert fused[0]["lat"] < 0.1
    assert fused[0]["lon"] < 0.1


def test_fuse_resampled_uses_weights():
    run_a = [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.0}]
    run_b = [{"lat": 1.0, "lon": 1.0}, {"lat": 1.0, "lon": 1.0}]

    fused = fuse_resampled([run_a, run_b], weights=[2.0, 0.5], drop_outlier_fraction=0.0)

    assert fused[0]["lat"] < 0.4
    assert fused[0]["lon"] < 0.4


def test_smooth_polyline_reduces_peak():
    points = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 2.0, "lon": 0.0},
        {"lat": 0.0, "lon": 0.0},
    ]

    smoothed = smooth_polyline(points, iterations=1)

    assert smoothed[1]["lat"] == pytest.approx(2 / 3, rel=1e-6)
    assert smoothed[0] == points[0]
    assert smoothed[-1] == points[-1]


def test_normalize_direction_reverses_when_closer():
    reference = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 0.0, "lon": 1.0},
    ]
    candidate = [
        {"lat": 0.0, "lon": 1.0},
        {"lat": 0.0, "lon": 0.0},
    ]

    normalized = normalize_direction(reference, candidate)

    assert normalized[0] == candidate[-1]


def test_map_match_runs_with_hmm_prefers_close_candidate():
    run = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 0.0, "lon": 0.0005},
        {"lat": 0.0, "lon": 0.001},
    ]

    summary = map_match_runs_with_hmm(
        [run],
        [
            (99, [{"lat": 1.0, "lon": 1.0}, {"lat": 1.0, "lon": 1.001}]),
            (1, [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.001}]),
        ],
        max_link_distance_m=200.0,
    )

    assert summary["matched_link_id"] == 1
    assert summary["matched_ratio"] is not None
    assert summary["matched_ratio"] > 0.8


def test_map_match_runs_with_hmm_returns_none_when_far():
    run = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 0.0, "lon": 0.0005},
        {"lat": 0.0, "lon": 0.001},
    ]

    summary = map_match_runs_with_hmm(
        [run],
        [(2, [{"lat": 5.0, "lon": 5.0}, {"lat": 5.0, "lon": 5.001}])],
        max_link_distance_m=20.0,
    )

    assert summary["matched_link_id"] is None
