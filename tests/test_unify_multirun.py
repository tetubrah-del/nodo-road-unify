import pytest

from statistics import median

from geodesic_utils import geodesic_distance_m
from unify_multirun import (
    MultirunParams,
    Run,
    RunQualityStats,
    build_run_quality_stats,
    build_alignment_mappings,
    compute_and_filter_runs_by_quality,
    compute_run_quality,
    dtw_align_run_to_reference,
    fuse_resampled,
    map_match_runs_with_hmm,
    normalize_direction,
    resample_polyline,
    select_reference_run,
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


def test_compute_run_quality_prefers_better_run():
    params = MultirunParams(use_quality_filter=False)
    good = RunQualityStats(run_id=1, length_m=100.0, hmm_score=0.9)
    bad = RunQualityStats(run_id=2, length_m=50.0, hmm_score=0.2)

    median_len = 75.0
    good_q = compute_run_quality(good, median_len, params)
    bad_q = compute_run_quality(bad, median_len, params)

    assert good_q > bad_q


def test_filtering_removes_low_quality_runs():
    params = MultirunParams(quality_min=0.3, outlier_sigma=2.0, use_quality_filter=True)
    stats = [
        RunQualityStats(run_id=1, length_m=100.0, hmm_score=0.9, quality=0.9),
        RunQualityStats(run_id=2, length_m=100.0, hmm_score=0.85, quality=0.8),
        RunQualityStats(run_id=3, length_m=100.0, hmm_score=0.01, quality=0.01),
    ]

    kept, removed = compute_and_filter_runs_by_quality(stats, params)

    assert any(s.run_id == 3 for s in removed)
    assert all(s.run_id in {1, 2} for s in kept)


def test_reference_run_selects_best_quality():
    params = MultirunParams(reference_method="best_quality")
    stats = [
        RunQualityStats(run_id=1, length_m=100.0, hmm_score=0.5, quality=0.5),
        RunQualityStats(run_id=2, length_m=100.0, hmm_score=0.9, quality=0.9),
    ]

    ref = select_reference_run(stats, params)

    assert ref is not None
    assert ref.run_id == 2


def test_dtw_alignment_straight_line_monotonic_and_close():
    ref_points = [
        {"lat": 0.0, "lon": i * 0.00005} for i in range(20)
    ]
    run_points = [
        {"lat": 0.00001, "lon": i * 0.000025} for i in range(40)
    ]

    mapping, cost = dtw_align_run_to_reference(ref_points, run_points)

    assert len(mapping) == len(run_points)
    assert mapping == sorted(mapping)
    assert cost < 10.0
    assert all(
        geodesic_distance_m(run_points[i]["lat"], run_points[i]["lon"], ref_points[mapping[i]]["lat"], ref_points[mapping[i]]["lon"]) < 5.0
        for i in range(len(run_points))
    )


def test_dtw_alignment_handles_shifted_curve():
    reference = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 0.0005, "lon": 0.0002},
        {"lat": 0.001, "lon": -0.0002},
        {"lat": 0.0015, "lon": 0.0002},
        {"lat": 0.002, "lon": 0.0},
    ]
    shifted_run = [
        {"lat": 0.00055, "lon": 0.00022},
        {"lat": 0.00105, "lon": -0.00018},
        {"lat": 0.00155, "lon": 0.00018},
    ]

    mapping, cost = dtw_align_run_to_reference(reference, shifted_run)

    assert mapping == sorted(mapping)
    assert len(mapping) == len(shifted_run)
    assert max(mapping) <= len(reference) - 1
    assert cost < 80.0
    assert all(
        geodesic_distance_m(
            shifted_run[i]["lat"],
            shifted_run[i]["lon"],
            reference[mapping[i]]["lat"],
            reference[mapping[i]]["lon"],
        )
        < 70.0
        for i in range(len(shifted_run))
    )


def test_index_and_dtw_alignment_match_on_identical_runs():
    params_index = MultirunParams(align_method="index")
    params_dtw = MultirunParams(align_method="dtw")
    reference = Run(
        link_id=1,
        points=[{"lat": 0.0, "lon": i * 0.0001} for i in range(10)],
    )
    other = Run(
        link_id=2,
        points=[{"lat": 0.00001, "lon": i * 0.0001} for i in range(10)],
    )

    index_alignments, _ = build_alignment_mappings(reference, [other], params_index)
    dtw_alignments, _ = build_alignment_mappings(reference, [other], params_dtw)

    assert index_alignments[other.link_id] == dtw_alignments[other.link_id]


def test_alignment_cost_used_for_filtering():
    reference = Run(
        link_id=1,
        points=[{"lat": 0.0, "lon": i * 0.0001} for i in range(10)],
    )
    good_run = Run(
        link_id=2,
        points=[{"lat": 0.00002, "lon": i * 0.0001} for i in range(10)],
    )
    bad_run = Run(
        link_id=3,
        points=[{"lat": 0.001, "lon": i * 0.0001} for i in range(10)],
    )

    params = MultirunParams(
        align_method="dtw",
        max_alignment_cost=30.0,
        quality_min=0.0,
        use_quality_filter=True,
    )

    runs = [reference, good_run, bad_run]
    stats = build_run_quality_stats(runs)
    length_med = median([s.length_m for s in stats])
    for s in stats:
        compute_run_quality(s, length_med, params)

    ref_stats = select_reference_run(stats, params)
    ref_run = next(r for r in runs if r.link_id == ref_stats.run_id)
    normalized = [
        Run(
            link_id=r.link_id,
            points=(normalize_direction(ref_run.points, r.points) if r.link_id != ref_run.link_id else r.points),
            metadata=r.metadata,
        )
        for r in runs
    ]
    others = [r for r in normalized if r.link_id != ref_run.link_id]
    alignments, alignment_costs = build_alignment_mappings(ref_run, others, params)
    alignment_costs[ref_run.link_id] = None

    kept, removed = compute_and_filter_runs_by_quality(
        stats,
        params,
        alignment_costs=alignment_costs,
        alignment_mappings=alignments,
        reference_run_id=ref_run.link_id,
    )

    kept_ids = {s.run_id for s in kept}
    removed_ids = {s.run_id for s in removed}

    assert ref_run.link_id in kept_ids
    assert good_run.link_id in kept_ids
    assert bad_run.link_id in removed_ids
    aligned_stats = [s for s in kept if s.run_id != ref_run.link_id]
    assert all(s.alignment_cost is not None for s in aligned_stats)


def test_outlier_filter_combines_multiple_metrics():
    reference = Run(
        link_id=1,
        points=[{"lat": 0.0, "lon": i * 0.0001} for i in range(10)],
    )
    normal_run = Run(
        link_id=2,
        points=[{"lat": 0.00001, "lon": i * 0.0001} for i in range(10)],
    )
    long_run = Run(
        link_id=3,
        points=[{"lat": 0.00002, "lon": i * 0.0001} for i in range(20)],
    )
    misaligned_run = Run(
        link_id=4,
        points=[{"lat": 0.001, "lon": i * 0.0001} for i in range(10)],
    )
    low_quality_run = Run(
        link_id=5,
        points=[{"lat": 0.00003, "lon": i * 0.0001} for i in range(10)],
    )

    params = MultirunParams(
        align_method="dtw",
        use_quality_filter=True,
        quality_min=0.0,
        outlier_sigma=10.0,
        max_alignment_cost=80.0,
        max_length_ratio_from_median=1.6,
        min_quality_score=0.3,
    )

    runs = [reference, normal_run, long_run, misaligned_run, low_quality_run]
    stats = build_run_quality_stats(runs)

    length_med = median([s.length_m for s in stats])
    for s in stats:
        if s.run_id == reference.link_id:
            s.hmm_score = 0.9
        elif s.run_id == low_quality_run.link_id:
            s.hmm_score = 0.0
        else:
            s.hmm_score = 0.8
        compute_run_quality(s, length_med, params)

    ref_stats = select_reference_run(stats, params)
    ref_run = next(r for r in runs if r.link_id == ref_stats.run_id)
    normalized = [
        Run(
            link_id=r.link_id,
            points=(
                normalize_direction(ref_run.points, r.points)
                if r.link_id != ref_run.link_id
                else r.points
            ),
            metadata=r.metadata,
        )
        for r in runs
    ]

    others = [r for r in normalized if r.link_id != ref_run.link_id]
    alignments, alignment_costs = build_alignment_mappings(ref_run, others, params)
    alignment_costs[ref_run.link_id] = None

    kept, removed = compute_and_filter_runs_by_quality(
        stats,
        params,
        alignment_costs=alignment_costs,
        alignment_mappings=alignments,
        reference_run_id=ref_run.link_id,
    )

    kept_ids = {s.run_id for s in kept}
    removed_ids = {s.run_id for s in removed}

    assert ref_run.link_id in kept_ids
    assert normal_run.link_id in kept_ids
    assert long_run.link_id in removed_ids
    assert misaligned_run.link_id in removed_ids
    assert low_quality_run.link_id in removed_ids

    long_stats = next(s for s in removed if s.run_id == long_run.link_id)
    misaligned_stats = next(s for s in removed if s.run_id == misaligned_run.link_id)
    low_quality_stats = next(s for s in removed if s.run_id == low_quality_run.link_id)

    assert "length_ratio" in long_stats.outlier_reasons
    assert "alignment_cost" in misaligned_stats.outlier_reasons
    assert "quality_score" in low_quality_stats.outlier_reasons
