import math

from shapely.geometry import LineString

from segment_scoring import (
    build_segment_geometries,
    compute_local_geom_samples,
    compute_local_intensities,
    sample_link_geometry,
    segment_link_by_intensity,
    smooth_intensities,
)


def test_segments_identify_curvy_section():
    # Construct a line with a sharp 90-degree bend in the middle.
    geom = LineString(
        [
            (0.0, 0.0),
            (0.001, 0.0),
            (0.001, 0.001),
            (0.002, 0.001),
        ]
    )

    sampled = sample_link_geometry(geom, step_m=10.0)
    samples = compute_local_geom_samples(sampled)
    fracs = [s.frac for s in samples]
    intensities = compute_local_intensities(samples)
    smoothed = smooth_intensities(intensities, window=3)

    # Intensities should always be clamped to [0, 1].
    assert all(0.0 <= val <= 1.0 for val in smoothed)

    segments = segment_link_by_intensity(fracs, smoothed, threshold=0.4)
    built_segments = build_segment_geometries(geom, segments)

    assert built_segments, "Expected at least one dangerous segment around the bend"

    # Choose the strongest segment and compute its derived danger score.
    best_segment, best_geom = max(
        built_segments, key=lambda pair: pair[0].intensity_mean
    )
    danger_v5 = max(1.0, min(5.0, 1.0 + best_segment.intensity_mean * 4.0))

    assert danger_v5 > 3.0
    assert 1.0 <= danger_v5 <= 5.0
    assert 0.0 <= best_segment.start_frac < best_segment.end_frac <= 1.0
    assert isinstance(best_geom, LineString)
