from types import SimpleNamespace

from danger_score_utils import (
    _compute_length,
    _compute_track_angles,
    compute_danger_score_v2,
)


class DummyPoint:
    """Simple point object mirroring CollectorPoint fields used in helpers."""

    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon


# --- _compute_track_angles / _compute_length ----------------------------------------------------

def test_compute_track_angles_straight_line_near_zero():
    """Straight three-point line should have angles close to 0 degrees."""
    points = [DummyPoint(0, 0), DummyPoint(0, 0.001), DummyPoint(0, 0.002)]

    angles = _compute_track_angles(points)
    assert len(angles) == 1
    assert abs(angles[0]) < 1.0

    # Length should be positive and increase with more points
    length_two = _compute_length(points[:2])
    length_three = _compute_length(points)
    assert length_two > 0
    assert length_three > length_two


def test_compute_track_angles_l_shape_around_90_degrees():
    """Clear 90-degree turn should yield an angle near 90 degrees."""
    points = [DummyPoint(0, 0), DummyPoint(0, 0.001), DummyPoint(0.001, 0.001)]

    angles = _compute_track_angles(points)
    assert len(angles) == 1
    assert 80 <= angles[0] <= 100


def test_compute_track_angles_zig_zag_multiple_curves():
    """Zig-zag path yields multiple angles with a reasonably large maximum."""
    points = [
        DummyPoint(0, 0),
        DummyPoint(0, 0.001),
        DummyPoint(0.001, 0.001),
        DummyPoint(0.001, 0.002),
        DummyPoint(0.002, 0.002),
    ]

    angles = _compute_track_angles(points)
    assert len(angles) == len(points) - 2
    assert max(angles) > 30


# --- compute_danger_score_v2 ---------------------------------------------------------------------

def _build_meta(**kwargs):
    return SimpleNamespace(**kwargs)


def _build_sensor(mode="vehicle", **kwargs):
    return SimpleNamespace(mode=mode, **kwargs)


def test_danger_score_v2_short_straight_no_sensors():
    """Short straight GPS track with no sensors should stay near the base score."""
    points = [DummyPoint(0, 0), DummyPoint(0, 0.0001)]
    score = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=None)
    assert 1.0 <= score <= 2.0


def test_danger_score_v2_long_curvy_no_sensors():
    """Long curvy track accumulates length and curvature scores but remains capped."""
    points = [
        DummyPoint(0, 0),
        DummyPoint(0, 0.003),
        DummyPoint(0.003, 0.003),
        DummyPoint(0.006, 0.006),
        DummyPoint(0.009, 0.009),
    ]

    score = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=None)
    assert score > 2.0
    assert score <= 5.0


def test_danger_score_v2_long_curvy_with_vehicle_sensors_hits_cap():
    """Vehicle sensor readings with rough metrics should push score to the cap (5.0)."""
    points = [
        DummyPoint(0, 0),
        DummyPoint(0, 0.003),
        DummyPoint(0.003, 0.003),
        DummyPoint(0.006, 0.006),
        DummyPoint(0.009, 0.009),
    ]

    sensor = _build_sensor(vertical_rms=1.3, vertical_max=2.6, pitch_mean_deg=7.0, sensor_samples=100)
    score = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=sensor)
    assert score == 5.0


def test_danger_score_v2_long_curvy_with_vehicle_sensors_moderate():
    """Moderate vehicle sensor readings add partial contributions without hitting the cap."""
    points = [
        DummyPoint(0, 0),
        DummyPoint(0, 0.003),
        DummyPoint(0.003, 0.003),
        DummyPoint(0.006, 0.006),
        DummyPoint(0.009, 0.009),
    ]

    sensor = _build_sensor(vertical_rms=0.9, vertical_max=2.0, pitch_mean_deg=4.0, sensor_samples=100)
    score = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=sensor)
    assert 2.0 < score < 5.0


def test_danger_score_v2_sensors_ignored_when_not_vehicle_mode():
    """Non-vehicle mode should nullify sensor contributions."""
    points = [
        DummyPoint(0, 0),
        DummyPoint(0, 0.003),
        DummyPoint(0.003, 0.003),
        DummyPoint(0.006, 0.006),
        DummyPoint(0.009, 0.009),
    ]

    sensor = _build_sensor(mode="off", vertical_rms=2.0, vertical_max=3.0, pitch_mean_deg=10.0, sensor_samples=50)
    score = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=sensor)

    # Score should match the no-sensor scenario for the same geometry and source
    baseline = compute_danger_score_v2(points, source="gps", meta=_build_meta(), sensor_summary=None)
    assert score == baseline
