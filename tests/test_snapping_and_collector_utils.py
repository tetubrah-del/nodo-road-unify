from main import (
    CollectorMeta,
    CollectorPoint,
    CollectorRequest,
    CollectorSensorSummary,
    _build_collector_metadata,
    _smooth_collector_points,
)


class DummyPoint:
    def __init__(self, lat: float, lon: float, timestamp_ms: int):
        self.lat = lat
        self.lon = lon
        self.timestamp_ms = timestamp_ms


# --- _smooth_collector_points ---------------------------------------------------------------------

def test_smooth_collector_points_preserves_length_and_timestamps():
    """Smoothing should keep length/timestamps while pulling values toward local averages."""
    points = [
        DummyPoint(35.0, 139.0, 0),
        DummyPoint(35.0005, 139.0005, 1),
        DummyPoint(35.0001, 139.0002, 2),
        DummyPoint(35.0004, 139.0004, 3),
        DummyPoint(35.0002, 139.0001, 4),
    ]

    smoothed = _smooth_collector_points(points)
    assert len(smoothed) == len(points)

    # Timestamps should be copied from the original points
    for original, result in zip(points, smoothed):
        assert result.timestamp_ms == original.timestamp_ms

    # Each smoothed coordinate should lie within the min/max of its neighborhood
    for i, result in enumerate(smoothed):
        # define neighborhood window used by function (half window size = 1 for default window_size=3)
        start = max(0, i - 1)
        end = min(len(points) - 1, i + 1)
        window = points[start : end + 1]
        lat_values = [p.lat for p in window]
        lon_values = [p.lon for p in window]
        assert min(lat_values) <= result.lat <= max(lat_values)
        assert min(lon_values) <= result.lon <= max(lon_values)


# --- _build_collector_metadata --------------------------------------------------------------------

def _make_payload(with_sensor: bool = True):
    collector_points = [
        CollectorPoint(lat=35.0, lon=139.0, timestamp_ms=0),
        CollectorPoint(lat=35.0001, lon=139.0002, timestamp_ms=1),
    ]
    meta = CollectorMeta(width_m=3.5, slope_deg=2.0, curvature=0.1, visibility=0.8, ground_condition=1)
    sensor_summary = (
        CollectorSensorSummary(
            mode="vehicle",
            vertical_rms=0.9,
            vertical_max=1.8,
            pitch_mean_deg=2.5,
            sensor_samples=50,
        )
        if with_sensor
        else None
    )
    return CollectorRequest(points=collector_points, meta=meta, sensor_summary=sensor_summary)


def test_build_collector_metadata_includes_sensor_summary_when_present():
    """Metadata should include sensor_summary when provided."""
    payload = _make_payload(with_sensor=True)
    smoothed = [p.copy(update={"lat": p.lat + 0.0001}) for p in payload.points]

    metadata = _build_collector_metadata(payload, smoothed_points=smoothed)

    assert metadata["collector"]["name"] == "web_pwa"
    assert metadata["num_points"] == len(payload.points)
    assert len(metadata["raw_points"]) == len(payload.points)
    assert "sensor_summary" in metadata
    assert metadata["sensor_summary"]["mode"] == "vehicle"
    assert "smoothed_points" in metadata
    assert len(metadata["smoothed_points"]) == len(payload.points)


def test_build_collector_metadata_omits_sensor_summary_when_absent():
    """Metadata should omit sensor_summary when payload lacks it."""
    payload = _make_payload(with_sensor=False)
    metadata = _build_collector_metadata(payload)

    assert metadata["collector"]["name"] == "web_pwa"
    assert metadata["num_points"] == len(payload.points)
    assert len(metadata["raw_points"]) == len(payload.points)
    assert "sensor_summary" not in metadata
