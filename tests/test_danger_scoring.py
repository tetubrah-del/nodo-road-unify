import json

import pytest

import main
from danger_scoring import DangerScoreParams, compute_danger_score_v3


class DummyCursor:
    def __init__(self, conn):
        self.conn = conn
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        if "SELECT" in query:
            self._result = [dict(row) for row in self.conn.rows]
        elif "UPDATE" in query and params is not None:
            danger_score, metadata_json, link_id = params
            for row in self.conn.rows:
                if row["link_id"] == link_id:
                    row["danger_score"] = danger_score
                    row["metadata"] = (
                        json.loads(metadata_json)
                        if isinstance(metadata_json, str)
                        else metadata_json
                    )
            self.conn.updated += 1

    def fetchall(self):
        return self._result


class DummyConnection:
    def __init__(self, rows):
        self.rows = rows
        self.updated = 0
        self.committed = False

    def cursor(self, cursor_factory=None):
        return DummyCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_compute_danger_score_v3_safe_case():
    params = DangerScoreParams()

    score = compute_danger_score_v3(
        curvature=0.0001,
        slope_deg=1.0,
        width_m=5.0,
        sensor_summary={"vertical_rms": 0.05},
        quality_info={"avg_quality": 0.95},
        params=params,
    )

    assert 1.0 <= score < 2.5


def test_compute_danger_score_v3_dangerous_case():
    params = DangerScoreParams()

    score = compute_danger_score_v3(
        curvature=0.02,
        slope_deg=20.0,
        width_m=2.0,
        sensor_summary={"vertical_rms": 0.6},
        quality_info={"avg_quality": 0.2},
        params=params,
    )

    assert score > 4.0


def test_compute_danger_score_v3_monotonic_with_roughness():
    params = DangerScoreParams()

    low = compute_danger_score_v3(
        curvature=0.001,
        slope_deg=2.0,
        width_m=3.0,
        sensor_summary={"vertical_rms": 0.05},
        quality_info=None,
        params=params,
    )
    high = compute_danger_score_v3(
        curvature=0.001,
        slope_deg=2.0,
        width_m=3.0,
        sensor_summary={"vertical_rms": 0.3},
        quality_info=None,
        params=params,
    )

    assert high >= low


def test_recompute_unified_danger_scores_updates_rows(monkeypatch):
    params = DangerScoreParams()
    rows = [
        {
            "link_id": 1,
            "width_m": 2.5,
            "slope_deg": 5.0,
            "curvature": 0.01,
            "metadata": {"sensor_summary": {"vertical_rms": 0.2}},
        }
    ]
    dummy_conn = DummyConnection(rows)
    monkeypatch.setattr(main, "get_connection", lambda: dummy_conn)

    updated = main.recompute_unified_danger_scores(params)

    assert updated == 1
    assert rows[0].get("danger_score") is not None
    assert rows[0]["metadata"].get("danger_v3", {}).get("score") == rows[0][
        "danger_score"
    ]
    assert dummy_conn.committed is True
