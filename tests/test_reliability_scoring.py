import json

import main
from danger_scoring import (
    DangerScoreParams,
    ReliabilityScoreParams,
    compute_reliability_score,
)


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


def test_compute_reliability_score_high():
    params = ReliabilityScoreParams()
    reliability = compute_reliability_score(
        hmm_info={"match_ratio": 0.9},
        alignment_stats={"mean_cost": 0.1},
        run_count=6,
        sensor_mode_stats={"vehicle": 5, "gps_only": 1},
        params=params,
    )

    assert reliability > 0.8


def test_compute_reliability_score_low():
    params = ReliabilityScoreParams()
    reliability = compute_reliability_score(
        hmm_info={"match_ratio": 0.2},
        alignment_stats={"mean_cost": params.max_alignment_cost},
        run_count=1,
        sensor_mode_stats={"vehicle": 0, "gps_only": 1},
        params=params,
    )

    assert reliability < 0.4


def test_recompute_unified_includes_reliability(monkeypatch):
    danger_params = DangerScoreParams()
    reliability_params = ReliabilityScoreParams()
    rows = [
        {
            "link_id": 10,
            "width_m": 3.0,
            "slope_deg": 2.0,
            "curvature": 0.001,
            "metadata": {
                "sensor_summary": {"vertical_rms": 0.1},
                "multirun_summary": {
                    "run_count": 4,
                    "alignment_cost_mean": 0.2,
                    "alignment_cost_max": 0.3,
                    "sensor_modes": {"vehicle": 3, "gps_only": 1},
                    "hmm_match_ratio_mean": 0.85,
                },
            },
        }
    ]
    dummy_conn = DummyConnection(rows)
    monkeypatch.setattr(main, "get_connection", lambda: dummy_conn)

    updated = main.recompute_unified_danger_scores(
        danger_params, reliability_params
    )

    assert updated == 1
    reliability_meta = rows[0]["metadata"].get("reliability", {})
    assert 0.0 <= reliability_meta.get("score", -1) <= 1.0
    assert reliability_meta.get("components", {}).get("alignment") > 0.0
    assert dummy_conn.committed is True
