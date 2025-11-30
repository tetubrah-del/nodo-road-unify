import pytest
from fastapi.testclient import TestClient

import main


class DummyCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        # No-op for stubbed queries
        return None

    def fetchall(self):
        return []


class DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, cursor_factory=None):
        return DummyCursor()


@pytest.fixture(autouse=True)
def stub_database(monkeypatch):
    dummy_conn = DummyConnection()
    monkeypatch.setattr(main, "get_connection", lambda: dummy_conn)
    yield


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_roads_feature_collection(client):
    params = {"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1, "limit": 10}
    response = client.get("/api/roads", params=params)
    body = response.json()

    assert response.status_code == 200
    assert body.get("type") == "FeatureCollection"
    assert isinstance(body.get("features"), list)


def test_unified_roads_feature_collection(client):
    params = {"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1, "limit": 10}
    response = client.get("/api/roads/unified", params=params)
    body = response.json()

    assert response.status_code == 200
    assert body.get("type") == "FeatureCollection"
    assert isinstance(body.get("features"), list)
