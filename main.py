import json
import math
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import connection as PGConnection
from danger_score_utils import compute_danger_score_v2

# Run: uvicorn main:app --reload

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "nodo")
PG_USER = os.getenv("PG_USER", "nodo")
PG_PASS = os.getenv("PG_PASS", "nodo_password")


app = FastAPI(title="Nodo Safety API")


# === DB helpers ===
def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )


def get_connection():
    """PostGIS へのコネクションを返すヘルパー"""
    return get_conn()


# === Models (kept for compatibility) ===
class RoadLink(BaseModel):
    link_id: int
    geom: dict
    width_m: Optional[float] = None
    slope_deg: Optional[float] = None
    curvature: Optional[float] = None
    visibility: Optional[float] = None
    ground_condition: Optional[int] = None
    danger_score: Optional[float] = None


class LatLng(BaseModel):
    lat: float
    lng: float


class RoadInput(BaseModel):
    coords: List[LatLng]  # GPSで取った点列
    width_m: float | None = None
    slope_deg: float | None = None
    curvature: float | None = None
    visibility: float | None = None
    ground_condition: int | None = None


class GpsPoint(BaseModel):
    lat: float
    lon: float
    ts: Optional[str] = None


class GpsTrackInput(BaseModel):
    points: List[GpsPoint]
    meta: Optional[dict] = {}


class CollectorPoint(BaseModel):
    lat: float
    lon: float
    timestamp_ms: Optional[int] = None


class CollectorMeta(BaseModel):
    width_m: Optional[float] = None
    slope_deg: Optional[float] = None
    curvature: Optional[float] = None
    visibility: Optional[float] = None
    ground_condition: Optional[int] = None
    note: Optional[str] = None


class CollectorSensorSummary(BaseModel):
    mode: Literal["vehicle", "off"]
    vertical_rms: Optional[float] = None
    vertical_max: Optional[float] = None
    pitch_mean_deg: Optional[float] = None
    sensor_samples: int


class CollectorRequest(BaseModel):
    points: List[CollectorPoint]
    meta: CollectorMeta = Field(default_factory=CollectorMeta)
    sensor_summary: Optional[CollectorSensorSummary] = None


def _point_distance_m(p1: GpsPoint, p2: GpsPoint) -> float:
    """Rudimentary planar distance using lat/lon in meters."""
    lat_factor = 111_000
    mean_lat_rad = math.radians((p1.lat + p2.lat) / 2)
    lon_factor = 111_000 * math.cos(mean_lat_rad)
    d_lat = (p2.lat - p1.lat) * lat_factor
    d_lon = (p2.lon - p1.lon) * lon_factor
    return math.sqrt(d_lat ** 2 + d_lon ** 2)


def compute_track_danger_score(points: List[GpsPoint]) -> float:
    if len(points) < 2:
        return 1.0

    total_len = 0.0
    for i in range(len(points) - 1):
        total_len += _point_distance_m(points[i], points[i + 1])

    straight_len = _point_distance_m(points[0], points[-1])
    if straight_len < 1.0:
        curviness = 1.0
    else:
        curviness = total_len / straight_len

    base = 1.0
    factor = max(1.0, curviness)
    score = base + (factor - 1.0) * 4.0
    return max(1.0, min(5.0, score))


def _build_linestring_wkt(points) -> str:
    coords = ", ".join(f"{p.lon} {p.lat}" for p in points)
    return f"LINESTRING({coords})"


def _parse_linestring_wkt_to_points(wkt: str) -> List[CollectorPoint]:
    """Minimal parser to turn LINESTRING WKT into CollectorPoint list."""

    try:
        inner = wkt[wkt.index("(") + 1 : wkt.rindex(")")]
    except ValueError:
        return []

    coords: List[CollectorPoint] = []
    for part in inner.split(","):
        tokens = part.strip().split()
        if len(tokens) != 2:
            continue
        lon, lat = map(float, tokens)
        coords.append(CollectorPoint(lat=lat, lon=lon))
    return coords


def _smooth_collector_points(
    points: List[CollectorPoint],
    window_size: int = 3,
) -> List[CollectorPoint]:
    """
    Apply a simple moving-average smoothing to a sequence of CollectorPoint
    objects. We keep the same number of points and preserve timestamps.

    - window_size: odd integer >= 3 (default 3)
    - for each index i, average lat/lon over points in [i-half, i+half]
      clipped to the valid range.
    - timestamps are copied from the original point at i.
    - if len(points) < 3 or window_size < 3, just return the original list.
    """

    if len(points) < 3 or window_size < 3:
        return points

    half = window_size // 2
    smoothed: List[CollectorPoint] = []

    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points) - 1, i + half)
        window = points[start : end + 1]
        avg_lat = sum(p.lat for p in window) / len(window)
        avg_lon = sum(p.lon for p in window) / len(window)

        smoothed.append(
            CollectorPoint(
                lat=avg_lat,
                lon=avg_lon,
                timestamp_ms=points[i].timestamp_ms,
            )
        )

    return smoothed


def _build_collector_metadata(
    payload: CollectorRequest, smoothed_points: Optional[List[CollectorPoint]] = None
) -> dict:
    meta_payload = payload.meta.dict() if payload.meta else {}
    metadata = {
        "collector": {
            "name": "web_pwa",
        },
        "num_points": len(payload.points),
        "meta": meta_payload,
        "raw_points": [p.dict() for p in payload.points],
    }

    if smoothed_points is not None:
        metadata["smoothed_points"] = [p.dict() for p in smoothed_points]

    if payload.sensor_summary is not None:
        metadata["sensor_summary"] = payload.sensor_summary.dict()

    return metadata


def update_unified_gps_stats_for_track(
    conn: PGConnection,
    geom_wkt: str,
    danger_score: float,
    search_radius_m: float = 30.0,
) -> None:
    """
    For the given GPS track geometry (WKT, SRID 4326) and its danger_score,
    find the nearest road_links_unified row within search_radius_m meters
    and update its metadata->gps_stats (count, avg_danger, last_danger).
    """

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH track AS (
                SELECT ST_GeomFromText(%s, 4326) AS geom
            )
            SELECT
                u.link_id,
                u.metadata,
                ST_Distance(u.geom::geography, track.geom::geography) AS dist
            FROM road_links_unified u, track
            WHERE ST_DWithin(u.geom::geography, track.geom::geography, %s)
            ORDER BY dist
            LIMIT 1
            """,
            (geom_wkt, search_radius_m),
        )
        row = cur.fetchone()

    if not row:
        return

    link_id = row["link_id"]
    metadata = row.get("metadata") or {}
    gps_stats = metadata.get("gps_stats") or {}

    old_count = gps_stats.get("count") or 0
    old_avg = gps_stats.get("avg_danger")

    new_count = old_count + 1
    if old_count == 0 or old_avg is None:
        new_avg = danger_score
    else:
        new_avg = (old_avg * old_count + danger_score) / new_count

    new_stats = {
        "count": new_count,
        "avg_danger": new_avg,
        "last_danger": danger_score,
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE road_links_unified
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{gps_stats}',
                %s::jsonb,
                true
            )
            WHERE link_id = %s
            """,
            (json.dumps(new_stats, ensure_ascii=False), link_id),
        )
def snap_linestring_to_unified_roads(
    conn: PGConnection, raw_wkt: str, search_radius_m: float = 30.0
) -> Optional[str]:
    """
    Given a raw collector WKT (LINESTRING, SRID 4326), snap each vertex to the
    nearest road_links_unified centerline when it is within `search_radius_m`.

    - Vertices beyond the search radius stay unchanged.
    - After vertex snapping, the reconstructed line is optionally snapped again
      to a local union of unified roads with a small tolerance (≈2 m).
    - Returns WKT of the snapped geometry, or None if snapping produced an empty
      geometry or failed.
    """

    snap_tolerance_deg = 2.0 / 111_000.0

    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                WITH input_geom AS (
                    SELECT ST_SetSRID(ST_GeomFromText(%s), 4326)::geometry AS geom
                ), points AS (
                    SELECT (dp.path[1]) AS idx, dp.geom AS geom
                    FROM input_geom, ST_DumpPoints(input_geom.geom) dp
                ), nearest AS (
                    SELECT
                        p.idx,
                        p.geom AS input_point,
                        n.geom AS nearest_geom,
                        ST_Distance(n.geom::geography, p.geom::geography) AS dist_m,
                        ST_ClosestPoint(n.geom, p.geom) AS snapped_point
                    FROM points p
                    LEFT JOIN LATERAL (
                        SELECT geom
                        FROM road_links_unified
                        ORDER BY geom <-> p.geom
                        LIMIT 1
                    ) n ON TRUE
                ), adjusted AS (
                    SELECT
                        idx,
                        CASE
                            WHEN dist_m IS NOT NULL AND dist_m <= %s THEN snapped_point
                            ELSE input_point
                        END AS geom
                    FROM nearest
                ), reconstructed AS (
                    SELECT ST_MakeLine(geom ORDER BY idx) AS geom
                    FROM adjusted
                ), union_geom AS (
                    SELECT ST_Union(r.geom) AS geom
                    FROM road_links_unified r
                    JOIN reconstructed rc ON ST_DWithin(
                        r.geom::geography,
                        rc.geom::geography,
                        %s
                    )
                ), snapped AS (
                    SELECT
                        rc.geom AS reconstructed_geom,
                        CASE
                            WHEN ug.geom IS NOT NULL THEN ST_LineMerge(
                                ST_Snap(rc.geom, ug.geom, %s)
                            )
                            ELSE rc.geom
                        END AS snapped_geom
                    FROM reconstructed rc
                    LEFT JOIN union_geom ug ON TRUE
                )
                SELECT ST_AsText(snapped_geom) AS snapped_wkt
                FROM snapped
                WHERE snapped_geom IS NOT NULL AND NOT ST_IsEmpty(snapped_geom);
                """,
                (raw_wkt, search_radius_m, search_radius_m, snap_tolerance_deg),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            return None

    return None


# === API ===
@app.get("/health")
def health():
    return {"status": "ok"}


RAW_FILTER_SQL = """
    geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
    AND (
        source = 'gps'
        OR metadata->>'source' = 'manual_mask_google_1km'
        OR (
            source = 'satellite'
            AND metadata->'gpt_filter'->>'class' = 'FARMLANE'
            AND (metadata->'gpt_filter'->>'confidence')::float >= 0.85
            AND ST_Length(geom::geography) BETWEEN 30 AND 800
        )
    )
"""


def _build_raw_feature(row: dict) -> dict:
    metadata = row.get("metadata")
    collector_meta = metadata.get("collector") if isinstance(metadata, dict) else {}
    snap_info = None
    if isinstance(collector_meta, dict):
        snap_info = {
            "used": collector_meta.get("snapping_used"),
            "distance_m": collector_meta.get("snapping_distance_m"),
            "raw_geom": row.get("raw_geom"),
        }

    return {
        "type": "Feature",
        "geometry": row["geom"],
        "properties": {
            "link_id": row["link_id"],
            "sources": [row["source"]] if row.get("source") else [],
            "parent_link_ids": [],
            "width_m": row.get("width_m"),
            "slope_deg": row.get("slope_deg"),
            "curvature": row.get("curvature"),
            "visibility": row.get("visibility"),
            "ground_condition": row.get("ground_condition"),
            "danger_score": row.get("danger_score"),
            "source": row.get("source"),
            "gpt_class": row.get("gpt_class"),
            "metadata_source": row.get("metadata_source"),
            "metadata": row.get("metadata"),
            "snap": snap_info,
        },
    }


def _build_unified_feature(row: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": row["geom"],
        "properties": {
            "link_id": row["link_id"],
            "sources": row.get("sources") or [],
            "parent_link_ids": row.get("parent_link_ids") or [],
            "width_m": row.get("width_m"),
            "slope_deg": row.get("slope_deg"),
            "curvature": row.get("curvature"),
            "visibility": row.get("visibility"),
            "ground_condition": row.get("ground_condition"),
            "danger_score": row.get("danger_score"),
            "metadata": row.get("metadata"),
        },
    }


@app.get("/api/roads")
def list_roads(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    limit: int = 5000,
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    link_id,
                    source,
                    confidence,
                    (metadata->'gpt_filter'->>'class') AS gpt_class,
                    metadata->>'source' AS metadata_source,
                    metadata,
                    ST_AsGeoJSON(geom)::json AS geom,
                    CASE
                        WHEN metadata->'collector'->>'raw_wkt' IS NOT NULL THEN
                            ST_AsGeoJSON(ST_GeomFromText(metadata->'collector'->>'raw_wkt', 4326))::json
                        ELSE NULL
                    END AS raw_geom,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score
                FROM road_links
                WHERE {RAW_FILTER_SQL}
                ORDER BY link_id
                LIMIT %s
                """,
                (min_lon, min_lat, max_lon, max_lat, limit),
            )
            rows = cur.fetchall()

    return {
        "type": "FeatureCollection",
        "features": [_build_raw_feature(r) for r in rows],
    }


@app.get("/api/roads/unified")
def list_unified_roads(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    limit: int = 5000,
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    link_id,
                    sources,
                    parent_link_ids,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score,
                    metadata,
                    ST_AsGeoJSON(geom)::json AS geom
                FROM road_links_unified
                WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                ORDER BY link_id
                LIMIT %s
                """,
                (min_lon, min_lat, max_lon, max_lat, limit),
            )
            rows = cur.fetchall()

    return {
        "type": "FeatureCollection",
        "features": [_build_unified_feature(r) for r in rows],
    }


@app.post("/api/collector/submit")
def collector_submit(payload: CollectorRequest):
    if not payload.points or len(payload.points) < 2:
        raise HTTPException(status_code=400, detail="At least 2 points are required")

    smoothed_points = _smooth_collector_points(payload.points)

    raw_wkt = _build_linestring_wkt(smoothed_points)
    metadata = _build_collector_metadata(payload, smoothed_points=smoothed_points)

    with get_connection() as conn:
        snapped_wkt = snap_linestring_to_unified_roads(conn, raw_wkt)
        wkt_to_use = snapped_wkt or raw_wkt

        collector_meta = metadata.get("collector")
        if isinstance(collector_meta, dict):
            collector_meta["raw_wkt"] = raw_wkt
            collector_meta["snapped_wkt"] = snapped_wkt
            collector_meta["snapping_used"] = bool(snapped_wkt)
            if snapped_wkt:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT ST_Distance(
                            ST_GeomFromText(%s, 4326)::geography,
                            ST_GeomFromText(%s, 4326)::geography
                        ) AS dist_m
                        """,
                        (raw_wkt, snapped_wkt),
                    )
                    dist_row = cur.fetchone()
                    if dist_row and dist_row[0] is not None:
                        collector_meta["snapping_distance_m"] = float(dist_row[0])

        points_for_score = _parse_linestring_wkt_to_points(wkt_to_use) or smoothed_points

        danger_score = compute_danger_score_v2(
            points=points_for_score,
            source="gps",
            meta=payload.meta,
            sensor_summary=payload.sensor_summary,
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO road_links (
                    geom,
                    source,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score,
                    metadata
                )
                VALUES (
                    ST_GeomFromText(%s, 4326),
                    %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb
                )
                RETURNING link_id, danger_score;
                """,
                (
                    wkt_to_use,
                    "gps",
                    payload.meta.width_m,
                    payload.meta.slope_deg,
                    payload.meta.curvature,
                    payload.meta.visibility,
                    payload.meta.ground_condition,
                    danger_score,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            row = cur.fetchone()

        update_unified_gps_stats_for_track(
            conn=conn,
            geom_wkt=wkt_to_use,
            danger_score=danger_score,
        )

    print(
        f"[collector] points={len(payload.points)} danger_score={danger_score} "
        f"link_id={row['link_id']}"
    )

    return {
        "status": "ok",
        "link_id": row["link_id"],
        "danger_score": row["danger_score"],
        "snapping_used": bool(metadata.get("collector", {}).get("snapping_used")),
    }


@app.post("/api/gps_tracks")
def create_gps_track(track: GpsTrackInput):
    if len(track.points) < 2:
        raise HTTPException(status_code=400, detail="points must contain at least 2 items")

    try:
        wkt = _build_linestring_wkt(track.points)
        num_points = len(track.points)
        start_ts = track.points[0].ts if track.points else None
        end_ts = track.points[-1].ts if track.points else None
        collector_name = (track.meta or {}).get("client") or "pwa_v1"

        metadata = {
            "collector": collector_name,
            "points": num_points,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "meta": track.meta or {},
        }

        danger_score = compute_track_danger_score(track.points)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO road_links (
                        geom,
                        width_m,
                        slope_deg,
                        curvature,
                        visibility,
                        ground_condition,
                        danger_score,
                        metadata,
                        source,
                        confidence,
                        parent_link_id
                    )
                    VALUES (
                        ST_GeomFromText(%s, 4326),
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        %s,
                        %s::jsonb,
                        %s,
                        NULL,
                        NULL
                    )
                    RETURNING link_id
                    """,
                    (
                        wkt,
                        danger_score,
                        json.dumps(metadata, ensure_ascii=False),
                        "gps_mobile",
                    ),
                )
                link_id = cur.fetchone()[0]

        return {
            "status": "ok",
            "inserted_link_id": link_id,
            "num_points": num_points,
            "danger_score": danger_score,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/roads/geojson")
def list_roads_geojson():
    """互換用: 全件取得。"""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    link_id,
                    source,
                    confidence,
                    metadata->'gpt_filter'->>'class' AS gpt_class,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score,
                    metadata->>'source' AS metadata_source,
                    ST_AsGeoJSON(geom)::json AS geom,
                    CASE
                        WHEN metadata->'collector'->>'raw_wkt' IS NOT NULL THEN
                            ST_AsGeoJSON(ST_GeomFromText(metadata->'collector'->>'raw_wkt', 4326))::json
                        ELSE NULL
                    END AS raw_geom
                FROM road_links
                ORDER BY link_id
                """
            )
            rows = cur.fetchall()

    return {
        "type": "FeatureCollection",
        "features": [_build_raw_feature(r) for r in rows],
    }


@app.get("/map", response_class=HTMLResponse)
def map_page():
    html = """
<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <title>Nodo Safety Map</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link
    rel=\"stylesheet\"
    href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"
  />
  <style>
    html, body { margin: 0; padding: 0; height: 100%; }
    #map { width: 100%; height: 100vh; }
    .legend { background: white; padding: 8px 12px; border-radius: 4px; line-height: 1.4; box-shadow: 0 0 4px rgba(0,0,0,0.2); }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-line { width: 28px; height: 6px; border-radius: 3px; }
  </style>
</head>
<body>
  <div id=\"map\"></div>

  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const map = L.map('map').setView([33.1478, 131.5209], 16);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    let rawLayer = null;
    let unifiedLayer = null;
    let gpsRawLayer = L.layerGroup();
    let layerControl = null;
    let dangerMode = false;

    function formatList(list) {
      if (!list || list.length === 0) return '-';
      return list.join(', ');
    }

    function formatNumber(v, suffix = '') {
      if (v === null || v === undefined) return '?';
      return `${v}${suffix}`;
    }

    function groundLabel(code) {
      const mapping = {1: 'Good', 2: 'Normal', 3: 'Bad', 4: 'Very bad'};
      return mapping[code] || '-';
    }

    function rawStyle(feature) {
      const p = feature.properties || {};
      const src = p.source;
      const metaSrc = p.metadata_source;

      if (p.link_id === 48889) {
        return { color: '#00ffff', weight: 8 };
      }
      if (src === 'manual' || metaSrc === 'manual_mask_google_1km') {
        return { color: '#ff0000', weight: 3 };
      }
      if (src === 'gps') {
        return { color: '#0000ff', weight: 3 };
      }
      if (src === 'satellite') {
        return { color: '#888888', weight: 2 };
      }
      return { color: '#aaaaaa', weight: 2 };
    }

    function gpsRawStyle() {
      return { color: '#888888', weight: 3, dashArray: '4,4' };
    }

    function dangerColor(score) {
      if (score === null || score === undefined) {
        return '#000000'; // 未設定は黒
      }
      // スコアに応じて色分け（必要に応じて後で調整）
      if (score < 2) return '#00cc66';     // 低リスク: 緑
      if (score < 4) return '#ffcc00';     // 中リスク: 黄
      if (score < 6) return '#ff8800';     // 高リスク: オレンジ
      return '#ff0000';                    // 非常に危険: 赤
    }

    function unifiedStyle(feature) {
      const p = feature.properties || {};
      const meta = p.metadata || {};
      const gpsStats = meta.gps_stats || {};

      const avgDanger = typeof gpsStats.avg_danger === 'number'
        ? gpsStats.avg_danger
        : p.danger_score;

      const score = typeof avgDanger === 'number' ? avgDanger : 1.0;

      if (!dangerMode) {
        return { color: '#000000', weight: 5, opacity: 0.9, lineJoin: 'round', lineCap: 'round' };
      }

      const color = dangerColor(score);
      return { color, weight: 5, opacity: 0.9, lineJoin: 'round', lineCap: 'round' };
    }

    function bindUnifiedPopup(layer, p) {
      const meta = p.metadata || {};
      const gpsStats = meta.gps_stats || {};

      const html = [
        `<b>Unified link ${p.link_id}</b>`,
        `Sources: ${formatList(p.sources)}`,
        `Parents: ${formatList(p.parent_link_ids)}`,
        `Width: ${formatNumber(p.width_m, ' m')}`,
        `Slope: ${formatNumber(p.slope_deg, '°')}`,
        `Curvature: ${formatNumber(p.curvature)}`,
        `Visibility: ${formatNumber(p.visibility)}`,
        `Ground: ${groundLabel(p.ground_condition)}`,
        `Danger score: ${p.danger_score ?? '-'}`
      ].join('<br>');

      let gpsHtml = '';
      const hasGpsStats = gpsStats && (
        typeof gpsStats.count === 'number'
        || typeof gpsStats.avg_danger === 'number'
        || typeof gpsStats.last_danger === 'number'
      );

      if (hasGpsStats) {
        gpsHtml += '<hr>';
        gpsHtml += '<b>GPS stats</b><br>';
        if (typeof gpsStats.count === 'number') {
          gpsHtml += `Runs: ${gpsStats.count}<br>`;
        }
        if (typeof gpsStats.avg_danger === 'number') {
          gpsHtml += `Avg danger: ${gpsStats.avg_danger.toFixed(2)}<br>`;
        }
        if (typeof gpsStats.last_danger === 'number') {
          gpsHtml += `Last danger: ${gpsStats.last_danger.toFixed(2)}<br>`;
        }
      }

      layer.bindPopup(html + gpsHtml);
    }

    function bindRawPopup(layer, p) {
      const meta = p.metadata || {};
      const collectorMeta = meta.collector && typeof meta.collector === 'object'
        ? meta.collector
        : {};

      const snapInfo = p.snap || collectorMeta || {};
      const snappingUsed = snapInfo.used === true;
      const snappingDistance = typeof snapInfo.distance_m === 'number'
        ? snapInfo.distance_m.toFixed(2)
        : null;
      const snappingText = snappingUsed
        ? `Snapped: yes${snappingDistance ? ` (${snappingDistance} m)` : ''}`
        : 'Snapped: no';

      const html = [
        `<b>Raw link ${p.link_id}</b>`,
        `Source: ${p.source || '-'}`,
        `GPT class: ${p.gpt_class || '-'}`,
        `Meta source: ${p.metadata_source || '-'}`,
        `Width: ${formatNumber(p.width_m, ' m')}`,
        `Danger score: ${p.danger_score ?? '-'}`,
        snappingText,
      ].filter(Boolean).join('<br>');
      layer.bindPopup(html);
    }

    function addRawGpsDebug(feature) {
      const p = feature.properties || {};
      if (p.source !== 'gps') return;

      const snapMeta = p.snap || (p.metadata && p.metadata.collector) || null;
      if (!snapMeta || !snapMeta.raw_geom) return;

      const rawFeature = { type: 'Feature', geometry: snapMeta.raw_geom };
      L.geoJSON(rawFeature, { style: gpsRawStyle }).addTo(gpsRawLayer);
    }

    async function loadLayers() {
      const b = map.getBounds();
      const params = new URLSearchParams({
        min_lon: b.getWest(),
        min_lat: b.getSouth(),
        max_lon: b.getEast(),
        max_lat: b.getNorth(),
        limit: 5000,
      });

      const [rawRes, unifiedRes] = await Promise.all([
        fetch('/api/roads?' + params.toString()),
        fetch('/api/roads/unified?' + params.toString()),
      ]);

      const rawGeojson = await rawRes.json();
      const unifiedGeojson = await unifiedRes.json();

      gpsRawLayer.clearLayers();

      if (rawLayer) rawLayer.remove();
      if (unifiedLayer) unifiedLayer.remove();
      if (layerControl) {
        map.removeControl(layerControl);
        layerControl = null;
      }

      unifiedLayer = L.geoJSON(unifiedGeojson, {
        style: unifiedStyle,
        onEachFeature: (feature, layer) => bindUnifiedPopup(layer, feature.properties),
      }).addTo(map);

      rawLayer = L.geoJSON(rawGeojson, {
        style: rawStyle,
        onEachFeature: (feature, layer) => {
          bindRawPopup(layer, feature.properties);
          addRawGpsDebug(feature);
        },
      }).addTo(map);

      const overlayMaps = {
        'Raw（カラー分け）': rawLayer,
        'Unified（統合レイヤー・黒）': unifiedLayer,
        '生GPS軌跡（デバッグ）': gpsRawLayer,
      };
      layerControl = L.control.layers(null, overlayMaps).addTo(map);

      unifiedLayer.setZIndex(1);
      rawLayer.setZIndex(2);
      rawLayer.bringToFront();
    }

    function addLegend() {
      const legend = L.control({ position: 'bottomright' });
      legend.onAdd = function() {
        const div = L.DomUtil.create('div', 'legend');
        div.innerHTML = `
          <div class="legend-item">
            <span class="legend-line" style="background:#000000"></span>
            <span>Unified（黒）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#ff0000"></span>
            <span>Manual（赤）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#0000ff"></span>
            <span>GPS（青）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#888888"></span>
            <span>Satellite（灰）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#00cc66"></span>
            <span>Danger: 安全（緑）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#ffcc00"></span>
            <span>Danger: 注意（黄）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#ff8800"></span>
            <span>Danger: 危険（オレンジ）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#ff0000"></span>
            <span>Danger: 非常に危険（赤）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#000000"></span>
            <span>Danger: 未設定（黒）</span>
          </div>
        `;
        return div;
      };
      legend.addTo(map);
    }

    function addDangerToggle() {
      const toggle = L.control({ position: 'topright' });
      toggle.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar');
        div.style.background = 'white';
        div.style.padding = '6px 8px';
        div.style.fontSize = '14px';
        div.style.lineHeight = '1.2';
        const label = L.DomUtil.create('label', '', div);
        const checkbox = L.DomUtil.create('input', '', label);
        checkbox.type = 'checkbox';
        checkbox.checked = dangerMode;
        checkbox.style.marginRight = '6px';
        label.appendChild(document.createTextNode('Danger mode'));
        checkbox.addEventListener('change', (e) => {
          dangerMode = e.target.checked;
          loadLayers();
        });
        L.DomEvent.disableClickPropagation(div);
        return div;
      };
      toggle.addTo(map);
    }

    map.on('moveend', loadLayers);
    addLegend();
    addDangerToggle();
    loadLayers();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


BASE_DIR = Path(__file__).resolve().parent


@app.get("/collector", response_class=HTMLResponse)
def collector_page():
    html_path = BASE_DIR / "app" / "collector.html"
    return html_path.read_text(encoding="utf-8")


# Self-review:
#   - /api/gps_tracks で GPS ポイントを受け取り、road_links に source='gps_mobile' で1本の LINESTRING として挿入し、簡易な曲がり具合ベースの danger_score を付与するようにした。
#   - collector.html にブラウザの geolocation API を使った Start/Stop ロガーと、API への送信処理を追加した。

# Testing:
#   - ローカルで `uvicorn main:app --reload` を起動。
#   - ブラウザで `http://localhost:8000/collector` を開き、屋外 or 擬似位置情報で「記録開始」→少し移動→「記録停止＆送信」を実行。
#   - サーバのログで `/api/gps_tracks` のリクエストが成功し、レスポンス JSON に `status: ok` と `danger_score` が含まれていることを確認。
#   - `/map` を開き、GPS レイヤー（青）の中に、新しいログの LINESTRING が描画されていることを確認。
