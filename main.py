from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor

# main.py のどこか（FastAPI app 定義の下あたり）
import os
import json
import psycopg2
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "nodo")
PG_USER = os.getenv("PG_USER", "nodo")
PG_PASS = os.getenv("PG_PASS", "nodo_password")

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )

from psycopg2.extras import RealDictCursor
from fastapi.responses import JSONResponse

@app.get("/api/roads")
def get_roads(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    limit: int = 5000,
):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    sql = """
    SELECT
        link_id,
        source,
        confidence,
        (metadata->'gpt_filter'->>'class') AS gpt_class,
        ST_AsGeoJSON(geom)::json AS geom,
        width_m,
        slope_deg,
        curvature,
        visibility,
        ground_condition,
        danger_score,
        metadata
    FROM road_links
    WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
      AND (
            -- ① 既存の GPS（今見えている青線）
            source = 'gps'

            -- ② 手描き農道（今回追加した 14 本）
            OR metadata->>'source' = 'manual_mask_google_1km'

            -- ③ 衛星＋GPT フィルタ済 FARMLANE
            OR (
                source = 'satellite'
                AND metadata->'gpt_filter'->>'class' = 'FARMLANE'
                AND (metadata->'gpt_filter'->>'confidence')::float >= 0.85
                AND ST_Length(geom::geography) BETWEEN 30 AND 800
            )
        )
    ORDER BY link_id
    LIMIT %s;
    """

    cur.execute(sql, (min_lon, min_lat, max_lon, max_lat, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": r["geom"],
            "properties": {
                "link_id": r["link_id"],
                "source": r["source"],  # gps / satellite / None
                "gpt_class": r["gpt_class"],
                "confidence": float(r["confidence"] or 0.0) if "confidence" in r else None,
                "width_m": r["width_m"],
                "slope_deg": r["slope_deg"],
                "curvature": r["curvature"],
                "visibility": r["visibility"],
                "ground_condition": r["ground_condition"],
                "danger_score": r["danger_score"],
                "metadata_source": (r["metadata"] or {}).get("source") if r.get("metadata") else None,
            },
        })

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
    })


# === DB接続設定（docker-compose の設定と合わせてある） ===
DB_SETTINGS = dict(
    host="localhost",
    port=5432,
    dbname="nodo",
    user="nodo",
    password="nodo_password",
)


def get_connection():
    """PostGIS へのコネクションを返すヘルパー"""
    return psycopg2.connect(**DB_SETTINGS)


# === DangerScore の計算ロジック ===
def calc_danger_score(
    width_m: Optional[float],
    slope_deg: Optional[float],
    curvature: Optional[float],
    visibility: Optional[float],
    ground_condition: Optional[int],
) -> float:
    score = 0.0

    # 幅員：狭いほど危険
    if width_m is not None:
        if width_m >= 4:
            score += 0
        elif width_m >= 3:
            score += 20
        elif width_m >= 2.5:
            score += 40
        else:
            score += 60

    # 傾斜：急なほど危険
    if slope_deg is not None:
        if slope_deg <= 3:
            score += 0
        elif slope_deg <= 8:
            score += 10
        elif slope_deg <= 15:
            score += 25
        else:
            score += 40

    # カーブ：きついほど危険
    if curvature is not None:
        if curvature <= 0.2:
            score += 0
        elif curvature <= 0.4:
            score += 10
        elif curvature <= 0.6:
            score += 25
        else:
            score += 40

    # 視認性：悪いほど危険（値が小さいほど危険）
    if visibility is not None:
        if visibility >= 0.8:
            score += 0
        elif visibility >= 0.5:
            score += 10
        elif visibility >= 0.3:
            score += 25
        else:
            score += 40

    # ground_condition：路面状態（1:良い〜4:悪い）
    if ground_condition is not None:
        if ground_condition == 1:
            score += 0
        elif ground_condition == 2:
            score += 10
        elif ground_condition == 3:
            score += 25
        else:  # 4 以上は全部「悪い」と扱う
            score += 40

    # 0〜100 にクリップ
    return max(0.0, min(score, 100.0))

def calc_danger_score(slope_deg, curvature, visibility, ground_condition):
    return (
        0.3 * (slope_deg / 30.0) +
        0.2 * curvature +
        0.2 * (1 - visibility) +
        0.3 * (ground_condition / 4.0)
    )

# === Pydantic モデル ===
class RoadLink(BaseModel):
    link_id: int
    geom: dict
    width_m: Optional[float] = None
    slope_deg: Optional[float] = None
    curvature: Optional[float] = None
    visibility: Optional[float] = None
    ground_condition: Optional[int] = None
    danger_score: Optional[float] = None

from pydantic import BaseModel
from typing import List

class LatLng(BaseModel):
    lat: float
    lng: float

class RoadInput(BaseModel):
    coords: List[LatLng]         # GPSで取った点列
    width_m: float | None = None
    slope_deg: float | None = None
    curvature: float | None = None
    visibility: float | None = None
    ground_condition: int | None = None

# === FastAPI 本体 ===
app = FastAPI(title="Nodo Safety API")


@app.get("/health")
def health():
    return {"status": "ok"}


from fastapi import HTTPException  # もう入ってたらOK

# --- API: 全リンクを GeoJSON で返す ---
from psycopg2.extras import RealDictCursor

@app.get("/api/roads/geojson")
def list_roads_geojson():
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
                    ST_AsGeoJSON(geom)::json AS geom
                FROM road_links
                ORDER BY link_id
                """
            )
            rows = cur.fetchall()

    features = []
    for r in rows:
        features.append(
            {
                "type": "Feature",
                "geometry": r["geom"],
                "properties": {
                    "link_id": r["link_id"],
                    "source": r["source"],              # ← gps / satellite
                    "confidence": float(r["confidence"] or 0.0),
                    "gpt_class": r["gpt_class"],        # ← FARMLANE など
                    "width_m": r["width_m"],
                    "slope_deg": r["slope_deg"],
                    "curvature": r["curvature"],
                    "visibility": r["visibility"],
                    "ground_condition": r["ground_condition"],
                    "danger_score": float(r["danger_score"] or 0.0),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }

@app.get("/map", response_class=HTMLResponse)
def map_page():
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Nodo Safety Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  />
  <style>
    html, body { margin: 0; padding: 0; height: 100%; }
    #map { width: 100%; height: 100vh; }
  </style>
</head>
<body>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([33.1478, 131.5209], 16);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    let roadsLayer = null;

    function styleByProps(p) {
  // ① 手書き農道：metadata_source が manual_mask_google_1km
  if (p.metadata_source === 'manual_mask_google_1km') {
    return { color: '#ff3333', weight: 4 };
  }

  // ② GPS は青太線
  if (p.source === 'gps') {
    return { color: '#0066ff', weight: 5 };
  }

  // ③ 衛星 + FARMLANE は赤
  if (p.source === 'satellite' && p.gpt_class === 'FARMLANE') {
    return { color: '#ff0000', weight: 4 };
  }

  // ④ 衛星その他はグレー
  if (p.source === 'satellite') {
    return { color: '#999999', weight: 2 };
  }

  // ⑤ OSM やその他は濃いグレー
  return { color: '#555555', weight: 1 };
}



    async function loadRoads() {
      const b = map.getBounds();
      const params = new URLSearchParams({
        min_lon: b.getWest(),
        min_lat: b.getSouth(),
        max_lon: b.getEast(),
        max_lat: b.getNorth(),
        limit: 5000
      });

      const res = await fetch('/api/roads?' + params.toString());
      const geojson = await res.json();

      if (roadsLayer) {
        roadsLayer.remove();
      }

      roadsLayer = L.geoJSON(geojson, {
        style: (feature) => styleByProps(feature.properties),
        onEachFeature: (feature, layer) => {
          const p = feature.properties;
          layer.bindPopup(
            'link_id: ' + p.link_id +
            '<br>source: ' + (p.source ?? 'NULL') +
            '<br>gpt_class: ' + (p.gpt_class ?? 'N/A') +
            '<br>conf: ' + (p.confidence ?? 'N/A') +
            '<br>curvature: ' + (p.curvature ?? 'N/A')
          );
        }
      }).addTo(map);
    }

    map.on('moveend', loadRoads);
    loadRoads();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

@app.get("/api/roads")
def list_roads(min_lon: float, min_lat: float, max_lon: float, max_lat: float, limit: int = 5000):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    link_id,
                    source,
                    confidence,
                    (metadata->'gpt_filter'->>'class') AS gpt_class,
                    metadata->>'source' AS metadata_source,   -- ★追加
                    ST_AsGeoJSON(geom)::json AS geom,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score
                FROM road_links
                WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                ORDER BY link_id
                LIMIT %s
                """,
                (min_lon, min_lat, max_lon, max_lat, limit)
            )
            rows = cur.fetchall()

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": r["geom"],
                "properties": {
                    "link_id": r["link_id"],
                    "source": r["source"],
                    "confidence": r["confidence"],
                    "gpt_class": r["gpt_class"],
                    "metadata_source": r["metadata_source"],   # ★追加
                    "width_m": r["width_m"],
                    "slope_deg": r["slope_deg"],
                    "curvature": r["curvature"],
                    "visibility": r["visibility"],
                    "ground_condition": r["ground_condition"],
                    "danger_score": r["danger_score"]
                }
            }
            for r in rows
        ]
    }

from fastapi.responses import HTMLResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/collector", response_class=HTMLResponse)
def collector_page():
    html_path = BASE_DIR / "app" / "collector.html"
    return html_path.read_text(encoding="utf-8")

from fastapi.responses import JSONResponse, HTMLResponse
import psycopg2
import json
import os

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "nodo")
PG_USER = os.getenv("PG_USER", "nodo")
PG_PASS = os.getenv("PG_PASS", "nodo_password")

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )

# 👇 これが今回必要な GET 用の API
@app.get("/api/roads")
def get_roads(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    limit: int = 5000,
):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
   SELECT
    link_id,
    source,
    confidence,
    (metadata->'gpt_filter'->>'class') AS gpt_class,
    ST_AsGeoJSON(geom)::json AS geom,
    width_m,
    slope_deg,
    curvature,
    visibility,
    ground_condition,
    danger_score
FROM road_links
WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
  AND (
        source = 'gps'
        OR (
            source = 'satellite'
            AND metadata->'gpt_filter'->>'class' = 'FARMLANE'
            AND (metadata->'gpt_filter'->>'confidence')::float >= 0.85
            AND ST_Length(geom::geography) BETWEEN 30 AND 800
        )
    )
ORDER BY link_id
LIMIT %s
    """

    cur.execute(sql, (min_lon, min_lat, max_lon, max_lat, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    features = []
    for link_id, geom, highway, curvature, slope_deg, danger_score in rows:
        features.append({
            "type": "Feature",
            "geometry": geom,  # ST_AsGeoJSON(... )::json なので dict
            "properties": {
                "link_id": link_id,
                "highway": highway,
                "curvature": curvature,
                "slope_deg": slope_deg,
                "danger_score": danger_score,
            },
        })

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
    })
