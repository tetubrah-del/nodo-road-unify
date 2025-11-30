from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import psycopg2
from psycopg2.extras import RealDictCursor

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
                    ST_AsGeoJSON(geom)::json AS geom,
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
                    ST_AsGeoJSON(geom)::json AS geom
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

    let manualLayer = null;
    let gpsLayer = null;
    let satelliteLayer = null;
    let unifiedLayer = null;
    let layerControl = null;

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
      if (p.source === 'gps') {
        return { color: '#0000ff', weight: 3 };
      }
      if (p.source === 'manual' || p.metadata_source === 'manual_mask_google_1km') {
        return { color: '#ff0000', weight: 3 };
      }
      if (p.source === 'satellite') {
        return { color: '#888888', weight: 2 };
      }
      return { color: '#888888', weight: 2 };
    }

    function bindUnifiedPopup(layer, p) {
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
      layer.bindPopup(html);
    }

    function bindRawPopup(layer, p) {
      const html = [
        `<b>Raw link ${p.link_id}</b>`,
        `Source: ${p.source || '-'}`,
        `GPT class: ${p.gpt_class || '-'}`,
        `Meta source: ${p.metadata_source || '-'}`,
        `Width: ${formatNumber(p.width_m, ' m')}`,
        `Danger score: ${p.danger_score ?? '-'}`
      ].join('<br>');
      layer.bindPopup(html);
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

      const manualFeatures = [];
      const gpsFeatures = [];
      const satelliteFeatures = [];

      (rawGeojson.features || []).forEach((feature) => {
        const p = feature.properties || {};
        if (p.source === 'manual' || p.metadata_source === 'manual_mask_google_1km') {
          manualFeatures.push(feature);
        } else if (p.source === 'gps') {
          gpsFeatures.push(feature);
        } else if (p.source === 'satellite') {
          satelliteFeatures.push(feature);
        }
      });

      if (manualLayer) manualLayer.remove();
      if (gpsLayer) gpsLayer.remove();
      if (satelliteLayer) satelliteLayer.remove();
      if (unifiedLayer) unifiedLayer.remove();
      if (layerControl) {
        map.removeControl(layerControl);
        layerControl = null;
      }

      manualLayer = L.geoJSON({ type: 'FeatureCollection', features: manualFeatures }, {
        style: rawStyle,
        onEachFeature: (feature, layer) => bindRawPopup(layer, feature.properties),
      }).addTo(map);

      gpsLayer = L.geoJSON({ type: 'FeatureCollection', features: gpsFeatures }, {
        style: rawStyle,
        onEachFeature: (feature, layer) => bindRawPopup(layer, feature.properties),
      }).addTo(map);

      satelliteLayer = L.geoJSON({ type: 'FeatureCollection', features: satelliteFeatures }, {
        style: rawStyle,
        onEachFeature: (feature, layer) => bindRawPopup(layer, feature.properties),
      }).addTo(map);

      unifiedLayer = L.geoJSON(unifiedGeojson, {
        style: { color: '#000000', weight: 5, opacity: 0.9, lineJoin: 'round', lineCap: 'round' },
        onEachFeature: (feature, layer) => bindUnifiedPopup(layer, feature.properties),
      }).addTo(map);

      const overlayMaps = {
        'Raw / Manual（手描き・赤）': manualLayer,
        'Raw / GPS（青）': gpsLayer,
        'Raw / Satellite FARMLANE（薄グレー）': satelliteLayer,
        'Unified（統合レイヤー・黒）': unifiedLayer,
      };
      layerControl = L.control.layers(null, overlayMaps).addTo(map);

      manualLayer.setZIndex(1);
      gpsLayer.setZIndex(2);
      satelliteLayer.setZIndex(3);
      unifiedLayer.setZIndex(4);
    }

    function addLegend() {
      const legend = L.control({ position: 'bottomright' });
      legend.onAdd = function() {
        const div = L.DomUtil.create('div', 'legend');
        div.innerHTML = `
          <div class="legend-item">
            <span class="legend-line" style="background:#000000"></span>
            <span>Unified roads（統合レイヤー）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#ff0000"></span>
            <span>Raw / Manual（手描き）</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#0000ff"></span>
            <span>Raw / GPS</span>
          </div>
          <div class="legend-item">
            <span class="legend-line" style="background:#888888"></span>
            <span>Raw / Satellite FARMLANE</span>
          </div>
        `;
        return div;
      };
      legend.addTo(map);
    }

    map.on('moveend', loadLayers);
    addLegend();
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


# Self-review: updated road APIs and /map Leaflet rendering to include unified and raw layers.
