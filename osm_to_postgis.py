#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Any, Tuple, Optional

import requests
import psycopg2
from psycopg2.extras import execute_values

# =========================
# 設定
# =========================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 大分市周辺あたりのBBOX（ざっくり）
# (min_lon, min_lat, max_lon, max_lat)
BBOX = (131.4, 33.0, 131.9, 33.5)

# 農道寄りの highway を中心に取得
HIGHWAY_FILTER = [
    "track", "service", "unclassified",
    "living_street", "residential",
    "tertiary", "secondary", "primary",
]

# Postgres接続情報（環境変数 or 直書き）
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "nodo")
PG_USER = os.getenv("PG_USER", "nodo")
PG_PASS = os.getenv("PG_PASS", "nodo_password")

TARGET_TABLE = "road_links"


# =========================
# Overpass から取得
# =========================

def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    """
    指定bbox内の highway=* の way を取得する Overpassクエリ
    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    highway_regex = "|".join(HIGHWAY_FILTER)
    query = f"""
    [out:json][timeout:180];
    (
      way
        ["highway"]["highway"~"{highway_regex}"]
        ({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    return query


def fetch_osm_data(bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    print("=== fetch_osm_data: start ===")
    query = build_overpass_query(bbox)
    print("Sending Overpass query to:", OVERPASS_URL)
    print("BBOX:", bbox)
    resp = requests.post(OVERPASS_URL, data={"data": query})
    print("HTTP status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    print("Received elements:", len(data.get("elements", [])))
    return data


# =========================
# Overpass JSON → LineString
# =========================

def overpass_to_ways(data: Dict[str, Any]):
    """
    OverpassのJSONから node辞書 と wayリスト を抽出
    """
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way":
            ways.append(el)
    print(f"Parsed {len(nodes)} nodes, {len(ways)} ways")
    return nodes, ways


def build_linestring_wkt(
    node_ids: List[int],
    node_dict: Dict[int, Tuple[float, float]]
) -> Optional[str]:
    """
    ノードID列からWKT LineStringを生成
    """
    coords: List[str] = []
    for nid in node_ids:
        if nid not in node_dict:
            return None
        lon, lat = node_dict[nid]
        coords.append(f"{lon} {lat}")
    if len(coords) < 2:
        return None
    return "LINESTRING(" + ", ".join(coords) + ")"


def overpass_to_geoms(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Overpass JSON → (WKT, tags) のリスト
    """
    node_dict, ways = overpass_to_ways(data)
    results: List[Tuple[str, Dict[str, Any]]] = []
    for way in ways:
        node_ids = way.get("nodes", [])
        tags = way.get("tags", {})
        wkt = build_linestring_wkt(node_ids, node_dict)
        if not wkt:
            continue
        results.append((wkt, tags))
    print(f"Converted to {len(results)} valid LineStrings")
    return results


# =========================
# PostGISへINSERT
# =========================

def insert_roads_to_postgis(geoms_and_tags: List[Tuple[str, Dict[str, Any]]]) -> None:
    """
    PostGIS (road_links) へ一括INSERTする。
    """
    if not geoms_and_tags:
        print("No geometries to insert. Skipping DB insert.")
        return

    print(f"Connecting to Postgres at {PG_HOST}:{PG_PORT}, db={PG_DB} ...")

    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )
    cur = conn.cursor()

    # 外側のSQLは「VALUES %s」を1個だけ
    sql = f"""
    INSERT INTO {TARGET_TABLE} (
        geom,
        width_m,
        slope_deg,
        curvature,
        visibility,
        ground_condition,
        danger_score,
        metadata
    ) VALUES %s
    """

    # 1行分のテンプレート（ここに %s を使う）
    template = """
    (
        ST_SetSRID(ST_GeomFromText(%s), 4326),
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        %s::jsonb
    )
    """

    # (geom_wkt, tags_json) のタプルリスト
    values = [
        (wkt, json.dumps(tags))
        for (wkt, tags) in geoms_and_tags
    ]

    print(f"Inserting {len(values)} rows into {TARGET_TABLE} ...")

    execute_values(
        cur,
        sql,
        values,
        template=template,
        page_size=1000,
    )

    conn.commit()
    cur.close()
    conn.close()
    print("DB insert completed.")




# =========================
# メイン
# =========================

def main() -> None:
    print("=== osm_to_postgis.py: START ===")
    print("BBOX:", BBOX)
    try:
        data = fetch_osm_data(BBOX)
    except Exception as e:
        print("Error while fetching OSM data from Overpass:", repr(e))
        return

    geoms_and_tags = overpass_to_geoms(data)
    print("Number of geoms_and_tags:", len(geoms_and_tags))

    try:
        insert_roads_to_postgis(geoms_and_tags)
    except Exception as e:
        print("Error while inserting into PostGIS:", repr(e))
        return

    print("=== osm_to_postgis.py: END ===")


if __name__ == "__main__":
    main()
