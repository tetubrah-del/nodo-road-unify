# Run:
#   python 10_import_gps_gpx.py
#   python 10_import_gps_gpx.py --dir data/gps

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

from main import get_connection


def is_gpx_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            start = f.read(1024)
        return start.lstrip().startswith("<gpx")
    except Exception:
        return False


def extract_track_points(path: Path) -> Tuple[List[Tuple[float, float]], List[Optional[str]]]:
    tree = ET.parse(path)
    root = tree.getroot()
    track_points: List[Tuple[float, float]] = []
    times: List[Optional[str]] = []

    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.get("lat")
        lon = trkpt.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except ValueError:
            continue
        track_points.append((lon_f, lat_f))
        time_el = trkpt.find("{*}time")
        times.append(time_el.text if time_el is not None else None)

    return track_points, times


def already_imported(cur, filename: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM road_links
        WHERE source = 'gps' AND metadata->>'file' = %s
        LIMIT 1
        """,
        (filename,),
    )
    return cur.fetchone() is not None


def insert_track(cur, coords: List[Tuple[float, float]], times: List[Optional[str]], filename: str):
    wkt_coords = ", ".join(f"{lon} {lat}" for lon, lat in coords)
    linestring_wkt = f"LINESTRING({wkt_coords})"

    start_time = next((t for t in times if t), None) if times else None
    end_time = next((t for t in reversed(times) if t), None) if times else None

    metadata = {
        "file": filename,
        "num_points": len(coords),
        "start_time": start_time,
        "end_time": end_time,
        "type": "gps_track",
    }

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
            'gps',
            NULL, NULL, NULL, NULL, NULL, NULL,
            %s::jsonb
        )
        """,
        (linestring_wkt, json.dumps(metadata)),
    )


def process_file(path: Path, cur) -> bool:
    filename = path.name

    if not path.is_file():
        return False

    if not is_gpx_file(path):
        print(f"[skip] {filename}: not a GPX file")
        return False

    if already_imported(cur, filename):
        print(f"[skip] {filename}: skipped (already imported)")
        return False

    try:
        coords, times = extract_track_points(path)
        if len(coords) <= 1:
            print(f"[skip] {filename}: too few points")
            return False
        insert_track(cur, coords, times, filename)
        print(f"[ok] {filename}: imported {len(coords)} points")
        return True
    except Exception as exc:
        print(f"[error] failed to import {filename}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Import GPX tracks into road_links")
    parser.add_argument("--dir", default="data/gps", help="Directory containing GPX files")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    total_files = 0
    imported_files = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for path in sorted(target_dir.rglob("*")):
                if path.is_dir():
                    continue
                total_files += 1
                if process_file(path, cur):
                    imported_files += 1
            conn.commit()

    print(f"Processed {total_files} files, imported {imported_files} new files")


if __name__ == "__main__":
    main()
