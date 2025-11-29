import argparse
import psycopg2
import cv2
import numpy as np
from psycopg2.extras import RealDictCursor

def latlon_to_pixel(lon, lat, w, h, min_lat, max_lat, min_lon, max_lon):
    nx = (lon - min_lon) / (max_lon - min_lon)
    ny = (max_lat - lat) / (max_lat - min_lat)
    x = int(nx * (w - 1))
    y = int(ny * (h - 1))
    return x, y

def parse_linestring(wkt):
    body = wkt[len("LINESTRING("):-1]
    pts = []
    for part in body.split(","):
        lon, lat = part.strip().split()
        pts.append((float(lon), float(lat)))
    return pts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output", default="overlay.png")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--source", default="google_skeleton")
    args = parser.parse_args()

    # Load base image
    base = cv2.imread(args.image_path)
    if base is None:
        raise FileNotFoundError(args.image_path)
    h, w, _ = base.shape

    # DB connect
    conn = psycopg2.connect(args.db_url)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT link_id, ST_AsText(geom) AS wkt
            FROM road_links
            WHERE source = %s;
            """,
            (args.source,)
        )
        rows = cur.fetchall()

    print(f"[INFO] Loaded {len(rows)} polylines from DB.")

    # Draw overlay
    overlay = base.copy()

    for row in rows:
        pts_lonlat = parse_linestring(row["wkt"])
        pts_xy = [
            latlon_to_pixel(lon, lat, w, h,
                            args.min_lat, args.max_lat,
                            args.min_lon, args.max_lon)
            for lon, lat in pts_lonlat
        ]

        # draw
        for i in range(len(pts_xy)-1):
            cv2.line(
                overlay,
                pts_xy[i],
                pts_xy[i+1],
                (0, 0, 255),  # 🔴 red line
                thickness=3
            )

    # save
    cv2.imwrite(args.output, overlay)
    print(f"[INFO] Saved overlay -> {args.output}")

if __name__ == "__main__":
    main()
