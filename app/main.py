from fastapi import FastAPI
from fastapi.responses import JSONResponse
from db import get_connection

app = FastAPI()

@app.get("/road_links")
def get_road_links():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            link_id,
            ST_AsGeoJSON(geom) AS geometry,
            width_m,
            slope_deg,
            curvature,
            visibility,
            ground_condition,
            danger_score
        FROM road_links;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": r[1],
            "properties": {
                "link_id": r[0],
                "width_m": r[2],
                "slope_deg": r[3],
                "curvature": r[4],
                "visibility": r[5],
                "ground_condition": r[6],
                "danger_score": r[7],
            }
        })
    return {"type": "FeatureCollection", "features": features}
