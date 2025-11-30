"""Compute provisional danger_score for road link tables."""

import argparse
import os

import psycopg2
import psycopg2.extras
from psycopg2 import sql


def bucket(value, thresholds):
    """閾値リストに基づいて 0〜len(thresholds) のスコアを返すヘルパー。
    value が None の場合は最大スコアを返す。"""
    if value is None:
        return len(thresholds)
    for i, th in enumerate(thresholds):
        if value < th:
            return i
    return len(thresholds)


def compute_danger_score(row):
    width = row["width_m"]
    slope = row["slope_deg"]
    curv = row["curvature"]
    vis = row["visibility"]
    ground = row["ground_condition"]

    # 幅: 狭いほど危険
    width_score = bucket(width, [3.5, 3.0, 2.5, 2.0])

    # 勾配: 急なほど危険
    slope_score = bucket(slope, [5, 8, 12, 16])

    # カーブ: 大きいほど危険
    curvature_score = bucket(curv, [0.1, 0.2, 0.3, 0.4])

    # 見通し: 低いほど危険（visibility は 0〜1 を想定）
    if vis is None:
        visibility_score = 4
    else:
        # 高いほど安全なので 1 - vis で反転させてから bucket
        visibility_score = bucket(1.0 - vis, [0.2, 0.4, 0.6, 0.8])

    # 路面状態: 1〜4（良→悪）。None は 3 とみなす。
    if ground is None:
        ground = 3
    ground_score = float(ground)  # そのまま使う（最大 4）

    raw_total = (
        width_score * 1.0
        + slope_score * 1.2
        + curvature_score * 1.2
        + visibility_score * 1.0
        + ground_score * 0.8
    )

    danger = 1.0 + raw_total / 3.0
    # 1.0〜8.0 にクリップ
    if danger < 1.0:
        danger = 1.0
    if danger > 8.0:
        danger = 8.0
    return round(danger, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute provisional danger_score for road link tables.",
    )
    parser.add_argument(
        "--table",
        default="road_links_unified",
        help="Target table name (default: road_links_unified)",
    )
    return parser.parse_args()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASS"),
    )


def main():
    args = parse_args()

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            target_table = sql.Identifier(args.table)

            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        link_id,
                        width_m,
                        slope_deg,
                        curvature,
                        visibility,
                        ground_condition
                    FROM {};
                    """
                ).format(target_table)
            )
            rows = cur.fetchall()

            updates = [
                (compute_danger_score(row), row["link_id"])
                for row in rows
            ]

            print(f"Updating {len(updates)} rows in {args.table}...")

            update_query = sql.SQL(
                "UPDATE {} SET danger_score = %s WHERE link_id = %s"
            ).format(target_table)
            cur.executemany(update_query, updates)

    print("Done.")


if __name__ == "__main__":
    main()
