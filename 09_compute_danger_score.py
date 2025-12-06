"""Compute provisional danger_score for road link tables."""

import argparse

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from danger_score import compute_danger_score
from main import get_connection


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
                (
                    compute_danger_score(
                        width_m=row.get("width_m"),
                        slope_deg=row.get("slope_deg"),
                        curvature=row.get("curvature"),
                        visibility=row.get("visibility"),
                        ground_condition=row.get("ground_condition"),
                    ),
                    row["link_id"],
                )
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
