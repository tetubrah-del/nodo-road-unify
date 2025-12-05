"""Hidden Markov Model based map matching utilities."""
import json
import math
from typing import List, Optional, Sequence, Tuple

from psycopg2.extensions import connection as PGConnection

from geodesic_utils import Point, geodesic_distance_m


def emission_probability(point: Point, state_point: Point, sigma: float = 5.0) -> float:
    distance = geodesic_distance_m(point["lat"], point["lon"], state_point["lat"], state_point["lon"])
    return math.exp(-((distance / sigma) ** 2))


def transition_probability(idx_from: int, idx_to: int, delta_d: float = 1.0) -> float:
    return math.exp(-(((abs(idx_from - idx_to) * delta_d) ** 2)))


def hmm_viterbi(
    observations: Sequence[Point],
    states: Sequence[Point],
    sigma: float = 5.0,
    delta_d: float = 1.0,
) -> List[int]:
    if not observations or not states:
        return []

    num_states = len(states)
    log_emissions: List[List[float]] = []
    for obs in observations:
        row = []
        for st in states:
            prob = emission_probability(obs, st, sigma)
            row.append(math.log(prob + 1e-12))
        log_emissions.append(row)

    log_transitions = [
        [math.log(transition_probability(i, j, delta_d) + 1e-12) for j in range(num_states)]
        for i in range(num_states)
    ]

    dp: List[List[float]] = [[-math.inf for _ in range(num_states)] for _ in observations]
    backpointers: List[List[Optional[int]]] = [[None for _ in range(num_states)] for _ in observations]

    initial_log_prob = -math.log(num_states)
    for state_idx in range(num_states):
        dp[0][state_idx] = initial_log_prob + log_emissions[0][state_idx]

    for t in range(1, len(observations)):
        for state_idx in range(num_states):
            best_prev: Tuple[float, Optional[int]] = (-math.inf, None)
            for prev_idx in range(num_states):
                candidate = dp[t - 1][prev_idx] + log_transitions[prev_idx][state_idx]
                if candidate > best_prev[0]:
                    best_prev = (candidate, prev_idx)
            dp[t][state_idx] = best_prev[0] + log_emissions[t][state_idx]
            backpointers[t][state_idx] = best_prev[1]

    last_state = max(range(num_states), key=lambda s: dp[-1][s])
    path = [last_state]
    for t in range(len(observations) - 1, 0, -1):
        prev = backpointers[t][path[-1]]
        if prev is None:
            break
        path.append(prev)
    path.reverse()
    return path


def hmm_match_trace(
    points: Sequence[Point],
    centerline: Sequence[Point],
    sigma: float = 5.0,
    delta_d: float = 1.0,
) -> List[Point]:
    state_path = hmm_viterbi(points, centerline, sigma=sigma, delta_d=delta_d)
    return [centerline[i] for i in state_path]


def store_matched_trace(
    conn: PGConnection,
    matched_points: Sequence[Point],
    source_id: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> Optional[int]:
    if not matched_points:
        return None

    coords = ", ".join(f"{p['lon']} {p['lat']}" for p in matched_points)
    wkt = f"LINESTRING({coords})"
    meta = metadata or {}
    meta.update({"method": "hmm_map_match", "source_id": source_id})

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO road_links_unified (
                geom,
                source,
                metadata
            )
            VALUES (
                ST_GeomFromText(%s, 4326),
                'gps_matched',
                %s::jsonb
            )
            RETURNING link_id
            """,
            (wkt, json.dumps(meta, ensure_ascii=False)),
        )
        row = cur.fetchone()
        return row[0] if row else None


__all__ = [
    "emission_probability",
    "transition_probability",
    "hmm_viterbi",
    "hmm_match_trace",
    "store_matched_trace",
]
