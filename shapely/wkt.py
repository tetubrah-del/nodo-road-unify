from __future__ import annotations

from typing import Iterable, Tuple

from .geometry import LineString


def loads(wkt_str: str) -> LineString:
    if not isinstance(wkt_str, str):
        raise ValueError("WKT input must be a string")

    cleaned = wkt_str.strip().upper()
    if not cleaned.startswith("LINESTRING"):
        raise ValueError("Unsupported WKT geometry type")

    try:
        coord_text = wkt_str.strip()[wkt_str.index("(") + 1 : wkt_str.rindex(")")]
    except ValueError as exc:  # pragma: no cover - malformed input edge case
        raise ValueError("Invalid WKT format") from exc

    coords = []
    for part in coord_text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        pieces = stripped.split()
        if len(pieces) < 2:
            continue
        x, y = float(pieces[0]), float(pieces[1])
        coords.append((x, y))

    return LineString(coords)
