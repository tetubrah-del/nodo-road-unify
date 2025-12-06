from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


class LineString:
    def __init__(self, coords: Iterable[Sequence[float]]):
        self._coords: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in coords]

    @property
    def coords(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(self._coords)

    @property
    def is_empty(self) -> bool:
        return len(self._coords) == 0

    @property
    def wkt(self) -> str:
        joined = ", ".join(f"{x} {y}" for x, y in self._coords)
        return f"LINESTRING ({joined})"

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._coords)

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self._coords)


__all__ = ["LineString"]
