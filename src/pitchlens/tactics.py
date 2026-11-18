"""Spatial metrics on pitch coordinates: shape, lines, passing lanes, space."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .pitch import DEFAULT_PITCH, Pitch


@dataclass
class TeamShape:
    team: int
    n_players: int
    centroid: tuple[float, float]
    width_m: float               # spread across the pitch (y)
    depth_m: float               # spread along the pitch (x)
    area_m2: float               # convex hull area
    defends_left: bool           # True if the team defends the x=0 goal
    last_line_x: float | None    # deepest outfield player (defensive line), metres from own goal line
    block_height_m: float | None  # distance from own goal to the defensive line
    lines: list[list[int]] = field(default_factory=list)  # player indices, back to front
    line_x: list[float] = field(default_factory=list)      # mean x of each line
    shape: str | None = None     # e.g. "4-4-2" when all 10 outfield players are visible
    hull: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PassingLane:
    from_idx: int
    to_idx: int
    length_m: float
    nearest_opponent_m: float    # closest opponent to the pass line
    open: bool


def _hull(xy: np.ndarray) -> tuple[float, list[tuple[float, float]]]:
    if len(xy) < 3:
        return 0.0, [tuple(map(float, p)) for p in xy]
    try:
        h = ConvexHull(xy)
    except QhullError:
        return 0.0, [tuple(map(float, p)) for p in xy]
    return float(h.volume), [tuple(map(float, xy[i])) for i in h.vertices]


def split_lines(x: np.ndarray, max_lines: int = 4, max_line_depth: float = 7.0) -> list[np.ndarray]:
    """Group outfield players into lines along the pitch.

    Exact 1-D k-means for k = 2..max_lines; take the smallest k where no line
    is deeper than ``max_line_depth`` metres front to back (a staggered back
    four is still one line, a holding midfielder 10 m ahead of it is not).
    Returns index arrays sorted from deepest to most advanced (in ``x`` order).
    """
    n = len(x)
    if n == 0:
        return []
    if n <= 2:
        return [np.argsort(x)]
    order = np.argsort(x)
    xs = x[order]

    def best_split(k: int) -> tuple[float, list[np.ndarray]]:
        # Exact 1-D clustering by dynamic programming over sorted values.
        cs = np.concatenate([[0], np.cumsum(xs)])
        cs2 = np.concatenate([[0], np.cumsum(xs ** 2)])

        def cost(i: int, j: int) -> float:  # items i..j-1
            m = j - i
            s = cs[j] - cs[i]
            return float(cs2[j] - cs2[i] - s * s / m)

        INF = float("inf")
        D = np.full((k + 1, n + 1), INF)
        B = np.zeros((k + 1, n + 1), dtype=int)
        D[0, 0] = 0
        for kk in range(1, k + 1):
            for j in range(kk, n + 1):
                for i in range(kk - 1, j):
                    v = D[kk - 1, i] + cost(i, j)
                    if v < D[kk, j]:
                        D[kk, j], B[kk, j] = v, i
        cuts, j = [], n
        for kk in range(k, 0, -1):
            i = B[kk, j]
            cuts.append((i, j))
            j = i
        groups = [order[i:j] for i, j in reversed(cuts)]
        return float(D[k, n]), groups

    ks = list(range(2, min(max_lines, n) + 1))
    for k in ks:
        _, groups = best_split(k)
        if all(np.ptp(x[g]) <= max_line_depth for g in groups):
            return groups
    return best_split(ks[-1])[1]


def team_shape(xy: np.ndarray, team: int, defends_left: bool, gk_mask: np.ndarray | None = None,
               pitch: Pitch = DEFAULT_PITCH) -> TeamShape:
    gk_mask = np.zeros(len(xy), bool) if gk_mask is None else gk_mask
    outfield_idx = np.nonzero(~gk_mask)[0]
    of = xy[outfield_idx]
    area, hull = _hull(of if len(of) >= 3 else xy)
    # x measured from the team's own goal line, so "lines" go back -> front.
    own_x = of[:, 0] if defends_left else pitch.length - of[:, 0]
    lines = [outfield_idx[g].tolist() for g in split_lines(own_x)] if len(of) else []
    line_x = [float(np.mean(xy[g, 0])) for g in lines]
    last = float(own_x.min()) if len(of) else None
    shape = "-".join(str(len(g)) for g in lines) if len(of) == 10 else None
    return TeamShape(
        team=team,
        n_players=len(xy),
        centroid=tuple(float(v) for v in (of if len(of) else xy).mean(axis=0)),
        width_m=float(np.ptp(of[:, 1])) if len(of) else 0.0,
        depth_m=float(np.ptp(of[:, 0])) if len(of) else 0.0,
        area_m2=area,
        defends_left=defends_left,
        last_line_x=(last if defends_left else pitch.length - last) if last is not None else None,
        block_height_m=last,
        lines=lines,
        line_x=line_x,
        shape=shape,
        hull=hull,
    )


def infer_sides(xy0: np.ndarray, xy1: np.ndarray, gk0: np.ndarray | None = None,
                gk1: np.ndarray | None = None, pitch: Pitch = DEFAULT_PITCH) -> bool:
    """Return True if team 0 defends the left (x=0) goal.

    Uses a goalkeeper's position when one is known, otherwise the team whose
    deepest outfield player is nearer the left goal line.
    """
    for gk, xy, flip in ((gk0, xy0, False), (gk1, xy1, True)):
        if gk is not None and gk.any():
            left = xy[gk][:, 0].mean() < pitch.length / 2
            return bool(left != flip)
    if len(xy0) == 0 or len(xy1) == 0:
        return True
    return bool(xy0[:, 0].min() + xy0[:, 0].mean() < xy1[:, 0].min() + xy1[:, 0].mean())


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in ``p`` (N,2) to segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / denom, 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def passing_lanes(carrier: int, team_xy: np.ndarray, opp_xy: np.ndarray, max_len: float = 45.0,
                  base_clearance: float = 1.5, clearance_per_m: float = 0.08) -> list[PassingLane]:
    """Lanes from the carrier to each teammate.

    A lane is open if no opponent is within ``base_clearance + clearance_per_m *
    length`` metres of the straight pass line (longer passes give defenders
    more time to step in). Opponents right next to the carrier still count:
    they can block the pass at the source.
    """
    a = team_xy[carrier]
    lanes = []
    for j, b in enumerate(team_xy):
        if j == carrier:
            continue
        length = float(np.linalg.norm(b - a))
        if length > max_len or length < 1.0:
            continue
        d = float(point_segment_distance(opp_xy, a, b).min()) if len(opp_xy) else float("inf")
        need = base_clearance + clearance_per_m * length
        lanes.append(PassingLane(carrier, j, length, d, bool(d >= need)))
    return lanes


def space_control(points: np.ndarray, teams: np.ndarray, pitch: Pitch = DEFAULT_PITCH,
                  res: float = 1.0) -> tuple[np.ndarray, dict[int, float]]:
    """Nearest-player (Voronoi) control grid over the pitch.

    Returns a (rows, cols) grid of team ids and each team's share of the pitch.
    This is the simplest pitch-control model: whoever is closest owns the
    space. It ignores speed and direction, which a single image can't give.
    """
    xs = np.arange(res / 2, pitch.length, res)
    ys = np.arange(res / 2, pitch.width, res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d = np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2)
    owner = teams[np.argmin(d, axis=1)].reshape(gy.shape)
    share = {int(t): float((owner == t).mean()) for t in np.unique(teams)}
    return owner, share


def to_dict(obj) -> dict:
    return asdict(obj)
