"""Draw the annotated frame and the 2D pitch map."""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from .pipeline import Analysis

FALLBACK = {0: (230, 57, 70), 1: (69, 123, 157), None: (240, 200, 40)}
REF = (30, 30, 30)
BALL = (255, 255, 255)


def team_rgb(a: Analysis, team: int | None) -> tuple[int, int, int]:
    """The team's real shirt colour, made readable on green grass."""
    c = a.team_info.get("team_colours", {}).get(team)
    if c is None:
        return FALLBACK.get(team, FALLBACK[None])
    c = np.array(c, float)
    # Grass-like or very washed out shirts are hard to see: nudge them.
    hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if 30 <= hsv[0] <= 90 and hsv[1] > 60:
        hsv[1] = min(255, hsv[1] + 60)
        hsv[2] = min(255, hsv[2] + 40)
        c = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(v) for v in c)


def _hex(rgb) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def annotate_frame(image_bgr: np.ndarray, a: Analysis, show_pitch_lines: bool = True) -> np.ndarray:
    """Boxes coloured by team, plus the pitch markings projected back onto the image.

    The projected markings are the quickest way to check a calibration: if they
    sit on the painted lines, the 2D map is trustworthy.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    lw = max(2, w // 700)
    if show_pitch_lines and a.calibration is not None:
        for seg in a.pitch.segments():
            dense = []
            for p, q in zip(seg[:-1], seg[1:]):
                for t in np.linspace(0, 1, 12):
                    dense.append(p + (q - p) * t)
            pts = a.calibration.to_image(np.array(dense))
            ok = np.all(np.isfinite(pts), axis=1) & (np.abs(pts) < 4 * max(h, w)).all(axis=1)
            pts = pts[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(out, [pts], False, (255, 0, 255), lw, cv2.LINE_AA)
        for p in a.calibration.points:
            x, y = map(int, p.image)
            cv2.circle(out, (x, y), 4 * lw, (0, 255, 255), -1)
    for d in a.detections:
        x1, y1, x2, y2 = map(int, d.box)
        if d.role == "ball":
            cv2.circle(out, (int(d.centre[0]), int(d.centre[1])), 10 * lw, (255, 255, 255), lw)
            continue
        rgb = REF if d.role in ("referee", "official") else team_rgb(a, d.team)
        bgr = tuple(int(v) for v in rgb[::-1])
        cx = (x1 + x2) // 2
        cv2.ellipse(out, (cx, y2), (max(8, (x2 - x1) // 2 + 4), max(3, (x2 - x1) // 6)), 0, -45, 235,
                    bgr, lw + 1, cv2.LINE_AA)
        if d.role == "goalkeeper":
            cv2.putText(out, "GK", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, bgr, lw, cv2.LINE_AA)
        if d is a.carrier:
            cv2.drawMarker(out, (cx, y1 - 10 * lw), (0, 215, 255), cv2.MARKER_TRIANGLE_DOWN, 14 * lw, lw + 1)
    return out


def draw_pitch_map(a: Analysis, path: str | None = None, title: str | None = None,
                   show_control: bool = True, show_lines: bool = True, show_lanes: bool = True):
    """Top-down map: players, team shapes, lines, passing lanes and space control."""
    P = a.pitch
    fig, ax = plt.subplots(figsize=(12, 12 * (P.width + 6) / (P.length + 6) + 0.9), dpi=110)
    # A dark neutral pitch rather than green: many kits are green, and team
    # colours (taken from the real shirts) need to stand out.
    ax.set_facecolor("#1f2622")
    fig.patch.set_facecolor("#151a17")
    for i in range(10):  # mowing stripes
        ax.axvspan(i * P.length / 10, (i + 1) * P.length / 10, color="#232b26" if i % 2 else "#1f2622", zorder=0)
    c0, c1 = team_rgb(a, 0), team_rgb(a, 1)

    if show_control and a.control_grid is not None:
        grid = np.where(a.visible_grid, a.control_grid, np.nan)
        ax.imshow(grid, extent=(0, P.length, P.width, 0), origin="upper",
                  cmap=ListedColormap([_hex(c0), _hex(c1)]), vmin=0, vmax=1, alpha=0.18,
                  interpolation="nearest", zorder=1)
        # Dim what the camera can't see.
        ax.imshow(np.where(a.visible_grid, np.nan, 1.0), extent=(0, P.length, P.width, 0),
                  origin="upper", cmap=ListedColormap(["#000000"]), alpha=0.45, zorder=1)

    for seg in P.segments():
        ax.plot(seg[:, 0], seg[:, 1], color="#d9e2dc", lw=1.4, alpha=0.75, zorder=2)
    ax.scatter([P.penalty_spot_distance, P.length - P.penalty_spot_distance, P.length / 2],
               [P.width / 2] * 3, s=8, color="white", zorder=2)

    for t, s in a.shapes.items():
        col = _hex(team_rgb(a, t))
        if len(s.hull) >= 3:
            ax.add_patch(Polygon(s.hull, closed=True, fc=col, ec=col, alpha=0.15, lw=2, zorder=3))
            ax.add_patch(Polygon(s.hull, closed=True, fill=False, ec=col, alpha=0.8, lw=1.5, ls="--", zorder=3))
        if show_lines:
            players = a.players(t)
            for line in s.lines:
                pts = np.array([players[i].pitch_xy for i in line])
                if len(pts) >= 2:
                    pts = pts[np.argsort(pts[:, 1])]
                    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.2, alpha=0.75, zorder=4)
        if s.last_line_x is not None:
            ax.axvline(s.last_line_x, color=col, lw=1.2, ls=":", alpha=0.9, zorder=3)

    if show_lanes and a.carrier is not None:
        mates = a.players(a.carrier.team)
        src = np.array(a.carrier.pitch_xy)
        for lane in a.lanes:
            dst = np.array(mates[lane.to_idx].pitch_xy)
            ax.annotate("", xy=dst, xytext=src, zorder=5,
                        arrowprops=dict(arrowstyle="-|>", lw=2 if lane.open else 1.2,
                                        color="#ffd166" if lane.open else "#ff6b6b",
                                        ls="-" if lane.open else (0, (3, 3)), alpha=0.95 if lane.open else 0.7,
                                        shrinkA=6, shrinkB=8))

    for d in a.players() + [x for x in a.detections if x.role in ("referee", "official") and x.pitch_xy]:
        x, y = d.pitch_xy
        if d.role in ("referee", "official"):
            ax.scatter(x, y, s=110, c="#222222", edgecolors="#f1c40f", linewidths=1.5, zorder=6, marker="D")
            continue
        col = _hex(team_rgb(a, d.team))
        ax.scatter(x, y, s=260 if d.role == "goalkeeper" else 200, c=col, edgecolors="white",
                   linewidths=2.2 if d is a.carrier else 1.2, zorder=6,
                   marker="s" if d.role == "goalkeeper" else "o")
    if a.ball is not None and a.ball.pitch_xy is not None:
        ax.scatter(*a.ball.pitch_xy, s=60, c="white", edgecolors="black", linewidths=1.2, zorder=7)

    ax.set_xlim(-3, P.length + 3)
    ax.set_ylim(P.width + 3, -3)
    ax.set_aspect("equal")
    ax.axis("off")

    lines = []
    for t, s in sorted(a.shapes.items()):
        side = "attacks →" if s.defends_left else "← attacks"
        bits = [f"Team {t + 1} ({side})", f"{s.n_players} seen"]
        if s.shape:
            bits.append(s.shape)
        bits += [f"width {s.width_m:.0f} m", f"depth {s.depth_m:.0f} m"]
        if s.block_height_m is not None:
            bits.append(f"last line {s.block_height_m:.0f} m from goal")
        if t in a.control_share:
            bits.append(f"{a.control_share[t]:.0%} of visible space")
        lines.append(("  ·  ".join(bits), _hex(team_rgb(a, t))))
    if a.carrier is not None:
        n_open = sum(lane.open for lane in a.lanes)
        lines.append((f"Ball carrier: Team {a.carrier.team + 1}  ·  {n_open} of {len(a.lanes)} passing lanes open",
                      "#ffd166"))
    top = 0.965
    if title:
        fig.text(0.02, top, title, color="white", fontsize=14, fontweight="bold", va="top")
        top -= 0.045
    for txt, col in lines:
        fig.text(0.02, top, txt, color=col, fontsize=10.5, fontweight="bold", va="top")
        top -= 0.034
    fig.subplots_adjust(left=0.01, right=0.99, top=top - 0.01, bottom=0.01)
    if path:
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig
