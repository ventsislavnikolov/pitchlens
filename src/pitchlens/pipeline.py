"""End-to-end analysis of one broadcast frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tactics
from .detect import Detection, Detector, best_ball, filter_on_pitch, grass_mask
from .homography import Calibration
from .pitch import DEFAULT_PITCH, Pitch
from .teams import assign_teams, resolve_others


@dataclass
class Analysis:
    detections: list[Detection]
    ball: Detection | None
    calibration: Calibration | None
    team_info: dict
    shapes: dict[int, tactics.TeamShape] = field(default_factory=dict)
    lanes: list[tactics.PassingLane] = field(default_factory=list)
    carrier: Detection | None = None
    control_grid: np.ndarray | None = None
    visible_grid: np.ndarray | None = None
    control_share: dict[int, float] = field(default_factory=dict)
    pitch: Pitch = DEFAULT_PITCH

    def players(self, team: int | None = None) -> list[Detection]:
        ps = [d for d in self.detections if d.role in ("player", "goalkeeper") and d.pitch_xy is not None]
        return ps if team is None else [d for d in ps if d.team == team]

    def to_dict(self) -> dict:
        def det(d: Detection) -> dict:
            return {
                "role": d.role,
                "team": d.team,
                "conf": round(d.conf, 3),
                "box": [round(v, 1) for v in d.box],
                "shirt_rgb": list(d.colour) if d.colour else None,
                "pitch_xy": [round(v, 2) for v in d.pitch_xy] if d.pitch_xy else None,
            }

        out = {
            "pitch": {"length": self.pitch.length, "width": self.pitch.width},
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "team_colours_rgb": {str(k): list(v) for k, v in self.team_info.get("team_colours", {}).items()},
            "detections": [det(d) for d in self.detections],
            "ball": det(self.ball) if self.ball else None,
        }
        if self.shapes:
            out["teams"] = {}
            for t, s in self.shapes.items():
                sd = tactics.to_dict(s)
                sd.pop("hull")
                out["teams"][str(t)] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sd.items()}
        if self.carrier is not None:
            out["ball_carrier"] = det(self.carrier)
            out["passing_lanes"] = [
                {**tactics.to_dict(lane), "to": det(self.players(self.carrier.team)[lane.to_idx])}
                for lane in self.lanes
            ]
        if self.control_share:
            out["space_control_visible_share"] = {str(k): round(v, 3) for k, v in self.control_share.items()}
        return _clean(out)


def _clean(o):
    """Turn numpy scalars and tuples into plain JSON-friendly Python values."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def analyse(image_bgr: np.ndarray, calibration: Calibration | None = None,
            detector: Detector | None = None, detections: list[Detection] | None = None,
            pitch: Pitch = DEFAULT_PITCH, carrier_max_dist: float = 3.0,
            left_team: int | None = None) -> Analysis:
    """Run detection, team split, projection and tactics on one frame.

    Pass ``detections`` to skip the detector (useful for tests or when boxes
    come from another model). ``left_team`` (0 or 1) says which team defends
    the x=0 goal; without it, it is guessed from positions.
    """
    mask = grass_mask(image_bgr)
    if detections is None:
        detector = detector or Detector()
        detections = detector(image_bgr)
    dets = filter_on_pitch(detections, mask)
    team_info = assign_teams(image_bgr, dets)

    if calibration is not None:
        feet = np.array([d.foot for d in dets]) if dets else np.zeros((0, 2))
        proj = calibration.to_pitch(feet)
        kept = []
        for d, xy in zip(dets, proj):
            if pitch.contains(xy, margin=3.0)[0]:
                d.pitch_xy = (float(xy[0]), float(xy[1]))
                # People standing just off the pitch are assistant referees or
                # staff, not players.
                if d.role in ("player", "goalkeeper") and not pitch.contains(xy, margin=0.75)[0]:
                    d.role, d.team = "official", None
                kept.append(d)
        dets = kept
        resolve_others(dets, left_team)

    ball = best_ball(dets)
    a = Analysis(detections=dets, ball=ball, calibration=calibration, team_info=team_info, pitch=pitch)
    if calibration is None:
        return a

    xy = {t: np.array([d.pitch_xy for d in a.players(t)]).reshape(-1, 2) for t in (0, 1)}
    gk = {t: np.array([d.role == "goalkeeper" for d in a.players(t)], bool) for t in (0, 1)}
    if left_team is not None:
        team0_left = left_team == 0
    else:
        team0_left = tactics.infer_sides(xy[0], xy[1], gk[0], gk[1], pitch)
    for t in (0, 1):
        if len(xy[t]):
            a.shapes[t] = tactics.team_shape(xy[t], t, defends_left=team0_left if t == 0 else not team0_left,
                                             gk_mask=gk[t], pitch=pitch)

    # Ball carrier: nearest player to the ball, if close enough.
    if ball is not None and ball.pitch_xy is not None:
        everyone = [d for d in a.players() if d.team is not None]
        if everyone:
            dist = [np.hypot(d.pitch_xy[0] - ball.pitch_xy[0], d.pitch_xy[1] - ball.pitch_xy[1])
                    for d in everyone]
            i = int(np.argmin(dist))
            if dist[i] <= carrier_max_dist:
                a.carrier = everyone[i]
    if a.carrier is not None:
        t = a.carrier.team
        mates = a.players(t)
        a.lanes = tactics.passing_lanes(mates.index(a.carrier), xy[t], xy[1 - t])

    # Space control, only over the part of the pitch the camera can see.
    pts = np.vstack([xy[0], xy[1]])
    if len(xy[0]) and len(xy[1]):
        teams = np.array([0] * len(xy[0]) + [1] * len(xy[1]))
        owner, _ = tactics.space_control(pts, teams, pitch)
        ys, xs = np.mgrid[0:owner.shape[0], 0:owner.shape[1]]
        centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        img_pts = calibration.to_image(centres)
        h, w = mask.shape
        inside = ((img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h))
        vis = np.zeros(len(centres), bool)
        ii = np.nonzero(inside)[0]
        vis[ii] = mask[img_pts[ii, 1].astype(int), img_pts[ii, 0].astype(int)] > 0
        vis = vis.reshape(owner.shape)
        a.control_grid, a.visible_grid = owner, vis
        if vis.any():
            a.control_share = {t: float((owner[vis] == t).mean()) for t in (0, 1)}
    return a
