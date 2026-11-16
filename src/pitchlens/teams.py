"""Split players into two teams by shirt colour.

For each player we crop the torso, drop grass-coloured pixels, and take the
median colour in CIE Lab (where distance roughly matches how different two
colours look). K-means with k=2 gives the two teams. Players far from both
team colours are flagged as "other": usually the referee or a goalkeeper.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .detect import Detection


def shirt_colour(image_bgr: np.ndarray, box: tuple[float, float, float, float]
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (lab, rgb) median shirt colour for a box, or None if too small."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    # Torso: between 15% and 50% of the height, middle 60% of the width.
    cx1, cx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    cy1, cy2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
    H, W = image_bgr.shape[:2]
    cx1, cx2 = max(cx1, 0), min(cx2, W)
    cy1, cy2 = max(cy1, 0), min(cy2, H)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    pix = crop.reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 60) & (hsv[:, 2] >= 40)
    keep = pix[~grass]
    if len(keep) < max(6, 0.15 * len(pix)):
        keep = pix
    bgr = np.median(keep, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(float)
    return lab, bgr[::-1].astype(int)


def _cluster(Xw: np.ndarray, seed: int = 0, max_misc: int = 6) -> np.ndarray:
    """Label rows 0/1 for the two teams and -1 for odd colours.

    Fits k=3 first: two big clusters are the teams, and a small third cluster
    (at most ``max_misc`` people: two goalkeepers, the referee, assistants
    and a fourth official) collects the people whose
    kits are chosen to differ from both teams. If the third cluster is big,
    a team was split (e.g. sun and shadow), so fall back to k=2. Inside each
    team, a player far from the team colour is also marked -1.
    """
    n = len(Xw)
    labels = None
    if n >= 6:
        km3 = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(Xw)
        sizes = np.bincount(km3.labels_, minlength=3)
        small = int(np.argmin(sizes))
        if sizes[small] <= max_misc and sizes[small] < 0.75 * np.sort(sizes)[1]:
            big = [k for k in range(3) if k != small]
            remap = {big[0]: 0, big[1]: 1, small: -1}
            labels = np.array([remap[k] for k in km3.labels_])
    if labels is None:
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xw).labels_.copy()
    for t in (0, 1):
        m = labels == t
        if m.sum() < 3:
            continue
        centre = np.median(Xw[m], axis=0)
        dist = np.linalg.norm(Xw[m] - centre, axis=1)
        far = dist > np.maximum(18.0, 3.0 * np.median(dist))
        labels[np.nonzero(m)[0][far]] = -1
    return labels


def assign_teams(image_bgr: np.ndarray, dets: list[Detection], seed: int = 0) -> dict:
    """Label ``det.team`` in place for players. Returns team colour info.

    Detections already marked ``referee`` by the model are left as referees.
    Players whose colour is far from both clusters become ``referee`` if they
    are the single most distant, else stay as "other" (team None) so a
    goalkeeper can be attached later by position.
    """
    people = [d for d in dets if d.role in ("player", "goalkeeper")]
    feats, idx = [], []
    for i, d in enumerate(people):
        c = shirt_colour(image_bgr, d.box)
        if c is None:
            continue
        d.colour = tuple(int(v) for v in c[1])
        feats.append(c[0])
        idx.append(i)
    info: dict = {"team_colours": {}, "outliers": 0}
    if len(feats) < 2:
        return info
    X = np.array(feats)
    # Lightness matters a bit less than hue (shadows), so down-weight L.
    Xw = X * np.array([0.7, 1.0, 1.0])
    labels = _cluster(Xw, seed)
    for j, i in enumerate(idx):
        d = people[i]
        if d.role == "goalkeeper":
            d.team = None
        elif labels[j] < 0:
            d.team = None
            d.extra["colour_outlier"] = True
            info["outliers"] += 1
        else:
            d.team = int(labels[j])
    # Refit centres on inliers only, report them as RGB.
    for t in (0, 1):
        members = [people[i] for j, i in enumerate(idx) if people[i].team == t]
        if members:
            rgb = np.median(np.array([m.colour for m in members]), axis=0).astype(int)
            info["team_colours"][t] = tuple(int(v) for v in rgb)
    # Make team 0 the one with more detected players, for stable output.
    counts = [sum(1 for d in people if d.team == t) for t in (0, 1)]
    if counts[1] > counts[0]:
        for d in people:
            if d.team is not None:
                d.team = 1 - d.team
        info["team_colours"] = {1 - k: v for k, v in info["team_colours"].items()}
    return info


def resolve_others(dets: list[Detection], left_team: int | None = None) -> None:
    """Decide what the colour outliers are, using pitch positions.

    An outlier standing near a goal and deeper than everyone else becomes the
    goalkeeper of the team defending that goal. Any other outlier is the referee.
    Requires ``pitch_xy`` to be set.
    """
    from .pitch import DEFAULT_PITCH

    L = DEFAULT_PITCH.length
    teams = {t: [d for d in dets if d.team == t and d.pitch_xy is not None] for t in (0, 1)}
    for d in dets:
        if d.role not in ("player", "goalkeeper") or d.team is not None or d.pitch_xy is None:
            continue
        x = d.pitch_xy[0]
        near_goal = x < 22 or x > L - 22
        best = None
        if near_goal:
            # The keeper belongs to the team defending that goal. If the caller
            # knows which team defends the left goal, use it. Otherwise guess the
            # team that sits deeper towards that goal (deepest player + average
            # position), which is wrong when one team presses very high.
            left = x < L / 2
            if left_team is not None:
                best = left_team if left else 1 - left_team
            else:
                score = {}
                for t, members in teams.items():
                    xs = np.array([m.pitch_xy[0] for m in members])
                    if len(xs) and ((left and x <= xs.min()) or (not left and x >= xs.max())):
                        own = xs if left else L - xs
                        score[t] = own.min() + own.mean()
                if score:
                    best = min(score, key=score.get)
        if best is not None:
            d.team = best
            d.role = "goalkeeper"
        elif d.role != "goalkeeper":
            d.role = "referee"
