"""Make a fake broadcast frame with a known camera, for tests and the demo.

The pitch is drawn from a known pitch->image homography, and players are
simple coloured figures standing at known pitch positions. Because we know
the truth, tests can check that detections land in the right place.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.detect import Detection
from pitchlens.homography import Correspondence, apply_homography
from pitchlens.pitch import DEFAULT_PITCH

W, H = 1600, 900

# Pitch points (x, y in metres) and where a camera on the near side, looking at
# the left half, would see them.
_SRC = np.array([(0, 0), (60, 0), (60, 68), (0, 68)], dtype=np.float32)
_DST = np.array([(250, 220), (1450, 230), (1700, 900), (-300, 880)], dtype=np.float32)
PITCH_TO_IMAGE = cv2.getPerspectiveTransform(_SRC, _DST)
IMAGE_TO_PITCH = np.linalg.inv(PITCH_TO_IMAGE)

TEAM_A = (220, 30, 40)    # red shirts (RGB)
TEAM_B = (240, 240, 245)  # white shirts
REFEREE = (20, 20, 20)    # black
KEEPER_A = (250, 220, 0)  # yellow

KEEPER_A_POS = (8, 34)
PLAYERS_A = [(22, 15), (20, 28), (21, 41), (23, 54), (35, 12), (33, 30), (34, 44), (36, 58),
             (46, 26), (47, 42)]
PLAYERS_B = [(30, 20), (28, 36), (31, 50), (40, 18), (42, 34), (41, 52), (50, 8), (52, 60), (49, 34)]
REF_POS = (38, 40)


def frame(ball_at: tuple[float, float] | None = (35.5, 44.5)) -> tuple[np.ndarray, list[Detection]]:
    """Return (BGR image, ground-truth detections with pitch_xy set)."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 40, 40)  # stands
    # Grass: fill the projected pitch polygon (plus a run-off) with stripes.
    corners = apply_homography(PITCH_TO_IMAGE, [(-3, -3), (63, -3), (63, 71), (-3, 71)]).astype(np.int32)
    grass = np.zeros((H, W), np.uint8)
    cv2.fillPoly(grass, [corners], 255)
    stripes = np.zeros_like(img)
    for i in range(12):
        x0, x1 = i * 5, (i + 1) * 5
        poly = apply_homography(PITCH_TO_IMAGE, [(x0, -3), (x1, -3), (x1, 71), (x0, 71)]).astype(np.int32)
        cv2.fillPoly(stripes, [poly], (60, 140, 60) if i % 2 else (50, 125, 50))
    img[grass > 0] = stripes[grass > 0]
    for seg in DEFAULT_PITCH.segments():
        pts = apply_homography(PITCH_TO_IMAGE, seg).astype(np.int32)
        cv2.polylines(img, [pts], False, (235, 235, 235), 3, cv2.LINE_AA)

    dets: list[Detection] = []
    people = ([(p, TEAM_A, "player", 0) for p in PLAYERS_A] + [(p, TEAM_B, "player", 1) for p in PLAYERS_B]
              + [(KEEPER_A_POS, KEEPER_A, "goalkeeper", 0), (REF_POS, REFEREE, "referee", None)])
    # Draw far players first so near ones overlap them, like in real footage.
    for (px, py), rgb, role, team in sorted(people, key=lambda t: t[0][1]):
        fx, fy = apply_homography(PITCH_TO_IMAGE, [(px, py)])[0]
        h = 30 + 70 * fy / H        # nearer = bigger
        w = h * 0.38
        x1, y1, x2, y2 = fx - w / 2, fy - h, fx + w / 2, fy
        bgr = tuple(int(c) for c in rgb[::-1])
        cv2.rectangle(img, (int(x1 + w * 0.2), int(y1 + h * 0.5)), (int(x2 - w * 0.2), int(y2)), (30, 30, 30), -1)
        cv2.rectangle(img, (int(x1), int(y1 + h * 0.12)), (int(x2), int(y1 + h * 0.55)), bgr, -1)
        cv2.circle(img, (int(fx), int(y1 + h * 0.07)), max(2, int(h * 0.07)), (150, 180, 220), -1)
        dets.append(Detection(box=(x1, y1, x2, y2), conf=0.9, role=role, team=team, pitch_xy=(px, py)))
    if ball_at is not None:
        bx, by = apply_homography(PITCH_TO_IMAGE, [ball_at])[0]
        cv2.circle(img, (int(bx), int(by - 4)), 5, (255, 255, 255), -1)
        dets.append(Detection(box=(bx - 5, by - 9, bx + 5, by + 1), conf=0.5, role="ball", pitch_xy=ball_at))
    return img, dets


def landmarks(names: list[str] | None = None) -> list[Correspondence]:
    names = names or ["corner_far_left", "halfway_far", "halfway_near", "left_box_near_corner",
                      "left_box_far_corner", "centre_circle_left"]
    out = []
    for n in names:
        p = DEFAULT_PITCH.landmarks[n]
        img = apply_homography(PITCH_TO_IMAGE, [p])[0]
        out.append(Correspondence(image=(float(img[0]), float(img[1])), pitch=p, name=n))
    return out


def untagged(dets: list[Detection]) -> list[Detection]:
    """Strip the ground truth, as if the boxes came from a detector."""
    return [Detection(box=d.box, conf=d.conf, role="ball" if d.role == "ball" else "player") for d in dets]
