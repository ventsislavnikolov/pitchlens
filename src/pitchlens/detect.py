"""Detect players, referees and the ball in a broadcast frame.

By default this uses a general COCO YOLO model (``person`` + ``sports ball``),
which downloads automatically and needs no training. People in the stands,
on the bench and along the touchline are removed with a grass mask and, when
a calibration is available, by checking they stand on the pitch.

A football-specific model (for example one trained on Roboflow's
``football-players-detection`` dataset with classes ball / goalkeeper /
player / referee) can be passed with ``weights=`` and is used as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_WEIGHTS = "yolo11m.pt"

# Map whatever class names a model uses onto our four roles.
ROLE_ALIASES = {
    "person": "player",
    "player": "player",
    "players": "player",
    "goalkeeper": "goalkeeper",
    "goalkeepers": "goalkeeper",
    "referee": "referee",
    "referees": "referee",
    "sports ball": "ball",
    "ball": "ball",
    "football": "ball",
}


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    conf: float
    role: str                                # player | goalkeeper | referee | ball
    team: int | None = None                  # 0 / 1, or None for referee / unknown
    colour: tuple[int, int, int] | None = None  # mean shirt colour, RGB
    pitch_xy: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the player touches the grass."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def grass_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of the playing surface.

    Takes green pixels, keeps the largest connected area and fills it with its
    convex hull, so players standing on the grass are inside the mask.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    k = max(3, image_bgr.shape[1] // 200)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    mask = np.zeros_like(green)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == biggest)
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


class Detector:
    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str | None = None,
                 imgsz: int | None = None):
        from ultralytics import YOLO  # imported lazily: heavy

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.names = {int(k): v.lower() for k, v in self.model.names.items()}

    def __call__(self, image_bgr: np.ndarray, conf: float = 0.25,
                 ball_conf: float = 0.08) -> list[Detection]:
        # Players and the ball are small in a wide shot, so by default run at the
        # image's own resolution (capped at 1920) instead of shrinking it.
        imgsz = self.imgsz or int(min(1920, max(640, -(-image_bgr.shape[1] // 32) * 32)))
        res = self.model.predict(image_bgr, conf=min(conf, ball_conf), imgsz=imgsz,
                                 device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for box, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
            role = ROLE_ALIASES.get(self.names.get(int(k), ""))
            if role is None:
                continue
            if c < (ball_conf if role == "ball" else conf):
                continue
            out.append(Detection(box=tuple(map(float, box)), conf=float(c), role=role))
        return out


def filter_on_pitch(dets: list[Detection], mask: np.ndarray, min_rel_height: float = 0.35
                    ) -> list[Detection]:
    """Drop detections whose feet are off the grass, plus tiny/huge outliers.

    ``min_rel_height`` removes people much smaller than the median player (crowd
    members that leak through the mask) relative to others at the same image row
    is hard to know, so we use a simple global ratio.
    """
    h, w = mask.shape
    people = [d for d in dets if d.role != "ball"]
    if people:
        med = float(np.median([d.height for d in people]))
    else:
        med = 0.0
    keep = []
    for d in dets:
        fx, fy = d.foot
        xi = int(np.clip(fx, 0, w - 1))
        yi = int(np.clip(fy - 2, 0, h - 1))
        if mask[yi, xi] == 0:
            continue
        if d.role != "ball" and med and not (min_rel_height * med <= d.height <= 3 * med):
            continue
        keep.append(d)
    return keep


def best_ball(dets: list[Detection]) -> Detection | None:
    balls = [d for d in dets if d.role == "ball"]
    return max(balls, key=lambda d: d.conf) if balls else None
