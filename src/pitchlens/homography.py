"""Image <-> pitch homography from landmark correspondences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .pitch import DEFAULT_PITCH, Pitch


@dataclass
class Correspondence:
    image: tuple[float, float]
    pitch: tuple[float, float]
    name: str | None = None


@dataclass
class Calibration:
    """A fitted image->pitch homography plus the points it came from."""

    H: np.ndarray                      # 3x3, maps image pixels -> pitch metres
    points: list[Correspondence] = field(default_factory=list)
    inliers: np.ndarray | None = None
    source: str = "manual"

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_pitch(self, pts: np.ndarray) -> np.ndarray:
        return apply_homography(self.H, pts)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        return apply_homography(self.H_inv, pts)

    def reprojection_error(self) -> float:
        """Mean distance in metres between clicked landmarks and their mapped position."""
        if not self.points:
            return float("nan")
        img = np.array([p.image for p in self.points], dtype=float)
        ref = np.array([p.pitch for p in self.points], dtype=float)
        err = np.linalg.norm(self.to_pitch(img) - ref, axis=1)
        if self.inliers is not None and self.inliers.any():
            err = err[self.inliers]
        return float(err.mean())

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "H": self.H.tolist(),
            "reprojection_error_m": round(self.reprojection_error(), 3),
            "points": [
                {"name": p.name, "image": list(map(float, p.image)), "pitch": list(map(float, p.pitch))}
                for p in self.points
            ],
        }


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


def fit(points: list[Correspondence], ransac_m: float = 0.6) -> Calibration:
    """Fit a homography from at least 4 image<->pitch correspondences.

    ``ransac_m`` is the RANSAC threshold in pitch metres (the fit maps image
    to pitch), so 0.6 tolerates a click that is ~60 cm off.
    """
    if len(points) < 4:
        raise ValueError(f"Need at least 4 landmarks to fit a homography, got {len(points)}.")
    src = np.array([p.image for p in points], dtype=np.float64)
    dst = np.array([p.pitch for p in points], dtype=np.float64)
    if len(points) == 4:
        H, mask = cv2.findHomography(src, dst, 0)
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_m)
    if H is None:
        raise ValueError("Could not fit a homography: the landmarks may be collinear.")
    inliers = mask.ravel().astype(bool) if mask is not None else None
    return Calibration(H=H, points=list(points), inliers=inliers)


def parse_points(data: dict | list, pitch: Pitch = DEFAULT_PITCH) -> list[Correspondence]:
    """Parse a calibration JSON.

    Accepts ``{"points": [...]}`` or a bare list. Each point has ``image: [x, y]``
    and either ``name`` (a landmark from :class:`Pitch`) or ``pitch: [x, y]``.
    """
    items = data["points"] if isinstance(data, dict) else data
    out = []
    for item in items:
        name = item.get("name")
        if "pitch" in item and item["pitch"] is not None:
            pxy = tuple(item["pitch"])
        elif name in pitch.landmarks:
            pxy = pitch.landmarks[name]
        else:
            known = ", ".join(pitch.landmarks)
            raise ValueError(f"Unknown landmark {name!r}. Use one of: {known}, or give 'pitch': [x, y].")
        out.append(Correspondence(image=tuple(item["image"]), pitch=pxy, name=name))
    return out


def load(path: str | Path, pitch: Pitch = DEFAULT_PITCH) -> Calibration:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "H" in data and not data.get("points"):
        return Calibration(H=np.array(data["H"], dtype=float), source="file")
    return fit(parse_points(data, pitch))


def from_keypoints(xy: np.ndarray, conf: np.ndarray, pitch: Pitch = DEFAULT_PITCH,
                   min_conf: float = 0.5) -> Calibration:
    """Fit from a 32-keypoint pitch model output (Roboflow order)."""
    names = pitch.keypoint_names
    pts = [Correspondence(image=(float(x), float(y)), pitch=pitch.landmarks[n], name=n)
           for (x, y), c, n in zip(xy, conf, names) if c >= min_conf and x > 1 and y > 1]
    cal = fit(pts)
    cal.source = "keypoint-model"
    return cal


def calibration_summary(cal: Calibration) -> str:
    """One line on fit quality, naming any landmarks RANSAC ignored."""
    n = len(cal.points)
    msg = f"Calibration error {cal.reprojection_error():.2f} m"
    if cal.inliers is not None and not cal.inliers.all():
        bad = [p.name or f"#{i + 1}" for i, (p, ok) in enumerate(zip(cal.points, cal.inliers)) if not ok]
        msg += (f" over {int(cal.inliers.sum())} of {n} landmarks. Ignored as inconsistent: "
                f"{', '.join(bad)}. Check those clicks.")
    else:
        msg += f" over {n} landmarks."
    return msg
