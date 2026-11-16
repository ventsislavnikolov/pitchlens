"""Pitch model: dimensions, named landmarks and drawing helpers.

Coordinates are in metres. The origin is the corner on the left of the far
touchline as seen from the main broadcast camera:

    x runs along the touchline, 0 (left goal line) .. length (right goal line)
    y runs across the pitch,    0 (far touchline)  .. width (near touchline)

So in a standard wide shot, small y is the top of the image.

The 32 landmarks follow the same order as Roboflow's
``SoccerPitchConfiguration`` so that a keypoint model trained on their
"football-field-detection" dataset can be plugged in directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    goal_width: float = 7.32

    @cached_property
    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named landmarks, in Roboflow keypoint order (1..32)."""
        L, W = self.length, self.width
        pb_y0 = (W - self.penalty_box_width) / 2
        pb_y1 = (W + self.penalty_box_width) / 2
        gb_y0 = (W - self.goal_box_width) / 2
        gb_y1 = (W + self.goal_box_width) / 2
        pbl, gbl = self.penalty_box_length, self.goal_box_length
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        return {
            "corner_far_left": (0, 0),                                  # 1
            "left_box_far_goalline": (0, pb_y0),                        # 2
            "left_six_far_goalline": (0, gb_y0),                        # 3
            "left_six_near_goalline": (0, gb_y1),                       # 4
            "left_box_near_goalline": (0, pb_y1),                       # 5
            "corner_near_left": (0, W),                                 # 6
            "left_six_far_corner": (gbl, gb_y0),                        # 7
            "left_six_near_corner": (gbl, gb_y1),                       # 8
            "left_penalty_spot": (ps, W / 2),                           # 9
            "left_box_far_corner": (pbl, pb_y0),                        # 10
            "left_box_line_far_six": (pbl, gb_y0),                      # 11
            "left_box_line_near_six": (pbl, gb_y1),                     # 12
            "left_box_near_corner": (pbl, pb_y1),                       # 13
            "halfway_far": (L / 2, 0),                                  # 14
            "centre_circle_far": (L / 2, W / 2 - r),                    # 15
            "centre_circle_near": (L / 2, W / 2 + r),                   # 16
            "halfway_near": (L / 2, W),                                 # 17
            "right_box_far_corner": (L - pbl, pb_y0),                   # 18
            "right_box_line_far_six": (L - pbl, gb_y0),                 # 19
            "right_box_line_near_six": (L - pbl, gb_y1),                # 20
            "right_box_near_corner": (L - pbl, pb_y1),                  # 21
            "right_penalty_spot": (L - ps, W / 2),                      # 22
            "right_six_far_corner": (L - gbl, gb_y0),                   # 23
            "right_six_near_corner": (L - gbl, gb_y1),                  # 24
            "corner_far_right": (L, 0),                                 # 25
            "right_box_far_goalline": (L, pb_y0),                       # 26
            "right_six_far_goalline": (L, gb_y0),                       # 27
            "right_six_near_goalline": (L, gb_y1),                      # 28
            "right_box_near_goalline": (L, pb_y1),                      # 29
            "corner_near_right": (L, W),                                # 30
            "centre_circle_left": (L / 2 - r, W / 2),                   # 31
            "centre_circle_right": (L / 2 + r, W / 2),                  # 32
            # Extra landmarks that are easy to click by hand.
            "centre_spot": (L / 2, W / 2),
        }

    @property
    def keypoint_names(self) -> list[str]:
        """The 32 landmark names in keypoint-model order."""
        return list(self.landmarks)[:32]

    def segments(self) -> list[np.ndarray]:
        """Pitch markings as polylines (each an (N, 2) array), for drawing."""
        L, W = self.length, self.width
        segs: list[np.ndarray] = [
            np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]),
            np.array([(L / 2, 0), (L / 2, W)]),
        ]
        for side in (0, 1):
            def fx(x: float, side: int = side) -> float:
                return L - x if side else x

            pb0 = (W - self.penalty_box_width) / 2
            pb1 = (W + self.penalty_box_width) / 2
            gb0 = (W - self.goal_box_width) / 2
            gb1 = (W + self.goal_box_width) / 2
            segs.append(np.array([(fx(0), pb0), (fx(self.penalty_box_length), pb0),
                                  (fx(self.penalty_box_length), pb1), (fx(0), pb1)]))
            segs.append(np.array([(fx(0), gb0), (fx(self.goal_box_length), gb0),
                                  (fx(self.goal_box_length), gb1), (fx(0), gb1)]))
            # Penalty arc: the part of the 9.15 m circle around the spot that is
            # outside the box.
            dx = self.penalty_box_length - self.penalty_spot_distance
            half = np.arccos(dx / self.centre_circle_radius)
            t = np.linspace(-half, half, 30)
            arc = np.stack([self.penalty_spot_distance + self.centre_circle_radius * np.cos(t),
                            W / 2 + self.centre_circle_radius * np.sin(t)], axis=1)
            if side:
                arc[:, 0] = L - arc[:, 0]
            segs.append(arc)
        t = np.linspace(0, 2 * np.pi, 80)
        segs.append(np.stack([L / 2 + self.centre_circle_radius * np.cos(t),
                              W / 2 + self.centre_circle_radius * np.sin(t)], axis=1))
        return segs

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the pitch, expanded by ``margin`` metres."""
        xy = np.atleast_2d(xy)
        return ((xy[:, 0] >= -margin) & (xy[:, 0] <= self.length + margin)
                & (xy[:, 1] >= -margin) & (xy[:, 1] <= self.width + margin))


DEFAULT_PITCH = Pitch()
