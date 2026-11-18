"""Run the full pipeline on a generated frame, without downloading a model.

    python examples/synthetic_demo.py

Writes docs/synthetic_frame.jpg, docs/synthetic_annotated.jpg and
docs/synthetic_pitch.png. The frame is drawn from a known camera, so the
positions on the map can be checked against the truth.
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlens import homography
from pitchlens.pipeline import analyse
from pitchlens.render import annotate_frame, draw_pitch_map
from tests import synthetic

out = ROOT / "docs"
out.mkdir(exist_ok=True)
img, truth = synthetic.frame()
cal = homography.fit(synthetic.landmarks())
a = analyse(img, calibration=cal, detections=synthetic.untagged(truth))
cv2.imwrite(str(out / "synthetic_frame.jpg"), img)
cv2.imwrite(str(out / "synthetic_annotated.jpg"), annotate_frame(img, a))
draw_pitch_map(a, str(out / "synthetic_pitch.png"), title="Synthetic frame")
(ROOT / "examples" / "synthetic_calib.json").write_text(
    json.dumps({"points": [{"name": p.name, "image": [round(v, 1) for v in p.image]} for p in cal.points]}, indent=2)
)
print(homography.calibration_summary(cal))
for t, s in sorted(a.shapes.items()):
    print(f"Team {t + 1}: {s.n_players} players, shape {s.shape}, lines {[len(x) for x in s.lines]}")
