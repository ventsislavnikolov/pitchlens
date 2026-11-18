"""Command line entry point: ``pitchlens analyse frame.jpg --calib calib.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import homography
from .pitch import DEFAULT_PITCH


def _cmd_analyse(args: argparse.Namespace) -> int:
    from .detect import Detector
    from .pipeline import analyse
    from .render import annotate_frame, draw_pitch_map

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Could not read image {args.image}", file=sys.stderr)
        return 2
    cal = None
    if args.calib:
        cal = homography.load(args.calib)
    elif args.keypoint_weights:
        from ultralytics import YOLO

        res = YOLO(args.keypoint_weights).predict(img, verbose=False)[0]
        kp = res.keypoints
        cal = homography.from_keypoints(kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy())
    det = Detector(weights=args.weights, device=args.device, imgsz=args.imgsz)
    left = None if args.left_team is None else args.left_team - 1
    a = analyse(img, calibration=cal, detector=det, left_team=left)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"{stem}_annotated.jpg"), annotate_frame(img, a))
    result = a.to_dict()
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    print(f"Detected {sum(d.role != 'ball' for d in a.detections)} people, "
          f"ball {'found' if a.ball else 'not found'}.")
    if cal is None:
        print("No calibration given, so no pitch map. Pass --calib (see `pitchlens landmarks`).")
    else:
        draw_pitch_map(a, str(out / f"{stem}_pitch.png"), title=args.title or stem)
        print(homography.calibration_summary(cal))
        for t, s in sorted(a.shapes.items()):
            print(f"Team {t + 1}: {s.n_players} players, width {s.width_m:.1f} m, depth {s.depth_m:.1f} m, "
                  f"lines {[len(x) for x in s.lines]}" + (f", shape {s.shape}" if s.shape else ""))
        if a.carrier is not None:
            print(f"Ball carrier on team {a.carrier.team + 1}: "
                  f"{sum(lane.open for lane in a.lanes)}/{len(a.lanes)} passing lanes open.")
    print(f"Wrote results to {out}/")
    return 0


def _cmd_landmarks(_: argparse.Namespace) -> int:
    for name, (x, y) in DEFAULT_PITCH.landmarks.items():
        print(f"{name:28s} x={x:6.2f}  y={y:6.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pitchlens", description="Map players from a match image onto a 2D pitch.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", aliases=["analyze"], help="Analyse one image.")
    a.add_argument("image", type=Path)
    a.add_argument("--calib", type=Path, help="JSON with image<->pitch landmark points (or a saved H).")
    a.add_argument("--keypoint-weights", help="YOLO pose model with the 32 pitch keypoints (Roboflow order).")
    a.add_argument("--weights", default="yolo11m.pt", help="Detector weights (COCO or football-specific).")
    a.add_argument("--device", default=None)
    a.add_argument("--imgsz", type=int, default=None, help="Inference size (default: image width, max 1920).")
    a.add_argument("--out", default="out")
    a.add_argument("--title")
    a.add_argument("--left-team", type=int, choices=[1, 2],
                   help="Team (as numbered in the output) defending the left goal. Guessed if omitted.")
    a.set_defaults(func=_cmd_analyse)
    lm = sub.add_parser("landmarks", help="List the pitch landmark names usable in a calibration file.")
    lm.set_defaults(func=_cmd_landmarks)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
