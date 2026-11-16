"""Web demo: upload a frame, click pitch landmarks, get the 2D map.

Run with ``python app.py`` and open http://localhost:7860.
"""

from __future__ import annotations

import json
import tempfile

import cv2
import gradio as gr
import numpy as np

from pitchlens import homography
from pitchlens.detect import Detector
from pitchlens.homography import calibration_summary
from pitchlens.pipeline import analyse
from pitchlens.pitch import DEFAULT_PITCH
from pitchlens.render import annotate_frame, draw_pitch_map

LANDMARKS = list(DEFAULT_PITCH.landmarks)
_detector: Detector | None = None


def detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def _preview(image: np.ndarray | None, points: list[dict]) -> np.ndarray | None:
    if image is None:
        return None
    out = image.copy()
    r = max(4, image.shape[1] // 300)
    for i, p in enumerate(points):
        x, y = map(int, p["image"])
        cv2.circle(out, (x, y), r, (255, 230, 0), -1)
        cv2.putText(out, f"{i + 1} {p['name']}", (x + r + 2, y - r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + image.shape[1] / 4000, (255, 230, 0), max(1, r // 3), cv2.LINE_AA)
    return out


def _table(points: list[dict]) -> list[list]:
    return [[i + 1, p["name"], round(p["image"][0]), round(p["image"][1])] for i, p in enumerate(points)]


def on_upload(image):
    return image, [], _table([]), None, None, None, ""


def on_click(original, points, landmark, evt: gr.SelectData):
    if original is None:
        return gr.update(), points, _table(points), landmark
    x, y = evt.index
    # Clicking the same landmark again moves it.
    points = [p for p in points if p["name"] != landmark] + [{"name": landmark, "image": [float(x), float(y)]}]
    return _preview(original, points), points, _table(points), landmark


def on_undo(original, points):
    points = points[:-1]
    return _preview(original, points), points, _table(points)


def on_load_calib(original, file):
    if file is None:
        return gr.update(), [], _table([])
    data = json.loads(open(file).read())
    items = data["points"] if isinstance(data, dict) else data
    points = [{"name": p["name"], "image": list(p["image"])} for p in items if p.get("name") in LANDMARKS]
    return _preview(original, points), points, _table(points)


def on_analyse(original, points, left="Guess"):
    if original is None:
        raise gr.Error("Upload a frame first.")
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cal = None
    msg = []
    if len(points) >= 4:
        try:
            cal = homography.fit(homography.parse_points({"points": points}))
            msg.append(calibration_summary(cal))
        except ValueError as e:
            msg.append(f"Calibration failed: {e}")
    else:
        msg.append("Click at least 4 landmarks to get the pitch map. Showing detections only.")
    left_team = {"Team 1": 0, "Team 2": 1}.get(left)
    a = analyse(bgr, calibration=cal, detector=detector(), left_team=left_team)
    annotated = cv2.cvtColor(annotate_frame(bgr, a), cv2.COLOR_BGR2RGB)
    pitch_img = None
    if cal is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        draw_pitch_map(a, tmp)
        pitch_img = cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2RGB)
        for t, s in sorted(a.shapes.items()):
            msg.append(f"Team {t + 1}: {s.n_players} players seen, width {s.width_m:.0f} m, depth {s.depth_m:.0f} m"
                       + (f", shape {s.shape}" if s.shape else ""))
    result = a.to_dict()
    calib_json = json.dumps({"points": points}, indent=2)
    return annotated, pitch_img, result, "\n".join(msg), calib_json


with gr.Blocks(title="pitchlens") as demo:
    gr.Markdown(
        "# pitchlens\n"
        "Upload a broadcast frame. Pick a landmark in the list, then click where it is in the image. "
        "Four or more landmarks spread across the frame give the best map (for example both ends of the "
        "halfway line plus two points of the centre circle or a penalty box corner). Then press **Analyse**."
    )
    original = gr.State(None)
    points = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.Image(label="Frame", type="numpy", sources=["upload", "clipboard"])
            canvas = gr.Image(label="Click landmarks here", type="numpy", interactive=False)
        with gr.Column(scale=1):
            landmark = gr.Dropdown(LANDMARKS, value="halfway_far", label="Landmark to place")
            table = gr.Dataframe(headers=["#", "landmark", "x", "y"], label="Clicked landmarks",
                                 interactive=False)
            with gr.Row():
                undo = gr.Button("Undo last")
                run = gr.Button("Analyse", variant="primary")
            left = gr.Radio(["Guess", "Team 1", "Team 2"], value="Guess", label="Team defending the left goal")
            calib_file = gr.File(label="…or load a calibration JSON", file_types=[".json"])
            status = gr.Textbox(label="Summary", lines=4)
    pitch = gr.Image(label="2D pitch map")
    annotated = gr.Image(label="Detections + projected pitch lines (magenta should sit on the painted lines)")
    with gr.Accordion("JSON output", open=False):
        result = gr.JSON()
        calib_out = gr.Code(label="Calibration (save as calib.json for the CLI)", language="json")

    upload.change(on_upload, upload, [original, points, table, annotated, pitch, result, status]).then(
        lambda im: im, original, canvas)
    canvas.select(on_click, [original, points, landmark], [canvas, points, table, landmark])
    undo.click(on_undo, [original, points], [canvas, points, table])
    calib_file.change(on_load_calib, [original, calib_file], [canvas, points, table])
    run.click(on_analyse, [original, points, left], [annotated, pitch, result, status, calib_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
