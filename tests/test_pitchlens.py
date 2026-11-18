import json

import numpy as np
import pytest

from pitchlens import homography, tactics
from pitchlens.pipeline import analyse
from pitchlens.pitch import DEFAULT_PITCH
from pitchlens.render import annotate_frame, draw_pitch_map

from . import synthetic


def test_landmarks_follow_roboflow_order_and_are_on_pitch():
    names = DEFAULT_PITCH.keypoint_names
    assert len(names) == 32
    assert names[0] == "corner_far_left" and names[13] == "halfway_far" and names[31] == "centre_circle_right"
    pts = np.array(list(DEFAULT_PITCH.landmarks.values()))
    assert DEFAULT_PITCH.contains(pts).all()


def test_homography_recovers_known_camera():
    cal = homography.fit(synthetic.landmarks())
    assert cal.reprojection_error() < 1e-3
    probe = np.array([(10, 10), (30, 50), (55, 34)], float)
    img = homography.apply_homography(synthetic.PITCH_TO_IMAGE, probe)
    np.testing.assert_allclose(cal.to_pitch(img), probe, atol=1e-3)
    np.testing.assert_allclose(cal.to_image(probe), img, atol=1e-2)


def test_homography_ransac_rejects_a_bad_click():
    pts = synthetic.landmarks()
    bad = pts[2]
    pts[2] = homography.Correspondence(image=(bad.image[0] + 80, bad.image[1] - 60), pitch=bad.pitch, name=bad.name)
    cal = homography.fit(pts)
    assert not cal.inliers[2]
    assert cal.inliers.sum() == len(pts) - 1


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        homography.fit(synthetic.landmarks()[:3])


def test_calibration_json_roundtrip(tmp_path):
    pts = synthetic.landmarks()
    f = tmp_path / "calib.json"
    f.write_text(json.dumps({"points": [{"name": p.name, "image": list(p.image)} for p in pts]}))
    cal = homography.load(f)
    assert cal.reprojection_error() < 1e-3
    with pytest.raises(ValueError, match="Unknown landmark"):
        homography.parse_points([{"name": "nope", "image": [0, 0]}])


def test_split_lines_finds_a_back_four_and_midfield():
    x = np.array([20, 21, 22, 20.5, 35, 36, 34, 35.5, 48, 49], float)
    lines = tactics.split_lines(x)
    assert [len(g) for g in lines] == [4, 4, 2]


def test_passing_lane_blocked_by_opponent_on_the_line():
    team = np.array([(10, 30), (30, 30), (20, 50)], float)
    opp = np.array([(20, 30.5)])
    lanes = {lane.to_idx: lane for lane in tactics.passing_lanes(0, team, opp)}
    assert not lanes[1].open
    assert lanes[2].open


def test_space_control_shares_sum_to_one():
    pts = np.array([(32.5, 34), (72.5, 34)], float)
    owner, share = tactics.space_control(pts, np.array([0, 1]))
    assert owner.shape == (68, 105)
    assert abs(sum(share.values()) - 1) < 1e-9
    assert abs(share[0] - 0.5) < 0.02


def test_full_pipeline_on_synthetic_frame(tmp_path):
    img, truth = synthetic.frame()
    cal = homography.fit(synthetic.landmarks())
    a = analyse(img, calibration=cal, detections=synthetic.untagged(truth))

    # Every player lands within 0.5 m of where we put them.
    by_box = {d.box: d for d in truth}
    for d in a.detections:
        t = by_box[d.box]
        assert np.hypot(d.pitch_xy[0] - t.pitch_xy[0], d.pitch_xy[1] - t.pitch_xy[1]) < 0.5

    # Teams split by shirt colour exactly (cluster ids are arbitrary, so compare partitions).
    true_team = {d.box: d.team for d in truth if d.role in ("player", "goalkeeper")}
    pairs = {(true_team[d.box], d.team) for d in a.detections
             if d.box in true_team and d.role in ("player", "goalkeeper")}
    assert len(pairs) == 2 and len({p[1] for p in pairs}) == 2

    # The referee is not put on a team, and the keeper is found by position.
    ref = next(d for d in a.detections if by_box[d.box].role == "referee")
    assert ref.team is None and ref.role == "referee"
    red = next(d.team for d in a.detections if d.box in true_team and true_team[d.box] == 0)
    assert a.shapes[red].defends_left
    keeper = next(d for d in a.detections if by_box[d.box].role == "goalkeeper")
    assert keeper.role == "goalkeeper" and keeper.team == red
    assert a.shapes[red].shape == "4-4-2"

    # Ball carrier is the red player at (34, 44); some lanes open, some not.
    assert a.carrier is not None and a.carrier.pitch_xy == pytest.approx((34, 44), abs=0.5)
    assert any(lane.open for lane in a.lanes) and not all(lane.open for lane in a.lanes)
    assert 0.99 < sum(a.control_share.values()) < 1.01

    out = a.to_dict()
    json.dumps(out)  # must be serialisable
    assert out["teams"][str(red)]["shape"] == "4-4-2"

    draw_pitch_map(a, str(tmp_path / "pitch.png"), title="test")
    assert (tmp_path / "pitch.png").stat().st_size > 10_000
    assert annotate_frame(img, a).shape == img.shape


def test_pipeline_without_calibration_still_detects():
    img, truth = synthetic.frame()
    a = analyse(img, calibration=None, detections=synthetic.untagged(truth))
    assert len(a.detections) == len(truth)
    assert a.shapes == {}
