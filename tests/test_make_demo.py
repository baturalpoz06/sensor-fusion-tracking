"""Tests of scripts/make_demo.py on short scenes and tiny frames (no full demo is rendered).

The demo draws only recorded outputs of the simulation and the tracker; these tests check the
schedule that picks the frames, the geometry that is drawn, that the scene data equal what the
repository's own entry points produce, and that rendering and encoding work and are
deterministic. Each test docstring names the defect that makes it fail.
"""

import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.stats import chi2

from fusion.experiment import SCENARIOS as SINGLE_TARGET_SCENARIOS
from fusion.experiment import (
    FusionConfig,
    evaluation_masks,
    make_rngs,
    position_rmse,
    run_methods,
    simulate,
)
from fusion.mtt_simulation import make_mtt_rngs, simulate_mtt
from fusion.sensors.camera import CameraModel
from fusion.sensors.radar import RadarModel
from fusion.tracker.multi_target import TrackSnapshot, run_multi_target_tracking
from fusion.tracker.track import TrackStatus

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_demo.py"
TINY = (320, 180)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_script():
    spec = importlib.util.spec_from_file_location("make_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_demo"] = module
    spec.loader.exec_module(module)
    return module


demo = load_script()


@pytest.fixture(scope="module")
def fusion_scene():
    return demo.build_fusion_scene(config=FusionConfig(duration=12.0, burn_in=2.0))


@pytest.fixture(scope="module")
def crossing_scene():
    return demo.build_crossing_scene(duration=20.0, burn_in=2.0)


@pytest.fixture(scope="module")
def outage_scene():
    return demo.build_outage_scene(start=6.0, length=5.0, duration=20.0)


def image_of(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png))


def frame_durations(path: Path) -> list[int]:
    """Duration in ms of every stored frame of a GIF or an animated WebP.

    A GIF is read with Pillow. This Pillow version does not report WebP frame durations, so the
    ANMF chunks of the RIFF file are read directly (a 24-bit little-endian duration at byte 12
    of each chunk).
    """
    if path.suffix == ".gif":
        durations = []
        with Image.open(path) as image:
            for index in range(image.n_frames):
                image.seek(index)
                durations.append(image.info["duration"])
        return durations
    data = path.read_bytes()
    durations, position = [], 12
    while position + 8 <= len(data):
        tag = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        if tag == b"ANMF":
            start = position + 8 + 12
            durations.append(int.from_bytes(data[start : start + 3], "little"))
        position += 8 + size + (size & 1)
    return durations


# --- playback schedule ---------------------------------------------------------------------


def test_frames_follow_the_segments_and_reach_the_last_step():
    """Fails if a stride is not honored inside its segment, a segment does not start where it
    should, or the final step of the run gets no frame."""
    steps = demo.frame_steps(demo.CROSSING_SEGMENTS, 0.1, 600)
    assert steps[0] == 0 and steps[-1] == 600
    assert steps[:3] == [0, 2, 4]  # stride 2 during the acquisition
    assert 40 in steps and 45 in steps  # stride 5 from 4 s (step 40) to 24 s
    assert steps[steps.index(240) + 1] == 242  # stride 2 from 24 s
    assert all(b > a for a, b in zip(steps, steps[1:], strict=False))


def test_every_step_belongs_to_exactly_one_frame_window():
    """Fails if a step (and so a radar scan or camera bearing) is skipped between frames or
    drawn in two frames."""
    steps = demo.frame_steps(demo.CROSSING_SEGMENTS, 0.1, 600)
    covered = [k for i in range(len(steps)) for k in demo.window_steps(steps, i)]
    assert covered == list(range(steps[0], steps[-1] + 1))


def test_segment_stride_and_invalid_stride():
    """Fails if the playback speed is read from the wrong segment, or a stride of zero (an
    endless loop) is accepted."""
    assert demo.segment_stride(demo.CROSSING_SEGMENTS, 1.0) == 2
    assert demo.segment_stride(demo.CROSSING_SEGMENTS, 10.0) == 5
    assert demo.segment_stride(demo.CROSSING_SEGMENTS, 30.0) == 2
    assert demo.segment_stride(demo.CROSSING_SEGMENTS, 99.0) == 5
    with pytest.raises(ValueError, match="stride"):
        demo.frame_steps(((0.0, 1.0, 0),), 0.1, 10)


def test_fade_is_one_when_new_and_zero_at_the_lifetime():
    """Fails if an observation does not fade linearly or becomes visible again after its
    lifetime."""
    assert demo.fade(0.0, 1.5) == 1.0
    assert demo.fade(0.75, 1.5) == pytest.approx(0.5)
    assert demo.fade(1.5, 1.5) == 0.0 and demo.fade(9.0, 1.5) == 0.0


# --- geometry ----------------------------------------------------------------------------------


def test_ellipse_semi_axes_are_the_chi_square_scaled_standard_deviations():
    """Fails if the drawn ellipse is not the stated confidence region: semi-axes must be
    sqrt(chi2(0.95, 2) * eigenvalue), the width along the major axis, and the magnification must
    scale the size only."""
    width, height, angle = demo.ellipse_geometry(np.diag([4.0, 1.0]))
    scale = math.sqrt(chi2.ppf(0.95, df=2))
    assert width == pytest.approx(2.0 * scale * 2.0)
    assert height == pytest.approx(2.0 * scale * 1.0)
    assert angle == pytest.approx(0.0, abs=1e-9)
    big = demo.ellipse_geometry(np.diag([4.0, 1.0]), magnification=10.0)
    assert big[0] == pytest.approx(10 * width) and big[1] == pytest.approx(10 * height)
    narrow = demo.ellipse_geometry(np.diag([4.0, 1.0]), confidence=0.5)
    assert narrow[0] < width


def test_ellipse_angle_follows_the_major_axis_of_a_rotated_covariance():
    """Fails if the angle is not the counter-clockwise direction of the major axis (a 30 degree
    covariance would come out mirrored or perpendicular)."""
    theta = math.radians(30.0)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    cov = rotation @ np.diag([9.0, 1.0]) @ rotation.T
    width, height, angle = demo.ellipse_geometry(cov)
    assert width > height
    assert (angle - 30.0 + 90.0) % 180.0 - 90.0 == pytest.approx(0.0, abs=1e-6)


def test_ellipse_of_a_slightly_indefinite_covariance_has_no_nan():
    """Fails if a tiny negative eigenvalue from rounding gives a NaN axis."""
    width, height, _ = demo.ellipse_geometry(np.array([[1.0, 1.0], [1.0, 1.0 - 1e-15]]))
    assert np.isfinite(width) and np.isfinite(height)


def test_bearing_segments_point_along_the_bearing_for_any_quadrant():
    """Fails if a bearing ray is drawn with swapped sin/cos or mirrored, including a bearing
    beyond +-pi/2 (the target crossing the -x axis)."""
    up = demo.bearing_segment(math.pi / 2, 10.0, 30.0)
    np.testing.assert_allclose(up, [[0.0, 10.0], [0.0, 30.0]], atol=1e-12)
    left = demo.bearing_segment(math.pi, 0.0, 5.0)
    np.testing.assert_allclose(left, [[0.0, 0.0], [-5.0, 0.0]], atol=1e-12)
    wrapped = demo.bearing_segment(-3.0 * math.pi / 4.0, 0.0, math.sqrt(2.0))
    np.testing.assert_allclose(wrapped[1], [-1.0, -1.0], atol=1e-12)


# --- scene data --------------------------------------------------------------------------------


def test_fusion_scene_equals_the_experiments_own_runs(fusion_scene):
    """Fails if the demo's single-target data differ from fusion.experiment: the positions of
    both filters and the radar-step RMSE must be those of run_methods and position_rmse."""
    cfg = fusion_scene.config
    sim = simulate(SINGLE_TARGET_SCENARIOS["near"], cfg, make_rngs(demo.FUSION_SEED))
    positions = run_methods(sim, cfg)
    np.testing.assert_array_equal(fusion_scene.truth, sim.truth)
    np.testing.assert_array_equal(fusion_scene.fused.estimates[:, :2], positions["fusion"])
    np.testing.assert_array_equal(fusion_scene.radar_only.estimates[:, :2], positions["radar_only"])
    mask = evaluation_masks(len(sim.truth), cfg.radar_every, cfg.burn_in_steps)["radar"]
    assert fusion_scene.rmse["fused"] == position_rmse(positions["fusion"], sim.truth, mask)
    radar_only = positions["radar_only"]
    assert fusion_scene.rmse["radar_only"] == position_rmse(radar_only, sim.truth, mask)
    assert fusion_scene.fused.covariances.shape == (len(sim.truth), 4, 4)


def test_fusion_scene_is_deterministic():
    """Fails if two builds with the same seed differ (a hidden random source)."""
    config = FusionConfig(duration=6.0, burn_in=1.0)
    a, b = demo.build_fusion_scene(config=config), demo.build_fusion_scene(config=config)
    np.testing.assert_array_equal(a.fused.estimates, b.fused.estimates)
    np.testing.assert_array_equal(a.radar_z, b.radar_z)


def test_crossing_scene_history_is_the_trackers_history(crossing_scene):
    """Fails if the demo changes the tracker run: ids, status, state and covariance must equal
    run_multi_target_tracking bit for bit, and the camera log must be the tracker's own."""
    cfg = crossing_scene.config
    assert cfg.radar_clutter_rate == 0.0 and cfg.camera_clutter_rate == 0.0
    sim = simulate_mtt(demo.MTT_SCENARIOS["crossing"], cfg, make_mtt_rngs(demo.CROSSING_SEED))
    reference = run_multi_target_tracking(
        sim, cfg.tracker_config(), RadarModel(cfg.radar_range_std, cfg.radar_bearing_std),
        CameraModel(cfg.camera_bearing_std),
    )
    history = crossing_scene.run.history
    assert len(history) == len(reference.history)
    for mine, theirs in zip(history, reference.history, strict=True):
        assert [(s.track_id, s.status) for s in mine] == [(s.track_id, s.status) for s in theirs]
        for a, b in zip(mine, theirs, strict=True):
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.P, b.P)
    assert crossing_scene.run.camera_log == list(reference.tracker.camera_log)


def test_crossing_scene_facts_come_from_the_truth(crossing_scene):
    """Fails if the closest approach of targets 0 and 1 is not the minimum distance of their true
    positions, or the evaluation match has the wrong shape."""
    truth = crossing_scene.sim.truth
    separation = np.hypot(*(truth[0, :, :2] - truth[1, :, :2]).T)
    assert crossing_scene.closest == (float(separation.min()), float(separation.argmin() * 0.1))
    assert crossing_scene.matched_ids.shape == truth.shape[:2]


def test_outage_scene_applies_the_outage_to_the_data_both_policies_track(outage_scene):
    """Fails if the radar is not silent exactly in [start, start + length), the silenced scans
    still hold detections, or the two policies are not run on the same simulation."""
    cfg = outage_scene.config
    down = outage_scene.radar_down
    times = np.arange(len(down)) * cfg.dt
    expected = (times >= 6.0 - 1e-9) & (times < 11.0 - 1e-9)
    np.testing.assert_array_equal(down, expected)
    scheduled = [k for k, scan in enumerate(outage_scene.sim.radar_scans) if scan is not None]
    assert all(len(outage_scene.sim.radar_scans[k].z) == 0 for k in scheduled if down[k])
    assert sum(1 for k in scheduled if down[k]) == 5
    camera_counts = [len(scan.z) for scan in outage_scene.sim.camera_scans[60:110]]
    assert max(camera_counts) > 0  # the camera keeps delivering during the outage
    assert set(outage_scene.runs) == {"unaware", "aware"}
    assert all(len(r.history) == len(down) for r in outage_scene.runs.values())


# --- recorded tracks -----------------------------------------------------------------------------


def test_confirmed_series_keeps_only_confirmed_snapshots_with_their_covariance():
    """Fails if tentative snapshots enter the series, or the major axis is not the square root
    of the largest eigenvalue of P[:2, :2]."""
    def snap(track_id, status, x, var_x, var_y):
        return TrackSnapshot(track_id, status, np.array([x, 0.0, 0.0, 0.0]),
                             np.diag([var_x, var_y, 1.0, 1.0]))

    history = [
        [snap(7, TrackStatus.TENTATIVE, 0.0, 4.0, 1.0)],
        [snap(7, TrackStatus.CONFIRMED, 1.0, 4.0, 9.0)],
        [snap(7, TrackStatus.CONFIRMED, 2.0, 1.0, 1.0), snap(8, TrackStatus.TENTATIVE, 5.0, 1, 1)],
    ]
    series = demo.confirmed_series(history)
    assert set(series) == {7}
    np.testing.assert_array_equal(series[7].steps, [1, 2])
    np.testing.assert_allclose(series[7].xy[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(series[7].major, [3.0, 1.0])


def test_id_colors_are_stable_per_id_and_cycle_beyond_the_palette():
    """Fails if one id gets two colors across recordings, or more ids than colors crash."""
    first = {5: None, 2: None}
    second = {2: None, 9: None}
    colors = demo.id_colors(first, second)
    assert list(colors) == [5, 2, 9]
    assert colors[5] == demo.ID_COLORS[0] and colors[2] == demo.ID_COLORS[1]
    many = demo.id_colors({i: None for i in range(len(demo.ID_COLORS) + 3)})
    assert many[len(demo.ID_COLORS)] == demo.ID_COLORS[0]


# --- rendering and encoding ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fusion", "crossing", "outage"])
def test_each_scene_renders_a_png_of_the_requested_size_deterministically(
    name, fusion_scene, crossing_scene, outage_scene
):
    """Fails if a scene cannot be drawn at all (including the frame during the outage and after
    the radar returns), the size is not exact, or two renders of the same frame differ."""
    scene = {"fusion": fusion_scene, "crossing": crossing_scene, "outage": outage_scene}[name]
    segments, last, _ = demo.scene_inputs(name, scene)
    steps = demo.frame_steps(segments, scene.config.dt, last)
    picks = sorted({0, len(steps) // 2, len(steps) - 1})
    first, _ = demo.render_scene(name, scene, TINY, demo.FPS, only=picks)
    second, _ = demo.render_scene(name, scene, TINY, demo.FPS, only=picks)
    assert first == second
    for png in first:
        assert png.startswith(PNG_SIGNATURE)
        assert image_of(png).size == TINY


def test_the_outage_banner_appears_only_while_the_radar_is_down(outage_scene):
    """Fails if frames inside and outside the outage look the same (banner missing or stuck)."""
    segments, last, _ = demo.scene_inputs("outage", outage_scene)
    steps = demo.frame_steps(segments, outage_scene.config.dt, last)
    inside = next(i for i, k in enumerate(steps) if k * 0.1 > 8.0)
    after = next(i for i, k in enumerate(steps) if k * 0.1 > 12.0)
    frames, _ = demo.render_scene("outage", outage_scene, (640, 360), demo.FPS,
                                  only=[inside, after])
    # the warning color appears nowhere else in the band under the subtitle
    bands = [np.asarray(image_of(f).convert("RGB"))[25:75] for f in frames]
    red = [int(((b[..., 0] > 150) & (b[..., 1] < 100) & (b[..., 2] < 100)).sum()) for b in bands]
    assert red[0] > 20 and red[1] == 0


def test_hold_frames_repeat_the_last_frame(fusion_scene):
    """Fails if the final state is not held for HOLD_FRAMES frames."""
    segments, last, _ = demo.scene_inputs("fusion", fusion_scene)
    n = len(demo.frame_steps(segments, fusion_scene.config.dt, last))
    frames, _ = demo.render_scene("fusion", fusion_scene, TINY, demo.FPS, hold=3)
    assert len(frames) == n + 3
    assert frames[-1] == frames[-2] == frames[-3] == frames[n - 1]


def test_animations_have_the_frames_size_and_timing(tmp_path):
    """Fails if a frame is dropped, the frame duration is not 1000 / fps ms, or the loop is not
    endless, in either format."""
    frames = []
    for color in ((250, 250, 250), (200, 30, 30), (30, 30, 200)):
        buffer = io.BytesIO()
        Image.new("RGB", TINY, color).save(buffer, format="PNG")
        frames.append(buffer.getvalue())
    written = demo.encode_animations(frames, tmp_path, "clip", 10, ("gif", "webp"))
    assert set(written) == {"gif", "webp"}
    for path in written.values():
        with Image.open(path) as image:
            assert image.size == TINY
            assert image.n_frames == 3
            assert image.info["loop"] == 0
        assert frame_durations(path) == [100, 100, 100]


def test_a_held_frame_keeps_its_playing_time_in_both_formats(tmp_path):
    """Fails if repeated identical frames (the hold at the end of a scene) lose playing time:
    the encoders may merge them into one longer frame, but the total time must be 100 ms each."""
    frames = []
    for color in ((250, 250, 250), (200, 30, 30)):
        buffer = io.BytesIO()
        Image.new("RGB", TINY, color).save(buffer, format="PNG")
        frames.append(buffer.getvalue())
    frames = [frames[0], frames[1], frames[1], frames[1], frames[1]]
    for path in demo.encode_animations(frames, tmp_path, "held", 10, ("gif", "webp")).values():
        assert sum(frame_durations(path)) == 500


def test_unknown_animation_format_is_rejected(tmp_path):
    """Fails if a misspelled format silently writes a file in another format."""
    with pytest.raises(ValueError, match="format"):
        demo.encode_animations([b""], tmp_path, "clip", 10, ("mp4",))


def test_small_helpers():
    """Fails if the size argument or the preview picks are parsed wrongly."""
    assert demo.parse_size("1280x720") == (1280, 720)
    assert demo.preview_indices(101) == [0, 25, 50, 75, 100]
    assert demo.preview_indices(1) == [0]


def test_main_writes_the_animation_and_a_provenance_file(tmp_path, monkeypatch):
    """Fails if the command line does not produce the animation, the provenance file with its
    command line, frame count and checksum, or the requested preview frames."""
    short = demo.build_fusion_scene(config=FusionConfig(duration=6.0, burn_in=1.0))
    monkeypatch.setattr(demo, "build_fusion_scene", lambda: short)
    demo.main(["--out", str(tmp_path), "--scenes", "1", "--size", "320x180", "--formats", "gif",
               "--preview", "--name", "clip"])
    assert (tmp_path / "clip.gif").stat().st_size > 1000
    provenance = (tmp_path / "clip_provenance.txt").read_text(encoding="utf-8")
    assert provenance.startswith("# command: python ")
    assert "git HEAD" in provenance and "sha256" in provenance and "frames:" in provenance
    assert len(list((tmp_path / "preview").glob("fusion_*.png"))) >= 2
    frames = len(demo.frame_steps(demo.FUSION_SEGMENTS, 0.1, 60)) + demo.HOLD_FRAMES
    assert sum(frame_durations(tmp_path / "clip.gif")) == 100 * frames


def test_the_demo_uses_only_public_tracker_outputs():
    """Fails if the demo starts to observe tracker internals (the association method or the
    recording adapter): radar-to-track pairings are not part of the public outputs."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_associate" not in source
    assert "fusion.recording" not in source
    assert "_radar_rows" not in source
