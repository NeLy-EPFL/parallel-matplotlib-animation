"""Pytest tests for the parallel-video-io (pvio) encoding options exposed by
``Animator.make_video``.

These focus on the contract between ``make_video`` and
``pvio.write_frames_to_video``: that the user-facing ``video_*`` options are
forwarded correctly and that frames reach pvio as raw arrays, plus one real
end-to-end CPU encode.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

import parallel_animate.animator as animator_mod
from parallel_animate import Animator


class _SineAnimation(Animator):
    """Minimal animation used across the tests in this module."""

    def setup(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        self.x = np.linspace(0, 2 * np.pi, 50)
        (self.line,) = ax.plot(self.x, np.sin(self.x))
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        return fig

    def update(self, frame_idx, params):
        self.line.set_ydata(np.sin(self.x + params["phase"]))


def _params(n):
    return [{"phase": 2 * np.pi * i / n} for i in range(n)]


@pytest.fixture
def captured_encode(monkeypatch, tmp_path):
    """Patch pvio's writer with a recorder that still creates the output file.

    Returns a dict that, after ``make_video`` runs, holds the positional args
    (``video_path``, the materialised ``frames`` arrays, their backing
    ``frame_paths``, and ``fps``) and keyword args passed to
    ``pvio.write_frames_to_video``.
    """
    captured = {}

    def fake_write(video_path, frames, fps, **kwargs):
        # make_video reads the output file size afterwards, so it must exist.
        from pathlib import Path

        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).write_bytes(b"fake mp4")
        captured["video_path"] = video_path
        # Backing .npy paths of the lazy sequence, captured before the temp dir
        # is cleaned up; materialise the arrays too.
        captured["frame_paths"] = [str(p) for p in getattr(frames, "_paths", [])]
        captured["frames"] = list(frames)
        captured["fps"] = fps
        captured["kwargs"] = kwargs

    monkeypatch.setattr(animator_mod.pvio, "write_frames_to_video", fake_write)
    return captured


def _run(tmp_path, **make_video_kwargs):
    out = tmp_path / "out.mp4"
    _SineAnimation().make_video(
        output_file=out,
        param_by_frame=_params(make_video_kwargs.pop("n", 5)),
        fps=make_video_kwargs.pop("fps", 12),
        num_workers=1,
        disable_progress_bar=True,
        **make_video_kwargs,
    )
    return out


def test_defaults_forwarded_to_pvio(captured_encode, tmp_path):
    """With no encoding options set, pvio receives the documented defaults."""
    _run(tmp_path)

    kwargs = captured_encode["kwargs"]
    assert kwargs["mode"] == "auto"
    assert kwargs["preset"] is None
    assert kwargs["extra_ffmpeg_params"] is None
    # quality is left out entirely so pvio applies its own default.
    assert "quality" not in kwargs


@pytest.mark.parametrize("mode", ["auto", "gpu", "cpu"])
def test_video_mode_forwarded(captured_encode, tmp_path, mode):
    """video_mode (the GPU/CPU selector) is passed straight through as `mode`."""
    _run(tmp_path, video_mode=mode)
    assert captured_encode["kwargs"]["mode"] == mode


def test_quality_omitted_when_none(captured_encode, tmp_path):
    _run(tmp_path, video_quality=None)
    assert "quality" not in captured_encode["kwargs"]


def test_quality_forwarded_when_set(captured_encode, tmp_path):
    _run(tmp_path, video_quality=18)
    assert captured_encode["kwargs"]["quality"] == 18


def test_preset_and_extra_params_forwarded(captured_encode, tmp_path):
    _run(
        tmp_path,
        video_preset="p5",
        video_extra_ffmpeg_params=["-tune", "film"],
    )
    kwargs = captured_encode["kwargs"]
    assert kwargs["preset"] == "p5"
    assert kwargs["extra_ffmpeg_params"] == ["-tune", "film"]


def test_logging_and_progress_options_mapped(captured_encode, tmp_path):
    """saving_log_interval -> log_interval, disable_progress_bar -> quiet."""
    out = tmp_path / "out.mp4"
    _SineAnimation().make_video(
        output_file=out,
        param_by_frame=_params(5),
        fps=12,
        num_workers=1,
        disable_progress_bar=True,
        saving_log_interval=2,
    )
    kwargs = captured_encode["kwargs"]
    assert kwargs["log_interval"] == 2
    assert kwargs["quiet"] is True


def test_quiet_false_when_progress_enabled(captured_encode, tmp_path):
    _SineAnimation().make_video(
        output_file=tmp_path / "out.mp4",
        param_by_frame=_params(5),
        fps=12,
        num_workers=1,
        disable_progress_bar=None,
    )
    assert captured_encode["kwargs"]["quiet"] is False


def test_frames_passed_as_sorted_raw_arrays(captured_encode, tmp_path):
    """All rendered frames reach pvio as raw uint8 RGB arrays in frame order."""
    n = 7
    _run(tmp_path, n=n, fps=24)

    frames = captured_encode["frames"]
    assert len(frames) == n
    assert all(isinstance(f, np.ndarray) and f.dtype == np.uint8 for f in frames)
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames)

    paths = captured_encode["frame_paths"]
    assert len(paths) == n
    assert all(p.endswith(".npy") for p in paths)
    assert paths == sorted(paths)
    assert captured_encode["fps"] == 24


def test_dpi_in_savefig_params_changes_raster_resolution(captured_encode, tmp_path):
    """``dpi`` is honoured by the raw-buffer renderer (higher dpi -> bigger frames)."""
    _run(tmp_path, n=2, savefig_params={"dpi": 50})
    low_h, low_w = captured_encode["frames"][0].shape[:2]

    _run(tmp_path, n=2, savefig_params={"dpi": 100})
    high_h, high_w = captured_encode["frames"][0].shape[:2]

    assert high_h > low_h
    assert high_w > low_w


def test_cpu_encode_creates_real_video(tmp_path):
    """End-to-end: a real CPU encode through pvio produces a non-empty file."""
    out = tmp_path / "real.mp4"
    _SineAnimation().make_video(
        output_file=out,
        param_by_frame=_params(8),
        fps=10,
        num_workers=1,
        disable_progress_bar=True,
        video_mode="cpu",
    )
    assert out.exists()
    assert out.stat().st_size > 0
