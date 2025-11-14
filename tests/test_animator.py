"""AI-generated unit tests"""

import unittest
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_animate import Animator


class SimpleTestAnimation(Animator):
    """Minimal animation for testing."""

    def setup(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        self.x = np.linspace(0, 2 * np.pi, 50)
        (self.line,) = ax.plot(self.x, np.sin(self.x))
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        return fig

    def update(self, frame_idx, params):
        phase = params["phase"]
        self.line.set_ydata(np.sin(self.x + phase))


class BadSetupAnimation(Animator):
    """Animation that returns wrong type from setup()."""

    def setup(self):
        return "not a figure"

    def update(self, frame_idx, params):
        pass


class TestAnimatorBasics(unittest.TestCase):
    """Test basic animator functionality."""

    def test_animator_is_abstract(self):
        """Cannot instantiate abstract Animator class."""
        with self.assertRaises(TypeError):
            Animator()

    def test_simple_animation_instantiation(self):
        """Can create concrete animator subclass."""
        anim = SimpleTestAnimation()
        self.assertIsInstance(anim, Animator)

    def test_setup_returns_figure(self):
        """setup() should return a matplotlib Figure."""
        anim = SimpleTestAnimation()
        fig = anim.setup()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_setup_validation_wrong_type(self):
        """_setup_and_check() should raise TypeError for wrong return type."""
        anim = BadSetupAnimation()
        with self.assertRaises(TypeError):
            anim._setup_and_check()

    def test_update_callable(self):
        """update() should accept frame_idx and params."""
        anim = SimpleTestAnimation()
        fig = anim.setup()
        # Should not raise
        anim.update(0, {"phase": 0.0})
        anim.update(5, {"phase": 1.0})
        plt.close(fig)


class TestVideoCreation(unittest.TestCase):
    """Test video creation functionality."""

    def test_make_video_creates_file(self):
        """make_video() should create an output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"
            params = [{"phase": 2 * np.pi * i / 10} for i in range(10)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_make_video_with_string_path(self):
        """make_video() should accept string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test.mp4")
            params = [{"phase": 0.0}]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(Path(output_path).exists())

    def test_make_video_single_frame(self):
        """make_video() should work with a single frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single.mp4"
            params = [{"phase": 0.0}]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_make_video_empty_params_fails(self):
        """make_video() should fail gracefully with empty params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.mp4"
            params = []

            anim = SimpleTestAnimation()
            with self.assertRaises(RuntimeError):
                anim.make_video(
                    output_file=output_path,
                    param_by_frame=params,
                    fps=10,
                    num_workers=1,
                    disable_progress_bar=True,
                )


class TestSerialRendering(unittest.TestCase):
    """Test serial (single-worker) rendering."""

    def test_serial_mode_with_reuse(self):
        """Serial mode with figure reuse should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "serial_reuse.mp4"
            params = [{"phase": 2 * np.pi * i / 10} for i in range(10)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                reuse_figure_object=True,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_serial_mode_without_reuse(self):
        """Serial mode without figure reuse should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "serial_no_reuse.mp4"
            params = [{"phase": 2 * np.pi * i / 5} for i in range(5)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                reuse_figure_object=False,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())


class TestParallelRendering(unittest.TestCase):
    """Test parallel rendering."""

    def test_parallel_mode_with_reuse(self):
        """Parallel mode with figure reuse should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "parallel_reuse.mp4"
            params = [{"phase": 2 * np.pi * i / 10} for i in range(10)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=2,
                reuse_figure_object=True,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_parallel_mode_without_reuse(self):
        """Parallel mode without figure reuse should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "parallel_no_reuse.mp4"
            params = [{"phase": 2 * np.pi * i / 5} for i in range(5)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=2,
                reuse_figure_object=False,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_num_workers_auto(self):
        """num_workers=-1 should use all CPU cores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "auto_workers.mp4"
            params = [{"phase": 2 * np.pi * i / 5} for i in range(5)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=-1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())


class TestIterableParams(unittest.TestCase):
    """Test that param_by_frame accepts various iterable types."""

    def test_generator_serial(self):
        """Serial mode should work with a generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generator_serial.mp4"
            
            def param_generator():
                for i in range(5):
                    yield {"phase": 2 * np.pi * i / 5}

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=param_generator(),
                fps=10,
                n_frames=5,  # Must specify n_frames with generator
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_generator_parallel(self):
        """Parallel mode should work with a generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generator_parallel.mp4"
            
            def param_generator():
                for i in range(5):
                    yield {"phase": 2 * np.pi * i / 5}

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=param_generator(),
                fps=10,
                n_frames=5,  # Must specify n_frames with generator
                num_workers=2,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_tuple(self):
        """Should work with a tuple of parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tuple.mp4"
            params = tuple({"phase": 2 * np.pi * i / 5} for i in range(5))

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_range_object(self):
        """Should work with range objects (common use case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "range.mp4"
            # Using range directly as params (integers as parameters)
            
            class RangeAnimation(Animator):
                def setup(self):
                    fig, ax = plt.subplots(figsize=(4, 3))
                    self.ax = ax
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    return fig
                
                def update(self, frame_idx, params):
                    self.ax.clear()
                    self.ax.set_xlim(0, 10)
                    self.ax.set_ylim(0, 10)
                    self.ax.plot([params], [params], 'ro')
            
            anim = RangeAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=range(5),
                fps=10,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_generator_without_n_frames_fails(self):
        """Generator without n_frames should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "should_fail.mp4"
            
            def param_generator():
                for i in range(5):
                    yield {"phase": 2 * np.pi * i / 5}

            anim = SimpleTestAnimation()
            with self.assertRaises(ValueError) as cm:
                anim.make_video(
                    output_file=output_path,
                    param_by_frame=param_generator(),
                    fps=10,
                    num_workers=1,
                    disable_progress_bar=True,
                )
            
            self.assertIn("n_frames must be specified", str(cm.exception))


class TestVideoParameters(unittest.TestCase):
    """Test various video encoding parameters."""

    def test_custom_fps(self):
        """Should work with custom FPS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_fps.mp4"
            params = [{"phase": 2 * np.pi * i / 5} for i in range(5)]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=60,
                num_workers=1,
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())

    def test_savefig_params(self):
        """Should accept savefig parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_savefig.mp4"
            params = [{"phase": 0.0}]

            anim = SimpleTestAnimation()
            anim.make_video(
                output_file=output_path,
                param_by_frame=params,
                fps=10,
                num_workers=1,
                savefig_params={"dpi": 150, "bbox_inches": "tight"},
                disable_progress_bar=True,
            )

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
