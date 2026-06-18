import sys
import unittest
import tempfile
import numpy as np
from pathlib import Path

# The example scripts live in the top-level examples/ directory (they are
# standalone scripts, not part of the installable package), so make them
# importable by module name.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from simple_wave_animation import WaveAnimation
from multi_panel_animation import MultiPanelAnimation
from very_complex_animation import VeryComplexAnimation


class TestProvidedExamples(unittest.TestCase):
    """Test that provided example animations can be instantiated and set up."""

    params = [{"phase": 2 * np.pi * i / 10} for i in range(10)]  # 10 frames

    def _run_animation(self, anim_class, num_workers):
        with tempfile.TemporaryDirectory() as tmpdir:
            anim = anim_class()
            anim.make_video(
                output_file=Path(tmpdir)
                / f"{anim_class.__name__}_{num_workers}_workers.mp4",
                param_by_frame=self.params,
                fps=30,
                num_workers=num_workers,
            )

    def test_wave_animation_setup_parallel(self):
        self._run_animation(WaveAnimation, num_workers=4)

    def test_multi_panel_animation_setup_parallel(self):
        self._run_animation(MultiPanelAnimation, num_workers=4)

    def test_very_complex_animation_setup_parallel(self):
        self._run_animation(VeryComplexAnimation, num_workers=4)

    def test_wave_animation_setup_serial(self):
        self._run_animation(WaveAnimation, num_workers=1)

    def test_multi_panel_animation_setup_serial(self):
        self._run_animation(MultiPanelAnimation, num_workers=1)

    def test_very_complex_animation_setup_serial(self):
        self._run_animation(VeryComplexAnimation, num_workers=1)


if __name__ == "__main__":
    unittest.main()
