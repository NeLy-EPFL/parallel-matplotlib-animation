import unittest
import numpy as np

from parallel_animate.util import get_rendered_frame_ids


class TestGetRenderedFrameIds(unittest.TestCase):
    def test_stride_lt_one_stays_in_bounds(self):
        frame_ids = get_rendered_frame_ids(
            data_fps=330,
            play_speed=0.1,
            rendered_fps=30,
            n_data_frames=100,
        )

        self.assertGreater(len(frame_ids), 0)
        self.assertTrue(np.all(frame_ids >= 0))
        self.assertTrue(np.all(frame_ids <= 99))

    def test_uses_floor_for_fractional_stride(self):
        # stride = 1.5; floor([0, 1.5, 3.0]) -> [0, 1, 3]
        frame_ids = get_rendered_frame_ids(
            data_fps=3,
            play_speed=1.0,
            rendered_fps=2,
            n_data_frames=5,
        )

        np.testing.assert_array_equal(frame_ids, np.array([0, 1, 3]))

    def test_zero_data_frames_returns_empty(self):
        frame_ids = get_rendered_frame_ids(
            data_fps=30,
            play_speed=1.0,
            rendered_fps=30,
            n_data_frames=0,
        )

        self.assertTrue(np.issubdtype(frame_ids.dtype, np.integer))
        self.assertEqual(len(frame_ids), 0)

    def test_positive_data_frames_never_returns_empty(self):
        frame_ids = get_rendered_frame_ids(
            data_fps=30,
            play_speed=0.1,
            rendered_fps=30,
            n_data_frames=1,
        )

        self.assertGreaterEqual(len(frame_ids), 1)
        self.assertTrue(np.all(frame_ids >= 0))
        self.assertTrue(np.all(frame_ids <= 0))


if __name__ == "__main__":
    unittest.main()
