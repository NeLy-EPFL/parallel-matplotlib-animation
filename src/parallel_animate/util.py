import matplotlib
import numpy as np
import logging
from matplotlib import pyplot as plt
from fractions import Fraction

_logger = logging.getLogger(__name__)


def configure_matplotlib_style():
    """Use sans serif font and export text as texts (not shapes) in PDFs."""
    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    _logger.info("Configured matplotlib style.")
    # suppress matplotlib font manager warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_rendered_frame_ids(
    data_fps: Fraction | int,
    play_speed: float,
    rendered_fps: Fraction | int,
    n_data_frames: int,
) -> np.ndarray:
    """Get list of indices of input frames that should be rendered based on fps specs.

    Example: if data is recorded at 330 FPS, and we want to play it back at 0.1x speed
    at 30 FPS, then we need to render every `stride` frames in the original data, where
    stride = data_fps / (rendered_fps / play_speed) = 1.1 frames.

    Parameters
    ----------
    data_fps : Fraction | int
        The frame rate of the original data.
    play_speed : float
        The desired playback speed (e.g., 0.1 for 10% speed).
    rendered_fps : Fraction | int
        The frame rate at which the video will be rendered.
    n_data_frames : int
        The total number of data frames.


    Returns
    -------
     np.ndarray of int
         The indices of data frames that should be rendered.
    """
    if n_data_frames <= 0:
        return np.array([], dtype=int)

    stride = Fraction(data_fps) / (Fraction(rendered_fps) / play_speed)
    if stride < 1:
        _logger.warning(
            f"Calculated stride {stride} < 1. This will lead to repeated frames."
        )

    n_rendered_frames = max(1, int(n_data_frames / stride))
    # Use floor so we never map to a future data frame index, then clip to valid range.
    target_data_frame_ids = np.floor(
        np.arange(n_rendered_frames) * float(stride)
    ).astype(int)
    target_data_frame_ids = np.clip(target_data_frame_ids, 0, n_data_frames - 1)
    return target_data_frame_ids
