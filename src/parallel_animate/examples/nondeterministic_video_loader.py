import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_animate import Animator, IndexedFrameParams


class VideoFrameAnimation(Animator):
    def setup(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.imshow_artist = ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8))
        self.title_artist = ax.set_title("Video Frame X")
        ax.axis("off")
        return fig

    def update(self, frame_idx, params):
        self.imshow_artist.set_data(params["frame"])
        self.title_artist.set_text(f"Video Frame {frame_idx}")


def fake_video_loader(n_frames=64, frame_size=(128, 128)):
    """Emulates a video loader that yields in nondeterministic order (e.g. because
    loading is parallelized)."""
    frame_ids = np.arange(n_frames)
    np.random.shuffle(frame_ids)
    for frame_id in frame_ids:
        frame = np.zeros((*frame_size, 3), dtype=np.uint8)
        x = int((frame_id % frame_size[1]) * frame_size[1] / n_frames)
        frame[:, x : x + 20] = (0, 255, 0)  # moving green bar
        yield IndexedFrameParams(frame_id=frame_id, params={"frame": frame})


if __name__ == "__main__":
    # Create a "parallel video loader" that yields inputs out of order
    frame_loader = fake_video_loader(n_frames=64)

    # Create animation
    anim = VideoFrameAnimation()
    output_path = Path("example_output/nondeterministic_video_loader.mp4")
    anim.make_video(
        output_file=output_path,
        param_by_frame=frame_loader,
        n_frames=64,
        fps=30,
        num_workers=4,
    )
