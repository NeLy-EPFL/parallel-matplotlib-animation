from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parallel_animate import Animator


class WaveAnimation(Animator):
    def setup(self):
        fig, ax = plt.subplots()
        self.x = np.linspace(0, 4 * np.pi, 200)
        (self.line,) = ax.plot(self.x, np.cos(self.x))
        ax.set_xlim(0, 4 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cosine Wave")
        return fig

    def update(self, frame_idx, params):
        phase = params["phase"]
        self.line.set_ydata(np.cos(self.x + phase))


if __name__ == "__main__":
    # Generate parameters
    num_frames = 60
    params = [{"phase": 2 * np.pi * i / num_frames} for i in range(num_frames)]

    # Create and render
    anim = WaveAnimation()
    output_path = Path("examples/output/simple_wave_animation.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.make_video(
        output_file=output_path, param_by_frame=params, fps=30, num_workers=4
    )
