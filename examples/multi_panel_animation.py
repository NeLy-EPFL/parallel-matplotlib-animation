from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parallel_animate import Animator


class MultiPanelAnimation(Animator):
    def __init__(self):
        super().__init__()

    def setup(self):
        # Create figure with 1 row, 5 columns of subplots
        self.fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        self.axes = list(axes)

        # Replace 4th subplot with 3D version
        self.axes[3].remove()
        self.axes[3] = self.fig.add_subplot(1, 5, 4, projection="3d")

        # Initialize artists for each subplot
        self.artists = {}

        # Line chart - multiple overlapping waves
        x = np.linspace(0, 4 * np.pi, 100)
        (line,) = self.axes[0].plot(x, np.sin(x), "b-", linewidth=2)
        self.axes[0].set_xlim(0, 4 * np.pi)
        self.axes[0].set_ylim(-2, 2)
        self.axes[0].set_title("Line Chart")
        self.axes[0].grid(True, alpha=0.3)
        self.artists["line"] = line
        self.artists["line_x"] = x

        # Bar chart - dancing bars
        bars = self.axes[1].bar(range(10), np.random.rand(10), color="steelblue")
        self.axes[1].set_ylim(0, 2)
        self.axes[1].set_title("Bar Chart")
        self.axes[1].set_xlabel("Category")
        self.axes[1].set_ylabel("Value")
        self.artists["bars"] = bars

        # Imshow - wave interference pattern
        data = np.random.rand(30, 30)
        im = self.axes[2].imshow(data, cmap="viridis", vmin=0, vmax=2)
        self.axes[2].set_title("Wave Interference")
        self.axes[2].axis("off")
        self.fig.colorbar(im, ax=self.axes[2], fraction=0.046, pad=0.04)
        self.artists["imshow"] = im

        # 3D line - rotating helix
        t = np.linspace(0, 4 * np.pi, 100)
        (line3d,) = self.axes[3].plot(np.cos(t), np.sin(t), t, "b-", linewidth=2)
        self.axes[3].set_xlim(-2, 2)
        self.axes[3].set_ylim(-2, 2)
        self.axes[3].set_zlim(0, 4 * np.pi)
        self.axes[3].set_title("3D Helix")
        self.axes[3].set_xlabel("X")
        self.axes[3].set_ylabel("Y")
        self.axes[3].set_zlabel("Z")
        self.artists["line3d"] = line3d
        self.artists["line3d_t"] = t

        # Scatter - orbiting points with color gradient
        n_points = 50
        theta = np.linspace(0, 2 * np.pi, n_points)
        x_scatter = 0.5 + 0.3 * np.cos(theta)
        y_scatter = 0.5 + 0.3 * np.sin(theta)
        colors = np.linspace(0, 1, n_points)
        scatter = self.axes[4].scatter(
            x_scatter, y_scatter, c=colors, cmap="plasma", s=100, alpha=0.8
        )
        self.axes[4].set_xlim(0, 1)
        self.axes[4].set_ylim(0, 1)
        self.axes[4].set_title("Orbiting Points")
        self.axes[4].set_aspect("equal")
        self.artists["scatter"] = scatter
        self.artists["n_points"] = n_points

        self.fig.tight_layout()
        return self.fig

    def update(self, frame_idx, params):
        phase = params["phase"]

        # Update line chart - multiple frequency sine wave
        x = self.artists["line_x"]
        y = np.sin(x + phase) + 0.3 * np.sin(3 * x - phase)
        self.artists["line"].set_ydata(y)

        # Update bar chart - pulsating bars
        heights = np.abs(np.sin(np.arange(10) * 0.5 + phase)) * 1.5 + 0.2
        for bar, h in zip(self.artists["bars"], heights):
            bar.set_height(h)

        # Update imshow - wave interference from moving sources
        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)

        # Two moving wave sources
        x1, y1 = 2 * np.cos(phase), 2 * np.sin(phase)
        x2, y2 = 2 * np.cos(phase + np.pi), 2 * np.sin(phase + np.pi)

        R1 = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        R2 = np.sqrt((X - x2) ** 2 + (Y - y2) ** 2)

        Z = np.sin(3 * R1 - 2 * phase) / (R1 + 0.5) + np.sin(3 * R2 - 2 * phase) / (
            R2 + 0.5
        )
        Z = Z + 1  # Shift to positive range

        self.artists["imshow"].set_data(Z)

        # Update 3D line - rotating and pulsating helix
        t = self.artists["line3d_t"]
        radius = 1 + 0.3 * np.sin(2 * phase)
        x_3d = radius * np.cos(t + phase)
        y_3d = radius * np.sin(t + phase)
        z_3d = t
        self.artists["line3d"].set_data_3d(x_3d, y_3d, z_3d)

        # Update scatter - orbiting and pulsating points
        n_points = self.artists["n_points"]
        angle = phase
        r = 0.25 + 0.15 * np.sin(2 * angle)
        theta = np.linspace(0, 2 * np.pi, n_points) + angle
        x_scatter = 0.5 + r * np.cos(theta)
        y_scatter = 0.5 + r * np.sin(theta)
        self.artists["scatter"].set_offsets(np.c_[x_scatter, y_scatter])


if __name__ == "__main__":
    # Generate parameters
    num_frames = 90
    params = [{"phase": 2 * np.pi * i / num_frames} for i in range(num_frames)]

    # Create animation
    anim = MultiPanelAnimation()
    output_path = Path("examples/output/multi_panel_animation.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.make_video(
        output_file=output_path, param_by_frame=params, fps=30, num_workers=8
    )
