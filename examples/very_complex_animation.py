from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from parallel_animate import Animator


class VeryComplexAnimation(Animator):
    """
    A showcase animation with extremely complex layout to demonstrate the efficiency
    of setup-once, update-many approach. This creates 12+ diverse plots with:
    - GridSpec with panels of varying sizes
    - Custom formatters, locators, and tick labels
    - Multiple colorbars and legends
    - 3D projections
    - Polar plots
    - Complex annotations and text
    """

    def setup(self):
        # Create figure with complex GridSpec layout
        self.fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(
            6,
            6,
            figure=self.fig,
            hspace=0.4,
            wspace=0.4,
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.05,
        )

        # Dictionary to store all artists and data arrays
        self.artists = {}
        self.data_arrays = {}

        # ===== PLOT 1: Large 3D Surface (spans 2x2) =====
        self.ax1 = self.fig.add_subplot(gs[0:2, 0:2], projection="3d")
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        self.artists["surf"] = self.ax1.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            vmin=-1,
            vmax=1,
        )
        self.data_arrays["surf_X"] = X
        self.data_arrays["surf_Y"] = Y
        self.ax1.set_xlabel("X Axis", fontsize=10, labelpad=10)
        self.ax1.set_ylabel("Y Axis", fontsize=10, labelpad=10)
        self.ax1.set_zlabel("Z = f(X,Y,t)", fontsize=10, labelpad=10)
        self.ax1.set_title(
            "3D Surface Evolution", fontsize=12, fontweight="bold", pad=20
        )
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_zlim(-1.5, 1.5)
        # Custom tick labels
        self.ax1.set_xticks([-3, -1.5, 0, 1.5, 3])
        self.ax1.set_yticks([-3, -1.5, 0, 1.5, 3])
        self.ax1.set_zticks([-1, 0, 1])
        self.ax1.view_init(elev=25, azim=45)

        # ===== PLOT 2: Contour plot with colorbar (spans 2x2) =====
        self.ax2 = self.fig.add_subplot(gs[0:2, 2:4])
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        levels = np.linspace(-1, 1, 21)
        self.artists["contourf"] = self.ax2.contourf(
            X, Y, Z, levels=levels, cmap="RdBu_r"
        )
        self.artists["contour"] = self.ax2.contour(
            X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.4
        )
        self.data_arrays["contour_X"] = X
        self.data_arrays["contour_Y"] = Y
        cbar = self.fig.colorbar(
            self.artists["contourf"], ax=self.ax2, orientation="vertical"
        )
        cbar.set_label("Field Strength", rotation=270, labelpad=20, fontsize=10)
        self.ax2.set_xlabel("X Position (m)", fontsize=10)
        self.ax2.set_ylabel("Y Position (m)", fontsize=10)
        self.ax2.set_title("2D Scalar Field Contours", fontsize=12, fontweight="bold")
        self.ax2.grid(True, alpha=0.2, linestyle="--")
        self.ax2.set_aspect("equal")

        # ===== PLOT 3: Polar plot with custom angles =====
        self.ax3 = self.fig.add_subplot(gs[0:2, 4:6], projection="polar")
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1 + 0.5 * np.sin(5 * theta)
        (self.artists["polar_line"],) = self.ax3.plot(theta, r, "b-", linewidth=2)
        self.artists["polar_fill"] = self.ax3.fill(theta, r, alpha=0.3, color="blue")
        self.data_arrays["polar_theta"] = theta
        self.ax3.set_ylim(0, 2)
        self.ax3.set_title(
            "Polar Pattern Evolution", fontsize=12, fontweight="bold", pad=20
        )
        # Custom angle labels
        self.ax3.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        self.ax3.set_xticklabels(
            ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
        )
        self.ax3.set_yticks([0.5, 1.0, 1.5, 2.0])
        self.ax3.grid(True, alpha=0.3)

        # ===== PLOT 4: Heatmap with custom colormap (spans 1x2) =====
        self.ax4 = self.fig.add_subplot(gs[2, 0:2])
        data = np.random.randn(20, 40)
        self.artists["heatmap"] = self.ax4.imshow(
            data,
            aspect="auto",
            cmap="plasma",
            interpolation="bilinear",
            vmin=-3,
            vmax=3,
        )
        cbar = self.fig.colorbar(
            self.artists["heatmap"], ax=self.ax4, orientation="horizontal", pad=0.1
        )
        cbar.set_label("Intensity (a.u.)", fontsize=9)
        self.ax4.set_title("Temporal Heatmap", fontsize=12, fontweight="bold")
        self.ax4.set_xlabel("Time Index", fontsize=10)
        self.ax4.set_ylabel("Channel", fontsize=10)
        # Custom tick spacing
        self.ax4.set_xticks(np.linspace(0, 39, 5))
        self.ax4.set_xticklabels(["0", "10", "20", "30", "40"])
        self.ax4.set_yticks([0, 5, 10, 15, 19])

        # ===== PLOT 5: Multi-line plot with legend =====
        self.ax5 = self.fig.add_subplot(gs[2, 2:4])
        x = np.linspace(0, 10, 200)
        self.data_arrays["multiline_x"] = x
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        labels = ["Signal α", "Signal β", "Signal γ", "Signal δ"]
        self.artists["multilines"] = []
        for i, (color, label) in enumerate(zip(colors, labels)):
            (line,) = self.ax5.plot(
                x, np.sin(x + i * np.pi / 4), color=color, linewidth=2, label=label
            )
            self.artists["multilines"].append(line)
        self.ax5.set_xlim(0, 10)
        self.ax5.set_ylim(-2, 2)
        self.ax5.set_xlabel("Time (s)", fontsize=10)
        self.ax5.set_ylabel("Amplitude", fontsize=10)
        self.ax5.set_title("Multi-Channel Signals", fontsize=12, fontweight="bold")
        self.ax5.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.ax5.grid(True, alpha=0.3, linestyle=":")
        # Custom x-axis formatter
        self.ax5.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f s"))

        # ===== PLOT 6: Quiver (vector field) plot =====
        self.ax6 = self.fig.add_subplot(gs[2, 4:6])
        x = np.linspace(-2, 2, 15)
        y = np.linspace(-2, 2, 15)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X
        self.artists["quiver"] = self.ax6.quiver(
            X,
            Y,
            U,
            V,
            np.sqrt(U**2 + V**2),
            cmap="coolwarm",
            scale=30,
            width=0.004,
            alpha=0.8,
        )
        self.data_arrays["quiver_X"] = X
        self.data_arrays["quiver_Y"] = Y
        self.ax6.set_xlim(-2.5, 2.5)
        self.ax6.set_ylim(-2.5, 2.5)
        self.ax6.set_xlabel("X", fontsize=10)
        self.ax6.set_ylabel("Y", fontsize=10)
        self.ax6.set_title("Vector Field Dynamics", fontsize=12, fontweight="bold")
        self.ax6.set_aspect("equal")
        self.ax6.grid(True, alpha=0.2)
        cbar = self.fig.colorbar(
            self.artists["quiver"], ax=self.ax6, orientation="vertical"
        )
        cbar.set_label("Magnitude", rotation=270, labelpad=15, fontsize=9)

        # ===== PLOT 7: Bar chart with error bars =====
        self.ax7 = self.fig.add_subplot(gs[3, 0:2])
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        x_pos = np.arange(len(categories))
        values = np.random.rand(len(categories)) * 10 + 5
        errors = np.random.rand(len(categories)) * 2
        self.artists["bars"] = self.ax7.bar(
            x_pos,
            values,
            yerr=errors,
            color="steelblue",
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
        )
        self.ax7.set_ylabel("Measurement (units)", fontsize=10)
        self.ax7.set_xlabel("Category", fontsize=10)
        self.ax7.set_title(
            "Categorical Measurements with Uncertainty", fontsize=12, fontweight="bold"
        )
        self.ax7.set_xticks(x_pos)
        self.ax7.set_xticklabels(categories, fontsize=9)
        self.ax7.set_ylim(0, 20)
        self.ax7.grid(True, axis="y", alpha=0.3, linestyle="--")
        # Add value labels on bars
        self.artists["bar_labels"] = []
        for bar in self.artists["bars"]:
            height = bar.get_height()
            label = self.ax7.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            self.artists["bar_labels"].append(label)

        # ===== PLOT 8: Scatter plot with varying sizes and colors =====
        self.ax8 = self.fig.add_subplot(gs[3, 2:4])
        n_points = 100
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)
        colors = np.random.rand(n_points)
        sizes = np.random.rand(n_points) * 200 + 50
        self.artists["scatter"] = self.ax8.scatter(
            x,
            y,
            c=colors,
            s=sizes,
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        self.ax8.set_xlim(-3, 3)
        self.ax8.set_ylim(-3, 3)
        self.ax8.set_xlabel("Feature X", fontsize=10)
        self.ax8.set_ylabel("Feature Y", fontsize=10)
        self.ax8.set_title(
            "2D Point Cloud Distribution", fontsize=12, fontweight="bold"
        )
        self.ax8.grid(True, alpha=0.3)
        cbar = self.fig.colorbar(
            self.artists["scatter"], ax=self.ax8, orientation="vertical"
        )
        cbar.set_label("Color Value", rotation=270, labelpad=15, fontsize=9)
        self.ax8.set_aspect("equal")

        # ===== PLOT 9: Histogram with custom bins =====
        self.ax9 = self.fig.add_subplot(gs[3, 4:6])
        data = np.random.normal(0, 1, 1000)
        bins = np.linspace(-4, 4, 31)
        n, bins_edges, patches = self.ax9.hist(
            data, bins=bins, color="skyblue", edgecolor="black", linewidth=1, alpha=0.7
        )
        self.artists["hist_patches"] = patches
        self.data_arrays["hist_bins"] = bins
        self.ax9.set_xlabel("Value", fontsize=10)
        self.ax9.set_ylabel("Frequency", fontsize=10)
        self.ax9.set_title("Distribution Analysis", fontsize=12, fontweight="bold")
        self.ax9.grid(True, axis="y", alpha=0.3)
        # Add statistics text
        self.artists["stats_text"] = self.ax9.text(
            0.95,
            0.95,
            f"μ = 0.00\nσ = 1.00\nN = 1000",
            transform=self.ax9.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # ===== PLOT 10: Pie chart with custom formatting =====
        self.ax10 = self.fig.add_subplot(gs[4, 0:2])
        labels = ["Sector A", "Sector B", "Sector C", "Sector D", "Sector E"]
        sizes = [30, 25, 20, 15, 10]
        colors_pie = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
        explode = (0.05, 0, 0, 0, 0)
        self.artists["pie_wedges"], texts, autotexts = self.ax10.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        self.artists["pie_texts"] = texts
        self.artists["pie_autotexts"] = autotexts
        self.ax10.set_title("Market Share Distribution", fontsize=12, fontweight="bold")
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontweight("bold")

        # ===== PLOT 11: Step plot with filled regions =====
        self.ax11 = self.fig.add_subplot(gs[4, 2:4])
        x = np.arange(0, 20, 1)
        y = np.random.randint(0, 10, size=len(x))
        self.data_arrays["step_x"] = x
        (self.artists["step_line"],) = self.ax11.step(
            x, y, where="mid", linewidth=2, color="darkgreen", label="Signal"
        )
        self.artists["step_fill"] = self.ax11.fill_between(
            x, 0, y, step="mid", alpha=0.3, color="lightgreen"
        )
        self.ax11.set_xlim(-1, 20)
        self.ax11.set_ylim(0, 12)
        self.ax11.set_xlabel("Sample Index", fontsize=10)
        self.ax11.set_ylabel("Discrete Value", fontsize=10)
        self.ax11.set_title("Step Function Evolution", fontsize=12, fontweight="bold")
        self.ax11.grid(True, alpha=0.3, linestyle=":")
        self.ax11.legend(loc="upper right", fontsize=9)

        # ===== PLOT 12: Box plot with multiple groups =====
        self.ax12 = self.fig.add_subplot(gs[4, 4:6])
        # Create 5 groups of data
        data_groups = [np.random.normal(0, std, 100) for std in [1, 2, 1.5, 2.5, 1.8]]
        bp = self.ax12.boxplot(
            data_groups,
            tick_labels=["G1", "G2", "G3", "G4", "G5"],
            patch_artist=True,
            notch=True,
            showmeans=True,
        )
        self.artists["boxplot"] = bp
        # Color the boxes
        colors_box = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "plum"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        self.ax12.set_ylabel("Value Distribution", fontsize=10)
        self.ax12.set_xlabel("Group", fontsize=10)
        self.ax12.set_title("Statistical Comparison", fontsize=12, fontweight="bold")
        self.ax12.grid(True, axis="y", alpha=0.3)

        # ===== PLOT 13: Stream plot (spans 2x3) =====
        self.ax13 = self.fig.add_subplot(gs[5, 0:3])
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X
        speed = np.sqrt(U**2 + V**2)
        self.artists["streamplot"] = self.ax13.streamplot(
            X,
            Y,
            U,
            V,
            color=speed,
            cmap="autumn",
            linewidth=1.5,
            density=1.5,
            arrowsize=1.5,
            arrowstyle="->",
        )
        self.data_arrays["stream_X"] = X
        self.data_arrays["stream_Y"] = Y
        self.ax13.set_xlim(-3, 3)
        self.ax13.set_ylim(-3, 3)
        self.ax13.set_xlabel("X Coordinate", fontsize=10)
        self.ax13.set_ylabel("Y Coordinate", fontsize=10)
        self.ax13.set_title("Fluid Flow Streamlines", fontsize=12, fontweight="bold")
        self.ax13.set_aspect("equal")
        cbar = self.fig.colorbar(
            self.artists["streamplot"].lines,
            ax=self.ax13,
            orientation="horizontal",
            pad=0.1,
        )
        cbar.set_label("Flow Speed", fontsize=9)

        # ===== PLOT 14: Error band plot (spans 2x3) =====
        self.ax14 = self.fig.add_subplot(gs[5, 3:6])
        x = np.linspace(0, 10, 100)
        y_mean = np.sin(x)
        y_std = 0.2 + 0.1 * x / 10
        self.data_arrays["errorband_x"] = x
        (self.artists["errorband_line"],) = self.ax14.plot(
            x, y_mean, "b-", linewidth=2.5, label="Mean", zorder=3
        )
        self.artists["errorband_fill"] = self.ax14.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.3,
            color="blue",
            label="±1σ",
            zorder=1,
        )
        self.artists["errorband_fill2"] = self.ax14.fill_between(
            x,
            y_mean - 2 * y_std,
            y_mean + 2 * y_std,
            alpha=0.15,
            color="blue",
            label="±2σ",
            zorder=0,
        )
        self.ax14.set_xlim(0, 10)
        self.ax14.set_ylim(-2, 2)
        self.ax14.set_xlabel("Time (s)", fontsize=10)
        self.ax14.set_ylabel("Response", fontsize=10)
        self.ax14.set_title(
            "Time Series with Confidence Bands", fontsize=12, fontweight="bold"
        )
        self.ax14.grid(True, alpha=0.3, linestyle="--")
        self.ax14.legend(loc="upper right", fontsize=9, framealpha=0.9)
        # Custom minor ticks
        self.ax14.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax14.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax14.tick_params(which="minor", length=3, color="gray")

        # Add overall title
        self.fig.suptitle(
            "Complex Multi-Panel Scientific Visualization Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        return self.fig

    def update(self, frame_idx, params):
        """
        Update only the data, not the complex structure.
        This is where we save massive amounts of time by not recreating everything.
        """
        phase = params["phase"]

        # Update 3D surface
        X = self.data_arrays["surf_X"]
        Y = self.data_arrays["surf_Y"]
        Z = np.sin(np.sqrt(X**2 + Y**2) + phase)
        self.artists["surf"].remove()
        self.artists["surf"] = self.ax1.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            vmin=-1,
            vmax=1,
        )

        # Update contour plot
        X = self.data_arrays["contour_X"]
        Y = self.data_arrays["contour_Y"]
        Z = np.sin(X + phase) * np.cos(Y)
        # Remove old collections
        while self.ax2.collections:
            self.ax2.collections[0].remove()
        levels = np.linspace(-1, 1, 21)
        self.artists["contourf"] = self.ax2.contourf(
            X, Y, Z, levels=levels, cmap="RdBu_r"
        )
        self.artists["contour"] = self.ax2.contour(
            X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.4
        )

        # Update polar plot
        theta = self.data_arrays["polar_theta"]
        r = 1 + 0.5 * np.sin(5 * theta + phase)
        self.artists["polar_line"].set_ydata(r)
        self.artists["polar_fill"][0].remove()
        self.artists["polar_fill"] = self.ax3.fill(theta, r, alpha=0.3, color="blue")

        # Update heatmap
        data = np.random.randn(20, 40) * (1 + 0.3 * np.sin(phase))
        self.artists["heatmap"].set_data(data)

        # Update multi-line plot
        x = self.data_arrays["multiline_x"]
        for i, line in enumerate(self.artists["multilines"]):
            y = np.sin(x + i * np.pi / 4 + phase) * (1 + 0.2 * np.sin(phase * 2))
            line.set_ydata(y)

        # Update quiver plot
        X = self.data_arrays["quiver_X"]
        Y = self.data_arrays["quiver_Y"]
        rotation = phase
        U = -Y * np.cos(rotation) + X * np.sin(rotation)
        V = X * np.cos(rotation) + Y * np.sin(rotation)
        self.artists["quiver"].set_UVC(U, V, np.sqrt(U**2 + V**2))

        # Update bar chart
        new_values = np.abs(np.sin(np.arange(8) * 0.5 + phase)) * 10 + 5
        for bar, val in zip(self.artists["bars"], new_values):
            bar.set_height(val)
        # Update bar labels
        for bar, label in zip(self.artists["bars"], self.artists["bar_labels"]):
            height = bar.get_height()
            label.set_text(f"{height:.1f}")
            label.set_position((bar.get_x() + bar.get_width() / 2.0, height))

        # Update scatter plot
        n_points = 100
        angle = phase
        x = np.cos(angle) * np.random.randn(n_points) - np.sin(angle) * np.random.randn(
            n_points
        )
        y = np.sin(angle) * np.random.randn(n_points) + np.cos(angle) * np.random.randn(
            n_points
        )
        self.artists["scatter"].set_offsets(np.c_[x, y])

        # Update histogram
        data = np.random.normal(np.sin(phase), 1, 1000)
        n, _ = np.histogram(data, bins=self.data_arrays["hist_bins"])
        for count, patch in zip(n, self.artists["hist_patches"]):
            patch.set_height(count)
        mean_val = np.mean(data)
        std_val = np.std(data)
        self.artists["stats_text"].set_text(
            f"μ = {mean_val:.2f}\nσ = {std_val:.2f}\nN = 1000"
        )

        # Update pie chart
        sizes_new = [
            30 + 10 * np.sin(phase),
            25 + 8 * np.sin(phase + 1),
            20 + 6 * np.sin(phase + 2),
            15 + 4 * np.sin(phase + 3),
            10 + 2 * np.sin(phase + 4),
        ]
        sizes_new = [max(5, s) for s in sizes_new]  # Ensure positive
        total = sum(sizes_new)
        for i, (wedge, size) in enumerate(zip(self.artists["pie_wedges"], sizes_new)):
            wedge.set_theta1(wedge.theta1)
            wedge.set_theta2(wedge.theta2)
        # Recalculate angles
        angle_start = 90
        for i, (wedge, size) in enumerate(zip(self.artists["pie_wedges"], sizes_new)):
            angle = 360 * size / total
            wedge.set_theta1(angle_start)
            wedge.set_theta2(angle_start + angle)
            angle_start += angle

        # Update step plot
        x = self.data_arrays["step_x"]
        y = np.abs(np.sin(x * 0.5 + phase)) * 10
        self.artists["step_line"].set_ydata(y)
        self.artists["step_fill"].remove()
        self.artists["step_fill"] = self.ax11.fill_between(
            x, 0, y, step="mid", alpha=0.3, color="lightgreen"
        )

        # Box plot is static in this example (would be expensive to update)

        # Update stream plot (this is expensive, so we do it sparingly)
        if frame_idx % 3 == 0:  # Only update every 3 frames
            X = self.data_arrays["stream_X"]
            Y = self.data_arrays["stream_Y"]
            rotation = phase
            U = -Y * np.cos(rotation) + X * np.sin(rotation)
            V = X * np.cos(rotation) + Y * np.sin(rotation)
            speed = np.sqrt(U**2 + V**2)
            # Clear the axes and redraw
            self.ax13.clear()
            self.artists["streamplot"] = self.ax13.streamplot(
                X,
                Y,
                U,
                V,
                color=speed,
                cmap="autumn",
                linewidth=1.5,
                density=1.5,
                arrowsize=1.5,
                arrowstyle="->",
            )
            # Restore axes properties
            self.ax13.set_xlim(-3, 3)
            self.ax13.set_ylim(-3, 3)
            self.ax13.set_xlabel("X Coordinate", fontsize=10)
            self.ax13.set_ylabel("Y Coordinate", fontsize=10)
            self.ax13.set_title(
                "Fluid Flow Streamlines", fontsize=12, fontweight="bold"
            )
            self.ax13.set_aspect("equal")

        # Update error band plot
        x = self.data_arrays["errorband_x"]
        y_mean = np.sin(x + phase)
        y_std = 0.2 + 0.1 * x / 10
        self.artists["errorband_line"].set_ydata(y_mean)
        self.artists["errorband_fill"].remove()
        self.artists["errorband_fill2"].remove()
        self.artists["errorband_fill"] = self.ax14.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.3, color="blue", zorder=1
        )
        self.artists["errorband_fill2"] = self.ax14.fill_between(
            x,
            y_mean - 2 * y_std,
            y_mean + 2 * y_std,
            alpha=0.15,
            color="blue",
            zorder=0,
        )


if __name__ == "__main__":
    # Generate parameters
    num_frames = 120
    params = [{"phase": 2 * np.pi * i / num_frames} for i in range(num_frames)]

    # Create and render
    anim = VeryComplexAnimation()
    output_path = Path("examples/output/very_complex_animation.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.make_video(
        output_file=output_path, param_by_frame=params, fps=30, num_workers=8
    )
