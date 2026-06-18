# parallel-matplotlib-animation

Create matplotlib animations rendered to video **in parallel**, with efficient
reuse of figure resources.

## What it does

Rendering a matplotlib animation frame-by-frame in a single process is slow.
This library speeds it up by:

1. Spawning a pool of worker processes and creating the matplotlib resources
   (`plt.Figure`, `plt.Axes`, artists, ...) **once per worker**.
2. Distributing frames across workers via a dynamic queue.
3. Rendering each assigned frame by updating only the data that changes (rather
   than redrawing the whole plot from scratch).
4. Encoding the rendered frames into an H.264 MP4 with
   [parallel-video-io](https://github.com/sibocw/parallel-video-io) (FFmpeg
   under the hood, with automatic GPU/NVENC acceleration when available).

**Key design: figure reuse.** In each worker process, `setup()` runs once to
build the figure, then `update()` mutates it repeatedly. This combines the best
of both worlds:

- *Serial processing* — avoids the overhead of recreating complex layouts for
  every frame.
- *Parallel processing* — achieves speedup by using multiple CPU cores.

See the [benchmark](benchmark.md) for how this scales.

## Quick example

```python
import numpy as np
import matplotlib.pyplot as plt
from parallel_animate import Animator

# Step 1: Create a child class of parallel_animate.Animator
class WaveAnimation(Animator):

    # Step 2: Define how the plot should be set up
    def setup(self):
        fig, ax = plt.subplots()
        self.x = np.linspace(0, 4 * np.pi, 200)
        (self.line,) = ax.plot(self.x, np.cos(self.x))
        ax.set_xlim(0, 4 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cosine Wave")
        return fig  # <- return a plt.Figure object

    # Step 3: Define how plot elements should be updated for each frame
    # (given parameters that you define later)
    def update(self, frame_idx, params):
        phase = params["phase"]
        self.line.set_ydata(np.cos(self.x + phase))

# Step 4: Define a list of input parameters, one for each frame
params = [{"phase": 2 * np.pi * i / 60} for i in range(60)]

# Step 5: Make the video in parallel
anim = WaveAnimation()
anim.make_video("wave.mp4", param_by_frame=params, fps=30, num_workers=4)
```

![Simple wave animation](https://raw.githubusercontent.com/sibocw/parallel-matplotlib-animation/main/assets/simple_wave_animation.gif)

Continue to [Usage](usage.md) for the full API, or browse the
[examples](examples.md).

!!! note "Linux only"
    parallel-video-io (the encoding backend) is currently Linux-only.
