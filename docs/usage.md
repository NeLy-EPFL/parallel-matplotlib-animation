# Usage

This library has a single class: [`parallel_animate.Animator`](api/animator.md).
To make an animation you create your own subclass and define two methods.

## Define your animator

- **`setup(self)`** — no arguments except `self`. Build your figure however you
  like and **return the `plt.Figure` object**. Store the things you will need to
  update later (axes, the return values of plotting calls like `ax.plot`, etc.)
  as instance attributes so you can access them in `update()`.
- **`update(self, frame_idx, params)`** — given the frame index and the
  per-frame parameters, mutate the artists you created in `setup()`. Typically
  you call methods like `line.set_data(...)` on attributes you stored. `params`
  can be any picklable Python object (a dict, tuple, single value, ...).
- *(Optional)* **`__init__(self, ...)`** — add any custom construction logic. This
  is handy when you want to create many animators from the same class with
  different settings, e.g. `anim = RecordingAnimator(dataset_path=...)`.

!!! warning "Keep heavy data out of instance attributes"
    In parallel mode the whole `Animator` instance is pickled into **every**
    worker process. Stashing a large array (e.g. a full video tensor) on `self`
    multiplies its memory across workers. Pass per-frame data through
    `param_by_frame` instead, or load it lazily inside `setup()`.

## Render the video

Once your animator is defined, a single call renders the video:

```python
anim = MyAnimation()
anim.make_video(
    output_file="out.mp4",
    param_by_frame=params,
    fps=30,
    num_workers=-1,
)
```

Key arguments to `make_video`:

- **`output_file`** (`Path | str`) — output video path.
- **`param_by_frame`** (`Iterable`) — one element per frame, each passed to
  `update()` as `params`. Can be a list, tuple, or generator. Generators are
  useful for large per-frame data (e.g. bitmaps) to avoid loading everything
  into memory at once.
- **`fps`** (`int`) — frame rate of the output video.
- **`n_frames`** (`int | None`) — number of frames to render. If `None`, the
  length of `param_by_frame` is used (when available).
- **`num_workers`** (`int`) — worker processes to spawn. `-1` uses all CPU cores,
  `-2` all but one, etc. `1` runs in the main process (no child processes).
- **`video_mode`** (`str`) — encoder selection passed to parallel-video-io:
  `"auto"` (default) uses GPU/NVENC when a CUDA device is available and falls
  back to CPU/libx264 otherwise; `"gpu"` forces NVENC and `"cpu"` forces
  libx264. Output is always an H.264 MP4.
- **`video_quality`**, **`video_preset`**, **`video_extra_ffmpeg_params`** —
  optional encoding-quality controls forwarded to parallel-video-io. Leave as
  `None` for sensible defaults.

See the [API reference](api/animator.md) for the remaining, less commonly used
options (logging intervals, figure reuse, prefetching).

## Out-of-order frame parameters

Sometimes the frames in `param_by_frame` arrive out of order — for example when
the parameters are decoded from a video by a parallel dataloader that returns
frames non-deterministically. Wrap each item in
[`parallel_animate.IndexedFrameParams`](api/animator.md), which carries an
explicit `frame_id` that overrides the iterator ordering:

```python
from parallel_animate import IndexedFrameParams

param_by_frame = [
    IndexedFrameParams(frame_id=5, params={"frame": ...}),
    IndexedFrameParams(frame_id=0, params={"frame": ...}),
    # ...
]
```

See [`examples/nondeterministic_video_loader.py`](examples.md) for a full
example.
