# Examples

Runnable example scripts live in
[`examples/`](https://github.com/sibocw/parallel-matplotlib-animation/tree/main/examples).
Run all of them (except the benchmark) with:

```bash
./examples/run_all_except_benchmarks.sh
```

## `simple_wave_animation.py`

A single oscillating cosine wave — the [quick example](index.md#quick-example).

![Simple wave animation](https://raw.githubusercontent.com/sibocw/parallel-matplotlib-animation/main/assets/simple_wave_animation.gif)

## `multi_panel_animation.py`

Five subplots with different plot types updating together.

![Multi-panel animation](https://raw.githubusercontent.com/sibocw/parallel-matplotlib-animation/main/assets/multi_panel_animation.gif)

## `very_complex_animation.py`

A 14-subplot `GridSpec` layout — the figure used by the
[benchmark](benchmark.md). It showcases how much the setup-once / update-many
design saves when the layout is expensive to build.

![Very complex animation](https://raw.githubusercontent.com/sibocw/parallel-matplotlib-animation/main/assets/very_complex_animation.gif)

## `nondeterministic_video_loader.py`

Handling frames whose parameters arrive out of order, using
[`IndexedFrameParams`](api/animator.md). Useful when frames are decoded by a
parallel dataloader that returns them non-deterministically.

![Non-deterministic video loader](https://raw.githubusercontent.com/sibocw/parallel-matplotlib-animation/main/assets/nondeterministic_video_loader.gif)
