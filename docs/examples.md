# Examples

Runnable example scripts live in [`examples/`](https://github.com/sibocw/parallel-matplotlib-animation/tree/main/examples).

---

## `simple_wave_animation.py`

A single oscillating cosine wave — the [quick example](index.md#quick-example).

<div style="text-align:center;margin:1em 0;">
<a href="https://github.com/NeLy-EPFL/parallel-matplotlib-animation/blob/main/examples/simple_wave_animation.py" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.25em 0.9em;font-size:0.8em;font-weight:400;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
See source code ↗</a>
</div>

<div style="text-align:center;margin:1em 0;">
<video src="../output/simple_wave_animation.mp4" autoplay loop muted playsinline controls
       style="width:100%;border-radius:6px;">
Your browser does not support the video tag.
</video>
</div>

---

## `multi_panel_animation.py`

Five subplots with different plot types updating together.

<div style="text-align:center;margin:1em 0;">
<a href="https://github.com/NeLy-EPFL/parallel-matplotlib-animation/blob/main/examples/http://multi_panel_animation.py" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.25em 0.9em;font-size:0.8em;font-weight:400;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
See source code ↗</a>
</div>

<div style="text-align:center;margin:1em 0;">
<video src="../output/multi_panel_animation.mp4" autoplay loop muted playsinline controls
       style="width:100%;border-radius:6px;">
Your browser does not support the video tag.
</video>
</div>

---

## `very_complex_animation.py`

A 14-subplot `GridSpec` layout — the figure used by the
[benchmark](benchmark.md). It showcases how much the setup-once / update-many
design saves when the layout is expensive to build.

<div style="text-align:center;margin:1em 0;">
<a href="https://github.com/NeLy-EPFL/parallel-matplotlib-animation/blob/main/examples/very_complex_animation.py" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.25em 0.9em;font-size:0.8em;font-weight:400;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
See source code ↗</a>
</div>

<div style="text-align:center;margin:1em 0;">
<video src="../output/very_complex_animation.mp4" autoplay loop muted playsinline controls
       style="width:100%;border-radius:6px;">
Your browser does not support the video tag.
</video>
</div>

---

## `nondeterministic_video_loader.py`

Handling frames whose parameters arrive out of order, using
[`IndexedFrameParams`](api/animator.md). Useful when frames are decoded by a
parallel dataloader that returns them non-deterministically.

<div style="text-align:center;margin:1em 0;">
<a href="https://github.com/NeLy-EPFL/parallel-matplotlib-animation/blob/main/examples/nondeterministic_video_loader.py" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.25em 0.9em;font-size:0.8em;font-weight:400;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
See source code ↗</a>
</div>

<div style="text-align:center;margin:1em 0;">
<video src="../output/nondeterministic_video_loader.mp4" autoplay loop muted playsinline controls
       style="width:100%;border-radius:6px;">
Your browser does not support the video tag.
</video>
</div>
