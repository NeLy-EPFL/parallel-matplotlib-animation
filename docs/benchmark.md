# Benchmark

A [strong-scaling test](https://hpc-wiki.info/hpc/Scaling_tests#Strong_Scaling)
measures how render-to-video throughput scales with the number of worker
processes. It sweeps:

- **worker count** — 1, 2, 4, 8, 16;
- **figure reuse** — with vs without (`reuse_figure_object`); and
- **encoding backend** — parallel-video-io's CPU (libx264) and GPU (NVENC)
  encoders. The GPU backend is benchmarked only when a CUDA device is present.

The animation rendered is
[`very_complex_animation.py`](examples.md#very_complex_animationpy), a 14-subplot
`GridSpec` figure that is expensive to build — exactly the case where the
setup-once / update-many design pays off.

## How to run

```bash
pip install -e '.[benchmark]'
python examples/scaling_test.py            # full sweep
python examples/scaling_test.py --quick    # small, fast smoke run
```

Results are written to `examples/output/scaling_test/` (`results.csv`,
`results.json`, and the interactive figure embedded below as
`scaling_graph.html`).

## Result

Speedup is normalised to the serial (one worker), no-reuse, CPU-encode baseline,
so the curves capture both the parallel speedup and the extra gains from figure
reuse and GPU encoding. The dashed black line is ideal (zero-overhead) linear
scaling.

--8<-- "examples/output/scaling_test/scaling_graph.html"

The left-most dark-blue point is serial processing with figure reuse. Points at
2+ workers show the parallel speedup; reuse curves (darker) sit above the
no-reuse curves (lighter) because the expensive layout is built once per worker
instead of once per frame.
