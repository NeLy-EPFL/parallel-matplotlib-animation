# Installation

!!! warning "Linux only"
    This package depends on [parallel-video-io](https://github.com/sibocw/parallel-video-io)
    for encoding, which is currently Linux-only.

## Install with `pip`

```bash
pip install parallel-matplotlib-animation
```

## Install from a local copy

```bash
git clone https://github.com/sibocw/parallel-matplotlib-animation.git
cd parallel-matplotlib-animation
pip install -e . --config-settings editable_mode=compat
```

## Optional dependency groups

| Extra | Installs | For |
|-------|----------|-----|
| `dev` | pytest, ruff, mkdocs-material, mkdocstrings | running tests, formatting, building the docs |
| `benchmark` | plotly, pandas | running the [strong-scaling benchmark](benchmark.md) |

```bash
pip install -e '.[dev]'         # development
pip install -e '.[benchmark]'   # benchmarking
```

## Requirements

Make sure `ffmpeg` is available on your `$PATH` (required by parallel-video-io
via imageio-ffmpeg). On Debian/Ubuntu:

```bash
sudo apt-get install ffmpeg
```

GPU (NVENC) encoding is used automatically when a CUDA-capable NVIDIA GPU and the
matching drivers are present; it falls back to the CPU encoder otherwise.
