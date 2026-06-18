# Developer Info

Clone the repository and install the development extra:

```bash
git clone https://github.com/sibocw/parallel-matplotlib-animation.git
cd parallel-matplotlib-animation
pip install -e '.[dev]'
```

The `dev` extra installs everything needed for testing, formatting, and building
the documentation site.

## Testing

The test suite uses pytest. Run it from the repository root:

```bash
pytest
```

A few tests write small MP4 files via parallel-video-io / FFmpeg, so make sure
`ffmpeg` is on your `$PATH`.

## Documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) and the Material theme,
with API pages generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

```bash
mkdocs serve          # live-preview at http://127.0.0.1:8000
mkdocs build          # build the static site into site/
mkdocs gh-deploy      # build and push to the gh-pages branch
```

The [benchmark page](benchmark.md) embeds `assets/scaling_graph.html`, produced
by `python examples/scaling_test.py`. If that file is missing at build time, a
placeholder is substituted by the `docs/gen_benchmark.py` hook so the build still
succeeds.

## Code style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Use ASCII-only characters in `.py` docstrings and comments.
- Format with [ruff](https://docs.astral.sh/ruff/).
- On a new GitHub release, the package is published to PyPI and the docs are
  deployed to GitHub Pages automatically.
