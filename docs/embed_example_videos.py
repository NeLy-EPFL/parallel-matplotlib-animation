"""MkDocs hook: publish the gallery example videos into the built site.

The example scripts render their animations to ``examples/output/`` (git-ignored,
so the MP4s are not committed to ``main``/``dev`` branches). The docs embed them
via ``<video>`` tags pointing at ``output/<name>.mp4``. This hook copies the
gallery videos straight from ``examples/output/`` into the built site so they are
published to the ``gh-pages`` branch -- and only there -- by ``mkdocs gh-deploy``.

Only the named gallery videos are copied (not, e.g., the large benchmark
artifacts that ``scaling_test.py`` writes alongside them). Missing videos are
skipped with a warning so the build still succeeds on a fresh checkout where the
examples have not been rendered yet.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.embed_example_videos")

_SOURCE_DIR = Path("examples/output")
_GALLERY_VIDEOS = (
    "simple_wave_animation.mp4",
    "multi_panel_animation.mp4",
    "very_complex_animation.mp4",
    "nondeterministic_video_loader.mp4",
)


def on_post_build(config):  # noqa: ARG001
    dest_dir = Path(config["site_dir"]) / "output"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in _GALLERY_VIDEOS:
        src = _SOURCE_DIR / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)
        else:
            log.warning("Example video missing, not published: %s", src)
