"""Strong-scaling benchmark for parallel matplotlib animation.

Measures how render-to-video throughput scales with the number of worker
processes, with and without figure reuse, and across both parallel-video-io
encoding backends: CPU (libx264) and GPU (NVENC). The GPU backend is benchmarked
only when a CUDA device is available; otherwise it is skipped automatically.

See https://hpc-wiki.info/hpc/Scaling_tests#Strong_Scaling for background on
strong scaling.

Run from the repository root::

    python examples/scaling_test.py            # full sweep
    python examples/scaling_test.py --quick    # small, fast smoke run

The ``benchmark`` optional dependencies (plotly, pandas) are required::

    pip install -e '.[benchmark]'

Outputs:

* ``examples/output/scaling_test/results.csv``  -- one row per configuration
* ``examples/output/scaling_test/results.json`` -- same data as JSON
* ``examples/output/scaling_test/scaling_graph.html`` -- interactive Plotly
  figure (embedded in the documentation site)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from very_complex_animation import VeryComplexAnimation

# Repository-root-relative locations (this script lives in examples/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "examples" / "output" / "scaling_test"
_FIGURE_PATH = _OUTPUT_DIR / "scaling_graph.html"

# Two-tone colour scheme: CPU encode in blue, GPU encode in magenta; the darker
# shade is "with figure reuse" (the mode this library is built around).
_COLOR = {
    ("cpu", True): "#0c2e6e",  # CPU encode, with cache -- dark blue
    ("cpu", False): "#90c4e4",  # CPU encode, no cache    -- light blue
    ("gpu", True): "#7a0042",  # GPU encode, with cache  -- dark magenta
    ("gpu", False): "#f0a0c4",  # GPU encode, no cache    -- light pink
}


def _gpu_available() -> bool:
    """True when a CUDA device is present, so NVENC encoding can be benchmarked."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_scaling_test(
    num_frames: int,
    num_workers_list: list[int],
    video_modes: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run the strong-scaling sweep and return one row per configuration.

    Args:
        num_frames: Number of frames to render per run.
        num_workers_list: Worker counts to sweep (e.g. ``[1, 2, 4, 8, 16]``).
        video_modes: parallel-video-io encode backends to sweep, a subset of
            ``["cpu", "gpu"]``.
        output_dir: Directory for the rendered ``.mp4`` artifacts.

    Returns:
        A DataFrame with columns ``num_workers``, ``reuse_fig_obj``,
        ``video_mode``, and ``time_seconds``.
    """
    params = [{"phase": 2 * np.pi * i / num_frames} for i in range(num_frames)]

    rows: list[dict] = []
    for video_mode in video_modes:
        for num_workers in num_workers_list:
            for reuse_figure_object in (True, False):
                label = f"{num_workers}w_reuse{reuse_figure_object}_{video_mode}"
                print(f"Running test: {label} ...")
                output_file = output_dir / f"output_{label}.mp4"
                anim = VeryComplexAnimation()
                start_time = time.perf_counter()
                anim.make_video(
                    output_file=output_file,
                    param_by_frame=params,
                    fps=30,
                    num_workers=num_workers,
                    disable_progress_bar=True,
                    reuse_figure_object=reuse_figure_object,
                    video_mode=video_mode,
                )
                elapsed_time = time.perf_counter() - start_time
                rows.append(
                    {
                        "num_workers": num_workers,
                        "reuse_fig_obj": reuse_figure_object,
                        "video_mode": video_mode,
                        "time_seconds": elapsed_time,
                    }
                )
                print(f"  {label} completed in {elapsed_time:.2f} seconds")

    return pd.DataFrame(rows)


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``speedup`` column, normalised to the serial no-reuse CPU baseline.

    All configurations are normalised to a single baseline -- one worker, no
    figure reuse, CPU encode -- so the plot shows both the parallel speedup and
    the extra gains from figure reuse and GPU encoding on a common scale.
    """
    df = df.copy()
    baseline_mode = (
        "cpu" if "cpu" in df["video_mode"].values else df["video_mode"].iloc[0]
    )
    baseline = df[
        (df["num_workers"] == df["num_workers"].min())
        & (~df["reuse_fig_obj"])
        & (df["video_mode"] == baseline_mode)
    ]["time_seconds"]
    baseline_time = baseline.iloc[0]
    df["speedup"] = baseline_time / df["time_seconds"]
    return df


def plot_scaling_results(df: pd.DataFrame, output_path: Path) -> None:
    """Write an interactive Plotly speedup-vs-workers figure."""
    df = add_speedup(df)
    num_workers_list = sorted(df["num_workers"].unique())

    fig = go.Figure()
    for video_mode in sorted(df["video_mode"].unique()):
        for reuse in (True, False):
            g = df[(df["video_mode"] == video_mode) & (df["reuse_fig_obj"] == reuse)]
            if g.empty:
                continue
            g = g.sort_values("num_workers")
            cache = "with cache" if reuse else "without cache"
            fig.add_trace(
                go.Scatter(
                    x=g["num_workers"],
                    y=g["speedup"],
                    mode="lines+markers",
                    name=f"{video_mode.upper()} encode, {cache}",
                    line=dict(color=_COLOR.get((video_mode, reuse), "#888888")),
                    marker=dict(color=_COLOR.get((video_mode, reuse), "#888888")),
                    hovertemplate=(
                        f"<b>{video_mode.upper()} encode, {cache}</b><br>"
                        "%{x} workers<br>%{y:.2f}x speedup<extra></extra>"
                    ),
                )
            )

    # Ideal (zero-overhead) linear scaling.
    fig.add_trace(
        go.Scatter(
            x=[1, max(num_workers_list)],
            y=[1, max(num_workers_list)],
            mode="lines",
            name="ideal scaling",
            line=dict(color="black", dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Strong scaling test",
        xaxis_title="# workers",
        yaxis_title="speedup",
        xaxis=dict(type="log", dtick="L0.30103"),  # log base-2 ticks
        yaxis=dict(type="log"),
        hovermode="closest",
        legend_title_text="Configuration",
    )
    fig.update_xaxes(
        tickvals=num_workers_list, ticktext=[str(n) for n in num_workers_list]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    parser.add_argument(
        "--no-gpu", action="store_true", help="skip the GPU (NVENC) encode backend"
    )
    args = parser.parse_args()

    if args.quick:
        num_frames_to_draw = 24
        num_workers_to_test = [1, 2, 4]
    else:
        num_frames_to_draw = 320
        num_workers_to_test = [1, 2, 4, 8, 16]

    video_modes = ["cpu"]
    if not args.no_gpu and _gpu_available():
        video_modes.append("gpu")
        print("CUDA device detected: benchmarking both CPU and GPU encoding.")
    else:
        print("No CUDA device (or --no-gpu): benchmarking CPU encoding only.")

    _OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    df = run_scaling_test(
        num_frames_to_draw, num_workers_to_test, video_modes, _OUTPUT_DIR
    )
    df.to_csv(_OUTPUT_DIR / "results.csv", index=False)
    with open(_OUTPUT_DIR / "results.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)

    plot_scaling_results(df, output_path=_FIGURE_PATH)
    print(f"\nWrote results to {_OUTPUT_DIR}")
    print(f"Wrote figure to {_FIGURE_PATH}")


if __name__ == "__main__":
    main()
