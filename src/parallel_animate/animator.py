import matplotlib

matplotlib.use("Agg")

import tempfile
import itertools
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt
import av
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Iterable
from dataclasses import dataclass


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedFrameParams:
    """Special dataclass to hold parameters for each frame **if frames arrive out of
    order in the `param_by_frame` iterator**. The frame_id from this class will
    override the index in the iterator."""

    frame_id: int
    params: Any

    def __post_init__(self):
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {self.frame_id}")


class Animator(ABC):
    """
    Base class for creating matplotlib animations with efficient parallel rendering.
    """

    @abstractmethod
    def setup(self):
        """
        Set up the figure, axes, and any artists. Store whatever you need as instance
        attributes (self.ax, self.line, etc.); you will need them in update().

        This method MUST not accept any argument other than self (if you need to pass
        data to setup, store it as instance attributes in __init__).

        This method MUST return the plt.Figure object, and that alone.

        This method is called once before the animation loop (once per worker if
        parallelized), unless reuse_figure_object is False (basically never).
        """
        pass

    @abstractmethod
    def update(self, frame_idx: int, params: Any):
        """
        Update the plot for the given frame.

        This is called for each frame in the animation. Modify any figure, axes, and
        artists that you created in setup() and stored as instance attributes.

        Args:
            frame_idx (int): Current frame index (0 to num_frames-1)
            params (Any): Parameters for this frame from param_by_frame[frame_idx]. Can
                be any object you want as long as it's pickleable (required to
                distribute jobs to distributed workers).
        """
        pass

    def make_video(
        self,
        output_file: Path | str,
        param_by_frame: Iterable[Any],
        fps: int,
        n_frames: int | None = None,
        num_workers: int = -1,
        disable_progress_bar: bool | None = None,
        plotting_log_interval: int | None = None,
        saving_log_interval: int | None = None,
        savefig_params: dict[str, Any] | None = None,
        video_codec: str = "libx264",
        video_pixfmt: str = "yuv420p",
        video_params: dict[str, Any] | None = None,
        reuse_figure_object: bool = True,
        preload_factor: int = 8,
    ) -> None:
        """
        Render the animation to a video file.

        Args:
            output_file (Path | str): Path to output video file
            param_by_frame (Iterable[Any]): Iterable of parameters, one per frame
            fps (int): Frames per second for output video
            n_frames (int | None): Number of frames to render. If None, use the length
                of param_by_frame. If param_by_frame does not have __len__ implemented
                and n_frames is None, the progress bar won't show completion percentage.
            num_workers (int): Number of parallel workers. 1 for serial processing
                (in the main thread), -1 for all CPU cores, -2 for all but one CPU core,
                etc.
            disable_progress_bar (bool | None): Same behavior as tqdm: if True, disable
                progress bar. If None, auto-detect based on whether output is a TTY.
            plotting_log_interval (int | None): Log progress every N frames in each
                worker (None = no interval logging).
            saving_log_interval (int | None): Log progress every N frames when merging
                frames into video (None = no interval logging).
            savefig_params (dict[str, Any]): Additional keyword arguments to
                pass to plt.Figure.savefig() when saving frames (default: {}).
            video_codec (str): Codec to use for video encoding (default: "libx264").
            video_pixfmt (str): Pixel format for video encoding (default: "yuv420p").
            video_params (dict[str, Any]): Additional parameters to set on the video
                stream (default: {"crf": "23", "preset": "slow"}).
            reuse_figure_object (bool): If False, the figure will be re-created for each
                frame (i.e. setup() called every frame). There is basically no reason to
                set this to False. Use only for testing and benchmarking.
            preload_factor (int): Number of workloads to prefetch in the task queue per
                worker (approximately). I.e. each worker will have up to this many
                frames (on average) queued ahead of time to work on.
        """
        # Convert param_by_frame to list if it's not already
        if n_frames is None:
            try:
                n_frames = len(param_by_frame)
            except TypeError:
                _logger.warning(
                    "param_by_frame has no length and n_frames is not specified. "
                    "Progress bar won't show completion percentage."
                )

        # Determine number of workers
        if num_workers == 0:
            raise ValueError(
                "num_workers cannot be 0. Use 1 for serial mode or -1 for all cores."
            )
        elif num_workers == -1:
            num_workers = mp.cpu_count()
        elif num_workers < -1:
            num_workers = max(1, mp.cpu_count() + num_workers + 1)

        _logger.info(f"Rendering {n_frames} frames at {fps} fps")
        with tempfile.TemporaryDirectory(prefix="animator_frames_") as frames_dir:
            _logger.info(f"Using temporary directory: {frames_dir}")

            # Avoid mutable default arguments by creating defaults here
            if savefig_params is None:
                savefig_params = {}
            if video_params is None:
                video_params = {"crf": "23", "preset": "slow"}

            if num_workers == 1:
                _logger.info("Running in serial mode")
                self._render_serial(
                    param_by_frame,
                    n_frames,
                    frames_dir,
                    disable_progress_bar,
                    plotting_log_interval,
                    savefig_params,
                    reuse_figure_object,
                )
            else:
                _logger.info(f"Running in parallel mode with {num_workers} workers")
                self._render_parallel(
                    param_by_frame,
                    n_frames,
                    frames_dir,
                    num_workers,
                    disable_progress_bar,
                    plotting_log_interval,
                    savefig_params,
                    reuse_figure_object,
                    preload_factor,
                )

            _logger.info("Creating video with PyAV")
            _merge_frames_into_video(
                frames_dir,
                output_file,
                fps,
                video_codec,
                video_pixfmt,
                video_params,
                disable_progress_bar,
                _logger,
                log_interval=saving_log_interval,
            )

        _logger.info(f"Animation complete: {output_file}")

    def _setup_and_check(self) -> plt.Figure:
        """Call setup() and validate its return type."""
        fig = self.setup()
        if not isinstance(fig, plt.Figure):
            raise TypeError(
                f"`.setup()` must return a matplotlib Figure object, got {type(fig).__name__}"
            )
        return fig

    def _render_serial(
        self,
        param_by_frame: Iterable[Any],
        n_frames: int,
        frames_dir: Path | str,
        disable_progress_bar: bool | None,
        log_interval: int | None,
        savefig_params: dict[str, Any],
        reuse_figure_object: bool,
    ) -> None:
        """Render frames serially."""
        _logger.info("Serial rendering")

        fig = None
        if reuse_figure_object:
            fig = self._setup_and_check()

        seen_frame_ids: set[int] = set()
        frames_processed = 0
        for frame_idx, params in tqdm(
            enumerate(itertools.islice(param_by_frame, n_frames)),
            total=n_frames,
            disable=disable_progress_bar,
        ):
            if isinstance(params, IndexedFrameParams):
                if params.frame_id in seen_frame_ids:
                    raise ValueError(
                        f"Duplicate frame_id {params.frame_id} in param_by_frame"
                    )
                seen_frame_ids.add(params.frame_id)
                frame_idx = params.frame_id
                params = params.params

            if not reuse_figure_object:
                fig = self._setup_and_check()

            self.update(frame_idx, params)
            frame_path = Path(frames_dir) / f"frame_{frame_idx:09d}.png"
            fig.savefig(frame_path, **savefig_params)
            if not reuse_figure_object:
                plt.close(fig)

            frames_processed += 1
            if log_interval and frames_processed % log_interval == 0:
                _logger.info(f"Frame {frames_processed}/{n_frames} rendered")

        if reuse_figure_object and fig is not None:
            plt.close(fig)

    def _render_parallel(
        self,
        param_by_frame: Iterable[Any],
        n_frames: int,
        frames_dir: Path | str,
        num_workers: int,
        disable_progress_bar: bool | None,
        log_interval: int | None,
        savefig_params: dict[str, Any],
        reuse_figure_object: bool,
        preload_factor: int,
    ) -> None:
        """Render frames in parallel using dynamic work distribution."""
        _logger.info(f"Using dynamic work distribution with {num_workers} workers")

        # Create queues for task distribution and atomic counter for progress
        task_queue = mp.Queue(maxsize=num_workers * preload_factor)

        # Start worker processes
        workers = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=_worker_process,
                args=(
                    self,
                    worker_id,
                    task_queue,
                    frames_dir,
                    log_interval,
                    savefig_params,
                    reuse_figure_object,
                ),
            )
            p.start()
            workers.append(p)

        # Populate task queue with individual frames (batch_size = 1).
        # The bounded queue naturally throttles this loop so it only runs ahead of
        # workers by ~preload_factor frames. Note: tqdm here tracks enqueue speed,
        # which approximates render progress but hits 100% while workers finish the
        # last queued frames.
        seen_frame_ids: set[int] = set()
        for frame_idx, params in tqdm(
            enumerate(itertools.islice(param_by_frame, n_frames)),
            total=n_frames,
            disable=disable_progress_bar,
        ):
            if isinstance(params, IndexedFrameParams):
                if params.frame_id in seen_frame_ids:
                    raise ValueError(
                        f"Duplicate frame_id {params.frame_id} in param_by_frame"
                    )
                seen_frame_ids.add(params.frame_id)
                frame_idx = params.frame_id
                params = params.params

            task_queue.put((frame_idx, params))

        # Send sentinel values to signal workers to exit
        for _ in range(num_workers):
            task_queue.put(None)

        # Wait for all workers to finish and check for errors
        failed = []
        for p in workers:
            p.join()
            if p.exitcode != 0:
                failed.append((p.pid, p.exitcode))

        if failed:
            details = ", ".join(
                f"pid {pid} exited with code {code}" for pid, code in failed
            )
            raise RuntimeError(f"One or more worker processes failed: {details}")

        _logger.info("All workers completed")


def _worker_process(
    animator: Animator,
    worker_id: int,
    task_queue: mp.Queue,
    frames_dir: Path | str,
    log_interval: int | None,
    savefig_params: dict[str, Any],
    reuse_figure_object: bool,
) -> None:
    """
    Worker process that renders frames.

    Each worker:
    1. Calls setup() once to initialize the figure (unless reuse_figure_object is False)
    2. Repeatedly pulls individual frames from the task queue
    3. Renders each frame
    """
    fig = None
    if reuse_figure_object:
        fig = animator._setup_and_check()

    frames_processed = 0
    while True:
        task = task_queue.get()
        if task is None:
            break
        frame_idx, params = task

        if not reuse_figure_object:
            fig = animator._setup_and_check()
        animator.update(frame_idx, params)
        frame_path = Path(frames_dir) / f"frame_{frame_idx:09d}.png"
        fig.savefig(frame_path, **savefig_params)
        frames_processed += 1
        if not reuse_figure_object:
            plt.close(fig)

        if log_interval and frames_processed % log_interval == 0:
            _logger.info(f"Worker {worker_id}: processed {frames_processed} frames")

    if fig is not None:
        plt.close(fig)

    _logger.info(f"Worker {worker_id}: completed {frames_processed} frames")


def _merge_frames_into_video(
    frames_dir: Path | str,
    output_file: Path | str,
    fps: int,
    video_codec: str,
    video_pixfmt: str,
    video_params: dict[str, Any],
    disable_progress_bar: bool | None,
    logger: logging.Logger,
    log_interval: int | None,
) -> None:
    """Use PyAV to merge frames into video."""
    # Gather frame files in sorted order
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError("No frames found in temporary directory")

    # Open first image to determine size and ensure even dimensions
    with Image.open(frame_files[0]) as first_img:
        width, height = first_img.size
    # Make width/height even (required by many codecs)
    width = (width // 2) * 2
    height = (height // 2) * 2

    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(output_file), mode="w")
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = video_pixfmt
        stream.options.update(video_params)

        # Encode each frame
        for i, frame_path in tqdm(
            enumerate(frame_files),
            total=len(frame_files),
            desc="Merging frames",
            disable=disable_progress_bar,
        ):
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                # Ensure image has the right size
                if img.size != (width, height):
                    img = img.resize((width, height))

                video_frame = av.VideoFrame.from_image(img)
            video_frame = video_frame.reformat(width, height, format=video_pixfmt)
            for packet in stream.encode(video_frame):
                container.mux(packet)

            # Optional interval logging
            if log_interval and (i + 1) % log_interval == 0:
                logger.info(f"Frame written {i + 1}/{len(frame_files)} to video")

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()

        size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        logger.info(f"Video created: {output_file} ({size_mb:.2f} MB)")

    except Exception as e:
        logger.critical(f"PyAV failed to create video: {e}")
        raise
