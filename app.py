"""Local Gradio 5.49.1 frontend; the enhancement core is imported on demand.

Launch configuration (.env, beside this file): APP_HOST=127.0.0.1,
APP_PORT=7860, APP_MAX_UPLOAD_SIZE=2gb. No public share link is created.
The core owns CUDA detection, audio preservation, and temporary-file cleanup.
"""

from __future__ import annotations

import html
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from dotenv import load_dotenv

from pipeline.profiles import PROFILES
from pipeline.folder_batch import VIDEO_EXTENSIONS, discover_videos, select_local_folders


APP_ROOT = Path(__file__).resolve().parent
load_dotenv(APP_ROOT / ".env", override=False)
_output_path = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUTS = (_output_path if _output_path.is_absolute() else APP_ROOT / _output_path).resolve()
LOGGER = logging.getLogger(__name__)
GPU_QUEUE = "video-enhancement-gpu"
UPSCALE_MODELS = [
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3",
    "realesr-general-x4v3",
    "4x-UltraSharp",
    "SeedVR2",
]
LOG_LINES = 150
CSS = """
.gradio-container { max-width: 1220px !important; }
#hero { padding: 24px; border-radius: 18px; margin-bottom: 18px;
    background: linear-gradient(120deg, #10243e, #234960); color: #f0f7ff; }
#hero h1 { color: #f0f7ff; margin-bottom: 8px; }
#hero p { color: #d0e5f5; }
#job-status { border-radius: 14px; padding: 16px; background: var(--block-background-fill); }
.job-meter { height: 10px; overflow: hidden; border-radius: 9px;
    background: #d6dee8; margin: 12px 0; }
.job-meter-fill { height: 100%; background: linear-gradient(90deg, #0ea5e9, #14b8a6); }
.job-caption { font-size: 0.9rem; opacity: 0.85; }
"""


@dataclass(eq=False)
class Job:
    """One session's live job. Never use an Event-containing State default.

    Gradio stores runtime state by reference. __deepcopy__ also protects this
    explicitly reference-owned object if a framework path attempts to copy it.
    """

    sources: tuple[Path, ...]
    options: dict[str, Any]
    folder_batch: bool = False
    batch_mode: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    lock: Any = field(default_factory=threading.Lock)
    reports: queue.Queue[tuple[float, str]] = field(
        default_factory=lambda: queue.Queue(maxsize=256)
    )
    outputs: list[Path] = field(default_factory=list)
    errors: int = 0
    busy: bool = True
    claimed: bool = False
    fatal: bool = False
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    last_logged_at: float = 0.0
    last_logged_stage: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    video_started_at: float | None = None
    video_finished_at: float | None = None
    video_index: int = 0

    def elapsed_detail(self) -> str:
        with self.lock:
            now = time.monotonic()
            elapsed = max(0.0, (self.finished_at if self.finished_at is not None else now)
                          - self.started_at) if self.started_at is not None else 0.0
            detail = f"Elapsed: {elapsed:.0f} s"
            if len(self.sources) > 1 and self.video_started_at is not None:
                video_elapsed = max(0.0, (self.video_finished_at if self.video_finished_at is not None else now)
                                    - self.video_started_at)
                detail += f" · Video {self.video_index}/{len(self.sources)}: {video_elapsed:.0f} s"
            return detail

    def __deepcopy__(self, memo: dict[int, Any]) -> Job:
        memo[id(self)] = self
        return self

    def report(self, progress: float, message: str) -> None:
        """Never block GPU work on a slow or disconnected browser."""
        try:
            fraction = float(progress)
        except (TypeError, ValueError):
            fraction = 0.0
        fraction = min(1.0, max(0.0, fraction)) if math.isfinite(fraction) else 0.0
        item = (fraction, str(message)[:1500])
        stage = str(message).split(":", 1)[0]
        now = time.monotonic()
        with self.lock:
            persist = stage != self.last_logged_stage or now - self.last_logged_at >= 5 or fraction == 1
            if persist:
                self.last_logged_stage, self.last_logged_at = stage, now
        if persist:
            LOGGER.info("Job %s %.1f%% %s", self.job_id, fraction * 100, message)
        try:
            self.reports.put_nowait(item)
        except queue.Full:
            try:
                self.reports.get_nowait()
            except queue.Empty:
                pass
            try:
                self.reports.put_nowait(item)
            except queue.Full:
                pass


def release_session(job: Job | None) -> None:
    """Request cooperative cleanup when Gradio expires a session."""
    if job is not None:
        job.cancel.set()


def status_html(progress: float, title: str, detail: str = "") -> str:
    percent = round(min(1.0, max(0.0, progress)) * 100, 1)
    return (
        f"<strong>{html.escape(title)}</strong>"
        f'<div class="job-meter" role="progressbar" aria-label="Batch progress" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<div class="job-meter-fill" style="width:{percent}%"></div></div>'
        f'<div class="job-caption">{percent:.1f}% · {html.escape(detail)}</div>'
    )


def admit_job(
    current: Job | None,
    mode: str,
    single: str | None,
    batch: list[str] | None,
    upscale_model: str,
    scale: str,
    face_model: str,
    face_weight: float,
    rife_version: str,
    target_fps: float,
    fp16: bool,
    tile: float,
    rife_scale: str,
    keep_temp: bool,
    enable_upscale: bool = True,
    enable_face: bool = True,
    enable_rife: bool = True,
    batch_source: str = "Upload files",
    folder_text: str = "",
    output_resolution: str = "Scale",
    comfy_checkpoint: str = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    comfy_detector: str = "bbox/face_yolov8m.pt",
    comfy_positive: str = "photorealistic face, natural skin texture, subtle pores, preserve facial features, consistent lighting",
    comfy_negative: str = "plastic skin, waxy skin, oversharpened, deformed face, artifacts, excessive skin texture",
    comfy_denoise: float = 0.25,
    comfy_steps: float = 20,
    comfy_cfg: float = 5,
    comfy_seed: float = 42,
    comfy_guide_size: float = 768,
    seedvr_batch_size: float = 5,
    seedvr_noise: float = 0.03,
) -> tuple[Any, ...]:
    # This unqueued, synchronous callback reserves the session before the
    # streaming callback waits for the shared GPU concurrency slot.
    if current is not None:
        with current.lock:
            if current.busy:
                raise gr.Error("This session already has a queued or running job.")
    folder_batch = mode == "Batch" and batch_source == "Local folders"
    if folder_batch:
        try:
            sources = discover_videos(folder_text)
        except (OSError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
    else:
        uploads = [single] if mode == "Single video" and single else batch if mode == "Batch" else []
        if not uploads:
            raise gr.Error("Upload a video first (or select files for batch processing).")
        sources = tuple(Path(source).resolve() for source in uploads)
    for source in sources:
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise gr.Error(f"Unsupported or missing video: {source.name}")
    try:
        effective_scale = int(scale) if enable_upscale else 1
        effective_face = face_model if enable_face else "Off"
        effective_rife = rife_version if enable_rife else "Off"
        effective_resolution = output_resolution if enable_upscale else "Scale"
        if effective_scale == 1 and effective_resolution == "Scale" and effective_face == "Off" and effective_rife == "Off":
            raise ValueError("Select at least one operation. Upscaling needs scale 2 or 4; enabled face restoration and RIFE need a model other than Off.")
        tile_value = float(tile)
        if not math.isfinite(tile_value) or not tile_value.is_integer() or tile_value < 0:
            raise ValueError("Tile size must be a nonnegative integer; 0 disables tiling.")
        for name, value in (("steps", comfy_steps), ("seed", comfy_seed), ("guide size", comfy_guide_size),
                    ("SeedVR2 batch size", seedvr_batch_size)):
            if not math.isfinite(float(value)) or not float(value).is_integer():
                raise ValueError(f"FaceDetailer {name} must be a finite integer.")
        options = dict(
            upscale_model=upscale_model if enable_upscale else "RealESRGAN_x2plus",
            scale=effective_scale,
            face_model=effective_face,
            face_weight=float(face_weight),
            rife_version=effective_rife,
            target_fps=float(target_fps),
            fp16=bool(fp16),
            tile=int(tile_value),
            rife_scale=float(rife_scale),
            keep_temp=bool(keep_temp),
            output_resolution=effective_resolution,
            comfy_checkpoint=comfy_checkpoint, comfy_detector=comfy_detector,
            comfy_positive=comfy_positive, comfy_negative=comfy_negative,
            comfy_denoise=float(comfy_denoise), comfy_cfg=float(comfy_cfg),
            comfy_steps=int(comfy_steps), comfy_seed=int(comfy_seed),
            comfy_guide_size=int(comfy_guide_size),
            seedvr_batch_size=int(seedvr_batch_size), seedvr_noise=float(seedvr_noise),
        )
        from pipeline.config import Settings
        Settings(**options).validate()
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    job = Job(sources=sources, options=options, folder_batch=folder_batch, batch_mode=mode == "Batch")
    return (
        job,
        gr.update(interactive=False),
        gr.update(interactive=True),
        status_html(0, "Queued", "Waiting for the GPU processing slot"),
        "Job accepted. Cancellation remains available while queued.",
        [],
        None,
    )


def cancel_job(job: Job | None) -> None:
    if job is None:
        gr.Info("No active job in this session.")
        return
    with job.lock:
        if not job.busy:
            gr.Info("The job has already finished.")
            return
        job.cancel.set()
    # Do not cancel the Gradio generator: it must wait for core cleanup.
    job.report(0, "Cancellation requested; waiting for the current stage to stop safely.")
    gr.Info("Cancellation requested. Completed outputs will remain downloadable.")


def process_job(job: Job) -> None:
    """GPU worker: no Gradio component updates or request-local API calls."""
    with job.lock:
        job.started_at = time.monotonic()
    try:
        LOGGER.info("Job %s starting: inputs=%s settings=%s", job.job_id,
                    [str(source) for source in job.sources], job.options)
        if job.cancel.is_set():
            return
        from pipeline.config import ROOT, Settings
        from pipeline.runner import run_video

        if Path(ROOT).resolve() != APP_ROOT:
            raise ValueError("pipeline.config.ROOT must match the directory containing app.py.")
        settings = Settings(**job.options)
        settings.validate()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        total = len(job.sources)
        for index, source in enumerate(job.sources):
            if job.cancel.is_set():
                break
            with job.lock:
                job.video_started_at = time.monotonic()
                job.video_finished_at = None
                job.video_index = index + 1
            job.report(index / total, f"[{index + 1}/{total}] Starting {source.name}")

            def report(fraction: float, message: str, i: int = index) -> None:
                try:
                    value = float(fraction)
                except (TypeError, ValueError):
                    value = 0.0
                value = min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
                job.report((i + value) / total, f"[{i + 1}/{total}] {message}")

            try:
                if job.folder_batch:
                    output = Path(run_video(source, settings, job.cancel, report,
                                            save_beside_source=True, batch_mode=True)).resolve()
                    if output.parent != source.parent or output == source or not output.is_file():
                        raise ValueError("Folder-batch output must be a new file beside its source.")
                else:
                    if settings.upscale_model == "SeedVR2":
                        output = Path(run_video(source, settings, job.cancel, report,
                                                batch_mode=job.batch_mode)).resolve()
                    else:
                        output = Path(run_video(source, settings, job.cancel, report)).resolve()
                    if not output.is_relative_to(OUTPUTS) or not output.is_file():
                        raise ValueError("The core must return an existing file inside ROOT/outputs.")
                with job.lock:
                    job.outputs.append(output)
                job.report((index + 1) / total, f"Saved {output}")
            except Exception as exc:
                if job.cancel.is_set():
                    job.report(index / total, "Current video stopped by cancellation.")
                    break
                LOGGER.exception("Job %s enhancement failed for %s", job.job_id, source.name)
                with job.lock:
                    job.errors += 1
                job.report((index + 1) / total, f"Failed {source.name}: {exc}")
                # A failed batch member does not discard successful outputs.
            finally:
                with job.lock:
                    job.video_finished_at = time.monotonic()
                    video_seconds = job.video_finished_at - job.video_started_at
                job.report((index + 1) / total if not job.cancel.is_set() else index / total,
                           f"{source.name} processing ended after {video_seconds:.1f} s")
    except Exception as exc:
        LOGGER.exception("Unable to start enhancement")
        with job.lock:
            job.errors += 1
            job.fatal = True
        job.report(0, f"Cannot start the pipeline: {exc}")
    finally:
        with job.lock:
            job.finished_at = time.monotonic()
        LOGGER.info("Job %s %s", job.job_id, job.elapsed_detail())
        LOGGER.info("Job %s ended: cancelled=%s errors=%s outputs=%s", job.job_id,
                    job.cancel.is_set(), job.errors, [str(path) for path in job.outputs])
        job.done.set()


def stream_job(job: Job | None) -> Iterator[tuple[Any, ...]]:
    if job is None:
        raise gr.Error("No job was submitted.")
    with job.lock:
        if job.claimed:
            raise gr.Error("This job is already being processed.")
        job.claimed = True
    lines: deque[str] = deque(maxlen=LOG_LINES)
    progress = 0.0
    worker = threading.Thread(target=process_job, args=(job,), name="video-gpu-worker", daemon=True)

    def snapshot(title: str, finished: bool = False) -> tuple[Any, ...]:
        with job.lock:
            files = [str(path) for path in job.outputs]
            errors = job.errors
        # Do not expose source folders or copy their results into Gradio's cache:
        # users open folder-batch files locally; exact original timestamps remain.
        downloadable = [] if job.folder_batch else files
        # Gradio may transcode other containers for preview; only offer an
        # already browser-friendly MP4 here, avoiding hidden expensive work.
        preview = next((path for path in reversed(downloadable) if Path(path).suffix.lower() == ".mp4"), None)
        detail = f"{len(files)}/{len(job.sources)} outputs · {errors} errors · {job.elapsed_detail()}"
        if job.cancel.is_set() and not finished:
            detail += " · cancellation pending (waiting for safe cleanup)"
        return (
            None if finished else job,
            gr.update(interactive=finished),
            gr.update(interactive=not finished),
            status_html(progress, title, detail),
            "\n".join(lines),
            downloadable,
            preview,
        )

    try:
        worker.start()
        yield snapshot("Processing")
        last_timer_update = time.monotonic()
        while True:
            try:
                fraction, message = job.reports.get(timeout=0.1)
            except queue.Empty:
                if job.done.is_set():
                    break
                if time.monotonic() - last_timer_update >= 1:
                    yield snapshot("Stopping safely" if job.cancel.is_set() else "Processing")
                    last_timer_update = time.monotonic()
                continue
            # Drain a bounded group per update, keeping UI traffic bounded too.
            messages = [(fraction, message)]
            for _ in range(63):
                try:
                    messages.append(job.reports.get_nowait())
                except queue.Empty:
                    break
            for fraction, message in messages:
                progress = max(progress, fraction)
                timestamp = datetime.now().strftime("%H:%M:%S")
                # Bound individual lines as well as total stored log entries.
                for line in message.splitlines()[:10]:
                    lines.append(f"[{timestamp}] {line[:1500]}")
            yield snapshot("Stopping safely" if job.cancel.is_set() else "Processing")
            last_timer_update = time.monotonic()
        worker.join()
        with job.lock:
            errors, fatal = job.errors, job.fatal
            job.busy = False
        if job.cancel.is_set():
            title = "Cancelled"
        elif fatal:
            title = "Unable to start"
        else:
            progress = 1.0
            title = "Finished with errors" if errors else "Complete"
        lines.append(f"[{datetime.now():%H:%M:%S}] {title}.")
        yield snapshot(title, finished=True)
    finally:
        # Keep the shared Gradio GPU slot until the worker actually exits, even
        # when the browser closes the generator. Never overlap abandoned work.
        job.cancel.set()
        if worker.ident is not None:
            worker.join()
        with job.lock:
            job.busy = False


def operation_controls(enabled: bool, count: int) -> tuple[Any, ...]:
    """Disable related controls without clearing their remembered values."""
    return tuple(gr.update(interactive=bool(enabled)) for _ in range(count))


def source_controls(mode: str, source: str) -> tuple[Any, ...]:
    is_batch = mode == "Batch"
    return (gr.update(visible=not is_batch), gr.update(visible=is_batch),
            gr.update(visible=is_batch and source == "Upload files"),
            gr.update(visible=is_batch and source == "Local folders"))


def choose_folders() -> Any:
    try:
        selected = select_local_folders()
        return gr.update() if selected is None else selected
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"Cannot open local folder window: {exc}") from exc


def apply_profile(name: str) -> tuple[Any, ...]:
    """Populate enhancement controls without changing uploads or file retention.

    Jobs snapshot the actual controls at submission, so manual adjustments and
    already queued/running jobs are independent of subsequent preset changes.
    """
    if name not in PROFILES:
        raise gr.Error("Unknown processing profile.")
    profile = PROFILES[name]
    settings = profile.settings
    return (
        settings.upscale_model, str(settings.scale), settings.face_model,
        settings.face_weight, settings.rife_version, settings.target_fps,
        settings.fp16, settings.tile, str(settings.rife_scale).removesuffix(".0"),
        profile.description,
        settings.output_resolution,
    )


def build_app() -> gr.Blocks:
    defaults = PROFILES["Normal"].settings
    with gr.Blocks(
        title="Video Studio · RTX 4070 Ti",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="teal"),
        css=CSS,
        analytics_enabled=False,
    ) as demo:
        state = gr.State(value=None, delete_callback=release_session)
        gr.Markdown(
            "# Video Studio\nDetail faces → Interpolate → Upscale\n\n"
            "Local enhancement for RTX 4070 Ti / 12 GB · CUDA + FP16 · Original audio retained by the core",
            elem_id="hero",
        )
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 1 · Source videos")
                mode = gr.Radio(["Single video", "Batch"], value="Single video", label="Processing mode")
                single = gr.File(label="Upload video", type="filepath", file_count="single", file_types=["video"])
                batch_source = gr.Radio(["Upload files", "Local folders"], value="Upload files",
                                        label="Batch source", visible=False)
                batch = gr.File(
                    label="Upload batch (processed in order)", type="filepath",
                    file_count="multiple", file_types=["video"], visible=False,
                )
                with gr.Column(visible=False) as folder_panel:
                    folder_picker = gr.Button("Select / drag folders from Explorer")
                    folder_text = gr.Textbox(label="Local folders (one absolute path per line)", lines=4)
                    gr.Markdown("The button opens a Windows desktop window: drag one or more Explorer folders into it, or Browse to add them. Subfolders are included. Videos are processed in place; outputs are saved beside originals with matching creation/modified dates. Previous *_xxxxxxxx.mkv results are excluded. Folder results are listed in the log, not exposed for browser download. Use only on your own local desktop.")
                gr.Markdown("Files stay local. Only completed files inside `outputs/` are published for download.")
                gr.Markdown("### 2 · Enhancement settings")
                profile = gr.Dropdown(
                    list(PROFILES), value="Normal", label="Processing profile",
                    info="Selecting a profile replaces enhancement settings for single videos and batches; you can then customize them.",
                )
                profile_note = gr.Markdown(PROFILES["Normal"].description)
                gr.Markdown("#### Operations · select any combination")
                enable_upscale = gr.Checkbox(value=True, label="Apply Upscaling")
                upscale = gr.Dropdown(UPSCALE_MODELS, value=defaults.upscale_model, label="Upscale model (last stage)")
                scale = gr.Dropdown(["1", "2", "4"], value=str(defaults.scale), label="Output scale")
                resolution = gr.Dropdown(["FHD", "4K", "Scale"], value=defaults.output_resolution, label="Output resolution",
                                         info="FHD fits 1920×1080; 4K fits 3840×2160. Aspect ratio retained; Scale uses the multiplier above.")
                with gr.Accordion("SeedVR2 · 3B FP8 video upscale", open=False):
                    seed_batch = gr.Dropdown([1, 5, 9], value=5, label="SeedVR2 temporal batch size")
                    seed_noise = gr.Slider(0, 0.1, value=0.03, step=0.01, label="SeedVR2 latent noise")
                    gr.Markdown("Select SeedVR2 above and FHD/4K. CPU BlockSwap and VAE tiling are automatic. Single mode uses a ComfyUI video workflow (≤8 s / 240 processed frames); Batch uses the streaming CLI. These are starting settings, not guaranteed 12 GB fits. For video restoration alone, turn Face Restoration and RIFE off; RIFE before SeedVR2 increases frame count and processing time.")
                enable_face = gr.Checkbox(value=True, label="Apply Face Restoration")
                face = gr.Dropdown(["ComfyUI", "GFPGAN", "Off"], value=defaults.face_model, label="Face restoration")
                weight = gr.Slider(0, 1, value=defaults.face_weight, step=0.05, label="Face restoration weight")
                gr.Markdown("ComfyUI uses Impact FaceDetailer + Ultralytics + Juggernaut XL. GFPGAN weight applies only to GFPGAN. Generated detail may change identity or flicker; fixed seeds do not guarantee temporal consistency.")
                with gr.Accordion("ComfyUI FaceDetailer workflow", open=False):
                    checkpoint = gr.Textbox(value="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", label="ComfyUI checkpoint filename")
                    detector = gr.Textbox(value="bbox/face_yolov8m.pt", label="Ultralytics detector")
                    positive = gr.Textbox(value="photorealistic face, natural skin texture, subtle pores, preserve facial features, consistent lighting", label="Positive prompt")
                    negative = gr.Textbox(value="plastic skin, waxy skin, oversharpened, deformed face, artifacts, excessive skin texture", label="Negative prompt")
                    denoise = gr.Slider(0.05, 1, value=0.25, step=0.05, label="FaceDetailer denoise")
                    steps = gr.Number(value=20, minimum=1, maximum=100, precision=0, label="FaceDetailer steps")
                    cfg = gr.Slider(1, 20, value=5, step=0.5, label="FaceDetailer CFG")
                    seed = gr.Number(value=42, minimum=0, maximum=2**53-1, precision=0, label="Fixed seed (same each frame)")
                    guide = gr.Dropdown([512, 768, 1024], value=768, label="FaceDetailer guide size")
                    gr.Markdown("Start the project-contained dedicated ComfyUI server with scripts/Run-ComfyUI.ps1 before processing. Missing nodes/models or GPU-release failures stop the job; no GFPGAN fallback.")
                enable_rife = gr.Checkbox(value=True, label="Apply RIFE (frame interpolation)")
                with gr.Row():
                    rife = gr.Dropdown(["4.26", "4.25", "Off"], value=defaults.rife_version, label="RIFE version")
                    fps = gr.Number(value=defaults.target_fps, minimum=1, maximum=240, label="Target FPS")
                gr.Markdown("Unchecked operations are skipped; their settings are retained. Profiles do not change these checkboxes. Select at least one operation. Without RIFE, source FPS is retained.")
                with gr.Accordion("GPU & file options", open=False):
                    fp16 = gr.Checkbox(value=defaults.fp16, label="FP16 (recommended for RTX 4070 Ti)")
                    tile = gr.Number(value=defaults.tile, minimum=0, precision=0, label="Upscale tile size (0 = no tiling)")
                    rife_scale = gr.Dropdown(["1", "0.5", "0.25"], value="1", label="RIFE processing scale")
                    keep_temp = gr.Checkbox(value=False, label="Keep temporary frames")
                    gr.Markdown("For large frames, use tiling and a smaller RIFE scale. Cleanup and CUDA availability are handled by the pipeline.")
                with gr.Row():
                    run = gr.Button("Enhance video(s)", variant="primary")
                    cancel = gr.Button("Cancel safely", variant="stop", interactive=False)
            with gr.Column(scale=6):
                gr.Markdown("### 3 · Job activity")
                status = gr.HTML(status_html(0, "Ready", "Upload a video to begin"), elem_id="job-status")
                logs = gr.Textbox(label="Live pipeline log", lines=14, max_lines=14, interactive=False, autoscroll=True)
                gr.Markdown("### 4 · Results")
                preview = gr.Video(label="MP4 preview (MKV outputs are download-only)", interactive=False, format=None)
                downloads = gr.File(label="Download completed videos", file_count="multiple", interactive=False)
                gr.Markdown("Outputs are MKV to preserve original audio codecs. Download to play locally; no extra preview encode is performed. Batch failures do not remove completed videos.")

        for control in (mode, batch_source):
            control.change(source_controls, inputs=[mode, batch_source],
                           outputs=[single, batch_source, batch, folder_panel],
                           queue=False, show_progress="hidden")
        folder_picker.click(choose_folders, inputs=None, outputs=folder_text,
                            concurrency_limit=1, concurrency_id="local-folder-picker",
                            show_progress="hidden")
        profile.input(
            apply_profile, inputs=profile,
            outputs=[upscale, scale, face, weight, rife, fps, fp16, tile, rife_scale, profile_note, resolution],
            queue=False, show_progress="hidden",
        )
        for toggle, controls in (
            (enable_upscale, [upscale, scale, tile, resolution, seed_batch, seed_noise]),
            (enable_face, [face, weight, checkpoint, detector, positive, negative, denoise, steps, cfg, seed, guide]),
            (enable_rife, [rife, fps, rife_scale]),
        ):
            toggle.change(
                partial(operation_controls, count=len(controls)), inputs=toggle, outputs=controls,
                queue=False, show_progress="hidden",
            )
        outputs = [state, run, cancel, status, logs, downloads, preview]
        submission = run.click(
            admit_job,
            inputs=[state, mode, single, batch, upscale, scale, face, weight, rife, fps, fp16, tile, rife_scale, keep_temp,
                    enable_upscale, enable_face, enable_rife, batch_source, folder_text,
                    resolution, checkpoint, detector, positive, negative, denoise, steps, cfg, seed, guide,
                    seed_batch, seed_noise],
            outputs=outputs,
            queue=False,
            concurrency_limit=1,
            concurrency_id=GPU_QUEUE,
            trigger_mode="once",
            show_progress="hidden",
        )
        submission.success(
            stream_job, inputs=state, outputs=outputs,
            concurrency_limit=1, concurrency_id=GPU_QUEUE,
            show_progress="hidden", trigger_mode="once",
        )
        cancel.click(cancel_job, inputs=state, outputs=None, queue=False, show_progress="hidden")
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    from pipeline.logging_setup import configure_logging
    configure_logging()
    if gr.__version__ != "5.49.1":
        LOGGER.warning("This UI targets Gradio 5.49.1; found %s", gr.__version__)
    port = int(os.getenv("APP_PORT", os.getenv("PORT", "7860")))
    if not 1 <= port <= 65535:
        raise ValueError("APP_PORT must be between 1 and 65535.")
    build_app().launch(
        server_name=os.getenv("APP_HOST", os.getenv("HOST", "127.0.0.1")),
        server_port=port,
        share=False,
        inbrowser=False,
        max_file_size=os.getenv("APP_MAX_UPLOAD_SIZE", "2gb"),
        allowed_paths=[str(OUTPUTS)],
        show_error=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Application terminated with an error")
        raise