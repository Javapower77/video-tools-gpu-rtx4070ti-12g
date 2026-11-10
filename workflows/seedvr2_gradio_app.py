#!/usr/bin/env python3
"""
Minimal Gradio frontend for SeedVR2 Video Upscaler CLI
Optimized for RTX 4070 Ti 12 GB (3B FP8 + BlockSwap + VAE tiling)

Requirements:
  pip install gradio

Usage:
  1. Place this file next to (or point CLI_PATH to) inference_cli.py
     from the ComfyUI-SeedVR2_VideoUpscaler pack.
  2. python seedvr2_gradio_app.py
  3. Open the local URL shown in the terminal.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration – edit these if needed
# ---------------------------------------------------------------------------

# Path to inference_cli.py (absolute or relative to this script)
CLI_PATH = os.environ.get(
    "SEEDVR2_CLI",
    str(Path(__file__).resolve().parent / "inference_cli.py"),
)

# Default model directory (None = let CLI use its own default)
MODEL_DIR = os.environ.get("SEEDVR2_MODEL_DIR", None)

# Temporary working directory for uploads / outputs
WORK_DIR = Path(tempfile.gettempdir()) / "seedvr2_gradio"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Preset flag sets (12 GB optimized)
# ---------------------------------------------------------------------------

COMMON_FLAGS = [
    "--dit_model", "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "--vae_model", "ema_vae_fp16.safetensors",
    "--batch_size", "5",
    "--uniform_batch_size",
    "--temporal_overlap", "3",
    "--prepend_frames", "4",
    "--blocks_to_swap", "28",
    "--swap_io_components",
    "--dit_offload_device", "cpu",
    "--vae_offload_device", "cpu",
    "--vae_encode_tiled",
    "--vae_decode_tiled",
    "--color_correction", "lab",
    "--input_noise_scale", "0.0",
    "--latent_noise_scale", "0.03",
    "--video_backend", "ffmpeg",
    "--10bit",
    "--cache_dit",
    "--cache_vae",
    "--cuda_device", "0",
]

PRESETS = {
    "1080p (safe 12 GB)": {
        "resolution": "1080",
        "blocks_to_swap": "28",
        "vae_tile": "512",
        "vae_overlap": "64",
    },
    "2160p / 4K (aggressive 12 GB)": {
        "resolution": "2160",
        "blocks_to_swap": "32",
        "vae_tile": "384",
        "vae_overlap": "48",
    },
}


def build_command(
    input_path: str,
    output_path: str,
    preset_name: str,
    batch_size: int,
    latent_noise: float,
    extra_args: str = "",
) -> list[str]:
    """Assemble the full CLI command list."""
    p = PRESETS[preset_name]

    cmd = [
        sys.executable,
        CLI_PATH,
        input_path,
        "--output", output_path,
        "--resolution", p["resolution"],
        "--batch_size", str(batch_size),
        "--blocks_to_swap", p["blocks_to_swap"],
        "--vae_encode_tile_size", p["vae_tile"],
        "--vae_decode_tile_size", p["vae_tile"],
        "--vae_encode_tile_overlap", p["vae_overlap"],
        "--vae_decode_tile_overlap", p["vae_overlap"],
        "--latent_noise_scale", str(latent_noise),
    ]

    # Common flags (skip ones we already overrode)
    skip = {
        "--resolution", "--batch_size", "--blocks_to_swap",
        "--vae_encode_tile_size", "--vae_decode_tile_size",
        "--vae_encode_tile_overlap", "--vae_decode_tile_overlap",
        "--latent_noise_scale",
    }
    i = 0
    while i < len(COMMON_FLAGS):
        flag = COMMON_FLAGS[i]
        if flag in skip:
            # skip flag + its value (if any)
            if i + 1 < len(COMMON_FLAGS) and not COMMON_FLAGS[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        cmd.append(flag)
        if i + 1 < len(COMMON_FLAGS) and not COMMON_FLAGS[i + 1].startswith("--"):
            cmd.append(COMMON_FLAGS[i + 1])
            i += 2
        else:
            i += 1

    if MODEL_DIR:
        cmd.extend(["--model_dir", MODEL_DIR])

    if extra_args.strip():
        cmd.extend(extra_args.strip().split())

    return cmd


def run_upscale(
    video_file,
    preset: str,
    batch_size: int,
    latent_noise: float,
    extra_args: str,
    progress=gr.Progress(track_tqdm=False),
):
    if video_file is None:
        raise gr.Error("Please upload a video first.")

    if not Path(CLI_PATH).is_file():
        raise gr.Error(
            f"CLI not found at:\n{CLI_PATH}\n\n"
            "Set the SEEDVR2_CLI environment variable or edit CLI_PATH at the top of this script."
        )

    # Prepare paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_path = Path(video_file)
    out_name = f"seedvr2_{preset.split()[0]}_{ts}.mp4"
    out_path = WORK_DIR / out_name

    cmd = build_command(
        input_path=str(in_path),
        output_path=str(out_path),
        preset_name=preset,
        batch_size=int(batch_size),
        latent_noise=float(latent_noise),
        extra_args=extra_args,
    )

    progress(0.05, desc="Starting SeedVR2 CLI…")
    log_lines = [" ".join(cmd), ""]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            # crude progress hint
            if "Chunk" in line or "batch" in line.lower() or "%" in line:
                progress(0.3, desc=line[:80])
            yield None, "\n".join(log_lines[-40:])  # live log

        proc.wait()
        if proc.returncode != 0:
            raise gr.Error(
                f"CLI exited with code {proc.returncode}.\n"
                "See log for details."
            )

        if not out_path.exists():
            # CLI sometimes writes next to input or with a different name
            candidates = list(WORK_DIR.glob(f"*{ts}*")) + list(in_path.parent.glob("*seedvr*"))
            if candidates:
                out_path = candidates[0]
            else:
                raise gr.Error("Output file not found after successful run.")

        progress(1.0, desc="Done")
        yield str(out_path), "\n".join(log_lines)

    except Exception as e:
        log_lines.append(f"\n[ERROR] {e}")
        yield None, "\n".join(log_lines)
        raise


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SeedVR2 12 GB Upscaler") as demo:
    gr.Markdown(
        """
        # SeedVR2 Video Upscaler (12 GB optimized)
        Upload a short clip (≤ 8 s recommended).  
        Uses **3B FP8 + BlockSwap + VAE tiling** – safe for RTX 4070 Ti 12 GB.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Input video", sources=["upload"])
            preset = gr.Radio(
                choices=list(PRESETS.keys()),
                value="1080p (safe 12 GB)",
                label="Target preset",
            )
            batch_size = gr.Slider(
                minimum=1, maximum=21, step=4, value=5,
                label="batch_size (must be 4n+1: 1,5,9,13…)",
            )
            latent_noise = gr.Slider(
                minimum=0.0, maximum=0.15, step=0.01, value=0.03,
                label="latent_noise_scale",
            )
            extra = gr.Textbox(
                label="Extra CLI flags (optional)",
                placeholder="e.g. --enable_debug",
                lines=1,
            )
            run_btn = gr.Button("Upscale", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Upscaled result")
            log = gr.Textbox(label="CLI log", lines=18, max_lines=30)

    run_btn.click(
        fn=run_upscale,
        inputs=[video_in, preset, batch_size, latent_noise, extra],
        outputs=[video_out, log],
    )

    gr.Markdown(
        f"""
        **CLI path:** `{CLI_PATH}`  
        **Work dir:** `{WORK_DIR}`  

        First run downloads models automatically.  
        If you hit OOM → lower batch_size to 1 or switch to the 1080p preset.
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=2).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
