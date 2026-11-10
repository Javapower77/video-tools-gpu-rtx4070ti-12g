# Optional SeedVR2 setup (Windows / PowerShell 7)

## Verified scope and contracts

These optional scripts target only `d:\VSCode\video-tools-gpu-rtx4070ti-12g`.
Setup isolates Python 3.11 under
`vendor/ComfyUI/.venv`, constrains CUDA torch/torchvision, verifies official Git
origins and tracked-file cleanliness, and leaves existing revisions unchanged.
Launch already requires Impact-Pack/Subpack and excludes other custom nodes and
their prestartup scripts. Model downloads already require explicit selection,
SHA256 verification, and refuse to overwrite final or partial files.

This change only adds optional SeedVR2 provisioning, conditional launch
allowlisting, and explicit downloads; existing FaceDetector checksum requirements
and the Juggernaut URL/hash remain unchanged. No application or vendor-source
changes are needed.

## Explicit opt-in steps

1. Run `scripts/Setup-ComfyUI.ps1` with **`-SeedVR2`** when intentionally ready
   to clone/install dependencies. Without this switch, SeedVR2 is not provisioned.
   The official `numz/ComfyUI-SeedVR2_VideoUpscaler` checkout goes into
   `vendor/ComfyUI/custom_nodes/seedvr2_videoupscaler`, pinned to
   **`4490bd1f482e026674543386bb2a4d176da245b9`** (2.5.24).
   Only a newly cloned checkout receives an explicit detached checkout. Existing
   checkouts are never pulled, reset, cleaned, or checked out; a mismatched HEAD,
   unofficial origin, or tracked modifications fails setup for manual review.
   Its `requirements.txt` is installed using the same isolated ComfyUI Python,
   CUDA constraints, and pip policy as Impact. No SeedVR2 installer is executed.
2. Separately run `scripts/Download-ComfyUIModels.ps1` with **`-SeedVR2`** to
   manually opt in to downloading the 3B FP8 DiT and FP16 VAE. Review upstream
   licenses and available disk space first. No switch means no downloads. Existing
   files are hashed, never overwritten; failed `.part` files remain for manual
   inspection/removal. Downloading models does not install the node pack.
3. Run `scripts/Run-ComfyUI.ps1`. When `seedvr2_videoupscaler` is present, it is
   verified against the official numz origin and checked for tracked changes,
   and its HEAD must equal the pinned commit above,
   then appended to the existing Impact allowlist. When absent, old non-SeedVR2
   runs remain supported. Launch never installs dependencies or downloads models.
   Existing flags remain `-Listen`, `-Port`, `-LowVram`, and `-PassThru`.

Setup retains `-ComfyRevision`, `-ComfyRepository`, and `-TorchStack`.
Download retains `-Juggernaut`, `-FaceDetector`, and the mandatory
`-FaceDetectorSha256` when the face detector is selected. Switches can be combined.
The application `.env` and `.venv` are untouched.

## Published model integrity

Both files belong in **`vendor/ComfyUI/models/SEEDVR2/`**:

| File | Published SHA256 |
| --- | --- |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | `3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff` |
| `ema_vae_fp16.safetensors` | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |

Verified directly from the [official model registry at the pinned commit](https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/model_registry.py).
The registry defaults both entries to `numz/SeedVR2_comfyUI`. Official downloads:

- [3B FP8 DiT](https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors)
- [FP16 VAE](https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors)

The URLs use upstream `main`; published pinned-registry hashes determine accepted
bytes. If upstream replaces a file, verification fails rather than trusting it.
The pinned [constants.py](https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/constants.py)
defines `HUGGINGFACE_BASE_URL` as `https://huggingface.co/{repo}/resolve/main/{filename}`
and `SEEDVR2_FOLDER_NAME` as `SEEDVR2`. Both explicit resolve URLs returned HTTP
200 to HEAD requests on 2026-10-03; no model bodies were downloaded. Registry
hashes match `pipeline/seedvr2_backend.py`'s `MODEL_HASHES`.

## No first-use automatic download in the application

**Both local models are mandatory before submitting SeedVR2 work.** The app's
`pipeline.seedvr2_backend.check_models` checks existence, nonempty content, and
these SHA256 hashes before API/CLI execution. Missing or corrupt models fail
preflight; use the explicit downloader manually, not first-use auto-download.

Upstream [downloads.py](https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/downloads.py)
can download missing models and replace invalid ones using `urllib`. The launcher's
HF offline flags and Impact `skip_download_model` marker are **not** a network
sandbox and do not disable this direct upstream downloader. Direct ComfyUI UI/CLI
use bypasses the app preflight; do not rely on it to enforce no-download behavior.
Prevent concurrent modification of model files after preflight.

## Compatibility and validation limits

The pinned upstream README recommends Python 3.12+, while this project mandates
Python 3.11. Setup preserves 3.11 and uses the published requirements; successful
dependency resolution is not a claim of GPU/runtime compatibility. Start with a
short clip, 3B FP8, SDPA, CPU offload, VAE tiling, and batch size 5 on 12 GB VRAM.
Avoid optional compilation/attention packages until separately validated.

This edit does not install packages, clone repositories, fetch model weights, or
run GPU inference. Validation uses local PowerShell parsing and read-only source,
hash, and URL checks; real setup/download/inference remains an explicit user action.

## Standalone CLI (optional, separate from the server)

Run from `vendor/ComfyUI` using its dedicated `.venv/Scripts/python.exe` and
`custom_nodes/seedvr2_videoupscaler/inference_cli.py`; do not use the application's
Python or a separate SeedVR2 environment. The input is a positional video/image
path or a directory of media files. `--output` selects a destination;
`--output_format png` writes frames, while `--output_format mp4` writes video.
Use an unused output directory; upstream CLI writes are not protected by the
downloader's no-overwrite policy. Stop the server first to avoid GPU contention.

The pinned CLI supports `--model_dir models/SEEDVR2` and
`--dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors`. It has **no `--vae_model`**:
it always uses the registry's `DEFAULT_VAE`, `ema_vae_fp16.safetensors`.

For an initial landscape FHD experiment, the backend uses `--resolution 1080`,
`--max_resolution 1920`, `--batch_size 5`, `--uniform_batch_size`,
`--chunk_size 45`, `--temporal_overlap 3`, `--prepend_frames 0`,
`--cuda_device 0`, `--attention_mode sdpa`, `--blocks_to_swap 28`,
`--swap_io_components`, `--dit_offload_device cpu`, `--vae_offload_device cpu`,
`--tensor_offload_device cpu`, `--vae_encode_tiled`, `--vae_decode_tiled`,
`--vae_encode_tile_size 512`, `--vae_decode_tile_size 512`,
`--vae_encode_tile_overlap 64`, `--vae_decode_tile_overlap 64`,
`--color_correction lab`, `--input_noise_scale 0`, and `--latent_noise_scale 0.03`.
Resolution means the **short edge**, not a fixed width/height. Batch sizes must
follow 4n+1; chunk size counts new frames, separately from the inference batch.
Keep prepend at zero for this pin: its single-GPU CLI does not remove first-chunk
prepended frames correctly. These settings are not a 12 GB fit, speed, quality,
or successful-inference guarantee.

Direct CLI invokes upstream model validation/download code and bypasses the app's
strict model preflight. Independently verify both files before direct use;
offline environment flags alone do not block urllib. PNG output contains no audio;
the app runner separately owns encoding/audio muxing. Do not assume standalone
CLI video output preserves the original audio.

### Validation scope

Only local PowerShell parsing and read-only source/URL verification are performed
for this change. No setup, pip install, clone/checkout, model transfer, CLI/server
launch, or GPU inference is run. Runtime compatibility remains unvalidated.

