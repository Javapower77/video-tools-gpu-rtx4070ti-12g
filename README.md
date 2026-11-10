# RTX 4070 Ti Video Enhancement

## SeedVR2 video upscaling

Select **SeedVR2** in the upscale dropdown and choose **FHD** or **4K**. FHD fits the original aspect ratio inside 1920×1080; 4K fits inside 3840×2160, with even dimensions for encoding. Portrait/4:3 videos are not stretched or cropped. SeedVR2 replaces UltraSharp for that job; it does not run both upscalers. Normal/Boost presets still select UltraSharp, so choose SeedVR2 after selecting a profile.

- **Single video:** project-contained ComfyUI runs `workflows/seedvr2_api.json` with temporally grouped frames. Limited to 8 seconds and 240 processed frames because the video nodes materialize the full sequence. At 60 FPS the frame bound permits about four seconds; use Batch for larger sequences.
- **Batch (uploaded files or local folders):** the Gradio worker invokes the contained SeedVR2 CLI separately for each video, using 45-frame streaming chunks. It does not treat a PNG folder as a temporal sequence. Existing batch cancellation, final encoding/audio copying and original folder-output dates remain supported.
- **12 GB starting settings:** 3B FP8, temporal batch 5, SDPA, CPU offload and BlockSwap 28 for FHD / 32 for 4K; tiled VAE 512/64 for FHD / 384/48 for 4K. Temporal batch 1/5/9 and latent noise are editable. These settings are not benchmarked guarantees; 4K can still exhaust VRAM/RAM. The CLI uses prepend 0 to avoid a source-inspected duplication issue; frame counts are checked strictly.
- For SeedVR2-only restoration, turn Face Restoration and RIFE off. When enabled, the existing order remains faces → RIFE → SeedVR2; interpolation increases the frames SeedVR2 must process. Final output remains SDR 8-bit MKV with original audio; no 10-bit/HDR preservation is added.

See [SeedVR2 setup](docs/seedvr2-setup.md): explicitly run `scripts/Setup-ComfyUI.ps1 -SeedVR2`, then `scripts/Download-ComfyUIModels.ps1 -SeedVR2`, and restart the unified launcher. The pack is pinned to a reviewed revision and enabled in the backend allowlist when installed. Both local model hashes are checked before inference; missing/corrupt models fail rather than triggering implicit downloads. No installation/download was performed while implementing this feature. Runtime GPU inference and quality remain unvalidated.

Windows 10/11 Python **3.11-only** Gradio video enhancement application targeting an NVIDIA RTX 4070 Ti (12 GB VRAM) and 64 GB RAM.

**Status:** the UI, modular pipeline, BasicSR compatibility shim and loopback ComfyUI FaceDetailer integration are implemented. Full-video FaceDetailer inference, GPU handoff and performance still require validation with the installed upstream stack. This documentation update ran no installers or model downloads; no new full-suite test result is claimed here.

## Pipeline and defaults

`app.py` provides the Gradio UI; `pipeline/` contains configuration, extraction, upscaling, face restoration, interpolation, encoding and orchestration.

| Setting | Default |
| --- | --- |
| Upscaler / output | `4x-UltraSharp`, fitted within FHD 1920×1080 |
| Face restoration | ComfyUI Impact FaceDetailer + Impact Subpack face detector + Juggernaut XL v9 |
| Interpolation | Practical-RIFE 4.26, target 60 FPS |
| Precision | CUDA FP16 for upscaling/RIFE; ComfyUI manages its own precision; alternative GFPGAN uses FP32 |
| Upscaling tile | 256 pixels |
| Output | `.mkv`, original audio stream copy where compatible |

Processing order: **extract frames → ComfyUI FaceDetailer → existing Practical-RIFE → UltraSharp last → encode and mux original audio**. Face restoration and interpolation operate at extracted/source dimensions, before final upscaling. Select **GFPGAN** as a local, face-detection-aware alternative, or **Off** to skip restoration; ComfyUI failures never silently fall back to GFPGAN. Other existing upscalers remain selectable at the last stage.

The core releases local stage models/GPU memory. ComfyUI requires a separate, verified server-side release before RIFE (see below). RIFE still uses the application-owned helper subprocess, existing vendor checkout and selected model directory. Local stages detect CUDA and can fall back to CPU; the dedicated ComfyUI launcher requires CUDA. RIFE Off retains source FPS; an already-matching target FPS does not invoke interpolation. Existing folder-output locations and Windows creation/modified-time preservation are unchanged.

The UI supports single videos and sequential batches, model/scale/FPS controls, face restoration, FP16, temporary-file retention, synchronized live progress/logs, cancellation and completed-file downloads. Successful batch outputs survive later failures. **CodeFormer and TensorRT are not supported.**

### Normal and Boost profiles

After uploading a single video or batch, select **Processing profile** in Enhancement settings:

| Setting | Normal (default) | Boost |
| --- | --- | --- |
| Upscaler / output resolution | `4x-UltraSharp` / FHD fit (1920×1080 box) | `4x-UltraSharp` / 4K fit (3840×2160 box) |
| Face restoration | ComfyUI FaceDetailer, Juggernaut XL v9, denoise 0.25 | Same |
| Interpolation | Standard RIFE 4.26, 60 FPS | Same |
| Upscale / RIFE precision | CUDA FP16 | CUDA FP16 |
| Upscale tile / RIFE processing scale | 256 / 1 | 256 / 1 |

Selecting a profile replaces the main enhancement controls, including output resolution; every control remains editable afterward. The FaceDetailer accordion's prompts/checkpoint/detector/denoise/steps/CFG/seed/guide controls are initialized separately and are **not reset by profile selection**. Profiles apply to all videos in a submitted batch. Uploads, operation checkboxes and **Keep temporary frames** are not changed. Jobs capture the actual settings when submitted; changing the profile afterward does not modify queued/running jobs. Presets are defined in `pipeline/profiles.py`; selection is session-local, not saved to `.env`.

**Boost is a demanding quality candidate, not a benchmarked performance mode or a guarantee of better quality.** Use a dedicated ComfyUI instance, guide size 768 and upscaler tile 256 as starting points on 12 GB. RIFE runs **before** upscaling, so choosing 4K output does not feed final 4K frames into RIFE; large source frames can still require RIFE processing scale 0.5 or 0.25. Tiling only limits upscaler inference memory, not FaceDetailer/RIFE memory or native-scale output buffers. Test a short clip first. RIFE 4.26 Heavy is excluded until a trusted package's compatibility is verified. Selecting a profile downloads nothing.

### Output resolution versus legacy scale

**Output resolution** offers `FHD`, `4K` and `Scale`. FHD fits inside the **landscape 1920×1080 box**; 4K fits inside **3840×2160**. Frames are resized proportionally, without stretch or crop, with even dimensions for H.264/yuv420p (small rounding differences are possible). There is no padding to force the full box. A portrait 1080×1920 frame remains portrait: approximately 606×1080 for FHD or 1214×2160 for 4K, not a landscape crop.

FHD/4K override the multiplier; **Output scale** is the optional legacy mode used only when resolution is `Scale`. Scale 1 bypasses upscaling only in that mode; FHD/4K still invoke the upscaler at scale 1. Disabling Apply Upscaling resets the effective resolution to Scale and multiplier to 1, retaining source dimensions. `pipeline/config.py`'s `Settings()` defaults deliberately remain legacy-compatible (`RealESRGAN_x2plus`, ×2, GFPGAN, `Scale`); the **UI** starts from the new Normal preset instead.

### Select individual operations

Use **Apply Upscaling**, **Apply Face Restoration**, and **Apply RIFE (frame interpolation)** to choose any of the seven nonempty combinations, including face-only or RIFE-only processing. All three start checked. Unchecking an operation disables its related controls without clearing values and skips that stage and its model-file requirements. Recheck it to reuse those settings. Normal/Boost selection changes preset settings but preserves operation checkboxes.

At least one effective operation is required: output scale 1 with resolution `Scale` bypasses upscaling, and the `Off` model choices bypass face restoration/RIFE even if their checkbox is checked. Without upscaling, source dimensions are retained; without RIFE, source FPS is retained after the existing CFR normalization. Extraction, encoding and audio muxing still run for every accepted job. Selections apply to every batch member and are captured at submission, not changed in running jobs.

### Local folder batches (Windows desktop)

In **Batch** mode, choose **Local folders**, then **Select / drag folders from Explorer**. A native Windows window opens on the computer running the application. Drag one or more folders from Explorer into its list, or use Browse repeatedly; select **Use folders** to return their paths to the UI. You can also edit absolute paths directly, one folder per line. Cancel leaves the current selection unchanged. Browser folder drops are not used: browsers do not reliably expose original disk paths for saving beside sources.

All supported videos in the selected folders **and subfolders** are snapshotted at submission, deduplicated across overlapping folders, and processed sequentially using the current profile/operation settings. Symlink files and directories resolving outside the selected root are skipped; enumeration errors fail visibly. Names matching `*_xxxxxxxx.mkv` (eight hexadecimal characters) and `.partial.mkv` files are excluded to avoid processing previous results. A legitimate original with that naming pattern will also be excluded.

Each output is saved in its original video's directory as **`original_stem_<8 random hexadecimal characters>.mkv`**. For example, `holiday.mp4` becomes `holiday_a18c72ef.mkv`; the container remains MKV, not the original extension. Originals are never overwritten. Long filenames, insufficient disk space, or unwritable folders cause that item to fail without removing earlier successful outputs.

The Windows file API captures original **Date created** and **Date modified** before processing and applies them to the completed output before publication, with an exact FILETIME check. If the filesystem cannot retain those values, the item fails and its partial output is removed. The app does not change source creation/modified times (reading may affect access time). Preserving dates requires Windows and a filesystem supporting the source timestamp precision.

Folder results are saved **only beside their sources** and full paths appear in the live/persistent log. They are not browser downloads or previews: source folders are never added to Gradio's allowed paths and no browser-cache copy is made. Single uploads and uploaded-file batches retain the existing `outputs/` download behavior. Temporary frames still use `TEMP_DIR` and the existing cleanup option.

The native selector requires Windows PowerShell and an interactive desktop; it runs as the current user without elevation or execution-policy bypass. It opens on the **server computer**, not a remote browser's computer. Keep this application restricted to trusted local use: entering folder paths grants processing read/write access to those local folders. The window's interactive Explorer drag/drop needs a manual desktop check; no new packages are required.

## Prerequisites

- Windows 10/11 and the Windows Python launcher `py` with **Python 3.11** installed. The existing app setup accepts PowerShell 5.1/7; **`Run.ps1` and all three ComfyUI scripts require PowerShell 7 (`pwsh`)**, not Windows PowerShell 5.1.
- `ffmpeg`, `ffprobe` and `git` on PATH. FFmpeg must include `libx264`.
- Compatible NVIDIA driver for each environment's CUDA wheels: the app uses CUDA 12.4, while current ComfyUI checkouts select CUDA 12.8 and may need a newer driver. The isolated torch stack is selected independently of the app. A separately installed CUDA Toolkit is not required for PyTorch wheels; custom compilation may need it.
- Adequate free disk space and permission to write in this project directory.

Scripts do not install system prerequisites, request elevation or bypass execution policy. If policy blocks a script, review it and use your organization's approved execution-policy procedure. Installing Python packages and cloning repositories executes third-party code; review upstream licenses and sources first.

## Setup

From the project root, run `./scripts/Setup.ps1`, or use `setup_venv.bat`. The batch wrapper forwards parameters and uses PowerShell 7 when available, otherwise Windows PowerShell.

Setup validates Python 3.11, FFmpeg, ffprobe and Git, creates/checks `.venv`, upgrades pip, installs **torch 2.5.1 + torchvision 0.20.1 from the cu124 index first**, then installs `requirements.txt` and runs `pip check`. Exact `+cu124` pins prevent the requirements step from silently replacing the CUDA wheels. A direct requirements install into a fresh environment using only PyPI will not find those wheels; use the setup script. NumPy is fixed at 1.26.4 for the legacy restoration stack. FFmpeg is invoked directly; `ffmpeg-python` is not needed.

Setup clones `https://github.com/hzwer/Practical-RIFE.git` into `vendor/Practical-RIFE` without installing its requirements. An existing checkout is preserved without pull/reset. The commit hash is printed: record/review that revision for reproducibility, because the initial clone follows upstream HEAD rather than a frozen commit. Setup copies `.env.example` only if `.env` is absent and creates the default data directories. A partial environment is not deleted automatically on failure; inspect it and retry after resolving the error.

Use `./scripts/Setup.ps1 -DownloadModels` to opt into model downloads during setup, or run `./scripts/Download-Models.ps1` separately. These are network operations only when explicitly executed.

### BasicSR compatibility

`pipeline/compat.py` supplies BasicSR 1.4.2's removed `torchvision.transforms.functional_tensor` import before ML imports, exposing `rgb_to_grayscale` from the supported module. Installed site-packages are not patched. Dependency installation and `pip check` do **not** validate runtime ML or GPU compatibility.

### Dedicated ComfyUI setup (explicit opt-in)

ComfyUI is contained under **`vendor/ComfyUI/`**, including its own **`.venv/`**, `custom_nodes/`, `models/`, `input/` and `output/`. It does not use or modify the application's root `.venv`, requirements or `.env`. This is dependency isolation, **not a security sandbox**. The whole contained tree is ignored by Git. Do not commit its environments, checkouts or large model files.

Run these scripts yourself from the project root in **PowerShell 7**, only after reviewing their sources and upstream licenses:

1. **`./scripts/Setup-ComfyUI.ps1`** explicitly clones ComfyUI and the official `ltdrdata/ComfyUI-Impact-Pack` / `ComfyUI-Impact-Subpack`, creates the isolated Python 3.11 environment and installs upstream dependencies. It does **not** fetch models. Default repository is `Comfy-Org/ComfyUI`; `-ComfyRepository comfyanonymous/ComfyUI` is also accepted. `-ComfyRevision` selects a revision only for a new clone; existing checkouts are never pulled/reset/cleaned/revised automatically, and a requested revision must match existing HEAD. Origins and tracked modifications are checked; unfrozen HEAD hashes and upstream requirements are printed for review.
2. **`./scripts/Download-ComfyUIModels.ps1 -Juggernaut`** explicitly opts into the approximately **7.11 GB** Juggernaut XL v9 checkpoint download. The detector is a separate opt-in: `-FaceDetector` **requires `-FaceDetectorSha256` containing an independently verified 64-hex SHA256**. No detector hash is invented, and the switch fails before downloading if the hash is missing. Running with no selection downloads nothing.
3. **`./scripts/Run.ps1`** is the unified launcher: in one PowerShell 7 terminal it starts owned ComfyUI and Gradio as two separate Python processes, each using its own isolated environment. It waits up to 180 seconds for backend readiness (with up to one request-timeout overhead), including system stats and the required `FaceDetailer` / `UltralyticsDetectorProvider` nodes, before starting the UI. Stop any previously started manual backend yourself before unified launch: an occupied backend port is refused, not adopted or cleared. Launch never installs packages or downloads models. `./scripts/Run.ps1 -LowVram` forwards upstream low-VRAM mode to the owned backend (and disables dynamic VRAM when supported).

**Standalone backend launch remains available:** `./scripts/Run-ComfyUI.ps1` starts the dedicated foreground server on `127.0.0.1:8188`. Keep that terminal running and use `./scripts/Run.ps1 -AppOnly` in another PowerShell 7 terminal for Gradio. Standalone `-LowVram` remains supported; `-Listen` accepts only `127.0.0.1` or `::1`, and `-Port` must match `COMFYUI_URL`. App-only mode neither checks backend readiness nor terminates that separately managed server; FaceDetailer validates it when processing. See **Launch and tests** for ownership and exit behavior.

Setup's `-TorchStack Auto` detects `comfy-kitchen` / `comfy-aimdo` in ComfyUI requirements and selects **torch 2.8.0 / torchvision 0.23.0 / cu128** for those newer checkouts. Reviewed older checkouts default to **2.5.1 / 0.20.1 / cu124**. Explicit cu124 is rejected for detected modern dependencies. Constraints inside the dedicated venv prevent requirements from silently replacing the selected torch pair; future incompatible requirements fail. The script warns that current comfy-kitchen native wheels document CUDA 13 / r580+ and may need eager fallback. It changes no drivers. Setup checks CUDA with a small allocation; that is not full workflow validation.

The [upstream Impact Pack installer](https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Pack/Main/install.py) and [Subpack installer](https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Subpack/main/install.py), checked on 2026-10-02, both honor `custom_nodes/skip_download_model`. Setup creates this persistent marker **before** node dependency/installer execution, checks that both installers still mention it, and blocks the researched torchvision model-download helper. It runs Impact Pack's installer to write configuration; Subpack's model-download-only installer is **not run**, only its requirements are installed. SAM2 CUDA extension building is disabled. Upstream dependencies/build code remain third-party and unfrozen; these safeguards are not a universal network sandbox or reproducible lockfile.

Launch requires the marker and isolated CUDA environment, checks official Impact node origins/tracked files, and enables **only those two folders** with `--disable-all-custom-nodes` plus `--whitelist-custom-nodes`, including exclusion of other nodes' prestartup scripts. It uses the installed frontend, disables previews/browser auto-launch, sets Hugging Face/Transformers offline flags and adds upstream `--offline` if available. It refuses `extra_model_paths.yaml` and checkouts lacking required safety flags. Review untracked/local code and future upstream changes; an allowlist is not a sandbox. No ComfyUI Manager or arbitrary custom nodes are required.

#### ComfyUI model downloads and locations

| Model | Dedicated server location | Verification contract |
| --- | --- | --- |
| `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | `vendor/ComfyUI/models/checkpoints/` | Approximately 7.11 GB; SHA256 `c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca` checked before publication |
| `face_yolov8m.pt` | `vendor/ComfyUI/models/ultralytics/bbox/` | Independently verified caller-supplied `-FaceDetectorSha256` required; UI model name is `bbox/face_yolov8m.pt` |

Sources are [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) and [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer). The face detector URL matches upstream Subpack's installer; the detector is third-party, **not an Ultralytics-authored face model**. Hashes are checked for existing final files too; mismatches are refused without overwrite. New transfers use a same-directory `.part`, verify SHA256 and atomically publish without overwriting. Failed or existing partial files require manual inspection/removal; they are not silently resumed/deleted. No SAM/other models or unsafe pickle-loading overrides are installed. A hash verifies expected bytes, not license suitability or safe execution of a pickle-based `.pt` file. These ComfyUI downloads are separate from `Download-Models.ps1` and `WEIGHTS_DIR`.

### FaceDetailer controls, workflow and GPU handoff

In **ComfyUI FaceDetailer workflow**, edit checkpoint/detector names, positive/negative prompts, **denoise** (default 0.25), **steps** (20), **CFG** (5), **fixed seed** (42, reused each frame) and **guide size** (768; UI offers 512/768/1024). Lower denoise generally makes a more conservative edit; higher denoise/CFG or stronger texture prompts can invent detail or alter faces. Guide size controls face-detail processing, not final video resolution; increasing it can cost VRAM. The GFPGAN weight slider affects only GFPGAN. **Generated pores are synthesized detail, not recovered ground truth. Fixed seeds do not guarantee temporal consistency or preserved identity**; flicker and changed facial features remain possible.

`workflows/facedetailer_api.json` is an editable **API-format** template, not an arbitrary graph importer. The client strictly requires exactly seven node IDs/classes: LoadImage, CheckpointLoaderSimple, two CLIPTextEncode nodes, UltralyticsDetectorProvider, FaceDetailer and SaveImage. Required links are fixed: checkpoint model/CLIP/VAE and prompts/detector feed FaceDetailer; **full composited image output 0**, not face crops, feeds SaveImage output node 7. Only compatible scalar settings may be edited; no added nodes, changed links or nested graph inputs. `wildcard` must remain empty. Settings/UI overwrite checkpoint, detector, prompt and the five FaceDetailer controls listed above; per-frame upload/output paths are also assigned by the client. Remaining scalars (for example sampler, scheduler and detector thresholds) come from JSON and must match upstream `/object_info`. Missing models/nodes, schema changes or invalid PNGs stop processing with no fallback.

The backend uses bounded direct HTTP polling/upload/retrieval, **not WebSockets**, and accepts only a loopback HTTP(S) origin (literal loopback IP or `localhost` mapped to 127.0.0.1). No remote URLs, DNS hosts, redirects, proxies, credentials or URL paths/query/fragment are accepted. One static PNG is uploaded and one same-size PNG is retrieved per frame. Windows backslashes in server-returned subfolder paths are normalized to forward slashes before validation; traversal, absolute paths, unsafe filenames and paths outside the exact owned job folder remain rejected.

**A dedicated idle server with no other submitters is mandatory.** After owned work leaves the queue, the client sends `/free` with model-unload/free-memory flags and requires **two consecutive identical idle samples** of `/system_stats` CUDA `torch_vram_total` reservations. Only devices identified as **cudaMallocAsync** may retain up to **128 MiB per device**; the native allocator requires **zero reserved bytes**. Unknown allocators receive no residual allowance, and missing/invalid or unsupported device evidence fails closed. Reservations above the applicable cap are not accepted; failure to obtain stable below-cap evidence within the cleanup deadline stops the pipeline. HTTP 200 from `/free` alone does not prove release.

This is bounded reported allocator-reservation evidence, **not definitive proof that models unloaded, a zero-total-VRAM guarantee, or a guarantee that the next model fits**. It does not account for every driver allocation or other GPU process. The queue check and `/free` cannot be atomic in this API, hence no shared interactive workloads. Failure/unreachable server/unverifiable handoff stops the pipeline; **RIFE must not start after an unconfirmed handoff**.

Cancellation only deletes owned pending prompt IDs and sends a **targeted `/interrupt` with `prompt_id`** for owned running work, using unique client IDs to recover ambiguous submissions. It never globally interrupts or clears the server queue. A **current ComfyUI version supporting targeted interrupts is required**; older versions are not a safe cancellation fallback. Cleanup has its own bounded 60-second deadline, independent of the already-set cancel event; a failed cleanup is a handoff error, not successful cancellation. If cleanup also fails after an initial processing failure, the raised cleanup error includes that initial failure, and the persistent log records it rather than hiding it.

#### Retained server media

Each restoration call creates unique **`video_enhancer/<UUID>`** subfolders under ComfyUI's **`input/` and `output/`**. Uploaded and generated PNGs remain there even after success/cancellation; local `TEMP_DIR` frame cleanup and **Keep temporary frames** do not govern server files. The API has no scoped file-delete route used here, and the app **does not automatically delete server media**. Budget disk space for these extra copies. After jobs finish and the dedicated server is stopped/idle, inspect and manually remove only the intended job folders from those two trees; never clear another job's media. Server folders are not added to Gradio's downloadable paths.

## Configuration

Edit `.env` locally. The app loads it using `python-dotenv`. Relative paths resolve from the project root; `Run.ps1` sets that working directory.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEIGHTS_DIR` | `weights` | Model directory |
| `TEMP_DIR` | `temp` | Intermediate frame workspace |
| `OUTPUT_DIR` | `outputs` | Keep this default for the current UI download boundary |
| `RIFE_REPO` | `vendor/Practical-RIFE` | Reviewed upstream code checkout |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | Dedicated loopback-only ComfyUI origin; must match launcher address/port |
| `COMFYUI_TIMEOUT` | `900` | Seconds per frame including upload, execution and retrieval; finite, >0 and ≤86400; separate bounded cleanup |
| `FFMPEG` / `FFPROBE` | `ffmpeg` / `ffprobe` | Executable names or full paths |
| `HOST` / `PORT` | `127.0.0.1` / `7860` | Local UI binding; `APP_HOST` / `APP_PORT` override these |
| `APP_MAX_UPLOAD_SIZE` | `2gb` | Gradio per-file upload limit |

The entry point honors `HOST` / `PORT` and optional `APP_HOST` / `APP_PORT` overrides. Setup checks tools on PATH, not custom `.env` executable paths. Keep the UI on localhost; no public share link is created. Do not commit `.env` or secrets.

## Models

**4x-UltraSharp:** the new UI presets select it as the **last processing stage**. Its native inference scale is **×4**, even when the final FHD/4K fit or Scale ×2 output is smaller; the native result is resized afterward. Scale ×1 bypasses inference only with output resolution `Scale`. CUDA FP16 and tiling are supported, but smaller final dimensions do not eliminate native ×4 inference/output-buffer costs. The existing downloader includes the [user-selected Hugging Face checkpoint](https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth), saved as `weights/4x-UltraSharp.pth` (or your configured weights directory). Legacy ESRGAN keys are converted privately at load time; original weights are not modified. Review the source/license; model presence and compatibility checks are not a real-video inference guarantee.

`Download-Models.ps1` downloads official release assets for:

- `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth`, `RealESRGAN_x4plus_anime_6B.pth`.
- `realesr-animevideov3.pth`, `realesr-general-x4v3.pth`.
- `GFPGANv1.4.pth`.
- `facexlib/detection_Resnet50_Final.pth` from official facexlib v0.1.0.
- `facexlib/parsing_parsenet.pth` from official facexlib v0.2.2.

Sources are the `xinntao/Real-ESRGAN`, `TencentARC/GFPGAN` and `xinntao/facexlib` GitHub releases, with explicit URLs in the script. Downloaded files are written to unique same-directory `.part` files and atomically renamed after successful, nonempty transfer. Existing paths are skipped without overwrite or verification; inspect/remove an invalid existing file manually before retrying. No published checksum verification is claimed; a nonempty transfer is not authenticity or model-layout validation.

The downloader does not parse `.env`. For a custom `WEIGHTS_DIR`, pass `-WeightsDir` with the same path when running it separately. `Setup.ps1 -DownloadModels` uses the default `weights` directory.

### Manual RIFE installation (required)

1. Visit the [official Practical-RIFE repository](https://github.com/hzwer/Practical-RIFE) and follow its README's trusted model links for 4.26 (recommended) or 4.25. RIFE archives are **never downloaded or extracted automatically** by these scripts; there is no invented direct-download URL.
2. Inspect the archive before extracting: reject absolute paths, `..` traversal and unexpected executables. Only use trusted upstream artifacts. Extract manually into a staging location first.
3. Place the model package contents directly under `weights/rife/4.26/` (or `weights/rife/4.25/`): `RIFE_HDv3.py`, its `IFNet*.py` modules and `flownet.pkl`, plus any additional required package files. Avoid an extra nested `train_log/` or archive folder level.
4. Core preflight checks required assets; package/vendor import compatibility still needs runtime validation.

### facexlib / GFPGAN cache paths

Canonical downloads live under `weights/facexlib/` (or the configured model directory). Before GFPGAN loads, core preflight copies missing `detection_Resnet50_Final.pth` and `parsing_parsenet.pth` cache files into project-root `gfpgan/weights/`. Existing cache files are retained; manual secondary-cache copying is not required. No fully offline runtime validation is claimed.

## Video and storage limitations

- Frame extraction normalizes **VFR input to CFR using the probed average FPS** (nominal-rate fallback). This can duplicate/drop frames; original per-frame timing is not preserved.
- **SDR 8-bit only.** Probe rejects identified PQ/HLG and high-bit-depth pixel formats. No automatic tone mapping or HDR preservation is provided; incorrectly tagged inputs cannot be reliably detected.
- Output is MKV with H.264 (`libx264`, CRF 18) and original audio stream copy; unsupported muxing fails rather than silently transcoding audio. Subtitles and chapters are not supported. Source metadata copying is limited, not complete metadata/HDR preservation.
- **MKV is download-only by design:** the preview remains empty and no extra browser-preview encode is performed.
- PNG intermediate frames can consume enormous disk space; raw RGB size is roughly `width × height × 3 × frame_count` bytes per stage, before content-dependent PNG compression. One minute of 1080p at 30 FPS is about **11.2 GB decimal** per extracted/restored stage; source-resolution interpolation to 60 FPS is about **22.4 GB**. Its final 4K/60 FPS stage is about **89.6 GB**. Retained ComfyUI input/output copies add separate server storage, not included in local temp cleanup; batches add more. The local preflight estimate is not a guarantee of sufficient space on every output/server volume.
- Tiling limits upscaler inference VRAM, not disk requirements, native ×4 buffers or FaceDetailer/RIFE memory. RIFE works on source-sized restored frames before final upscaling; large inputs may still need a smaller RIFE processing scale. Performance figures and older processing-order guidance in `agents.md` are historical estimates, not verified benchmarks or the current runner contract.

## Launch and tests

For the default FaceDetailer presets, run **`./scripts/Run.ps1` in PowerShell 7** after explicit setup. No activation or second terminal is required: the unified launcher starts ComfyUI from `vendor/ComfyUI/.venv` and Gradio from the root `.venv`, with both processes' output in the same terminal. It checks `app.py`, root `.venv`, `.env`, Python 3.11 and a literal loopback `COMFYUI_URL` (`127.0.0.1` or `::1`), then waits for the owned backend's readiness and required nodes before starting the UI. Unified launch supports HTTP; separately managed HTTPS requires app-only mode. It never installs dependencies or downloads models.

- **Stop an existing manual ComfyUI backend yourself before unified launch.** The launcher refuses an occupied port and does not adopt or terminate an existing server.
- **`./scripts/Run.ps1 -LowVram`** configures the owned backend's low-VRAM mode.
- **`./scripts/Run.ps1 -AppOnly`** starts only Gradio for a separately managed backend, or for GFPGAN/Off without a backend. It performs no backend readiness check and never terminates a separately managed backend. FaceDetailer checks its backend when processing; select GFPGAN/Off or disable face restoration if no backend is running. `-LowVram` cannot be combined with `-AppOnly`.
- **`./scripts/Run-ComfyUI.ps1`** remains the standalone foreground backend launcher; pair it with `Run.ps1 -AppOnly`, not default unified launch.

On UI exit or startup/runtime failure, unified launch cleans up only its owned UI/backend processes. Ctrl+C cleanup is **best effort**. Closing or forcibly terminating the PowerShell host cannot guarantee cleanup; inspect any surviving owned processes manually. This process-lifetime cleanup is separate from the API GPU handoff performed between pipeline stages.

VS Code configures `.venv/Scripts/python.exe` and standard-library `unittest` discovery in `tests/` with `test_*.py`. Regression coverage includes configuration, resolution fitting, workflow/protocol handling, stage order, timing, cancellation and folder timestamps, with mocks and optional installed-stack checks. A passing mocked suite does not validate real FaceDetailer inference, server GPU release or full-video quality. This documentation change does not report a new suite result.

## Troubleshooting logs

Restart the application to enable persistent logging. Logs are written to `logs/app.log`, with rotation at 10 MiB and five backups (`app.log.1` through `app.log.5`). Console logs remain enabled. Defaults work without updating your existing `.env`.

Optional `.env` settings: `LOG_DIR=logs`, `LOG_LEVEL=INFO`, `LOG_MAX_BYTES=10485760`, `LOG_BACKUP_COUNT=5`. Relative paths resolve from the project root. Only run one application instance per log directory.

Logs record job IDs, input paths, settings, throttled progress, completion/cancellation, exception tracebacks, and FFmpeg/ffprobe/RIFE child diagnostics. Child output is copied into the rotating log after each subprocess exits, before temporary files are deleted. This includes child warnings and tracebacks; arbitrary parent-process `print()` output is not captured. Rotation eventually removes old diagnostics, so copy relevant logs promptly when reporting an issue.

The `logs/` folder is excluded from Git and is not published by the download UI. Logs can contain local paths and media metadata; review/redact them before sharing.

## Security and licensing

Python model modules and `.pth`/`.pkl` artifacts can execute code. Use trusted weights, archives and reviewed repository revisions. Official HTTPS URLs do not replace review or checksums. Check individual upstream code/model licenses, especially commercial-use and redistribution terms. Process only videos you are authorized to use.
