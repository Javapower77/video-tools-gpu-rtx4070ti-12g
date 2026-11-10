# Changelog

### SeedVR2 Windows CLI encoding — 2026-10-04

- Launch SeedVR2 CLI with explicit Python UTF-8 mode and unbuffered output. Prevents CP1252 `UnicodeEncodeError` on upstream startup emoji when subprocess output is redirected; vendor files and model weights remain unchanged.

### SeedVR2 integration — 2026-10-03

- Added selectable SeedVR2 3B FP8 video upscaling with FHD/4K aspect-preserving targets, temporal batch and latent-noise controls, CPU BlockSwap/offload and tiled VAE presets.
- Single mode uses a bounded ComfyUI API video workflow; Batch mode uses the project-contained memory-optimized streaming CLI. Exact output frame counts, cancellation, audio muxing and folder-batch creation/modified timestamps are retained.
- Added opt-in pinned SeedVR2 pack setup/model downloads and optional backend allowlist entry. No installs/downloads or real model inference performed during implementation. Existing UltraSharp profiles remain available.

### Compatibility fix validation — 2026-10-02

- All 70 Python unittest methods passed; standalone PowerShell launcher tests passed for parsing, environment isolation, mocked readiness, owned-process lifecycle and dotenv URL validation. No editor diagnostics in changed backend/test/launcher files.
- Real video inference and combined server startup were not rerun. Existing Gradio event-loop ResourceWarnings remain.

### ComfyUI integration validation — 2026-10-02

- All 67 unittest methods passed in the application's `.venv`, including mocked ComfyUI protocol/cancellation/GPU handoff, ordered FaceDetailer→RIFE→upscale checks, updated profiles and aspect-preserving FHD/4K fitting. All three ComfyUI PowerShell scripts parse without errors. Changed Python files have no editor diagnostics.
- No installation, checkpoint/detector download, ComfyUI server launch or real FaceDetailer video inference was performed. Actual upstream setup compatibility, VRAM handoff and visual/temporal quality still require runtime validation. Gradio construction tests emit event-loop ResourceWarnings.

## [Unreleased] — 2026-10-02

### Fixed — unified launch and handoff documentation

- Corrected startup instructions: PowerShell 7 `Run.ps1` launches owned ComfyUI and Gradio in two isolated Python processes in one terminal, waiting up to 180 seconds for backend readiness and required FaceDetailer/detector nodes before starting the UI. Stop an existing manual backend yourself first; occupied ports are refused, not adopted or cleared.
- Documented owned-process cleanup on exit/failure and best-effort Ctrl+C cleanup, without guarantees when the shell is closed or forcibly terminated. `-LowVram` configures the unified backend; `-AppOnly` supports a separately managed backend or GFPGAN/Off without backend readiness checks or backend termination. Standalone `Run-ComfyUI.ps1` remains available, paired with `Run.ps1 -AppOnly`.
- Corrected the earlier zero-reservation handoff description: `/free` must be followed by two consecutive identical idle reservation samples, allowing at most 128 MiB per CUDA device only for cudaMallocAsync; native requires zero, and unknown allocators receive no residual allowance. Above-cap reservations are not accepted and unknown/invalid evidence fails closed. This does not definitively establish model unloading, zero total VRAM use or that the next model fits; a dedicated server with no other submitters remains required.
- Documented Windows subfolder-separator normalization without weakening traversal, filename or exact owned-folder checks, and visibility of the initial processing failure when cleanup also fails.
- Documentation-only update to `README.md` and `Changelog.md`; no dependency changes, installation or downloads. No new test counts or runtime-validation results claimed by this entry; historical entries below are retained.

### Added — ComfyUI and folder batches

- Dedicated loopback ComfyUI FaceDetailer backend using Impact Pack/Subpack, Juggernaut XL v9 and a face detector; fixed seven-node API workflow with editable compatible scalar inputs and UI prompts/denoise/steps/CFG/seed/guide controls. No arbitrary graph import or silent GFPGAN fallback; GFPGAN remains selectable.
- Explicit PowerShell 7 ComfyUI setup, launch and opt-in model-download scripts contained in `vendor/ComfyUI/`, with an isolated Python 3.11 environment and independently selected CUDA torch stack. Persistent upstream download-skip marker, reviewed installer guards, official Impact-only launch allowlist and installed/offline frontend safeguards; no automatic provisioning at app launch.
- SHA256-verified approximately 7.11 GB Juggernaut download; face detector download requires an independently verified caller-supplied hash. Existing mismatches/partial files are refused without overwrite; no automatic SAM/other model downloads.
- Server GPU handoff requires idle queues, `/free` and `/system_stats` zero CUDA torch-reservation evidence before RIFE. Cancellation targets owned prompts only and requires a current targeted-interrupt API. Uploaded/generated PNGs remain in unique ComfyUI input/output job folders for manual cleanup, separate from local temp retention.
- FHD/4K aspect-ratio-preserving landscape-box fitting with even dimensions, no stretch/crop and portrait orientation retained; optional legacy Scale mode remains. UI Normal now selects FaceDetailer → RIFE → UltraSharp FHD, Boost selects the corresponding 4K fit; `Settings()` retains legacy defaults.
- Batch-only local folder sources, recursive/overlap-deduplicated discovery, and native Windows Explorer multi-folder drag/drop/Browse window. Previous eight-hex-suffix folder outputs and partial files are excluded.
- Folder-batch MKV outputs beside originals with full original stems and eight random hexadecimal characters. Original Windows creation/modified FILETIME values are captured before processing, applied and verified before publication; timestamp errors remove partial outputs.
- Folder results reported as local paths only, without publishing source directories through Gradio; single/file-upload batch behavior is unchanged. No additional dependencies, downloads or execution-policy bypass.

### Corrected — stage order and documentation

- Processing order is now extract → face restoration → existing Practical-RIFE → final upscaling → encode/audio mux. Documentation removes stale RIFE-after-upscale/profile VRAM guidance; existing folder outputs and creation/modified timestamps remain unchanged.
- Documented native ×4 UltraSharp inference even for smaller fitted output, generated rather than recovered skin detail, and lack of fixed-seed temporal/identity guarantees. Explicitly ignored the complete contained ComfyUI tree and safetensors model artifacts.

### Validation

- Documentation contracts checked against the backend, runner, UI, presets, resolution helper and ComfyUI scripts; upstream Impact installers and ComfyUI requirements reviewed. No installers, package changes or model downloads executed by this documentation update. Real FaceDetailer workflow/GPU handoff and full-video quality remain unvalidated; no new full-suite test count claimed here.

- All 44 unittest methods passed in `.venv`, including a real Windows creation/modified FILETIME round trip across rename, recursive discovery, timestamp-failure cleanup, folder admission and local-only output checks. Native PowerShell script parsed with zero errors; changed files have no editor diagnostics.
- Interactive Explorer drag/drop and real-video folder processing remain unvalidated. Gradio construction tests emit event-loop ResourceWarnings.

## [Unreleased] — 2026-10-01

### Added

- Independent Upscaling/Face Restoration/RIFE checkboxes support all seven nonempty operation combinations for single videos and batches. Disabled stages retain their control values and bypass model preflight; profiles preserve checkbox selections. Empty effective selections are rejected before queuing.
- Selectable Normal/Boost processing profiles for single videos and batches, backed by immutable presets. Normal preserves defaults; Boost selects RealESRGAN_x4plus ×4 with GFPGAN v1.4 and standard RIFE 4.26. Controls remain editable; uploads, file retention and submitted jobs are unaffected. Includes a 12 GB VRAM warning; unverified RIFE Heavy is not offered.
- Optional `4x-UltraSharp` upscaler, user-selected Hugging Face download source, and legacy ESRGAN-to-BasicSR checkpoint conversion with strict architecture validation.
- Progress caption shows elapsed processing seconds (excluding queue wait), updates during quiet stages, and retains the final duration; batch jobs also show current-video seconds. Per-video durations are logged.
- Rotating `logs/app.log` (10 MiB, five backups), configurable `.env` logging, correlated job lifecycle/progress, and child diagnostics preserved before temporary-file cleanup.
- Python 3.11 dependency configuration with torch 2.5.1 / torchvision 0.20.1 CUDA 12.4 local-version pins and NumPy 1.26.4.
- Environment template, generated-artifact ignore rules and VS Code interpreter/unittest settings.
- Checked Windows setup script, batch entry point and application launch guard.
- Implemented Gradio single/batch UI with synchronized activity reporting, cooperative cancellation and completed-file downloads.
- Implemented modular extraction, Real-ESRGAN upscaling, face-aware GFPGAN restoration, subprocess RIFE interpolation and MKV encoding/audio stream copy, with preflight and cleanup.
- Application-local BasicSR/torchvision compatibility shim; installed site-packages remain unchanged.
- Official Real-ESRGAN, GFPGAN and facexlib downloads with atomic publication, temporary-file cleanup and existing-file skipping.
- Manual trusted RIFE 4.26/4.25 installation instructions; no automatic RIFE archive download/extraction.
- `APP_MAX_UPLOAD_SIZE=2gb`; existing `HOST` / `PORT` settings retained.

### Corrected

- RIFE FP16 `grid_sample` crash (`expected scalar type Half but found Float`): install an application-owned dtype-safe warp before importing IFNet; vendor checkout and weights remain unchanged.
- Added CPU FP32 and CUDA mixed-precision warp regression tests. All 18 tests passed in `.venv`; installed RIFE 4.26 inference passed in FP32/FP16, and its FP16 worker produced four frames from two synthetic inputs.
- facexlib detector filename/URL: `detection_Resnet50_Final.pth`, official v0.1.0; parser URL: `parsing_parsenet.pth`, official v0.2.2.
- Documentation describes implemented code and preflight's missing-file copies into `gfpgan/weights/`.
- Documented intentional download-only MKV (no extra preview encode), unsupported subtitles/chapters and limited metadata copying.

### Scope and validation

- Operation selection update: explicit `.venv` unittest discovery passed all 36 test methods, including exhaustive checkbox combinations and mocked stage/FPS checks. No diagnostics in changed Python files. Gradio construction tests emitted event-loop ResourceWarnings; real-video operation combinations remain unvalidated.
- Profile update: all 40 regression tests passed, including six preset/UI/job-snapshot tests; no editor diagnostics in changed Python files. Full-video Boost inference and performance remain unvalidated. No dependencies or model downloads were changed.
- Standard-library regression tests cover configuration, timing, cancellation, cleanup, encoding and optional installed-stack precision checks.
- Probe rejects identified HDR/high-bit-depth inputs; extraction selects the actual video stream rather than attached cover art.
- `HOST` / `PORT` binding is supported with optional `APP_HOST` / `APP_PORT` overrides.
- Sixteen dependency-free regression tests passed under Python 3.11; Python and PowerShell syntax checks passed.
- Fail-fast Real-ESRGAN tiling and GFPGAN error propagation avoid silent restoration fallbacks.
- CodeFormer/TensorRT remain optional future extensions.
- Full-video processing and benchmarks remain unvalidated; small RIFE CUDA smoke tests passed as noted above. This update did not install/change packages or perform network downloads.
