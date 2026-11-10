# SeedVR2 for video

SeedVR2 (ByteDance Seed, Apache 2.0) is the stronger choice for video. It is a one-step diffusion video restoration/upscaler with temporal attention, so it keeps motion consistency far better than frame-by-frame image models. It is widely used on H3 (and similar AI-generated) clips and has mature ComfyUI support (both native nodes and the popular custom node pack).

Fit current hardware with proper quantization + tiling/block-swap. 

## Hardware fit (RTX 4070 Ti 12 GB)

- SeedVR2 3B FP8 / INT8 or GGUF Q8/Q4 + tiling + BlockSwap → comfortable on 12 GB.
- SeedVR2 7B only with aggressive quantization (GGUF Q4 + heavy BlockSwap + VAE tiling) and lower batch/chunk sizes; quality gain is real but slower and more memory-sensitive.
- 64 GB system RAM is excellent for offloading (BlockSwap / CPU offload).
- Typical realistic targets from H3 (often ~768p–1344×768 range): 2× → 1080p/2K-class, or mild 4× with tiling. Full 4K on long clips is possible but slower and needs careful settings.

From community benchmarks (similar 12 GB cards):

- 3B FP8, short clips, batch/chunk ~5–9, tile ~384–512 → workable without constant OOM.
- Higher batch (21) or larger tiles improves temporal consistency and speed when VRAM allows.

## Implementation overview (ComfyUI backend + Gradio frontend)

1. ComfyUI side

   - Install via Manager: search ComfyUI-SeedVR2_VideoUpscaler (numz / AInVFX) → Install → Restart.

    Manual:

    ```Bash
    ComfyUI/custom_nodes
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2_videoupscaler
    # then pip install -r requirements.txt in your ComfyUI Python env
    ```

   - Models auto-download on first run into `ComfyUI/models/SEEDVR2/` (or use Comfy-Org/SeedVR2 on Hugging Face). Prefer:
     - `seedvr2_ema_3b_fp8_e4m3fn` (or INT8/GGUF equivalents) for 12 GB.
     - Shared VAE: `ema_vae_fp16` / `seedvr2_ema_vae_fp16`.

   - Native ComfyUI SeedVR2 support also exists (newer builds); the custom pack is still the most battle-tested for video path/chunked processing.

2. Key settings for 12 GB (start here)

   - Model: 3B FP8 / GGUF Q8.
   - `batch_size` / chunk: follow 4n+1 rule (5 or 9 preferred; higher = better temporal consistency, more VRAM).
   - Tile size: 384–512 (lower if OOM).
   - Tile overlap: 48–64.
   - BlockSwap: 8–32 (or higher) + offload to CPU.
   - VAE tiling enabled, encode/decode tile ~512–1024.
   - Color correction: wavelet (default) or lab/adain if shift appears.
   - Latent / input noise scale: small values (0.01–0.1) for detail tuning.
   - Resolution: set target **short edge** (e.g. 1080).
   - For long clips: use the path/chunked video nodes (or CLI) so the whole sequence is never loaded as one giant tensor.

3. Gradio frontend (Python)

   Simple pattern:

   - Accept video upload (or path).
   - Call ComfyUI via its API (websocket/queue or `/prompt` + history polling) with a pre-built workflow JSON that contains the SeedVR2 nodes.
   - Or use the pack’s `inference_cli.py` for simpler non-UI batch jobs.
   - Progress feedback + download of the upscaled MP4.
   - Optional: pre-blur / mild denoise pass if SeedVR2 over-emphasizes skin texture (common tip for AI video).

## Practical workflow tips for MiniMax H3

- H3 often outputs around 768p-class or 1344×768; SeedVR2 is frequently used exactly for this "make it postable" step.
- Process short shots (≤15–20 s) first; longer clips benefit from the streaming/chunked path.
- Test a 2–5 s segment with different noise scales and color correction before committing a full render.
- If you see ghosting: increase temporal window / batch / prepend frames.
- System RAM offloading + preserve_vram / DynamicVRAM (newer ComfyUI) helps a lot on 12 GB.
  
---

## Memory-optimized CLI flags (standalone `inference_cli.py`)

Run from the `ComfyUI-SeedVR2_VideoUpscaler` folder (or wherever the pack lives). Models auto-download on first use into `models/SEEDVR2/`.

### **Recommended base command (12 GB safe – start here)**

```Bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors \
  --vae_model ema_vae_fp16.safetensors \
  --resolution 1080 \
  --batch_size 5 \
  --uniform_batch_size \
  --temporal_overlap 3 \
  --prepend_frames 4 \
  --blocks_to_swap 24 \
  --swap_io_components \
  --dit_offload_device cpu \
  --vae_offload_device cpu \
  --vae_encode_tiled \
  --vae_decode_tiled \
  --vae_encode_tile_size 512 \
  --vae_decode_tile_size 512 \
  --vae_encode_tile_overlap 64 \
  --vae_decode_tile_overlap 64 \
  --color_correction lab \
  --input_noise_scale 0.0 \
  --latent_noise_scale 0.03 \
  --video_backend ffmpeg \
  --10bit \
  --cache_dit \
  --cache_vae \
  --cuda_device 0 \
  --output output_1080p.mp4
```

### **For 2160p (4K) target – more aggressive memory saving**

```Bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors \
  --resolution 2160 \
  --batch_size 5 \
  --uniform_batch_size \
  --temporal_overlap 3 \
  --prepend_frames 4 \
  --blocks_to_swap 32 \
  --swap_io_components \
  --dit_offload_device cpu \
  --vae_offload_device cpu \
  --vae_encode_tiled \
  --vae_decode_tiled \
  --vae_encode_tile_size 384 \
  --vae_decode_tile_size 384 \
  --vae_encode_tile_overlap 48 \
  --vae_decode_tile_overlap 48 \
  --color_correction lab \
  --latent_noise_scale 0.03 \
  --video_backend ffmpeg \
  --10bit \
  --cache_dit --cache_vae \
  --cuda_device 0 \
  --output output_2160p.mp4
```

### **Key flag explanations for 12 GB**

Flag|Recommended (12 GB)|Why
-|-|-
--dit_model|seedvr2_ema_3b_fp8_e4m3fn.safetensors (or GGUF Q8/Q4)|3B FP8 fits; 7B needs extreme swap
--batch_size|5 (or 9 if stable)|"Must be 4n+1; higher = better temporal consistency, more VRAM"
--blocks_to_swap|24–32|Moves DiT blocks to system RAM (you have 64 GB)
--vae_*_tiled + tile 384–512|Required for 1080p→4K and safer for 768p→1080p|Prevents VAE OOM
--dit/vae_offload_device cpu|Always on 12 GB|Keeps peak VRAM low between phases
--chunk_size|Optional (e.g. 45–90) for safety|Streams long clips; not needed for ≤8 s
--temporal_overlap 3 + --prepend_frames 4|Keep|Reduces ghosting at chunk/batch edges
--color_correction lab|Preferred|More stable than wavelet on AI video
--video_backend ffmpeg --10bit|Recommended|"Better gradients, less banding"

### **Batch directory processing**

```Bash
python inference_cli.py /path/to/videos/ \
  --output /path/to/out/ \
  [same flags as above] \
  --cache_dit --cache_vae
```

If you still OOM:

1. Drop `--batch_size` to 5 (or even 1 temporarily).
2. Raise `--blocks_to_swap` to 36.
3. Lower VAE tile size to 384.
4. Add `--preserve_vram`if the flag exists in your version.

## Exact recommended ComfyUI node graph / settings (12 GB)

Use the official example as base and change only the values below.

### **Load the example**

`ComfyUI/custom_nodes/seedvr2_videoupscaler/example_workflows/SeedVR2_HD_video_upscale.json`

### **Node-by-node settings for 12 GB (480p/768p → 1080p)**

1. **LoadVideo**
   - Your input file (≤8 s).
2. **GetVideoComponents**
   - Default (images + audio + fps).
3. **SeedVR2LoadDiTModel**
   - `model:` **seedvr2_ema_3b_fp8_e4m3fn.safetensors** (or INT8/GGUF equivalent)
   - `device:` **cuda:0**
   - `blocks_to_swap:` **24–32**
   - `swap_io_components:` **true**
   - `offload_device:` **cpu**
   - `cache_model:` **true** (for repeated runs)
   - `attention_mode:` **sdpa** (or flash_attn_2 / sageattn if installed and stable)
4. **SeedVR2LoadVAEModel**
   - `model:` **ema_vae_fp16.safetensors**
   - `device:` **cuda:0**
   - `encode_tiled:` **true**
   - `encode_tile_size:` **512**
   - `encode_tile_overlap:` **64**
   - `decode_tiled:` **true**
   - `decode_tile_size:` **512**
   - `decode_tile_overlap:` **64**
   - `offload_device:` **cpu**
   - `cache_model:` **true**
5. **SeedVR2VideoUpscaler** (main node)
   - `resolution:` **1080** (short edge) → for 1080p output or **2160* for **4K** (then lower tiles/batch further)
   - `max_resolution:` **0** (unlimited) or **3840** if you want a hard cap
   - `batch_size:` **5** (safe) or **9** if it stays under ~11 GB
   - `uniform_batch_size:` **true**
   - `color_correction:` **lab**
   - `temporal_overlap:` **3**
   - `prepend_frames:` **4**
   - `input_noise_scale:` **0.0**
   - `latent_noise_scale:` **0.03** (0.01–0.08 range for detail tuning)
   - `offload_device:` **cpu**
   - `enable_debug:` **true** while testing
6. **CreateVideo + SaveVideo**
   - Keep original fps, use a good codec (or post-process with ffmpeg for 10-bit).

### **For 2160p target on the same graph**

- Set `resolution = 2160`
- Drop VAE tiles to 384 / overlap 48
- `blocks_to_swap = 32`
- `Keep batch_size = 5`

### **Optional torch.compile**

Connect the `SeedVR2TorchCompileSettings` node (backend `inductor`) only if you process many clips of similar length; first run has compile overhead.

---

## ComfyUI Workflows (Examples)

Two ready-to-use workflow JSONs are prepared with the 12 GB settings already baked in located in `workflow` folder of this project.

### What’s configured in both

Setting|1080p workflow|2160p workflow
-|-|-
DiT model|seedvr2_ema_3b_fp8_e4m3fn.safetensors|same
blocks_to_swap|28|32
swap_io_components|true|true
DiT / VAE offload|cpu|cpu
VAE encode/decode tiled|true|true
VAE tile size / overlap|512 / 64|384 / 48
resolution (short edge)|1080|2160
batch_size|5 (4n+1)|5
uniform_batch_size|true|true
temporal_overlap|3|3
prepend_frames|4|4
color_correction|lab|lab
input_noise_scale|0.0|0.0
latent_noise_scale|0.03|0.03
attention_mode|sdpa|sdpa
torch.compile|disabled (node mode=4)|disabled

### How to use

1. Download the JSON that matches your target.
2. In ComfyUI: Load → select the file (or drag & drop).
3. In the LoadVideo node pick your 480p / 768p clip (≤ 8 s).
4. Queue Prompt.
   First run will auto-download the 3B FP8 + VAE weights into ComfyUI/models/SEEDVR2/.

### Quick tuning if needed

- Still OOM → set `batch_size = 1` or raise `blocks_to_swap` to 36.
- More VRAM headroom → try `batch_size = 9`.
- Want slightly more detail → raise `latent_noise_scale` to 0.05–0.08.
- Enable torch.compile only when you process many similar-length clips (connect the compile node and set its mode to 0).

## Minimal Gradio wrapper (Example)

A python gradio wrapper example is located at `/worflows/seedvr2_gradio_app.py`

### What the UI gives you

- Video upload
- Two presets matching the workflows we built:
  - **1080p (safe 12 GB)** – batch 5, tiles 512, BlockSwap 28
  - **2160p / 4K (aggressive 12 GB)** – batch 5, tiles 384, BlockSwap 32
- Adjustable `batch_size` (4n+1) and `latent_noise_scale`
- Optional extra CLI flags
- Live log + downloadable result video

### Notes

- First run will download the 3B FP8 + VAE models.
- Clips ≤ 8 s are recommended for best stability on 12 GB.
- If you still OOM, drop `batch_size` to 1 in the UI.
- The script uses `ffmpeg` 10-bit output when available (same as the CLI flags we defined earlier).