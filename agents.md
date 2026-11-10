# AGENTS.md

# Video Enhancement Pipeline – Agent Context & Instructions

**Project Goal**  
Build a complete, production-ready Python application that performs in a simple or batch job:

1. Video Upscaling (×2 or ×4) using Real-ESRGAN
2. Face Restoration using GFPGAN (or CodeFormer)
3. Frame Interpolation to 60 FPS using RIFE

**Target Platform**: Windows 10/11  
**Python version**: 3.11 (mandatory)  
**Interface**: Gradio web UI (modern web interface)
**Backend**: ComfyUI
**Hardware**: NVIDIA RTX 4070 Ti 12GB VRAM + 64GB DDR5 RAM

**Frontend**: Gradio handles the video upload, UI controls, and displays the final result.
**Backend Engine**: ComfyUI runs in the background. Gradio sends API requests to ComfyUI via HTTP/WebSockets to process the video.


---

## 1. Hardware Profile & Strategy

- **GPU**: RTX 4070 Ti (12 GB VRAM) → Excellent performance, can run high-quality models
- **RAM**: 64 GB → Comfortable for large frame buffers and parallel processing
- Strategy: Prefer **CUDA + TensorRT** (when available) over NCNN
- Use FP16 by default for maximum speed
- Full resolution processing is viable up to 1080p/1440p; use scale=0.5 or tiling for 4K

---

## 2. Recommended Technology Stack

| Component              | Preferred Implementation                          | Notes |
|------------------------|---------------------------------------------------|-------|
| Upscaling              | Real-ESRGAN (PyTorch + CUDA)                      | Primary |
| Face Restoration       | GFPGAN v1.4 (default) / CodeFormer (heavy cases)  | - |
| Frame Interpolation    | Practical-RIFE 4.26 (CUDA)                        | Best quality |
| Acceleration           | CUDA + FP16 + (optional TensorRT)                 | - |
| Web UI                 | Gradio ≥ 4.x                                      | - |
| Video I/O              | FFmpeg                                            | - |
| Face Detection         | facexlib / OpenCV DNN                             | Skip empty frames |

---

## 3. Best Model Combination for RTX 4070 Ti

### Default (Best Overall Quality)

| Stage                  | Model                          | Scale / Setting      |
|------------------------|--------------------------------|----------------------|
| Upscaling              | `RealESRGAN_x2plus`            | ×2                   |
| Face Restoration       | `GFPGANv1.4`                   | weight = 0.5         |
| Frame Interpolation    | `RIFE 4.26`                    | multi=2 → 60 fps     |

### Alternative Combinations

**Anime focused**

- Upscale: `realesr-animevideov3` or `RealESRGAN_x4plus_anime_6B`
- Face: `GFPGANv1.4`
- Interpolate: `RIFE 4.25` or `4.26`

**Maximum Quality**

- Upscale: `RealESRGAN_x4plus`
- Face: `CodeFormer` (weight 0.6–0.7)
- Interpolate: `RIFE 4.26 Heavy`

**Speed Priority**

- Upscale: `realesr-general-x4v3`
- Face: `GFPGANv1.4` (only on frames with faces)
- Interpolate: `RIFE 4.26`

---

## 4. Official Model Download Links

### Real-ESRGAN

- RealESRGAN_x2plus.pth  
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
- RealESRGAN_x4plus.pth  
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- realesr-animevideov3.pth  
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
- realesr-general-x4v3.pth  
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

### GFPGAN

- GFPGANv1.4.pth  
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

### RIFE (Practical-RIFE)

- RIFE 4.26 (recommended)  
  https://drive.google.com/file/d/1gViYvvQrtETBgU1w8axZSsr7YUuw31uy/view
- RIFE 4.25  
  https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view
- RIFE 4.26 Heavy (optional)  
  Search in Practical-RIFE releases / community mirrors

Place all models inside a `weights/` folder.

---

## 5. Project Structure (Recommended)

video-enhancer/
├── app.py                     # Gradio interface
├── pipeline/
│   ├── init.py
│   ├── extractor.py
│   ├── upscaler.py            # Real-ESRGAN CUDA
│   ├── face_restorer.py       # GFPGAN / CodeFormer
│   ├── interpolator.py        # Practical-RIFE 4.26
│   └── encoder.py
├── weights/                   # All .pth / flownet.pkl models
├── temp/                      # Temporary frames (auto-cleaned)
├── requirements.txt
├── setup_venv.bat             # Windows venv setup
└── README.md

---

## 6. Environment Setup (Windows + Python 3.11)

```bat
:: Create virtual environment
python -m venv .venv
.venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA 12.1/12.4 (choose matching version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: Core dependencies
pip install realesrgan gfpgan basicsr facexlib
pip install opencv-python gradio tqdm numpy pillow
pip install ffmpeg-python
```

System requirements:

- NVIDIA drivers up to date
- FFmpeg installed and added to PATH
- CUDA Toolkit matching the PyTorch version (optional but recommended)

---

## 7. Gradio UI Requirements

The interface must include:

- Video upload component
- Model selection dropdowns
  - Upscale model
  - Face restoration model (GFPGAN / CodeFormer / Off)
  - RIFE model version
- Scale factor (1x / 2x / 4x)
- Target FPS (default 60)
- Options: Enable face restoration, Keep temporary files, FP16
- Real-time progress bar + log output (keep synch properly with the backend, only show in one section)
- Download button for the final video
- Ability to cancel processing

---

## 8. Critical Implementation Notes for the Agent

- Always detect and use CUDA (torch.cuda.is_available()).
- Use half=True (FP16) by default on the 4070 Ti.
- Implement face detection → process only frames that contain faces for GFPGAN/CodeFormer.
- Preferred processing order:
  Extract frames → Upscale → Face Restore → Interpolate → Encode + merge audio
- Clean the temp/ directory after successful processing (or provide a toggle).
- Preserve original audio track using FFmpeg.
- Provide clear logging so progress is visible in the Gradio UI.
- Proper refresh live log and progress between front and backend.
- Make the pipeline modular so models can be easily swapped.
- Support both single video and batch processing.
- Create documentation of everything and Changelog.md
- Use .env to store environment variables.
- Create powershell scripts to download models, configure app and run the app.

---

## 9. Performance Expectations (RTX 4070 Ti)

Resolution,Pipeline,Approximate Time (8–10s video)
480p,Full quality (x2 + GFPGAN + 4.26),1–3 minutes
720p,Full quality,2–5 minutes
1080p,Full quality,4–10 minutes
1440p/4K,With scale=0.5 or tiling,8–20+ minutes

---

## 10. Preferred Final Pipeline

Input Video
    ↓
FFmpeg → Extract high-quality frames
    ↓
Real-ESRGAN (CUDA + FP16) → Upscale
    ↓
Face Detection → GFPGAN / CodeFormer (only on faces)
    ↓
RIFE 4.26 (CUDA + FP16) → Interpolate to 60 fps
    ↓
FFmpeg → Encode video + copy original audio
    ↓
Output Video

---

## 11. Current Context Summary

- Original discussion started with CPU-only high-core machine.
- User now has RTX 4070 Ti 12GB + 64GB RAM.
- Best models for this hardware were selected:
  - Upscale: RealESRGAN_x2plus / x4plus
  - Face: GFPGAN v1.4
  - Interpolation: RIFE 4.26
- Target environment changed to Windows + Python 3.11 + .venv + Gradio.

---

## 12. Agent Instructions

When generating code:

- Target Windows + Python 3.11 + .venv.
- Prioritize CUDA + FP16 implementations.
- Use the recommended model combination above as default.
- Keep code clean, modular, and well-commented.
- Build a polished and informative Gradio interface with modern UI components.
- Include proper error handling, progress reporting, logging, and temp file cleanup.
- Prefer official model sources and stable library versions.
- Ask. A 30-second clarifying question is cheaper than a 30-minute revert.
