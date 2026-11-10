#requires -Version 7.0
<#
.SYNOPSIS
Explicit, opt-in model downloads only; no packages or models loaded/executed.
.DESCRIPTION
Juggernaut XL v9: approximately 7.11 GB, SHA256 supplied by the caller's specification.
Detector URL verified in official ltdrdata/ComfyUI-Impact-Subpack/main/install.py:
https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Subpack/main/install.py
README lists Bingsu/adetailer as its face-model source. The detector itself is
third-party, not an Ultralytics-authored face model. No upstream checksum was
specified: -FaceDetectorSha256 is REQUIRED with -FaceDetector (obtain independently).
Hash verification does not make a pickle-based .pt file safe to execute.
Never overwrites existing final files or .part files. Failed partial downloads
are left for manual inspection/removal, not silently resumed or replaced.
#>
[CmdletBinding()]
param(
    [switch]$Juggernaut,
    [switch]$FaceDetector,
    [ValidatePattern('^[a-fA-F0-9]{64}$')][string]$FaceDetectorSha256,
    [switch]$SeedVR2
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $Juggernaut -and -not $FaceDetector -and -not $SeedVR2) {
    Write-Host 'No downloads selected. Opt in with -SeedVR2, -Juggernaut and/or -FaceDetector -FaceDetectorSha256 <independently verified SHA256>.'
    return
}
if ($FaceDetector -and -not $FaceDetectorSha256) { throw '-FaceDetector requires an independently verified -FaceDetectorSha256; no invented checksum or unverified .pt downloads.' }
$Comfy = Join-Path (Split-Path -Parent $PSScriptRoot) 'vendor/ComfyUI'
if (-not (Test-Path -LiteralPath (Join-Path $Comfy 'main.py') -PathType Leaf)) { throw 'ComfyUI checkout missing. Run explicit Setup-ComfyUI.ps1 first.' }

function Receive-VerifiedModel {
    param([string]$Url, [string]$Destination, [string]$Sha256)
    if (Test-Path -LiteralPath $Destination) {
        if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) { throw "Destination is not a file: $Destination" }
        $Actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($Actual -ne $Sha256) { throw "Existing model hash mismatch: $Destination (actual $Actual). No overwrite; inspect manually." }
        Write-Host "Already present, SHA256 verified: $Destination"
        return
    }
    $Partial = "$Destination.part"
    if (Test-Path -LiteralPath $Partial) { throw "Existing partial file: $Partial. Inspect/remove manually; no overwrite or unverified resume." }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    $Client = [System.Net.Http.HttpClient]::new()
    $Client.Timeout = [TimeSpan]::FromHours(12)
    $Response = $null
    $InputStream = $null
    $OutputStream = $null
    try {
        Write-Host "Downloading $Url -> $Partial"
        # CreateNew is race-safe; .part is on the destination volume for atomic rename.
        $OutputStream = [IO.File]::Open($Partial, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $Response = $Client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $Response.EnsureSuccessStatusCode() | Out-Null
        if ($Response.RequestMessage.RequestUri.Scheme -ne 'https') { throw 'Refusing non-HTTPS redirect.' }
        $InputStream = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $InputStream.CopyToAsync($OutputStream).GetAwaiter().GetResult()
        $OutputStream.Flush($true)
        $OutputStream.Dispose()
        $OutputStream = $null
        $Length = (Get-Item -LiteralPath $Partial).Length
        if ($Length -eq 0 -or ($null -ne $Response.Content.Headers.ContentLength -and $Length -ne $Response.Content.Headers.ContentLength)) { throw 'Incomplete model response.' }
        $Actual = (Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash
        if ($Actual -ne $Sha256) { throw "SHA256 mismatch: expected $Sha256; actual $Actual. Partial left for inspection." }
        # Two-argument File.Move refuses an existing destination (including racing downloads).
        [IO.File]::Move($Partial, $Destination)
        Write-Host "Verified SHA256 $Actual ($Length bytes): $Destination"
    } finally {
        if ($OutputStream) { $OutputStream.Dispose() }
        if ($InputStream) { $InputStream.Dispose() }
        if ($Response) { $Response.Dispose() }
        $Client.Dispose()
    }
}

if ($Juggernaut) {
    Write-Host 'Juggernaut XL v9: approximately 7.11 GB; allow additional free space for ComfyUI dependencies. Review model license before use.'
    Receive-VerifiedModel 'https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors' (Join-Path $Comfy 'models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors') 'c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca'
}
if ($FaceDetector) {
    Receive-VerifiedModel 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt' (Join-Path $Comfy 'models/ultralytics/bbox/face_yolov8m.pt') $FaceDetectorSha256
}
if ($SeedVR2) {
    # Published hashes/repo verified from the official registry, not guessed:
    # https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/model_registry.py
    # DEFAULT_DIT/DEFAULT_VAE inherit ModelInfo.repo = numz/SeedVR2_comfyUI.
    # URL template and SEEDVR2 directory verified at the same commit:
    # https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/constants.py
    # Both resolve URLs returned HTTP 200 to HEAD on 2026-10-03 (no weights fetched).
    # Accepted SHA256 values also match pipeline/seedvr2_backend.py MODEL_HASHES.
    Write-Host 'SeedVR2: explicit 3B FP8 DiT + FP16 VAE downloads. Review model licenses and free disk space before use.'
    Receive-VerifiedModel 'https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors' (Join-Path $Comfy 'models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors') '3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff'
    Receive-VerifiedModel 'https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors' (Join-Path $Comfy 'models/SEEDVR2/ema_vae_fp16.safetensors') '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1'
}
Write-Host 'Selected downloads complete. No SAM/other models fetched; no model whitelist or unsafe pickle-loading override written.'