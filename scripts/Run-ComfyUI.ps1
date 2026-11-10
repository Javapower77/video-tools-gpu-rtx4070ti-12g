#requires -Version 7.0
<#
.SYNOPSIS
Run the dedicated ComfyUI Python subprocess locally, without installing anything.
.DESCRIPTION
Uses upstream --disable-all-custom-nodes + --whitelist-custom-nodes to exclude
other custom nodes AND their prestartup scripts. This is an allowlist, not a
sandbox: locally modified code or upstream node behavior still needs review.
Uses the installed frontend explicitly, preventing startup frontend downloads.
HF offline flags and the setup marker cover Impact's researched download paths,
not SeedVR2's direct urllib downloader. App model preflight is required before use;
this launcher is not a network sandbox or a guarantee against upstream downloads.
No environment activation, execution-policy change, elevation, or pip invocation.
#>
[CmdletBinding()]
param(
    [ValidateSet('127.0.0.1', '::1')][string]$Listen = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 8188,
    [switch]$LowVram,
    # Transfers lifetime ownership to the caller; does not imply readiness.
    [switch]$PassThru
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Comfy = Join-Path (Split-Path -Parent $PSScriptRoot) 'vendor/ComfyUI'
$Python = Join-Path $Comfy '.venv/Scripts/python.exe'
foreach ($File in @($Python, (Join-Path $Comfy 'main.py'), (Join-Path $Comfy 'custom_nodes/skip_download_model'))) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing $File. Run Setup-ComfyUI.ps1 explicitly first; launch never installs anything." }
}
if (Test-Path -LiteralPath (Join-Path $Comfy 'extra_model_paths.yaml')) {
    throw 'extra_model_paths.yaml can introduce external custom-node folders with allowlisted names. Review/remove it manually before using this isolated launcher.'
}
$NodeRepositories = [ordered]@{
    'ComfyUI-Impact-Pack' = 'ltdrdata/ComfyUI-Impact-Pack'
    'ComfyUI-Impact-Subpack' = 'ltdrdata/ComfyUI-Impact-Subpack'
}
# Absent optional pack must not break existing non-SeedVR2 runs.
if (Test-Path -LiteralPath (Join-Path $Comfy 'custom_nodes/seedvr2_videoupscaler')) {
    $NodeRepositories['seedvr2_videoupscaler'] = 'numz/ComfyUI-SeedVR2_VideoUpscaler'
}
foreach ($Name in $NodeRepositories.Keys) {
    $Path = Join-Path $Comfy "custom_nodes/$Name"
    if (-not (Get-Command git -CommandType Application -ErrorAction SilentlyContinue)) { throw 'Git is required to verify node provenance (no installation at launch).' }
    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) { throw "Missing official node checkout: $Path" }
    $Origin = & git -C $Path remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw "Cannot verify $Name origin." }
    $Normalized = $Origin.Trim().TrimEnd('/').Replace('git@github.com:', 'https://github.com/') -replace '\.git$', ''
    if ($Normalized -ne "https://github.com/$($NodeRepositories[$Name])") { throw "Unexpected node origin: $Origin" }
    $Changes = & git -C $Path status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $Changes) { throw "Cannot trust tracked node files at $Path; inspect manually." }
    if ($Name -eq 'seedvr2_videoupscaler') {
        $Head = & git -C $Path rev-parse --verify HEAD
        if ($LASTEXITCODE -ne 0 -or $Head.Trim() -ne '4490bd1f482e026674543386bb2a4d176da245b9') {
            throw 'SeedVR2 HEAD differs from the verified pin. Review manually; launch never pulls, resets, or checks out revisions.'
        }
    }
}

$Probe = @'
import sys, pathlib, torch, torchvision, importlib.util
assert sys.version_info[:2] == (3,11), 'Python 3.11 required'
assert sys.prefix != sys.base_prefix, 'Dedicated venv required'
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve()
assert 'include-system-site-packages = false' in (pathlib.Path(sys.prefix)/'pyvenv.cfg').read_text().lower()
assert torch.version.cuda and torch.cuda.is_available(), 'CUDA unavailable: check dedicated venv and NVIDIA driver'
print(f'Python: {sys.executable}; torch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}', file=sys.stderr)
spec = importlib.util.find_spec('comfyui_frontend_package')
assert spec and spec.origin, 'Installed ComfyUI frontend missing; rerun explicit setup'
frontend = pathlib.Path(spec.origin).parent / 'static'
assert (frontend / 'index.html').is_file(), 'Installed frontend static/index.html missing'
print(frontend)
'@
$Frontend = & $Python -I -c $Probe (Join-Path $Comfy '.venv')
if ($LASTEXITCODE -ne 0) { throw 'ComfyUI preflight failed. No installation attempted.' }
$Cli = Get-Content -LiteralPath (Join-Path $Comfy 'comfy/cli_args.py') -Raw
foreach ($Flag in @('--disable-all-custom-nodes', '--whitelist-custom-nodes', '--front-end-root')) {
    if (-not $Cli.Contains($Flag)) { throw "Checkout lacks required safety flag $Flag; choose a compatible revision manually." }
}
$Address = [System.Net.IPAddress]::Parse($Listen)
$Listener = [System.Net.Sockets.TcpListener]::new($Address, $Port)
try { $Listener.Start() } catch { throw "Cannot bind ${Listen}:${Port}: port occupied or unavailable." } finally { $Listener.Stop() }

$Start = [System.Diagnostics.ProcessStartInfo]::new()
$Start.FileName = $Python
$Start.WorkingDirectory = $Comfy
$Start.UseShellExecute = $false
$Arguments = @('-E', '-s', '-u', 'main.py', '--listen', $Listen, '--port', "$Port", '--preview-method', 'none', '--front-end-root', $Frontend.Trim(), '--disable-all-custom-nodes', '--whitelist-custom-nodes')
$Arguments += @($NodeRepositories.Keys)
$Arguments += '--disable-auto-launch'
if ($Cli.Contains('--offline')) { $Arguments += '--offline' }
if ($LowVram) {
    $Arguments += '--lowvram'
    if ($Cli.Contains('--disable-dynamic-vram')) { $Arguments += '--disable-dynamic-vram' }
}
foreach ($Argument in $Arguments) { $Start.ArgumentList.Add($Argument) }
foreach ($Key in @('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_CONSTRAINT')) { $Start.Environment.Remove($Key) | Out-Null }
$Start.Environment['VIRTUAL_ENV'] = Join-Path $Comfy '.venv'
$Start.Environment['PATH'] = (Join-Path $Comfy '.venv/Scripts') + [IO.Path]::PathSeparator + $env:PATH
$Start.Environment['COMFYUI_PATH'] = $Comfy
$Start.Environment['COMFYUI_MODEL_PATH'] = Join-Path $Comfy 'models'
$Start.Environment['HF_HUB_OFFLINE'] = '1'
$Start.Environment['TRANSFORMERS_OFFLINE'] = '1'
Write-Host "Starting dedicated subprocess: $Python $($Arguments -join ' ')"
Write-Host "Only verified official folders enabled: $($NodeRepositories.Keys -join ', '); no model/package installs. Ctrl+C stops the foreground server."
$Process = [System.Diagnostics.Process]::new()
$Process.StartInfo = $Start
$Started = $false
$Transferred = $false
try {
    if (-not $Process.Start()) { throw 'ComfyUI process did not start.' }
    $Started = $true
    Write-Host "ComfyUI PID: $($Process.Id) (server readiness is reported by ComfyUI)."
    if ($PassThru) {
        $Transferred = $true
        return $Process
    }
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) { throw "ComfyUI exited with code $($Process.ExitCode). See server output." }
} finally {
    if (-not $Transferred) {
        try {
            if ($Started -and -not $Process.HasExited) { $Process.Kill($true) }
        } finally { $Process.Dispose() }
    }
}