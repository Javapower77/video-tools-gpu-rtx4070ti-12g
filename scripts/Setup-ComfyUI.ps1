#requires -Version 7.0
<#
.SYNOPSIS
Explicitly provision an isolated Python 3.11 ComfyUI environment; never fetch models.
.DESCRIPTION
Research (2026-10-02): Comfy-Org/ComfyUI/master/requirements.txt includes
comfy-kitchen==0.2.36 (documented torch>=2.7) and comfy-aimdo==0.5.5
(DynamicVRAM needs torch>=2.8). Use torch 2.8.0 / torchvision 0.23.0 / cu128
for these checkouts, not the application's 2.5.1 / 0.20.1 / cu124.
Older checkouts without these dependencies default to the requested cu124 pair.
Sources: https://pypi.org/project/comfy-kitchen/0.2.36/
https://pypi.org/project/comfy-aimdo/0.5.5/
https://pytorch.org/get-started/previous-versions/
Impact-Pack/Main/install.py and Impact-Subpack/main/install.py both check
custom_nodes/skip_download_model. Subpack manual installation only needs requirements.
Unfrozen upstream requirements (including facebookresearch/sam2) execute trusted
third-party build code. This is NOT a sandbox or a fully reproducible lockfile.
No existing checkout is pulled, reset, cleaned, or checked out to a new revision.
#>
[CmdletBinding()]
param(
    [string]$ComfyRevision = '',
    [ValidateSet('Comfy-Org/ComfyUI', 'comfyanonymous/ComfyUI')]
    [string]$ComfyRepository = 'Comfy-Org/ComfyUI',
    [ValidateSet('Auto', 'cu124', 'cu128')][string]$TorchStack = 'Auto',
    # Optional official pack; never download its models during setup.
    [switch]$SeedVR2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Comfy = Join-Path $Root 'vendor/ComfyUI'
$Nodes = Join-Path $Comfy 'custom_nodes'
$Python = Join-Path $Comfy '.venv/Scripts/python.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed ($LASTEXITCODE). See preceding output; no automatic rollback performed." }
}

function Confirm-Repository {
    param([string]$Path, [string[]]$Allowed)
    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) { throw "Not a Git checkout: $Path" }
    $Origin = & git -C $Path remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw "Cannot read origin: $Path" }
    $Normalized = $Origin.Trim().TrimEnd('/').Replace('git@github.com:', 'https://github.com/') -replace '\.git$', ''
    if ($Normalized -notin $Allowed) { throw "Untrusted origin '$Origin' at $Path. Inspect manually." }
    $Hash = & git -C $Path rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Cannot read HEAD: $Path" }
    Write-Host "$Path HEAD: $Hash"
    $Changes = & git -C $Path status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect tracked changes: $Path" }
    if ($Changes) { throw "Tracked files are modified at $Path. Refusing to execute modified upstream code." }
    return $Hash.Trim()
}

function Add-Repository {
    param([string]$Path, [string]$Repository, [string[]]$Allowed, [string]$Revision = '')
    if (-not (Test-Path -LiteralPath $Path)) {
        Invoke-Checked git @('clone', '--', "https://github.com/$Repository.git", $Path)
        if ($Revision) {
            # Full clone, local object only: never fetch or change an existing checkout.
            Invoke-Checked git @('-C', $Path, 'checkout', '--detach', $Revision)
        }
    } else { Write-Host "Keeping existing checkout: $Path (no pull/reset/checkout)." }
    $Hash = Confirm-Repository $Path $Allowed
    if ($Revision) {
        $Requested = & git -C $Path rev-parse --verify "$Revision^{commit}"
        if ($LASTEXITCODE -ne 0 -or $Requested.Trim() -ne $Hash) {
            throw "Existing HEAD does not match revision '$Revision'. Select the revision manually; nothing was reset."
        }
    } else { Write-Warning "$Repository uses explicitly UNFROZEN HEAD ($Hash); dependency versions are not fully frozen." }
}

foreach ($Tool in @('py', 'git')) {
    if (-not (Get-Command $Tool -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Missing $Tool. Install Python 3.11 / Git manually; this script never elevates or installs system prerequisites."
    }
}
if ($ComfyRevision.StartsWith('-')) { throw 'ComfyRevision cannot be a Git option.' }
Invoke-Checked py @('-3.11', '-I', '-c', 'import sys; assert sys.version_info[:2] == (3,11); print(sys.version)')
New-Item -ItemType Directory -Path (Join-Path $Root 'vendor') -Force | Out-Null
Add-Repository $Comfy $ComfyRepository @('https://github.com/Comfy-Org/ComfyUI', 'https://github.com/comfyanonymous/ComfyUI') $ComfyRevision
New-Item -ItemType Directory -Path $Nodes -Force | Out-Null

# Persist BEFORE any node requirements/installer execution. Never remove this marker.
$Marker = Join-Path $Nodes 'skip_download_model'
if (Test-Path -LiteralPath $Marker) {
    if (-not (Test-Path -LiteralPath $Marker -PathType Leaf)) { throw "Marker must be a file: $Marker" }
} else { New-Item -ItemType File -Path $Marker | Out-Null }
foreach ($Name in @('ComfyUI-Impact-Pack', 'ComfyUI-Impact-Subpack')) {
    Add-Repository (Join-Path $Nodes $Name) "ltdrdata/$Name" @("https://github.com/ltdrdata/$Name")
    $Installer = Join-Path $Nodes "$Name/install.py"
    if (-not (Test-Path -LiteralPath $Installer -PathType Leaf) -or
        (Get-Content -LiteralPath $Installer -Raw) -notmatch 'skip_download_model') {
        throw "Upstream installer changed: $Installer. Review its download safeguards before continuing."
    }
}

$Requirements = @(
    (Join-Path $Comfy 'requirements.txt'),
    (Join-Path $Nodes 'ComfyUI-Impact-Pack/requirements.txt'),
    (Join-Path $Nodes 'ComfyUI-Impact-Subpack/requirements.txt')
)
if ($SeedVR2) {
    $SeedRevision = '4490bd1f482e026674543386bb2a4d176da245b9'
    Add-Repository (Join-Path $Nodes 'seedvr2_videoupscaler') 'numz/ComfyUI-SeedVR2_VideoUpscaler' @('https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler') $SeedRevision
    $Requirements += Join-Path $Nodes 'seedvr2_videoupscaler/requirements.txt'
    Write-Host "SeedVR2 requirements will use only $Python with the ComfyUI CUDA constraints; the application venv is not used."
    Write-Warning 'Pinned SeedVR2 upstream recommends Python 3.12+; this project retains Python 3.11. Dependencies/inference need local validation. No SeedVR2 installer or model download is run.'
}
foreach ($File in $Requirements) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing upstream requirements: $File" }
    Write-Host "Upstream requirements: $File"
    Get-Content -LiteralPath $File | ForEach-Object { Write-Host "  $_" }
}
$Modern = (Get-Content -LiteralPath $Requirements[0] -Raw) -match '(?m)^\s*comfy-(kitchen|aimdo)\b'
if ($TorchStack -eq 'Auto') { $TorchStack = if ($Modern) { 'cu128' } else { 'cu124' } }
if ($Modern -and $TorchStack -eq 'cu124') {
    throw 'This checkout includes newer kernel/offloader dependencies; cu124 torch 2.5.1 is incompatible. Use Auto/cu128 or a reviewed older ComfyRevision.'
}
$TorchVersion, $VisionVersion = if ($TorchStack -eq 'cu124') { '2.5.1', '0.20.1' } else { '2.8.0', '0.23.0' }
Write-Host "Isolated CUDA stack: torch $TorchVersion, torchvision $VisionVersion, $TorchStack."
if ($TorchStack -eq 'cu128') { Write-Warning 'CUDA 12.8 needs a compatible NVIDIA driver. Current comfy-kitchen native wheels document CUDA 13 / r580+; eager fallback may be necessary. No driver changes are made.' }

# Save/restore process environment; never read or modify the application .env/.venv.
$Overrides = @{
    PYTHONHOME = $null; PYTHONPATH = $null; PYTHONNOUSERSITE = '1'
    VIRTUAL_ENV = (Join-Path $Comfy '.venv'); PIP_CONFIG_FILE = 'NUL'
    PIP_TARGET = $null; PIP_PREFIX = $null; PIP_USER = $null
    PIP_INDEX_URL = $null; PIP_EXTRA_INDEX_URL = $null; PIP_REQUIREMENT = $null
    PIP_CONSTRAINT = $null; PIP_BUILD_CONSTRAINT = $null
    COMFYUI_PATH = $Comfy; COMFYUI_MODEL_PATH = (Join-Path $Comfy 'models')
    HF_HUB_OFFLINE = '1'; TRANSFORMERS_OFFLINE = '1'; SAM2_BUILD_CUDA = '0'
}
$Saved = @{}
foreach ($Key in $Overrides.Keys) {
    $Saved[$Key] = [Environment]::GetEnvironmentVariable($Key, 'Process')
    [Environment]::SetEnvironmentVariable($Key, $Overrides[$Key], 'Process')
}
Push-Location $Comfy
try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        if (Test-Path -LiteralPath (Join-Path $Comfy '.venv')) { throw 'Incomplete ComfyUI .venv; repair manually. App .venv is not used.' }
        Invoke-Checked py @('-3.11', '-I', '-m', 'venv', (Join-Path $Comfy '.venv'))
    }
    Invoke-Checked $Python @('-I', '-c', 'import sys,pathlib; assert sys.version_info[:2] == (3,11); assert sys.prefix != sys.base_prefix; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(); assert "include-system-site-packages = false" in (pathlib.Path(sys.prefix)/"pyvenv.cfg").read_text().lower(); print(sys.executable)', (Join-Path $Comfy '.venv'))
    Invoke-Checked $Python @('-I', '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', 'packaging', '--index-url', 'https://pypi.org/simple')
    Invoke-Checked $Python @('-I', '-m', 'pip', 'install', "torch==$TorchVersion+$TorchStack", "torchvision==$VisionVersion+$TorchStack", '--index-url', "https://download.pytorch.org/whl/$TorchStack")
    # Runtime artifact stays INSIDE the dedicated venv. Constraints prevent silent
    # CPU/newer torch replacement; incompatible future requirements fail loudly.
    $Constraints = Join-Path $Comfy '.venv/comfy-cuda-constraints.txt'
    @("torch==$TorchVersion+$TorchStack", "torchvision==$VisionVersion+$TorchStack") | Set-Content -LiteralPath $Constraints -Encoding utf8
    [Environment]::SetEnvironmentVariable('PIP_CONSTRAINT', $Constraints, 'Process')
    foreach ($File in $Requirements) {
        Invoke-Checked $Python @('-I', '-m', 'pip', 'install', '--no-build-isolation', '--index-url', 'https://pypi.org/simple', '-r', $File)
    }
    # Impact's optional SAM2 native post-processing needs a toolkit/compiler;
    # disabling its build avoids implicit system prerequisites on Windows.
    Write-Warning 'SAM2 CUDA extension build disabled; some SAM2 post-processing is unavailable.'
    $InstallImpact = @'
import pathlib, runpy, sys
from torchvision.datasets import utils
root = pathlib.Path(sys.argv[1])
assert (root / 'custom_nodes' / 'skip_download_model').is_file()
def blocked(*args, **kwargs):
    raise RuntimeError('Model download blocked: use Download-ComfyUIModels.ps1 explicitly')
utils.download_url = blocked
sys.path.insert(0, str(root))
installer = root / 'custom_nodes' / 'ComfyUI-Impact-Pack' / 'install.py'
runpy.run_path(str(installer), run_name='__main__')
assert (installer.parent / 'impact-pack.ini').is_file(), 'Impact installer did not produce configuration'
'@
    Invoke-Checked $Python @('-I', '-c', $InstallImpact, $Comfy)
    Write-Host 'Subpack: requirements installed; install.py intentionally NOT run (it only downloads optional models).'
    Invoke-Checked $Python @('-I', '-m', 'pip', 'check')
    Invoke-Checked $Python @('-I', '-c', 'import torch,torchvision; print("torch",torch.__version__,"vision",torchvision.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver"; print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())')
    Write-Host 'Setup complete; no models downloaded. App requirements and environment unchanged.'
    Write-Host 'Run-ComfyUI.ps1 allowlists official Impact folders plus SeedVR2 when present. Full workflow inference remains unvalidated.'
} finally {
    Pop-Location
    foreach ($Key in $Saved.Keys) { [Environment]::SetEnvironmentVariable($Key, $Saved[$Key], 'Process') }
}