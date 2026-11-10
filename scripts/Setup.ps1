# Run explicitly to install dependencies. No elevation or system package installs.
[CmdletBinding()]
param([switch]$DownloadModels)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

Push-Location $Root
try {
    foreach ($Tool in @('py', 'ffmpeg', 'ffprobe', 'git')) {
        if (-not (Get-Command $Tool -CommandType Application -ErrorAction SilentlyContinue)) {
            throw "Missing prerequisite: $Tool. Install it manually and add it to PATH, then retry."
        }
    }
    $VersionCheck = 'import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'
    Invoke-Checked -Executable 'py' -Arguments @('-3.11', '-c', $VersionCheck)
    Invoke-Checked -Executable 'ffmpeg' -Arguments @('-version')
    Invoke-Checked -Executable 'ffprobe' -Arguments @('-version')
    Invoke-Checked -Executable 'git' -Arguments @('--version')

    $Python = Join-Path $Root '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        if (Test-Path -LiteralPath (Join-Path $Root '.venv')) {
            throw 'An incomplete .venv already exists. Repair or remove it manually before retrying.'
        }
        Invoke-Checked -Executable 'py' -Arguments @('-3.11', '-m', 'venv', '.venv')
    }
    Invoke-Checked -Executable $Python -Arguments @('-c', $VersionCheck)
    Invoke-Checked -Executable $Python -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked -Executable $Python -Arguments @(
        '-m', 'pip', 'install', 'torch==2.5.1+cu124', 'torchvision==0.20.1+cu124',
        '--index-url', 'https://download.pytorch.org/whl/cu124'
    )
    Invoke-Checked -Executable $Python -Arguments @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked -Executable $Python -Arguments @('-m', 'pip', 'check')

    $Vendor = Join-Path $Root 'vendor/Practical-RIFE'
    if (-not (Test-Path -LiteralPath $Vendor)) {
        New-Item -ItemType Directory -Path (Join-Path $Root 'vendor') -Force | Out-Null
        Invoke-Checked -Executable 'git' -Arguments @(
            'clone', 'https://github.com/hzwer/Practical-RIFE.git', $Vendor
        )
    } elseif (-not (Test-Path -LiteralPath (Join-Path $Vendor '.git'))) {
        throw 'vendor/Practical-RIFE exists but is not a Git checkout. Inspect it manually.'
    } else {
        Write-Host 'Keeping existing Practical-RIFE checkout; no pull or reset performed.'
    }
    # Record the revision in terminal output; upstream is not a frozen dependency.
    Invoke-Checked -Executable 'git' -Arguments @('-C', $Vendor, 'rev-parse', 'HEAD')
    # Do not install vendor requirements: they can conflict with the pinned stack.
    if (-not (Test-Path -LiteralPath (Join-Path $Root '.env'))) {
        Copy-Item -LiteralPath (Join-Path $Root '.env.example') -Destination (Join-Path $Root '.env')
    }
    foreach ($Directory in @('weights', 'temp', 'outputs')) {
        New-Item -ItemType Directory -Path (Join-Path $Root $Directory) -Force | Out-Null
    }
    if ($DownloadModels) {
        & (Join-Path $PSScriptRoot 'Download-Models.ps1')
    }
    Write-Host 'Setup complete. RIFE model installation is manual; see README.md.'
    Write-Host 'The application includes a BasicSR compatibility shim for torchvision 0.20.1.'
    Write-Host 'Application and core are implemented; setup does not validate runtime ML or GPU inference.'
} finally {
    Pop-Location
}