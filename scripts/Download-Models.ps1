[CmdletBinding()]
param(
    # Match WEIGHTS_DIR in .env if you customize it. This script does not load .env.
    [string]$WeightsDir = 'weights'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($WeightsDir)) {
    $WeightsDir = Join-Path $Root $WeightsDir
}
$WeightsDir = [System.IO.Path]::GetFullPath($WeightsDir)
# Required by older Windows PowerShell HTTPS clients; never disable certificate checks.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Models = @(
    @{ Name = 'RealESRGAN_x2plus.pth'; Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth' },
    @{ Name = 'RealESRGAN_x4plus.pth'; Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' },
    @{ Name = 'RealESRGAN_x4plus_anime_6B.pth'; Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth' },
    @{ Name = 'realesr-animevideov3.pth'; Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth' },
    @{ Name = 'realesr-general-x4v3.pth'; Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth' },
    @{ Name = '4x-UltraSharp.pth'; Url = 'https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth' },
    @{ Name = 'GFPGANv1.4.pth'; Url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' },
    @{ Name = 'facexlib/detection_Resnet50_Final.pth'; Url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth' },
    @{ Name = 'facexlib/parsing_parsenet.pth'; Url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth' }
)

foreach ($Model in $Models) {
    $Destination = Join-Path $WeightsDir $Model.Name
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Skipping existing path: $Destination (not verified)."
        continue
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    # Same-directory temporary file makes final publication an atomic rename.
    $Temporary = "$Destination.$([Guid]::NewGuid().ToString('N')).part"
    try {
        Write-Host "Downloading $($Model.Name)"
        Invoke-WebRequest -Uri $Model.Url -OutFile $Temporary -UseBasicParsing
        if ((Get-Item -LiteralPath $Temporary).Length -eq 0) {
            throw "Empty download: $($Model.Name)"
        }
        # File.Move does not overwrite a destination created concurrently.
        [System.IO.File]::Move($Temporary, $Destination)
    } finally {
        if (Test-Path -LiteralPath $Temporary) {
            Remove-Item -LiteralPath $Temporary
        }
    }
}

Write-Host 'RIFE is NOT downloaded automatically.'
Write-Host 'Obtain a trusted model archive from https://github.com/hzwer/Practical-RIFE (official README links).'
Write-Host "Manually place RIFE_HDv3.py, IFNet*.py and flownet.pkl directly in $WeightsDir/rife/4.26 (or 4.25)."
Write-Host 'Keep any additional model package files/import dependencies. Do not leave an extra train_log folder level.'
Write-Host 'Inspect archives for absolute paths and .. traversal before extracting. Model Python/pickle can execute code.'
Write-Host 'Core preflight copies missing facexlib detector/parser cache files into project-root gfpgan/weights before GFPGAN loads.'