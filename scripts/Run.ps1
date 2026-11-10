#requires -Version 7.0
<#
.SYNOPSIS
Start owned ComfyUI and Gradio processes; stop both when the UI exits.
.DESCRIPTION
No activation, installs or downloads. -AppOnly opts into a separately managed
loopback ComfyUI server, which is never stopped by this launcher. -LowVram is
forwarded to the dedicated backend launcher. Ctrl+C runs best-effort cleanup;
forcibly terminating PowerShell cannot guarantee cleanup.
#>
[CmdletBinding()]
param(
    [switch]$AppOnly,
    [switch]$LowVram
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv/Scripts/python.exe'
$App = Join-Path $Root 'app.py'

function New-AppStartInfo {
    $Start = [System.Diagnostics.ProcessStartInfo]::new()
    $Start.FileName = $Python
    $Start.WorkingDirectory = $Root
    $Start.UseShellExecute = $false
    foreach ($Argument in @('-E', '-s', '-u', $App)) { $Start.ArgumentList.Add($Argument) }
    foreach ($Key in @('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_CONSTRAINT')) {
        $Start.Environment.Remove($Key) | Out-Null
    }
    $Start.Environment['VIRTUAL_ENV'] = Join-Path $Root '.venv'
    $Start.Environment['PATH'] = (Join-Path $Root '.venv/Scripts') + [IO.Path]::PathSeparator + $env:PATH
    return $Start
}

function Assert-BackendAlive {
    param([System.Diagnostics.Process]$Backend)
    if ($null -ne $Backend -and $Backend.HasExited) {
        throw "Owned ComfyUI exited with code $($Backend.ExitCode). See backend output."
    }
}

function Wait-BackendReady {
    param([string]$Origin, [System.Diagnostics.Process]$Backend)
    $Clock = [System.Diagnostics.Stopwatch]::StartNew()
    $LastFailure = 'Server has not responded.'
    while ($Clock.Elapsed.TotalSeconds -lt 180) {
        Assert-BackendAlive $Backend
        try {
            # Never follow redirects or use an HTTP proxy for local requests.
            $Stats = Invoke-RestMethod -Uri "$Origin/system_stats" -TimeoutSec 3 -MaximumRedirection 0 -NoProxy
            Assert-BackendAlive $Backend
            if ($Clock.Elapsed.TotalSeconds -ge 180) { break }
            $Nodes = Invoke-RestMethod -Uri "$Origin/object_info" -TimeoutSec 3 -MaximumRedirection 0 -NoProxy
            Assert-BackendAlive $Backend
            if ($null -eq $Stats -or $null -eq $Stats.PSObject.Properties['system'] -or
                $null -eq $Nodes -or $null -eq $Nodes.PSObject.Properties['FaceDetailer'] -or
                $null -eq $Nodes.PSObject.Properties['UltralyticsDetectorProvider']) {
                throw 'Required system stats / FaceDetailer / UltralyticsDetectorProvider are unavailable.'
            }
            if ($Clock.Elapsed.TotalSeconds -lt 180) { return }
        } catch {
            $LastFailure = $_.Exception.Message
            Assert-BackendAlive $Backend
        }
        if ($null -ne $Backend) {
            $Backend.WaitForExit(250) | Out-Null
        } else {
            [System.Threading.Tasks.Task]::Delay(250).GetAwaiter().GetResult()
        }
    }
    throw "ComfyUI readiness timed out after 180 seconds (up to one request timeout overhead). $LastFailure"
}

function Stop-OwnedProcess {
    param([System.Diagnostics.Process]$OwnedProcess)
    if ($null -eq $OwnedProcess) { return }
    try {
        if (-not $OwnedProcess.HasExited) {
            $OwnedProcess.Kill($true)
            if (-not $OwnedProcess.WaitForExit(5000)) { Write-Warning 'Owned process did not exit within cleanup timeout.' }
        }
    } catch {
        Write-Warning "Owned-process cleanup failed: $($_.Exception.Message)"
    } finally { $OwnedProcess.Dispose() }
}

if (-not (Test-Path -LiteralPath $App -PathType Leaf)) {
    throw 'app.py is missing. Restore the application entry point before running.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Virtual environment missing. Run scripts/Setup.ps1 explicitly first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Root '.env') -PathType Leaf)) {
    throw '.env missing. Copy .env.example to .env and configure it first.'
}

# Use the same dotenv loading/precedence as app.py, without importing the UI.
# Validation occurs before any HTTP requests or backend startup. No DNS lookup.
$Probe = @'
import sys, os, json, ipaddress
from urllib.parse import urlsplit
from dotenv import load_dotenv
assert sys.version_info[:2] == (3, 11), 'Python 3.11 required'
load_dotenv(sys.argv[1], override=False)
raw = os.getenv('COMFYUI_URL', 'http://127.0.0.1:8188')
assert not any(c.isspace() or ord(c) < 32 for c in raw), 'Invalid COMFYUI_URL'
url = urlsplit(raw)
assert url.scheme in ('http', 'https'), 'COMFYUI_URL must use HTTP(S)'
assert url.username is None and url.password is None, 'COMFYUI_URL cannot contain credentials'
assert url.path in ('', '/') and not url.query and not url.fragment, 'COMFYUI_URL must be an origin'
assert '?' not in raw and '#' not in raw, 'COMFYUI_URL must be an origin'
assert url.hostname in ('127.0.0.1', '::1'), 'COMFYUI_URL must use literal 127.0.0.1 or ::1'
assert ipaddress.ip_address(url.hostname).is_loopback
port = url.port if url.port is not None else (443 if url.scheme == 'https' else 80)
assert 1 <= port <= 65535, 'Invalid COMFYUI_URL port'
host = '[' + url.hostname + ']' if ':' in url.hostname else url.hostname
print(json.dumps(dict(origin=f'{url.scheme}://{host}:{port}', listen=url.hostname, port=port, scheme=url.scheme)))
'@
$Backend = $null
$UI = $null
try {
    $ConfigJson = & $Python -I -c $Probe (Join-Path $Root '.env')
    if ($LASTEXITCODE -ne 0) { throw 'Application Python / COMFYUI_URL preflight failed. No server started.' }
    $Config = ($ConfigJson -join "`n") | ConvertFrom-Json
    if (-not $AppOnly) {
        if ($Config.scheme -ne 'http') { throw 'Unified launch supports HTTP only; use -AppOnly for separately managed HTTPS.' }
        # Run-ComfyUI performs its own port bind check and refuses occupied ports.
        $Backend = & (Join-Path $PSScriptRoot 'Run-ComfyUI.ps1') -PassThru -Listen $Config.listen -Port $Config.port -LowVram:$LowVram
        if ($Backend -isnot [System.Diagnostics.Process]) { throw 'Backend launcher must return exactly one Process.' }
    } elseif ($LowVram) {
        throw '-LowVram cannot configure a separately managed backend; omit it with -AppOnly.'
    }
    if (-not $AppOnly) {
        Write-Host "Waiting for ComfyUI at $($Config.origin) with required FaceDetailer nodes..."
        Wait-BackendReady -Origin $Config.origin -Backend $Backend
    } else {
        Write-Host 'App-only mode: no ComfyUI startup/readiness check. FaceDetailer checks its backend when processing.'
    }
    Assert-BackendAlive $Backend
    $Start = New-AppStartInfo
    $Start.Environment['COMFYUI_URL'] = $Config.origin
    $UI = [System.Diagnostics.Process]::new()
    $UI.StartInfo = $Start
    if (-not $UI.Start()) { throw 'Application process did not start.' }
    Write-Host "Gradio PID: $($UI.Id). Ctrl+C stops owned processes."
    while (-not $UI.WaitForExit(250)) { Assert-BackendAlive $Backend }
    Assert-BackendAlive $Backend
    if ($UI.ExitCode -ne 0) { throw "Application exited with code $($UI.ExitCode). See UI output." }
} finally {
    # Never enumerate/kill processes by name or stop a separately managed server.
    try { Stop-OwnedProcess $UI } finally { Stop-OwnedProcess $Backend }
}