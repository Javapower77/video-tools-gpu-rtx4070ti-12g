#requires -Version 7.0
# Standalone tests: no Pester dependency, GPU imports, or real application servers.
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Assert-Throws {
    param([scriptblock]$Action, [string]$Pattern)
    $Failure = $null
    try { & $Action | Out-Null } catch { $Failure = $_.Exception.Message }
    Assert-True ($null -ne $Failure -and $Failure -match $Pattern) "Expected failure matching '$Pattern'; got '$Failure'."
}
function Read-ScriptAst {
    param([string]$Path)
    $Tokens = $null
    $ParseErrors = $null
    $Ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$Tokens, [ref]$ParseErrors)
    Assert-True ($ParseErrors.Count -eq 0) "Parse errors in ${Path}: $ParseErrors"
    return $Ast
}
function New-TestProcess {
    param([string]$Code)
    $Info = [System.Diagnostics.ProcessStartInfo]::new()
    $Info.FileName = (Get-Process -Id $PID).Path
    $Info.UseShellExecute = $false
    foreach ($Arg in @('-NoProfile', '-NonInteractive', '-Command', $Code)) { $Info.ArgumentList.Add($Arg) }
    return [System.Diagnostics.Process]::Start($Info)
}

$RunAst = Read-ScriptAst (Join-Path $Root 'scripts/Run.ps1')
$ComfyAst = Read-ScriptAst (Join-Path $Root 'scripts/Run-ComfyUI.ps1')
# Load only function definitions, never the scripts' real entry points.
foreach ($Definition in $RunAst.FindAll({ param($Node) $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($Definition.Extent.Text))
}
$Python = Join-Path $Root '.venv/Scripts/python.exe'
$App = Join-Path $Root 'app.py'
$Info = New-AppStartInfo
Assert-True (-not $Info.UseShellExecute) 'UI must not use a shell.'
Assert-True (($Info.ArgumentList -join '|') -eq "-E|-s|-u|$App") 'UI Python isolation flags missing.'
foreach ($Key in @('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_CONSTRAINT')) {
    Assert-True (-not $Info.Environment.ContainsKey($Key)) "Unsafe environment key retained: $Key"
}

# Mock HTTP at the function boundary: enforce local/no-redirect/no-proxy requests.
$script:Calls = @()
$script:MissingNodes = $false
function Invoke-RestMethod {
    param($Uri, $TimeoutSec, $MaximumRedirection, [switch]$NoProxy)
    Assert-True ($Uri -match '^http://127\.0\.0\.1:8188/(system_stats|object_info)$') 'Unexpected request origin.'
    Assert-True ($TimeoutSec -eq 3 -and $MaximumRedirection -eq 0 -and $NoProxy) 'Unsafe request options.'
    $script:Calls += $Uri
    if ($Uri.EndsWith('/system_stats')) { return [pscustomobject]@{ system = @{} } }
    if ($script:MissingNodes) { return [pscustomobject]@{ FaceDetailer = @{} } }
    return [pscustomobject]@{ FaceDetailer = @{}; UltralyticsDetectorProvider = @{} }
}
Wait-BackendReady -Origin 'http://127.0.0.1:8188' -Backend $null
Assert-True ($script:Calls.Count -eq 2) 'Readiness must inspect both endpoints.'
$Exited = New-TestProcess 'exit 7'
try {
    Assert-True ($Exited.WaitForExit(10000)) 'Short-lived test process hung.'
    Assert-Throws { Wait-BackendReady -Origin 'http://127.0.0.1:8188' -Backend $Exited } 'exited with code 7'
    Assert-True ($script:Calls.Count -eq 2) 'Exited backend must fail before HTTP.'
} finally { Stop-OwnedProcess $Exited }
# Shorten only the readiness deadline in a test copy; keep production at 180s.
$ReadyDefinition = $RunAst.Find({ param($Node)
    $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Wait-BackendReady'
}, $false)
$ReadyBody = $ReadyDefinition.Body.Extent.Text
$FastReady = [scriptblock]::Create(($ReadyBody.Substring(1, $ReadyBody.Length - 2) -replace '180', '0.05'))
$script:MissingNodes = $true
Assert-Throws { & $FastReady -Origin 'http://127.0.0.1:8188' -Backend $null } 'readiness timed out.*UltralyticsDetectorProvider'
$script:MissingNodes = $false

# Exercise the actual backend lifecycle tail with a harmless process instead of
# running GPU/provenance preflights or starting ComfyUI.
$ComfyText = $ComfyAst.Extent.Text
$Tail = [scriptblock]::Create($ComfyText.Substring($ComfyText.IndexOf('$Process = [System.Diagnostics.Process]::new()')))
$Start = [System.Diagnostics.ProcessStartInfo]::new()
$Start.FileName = (Get-Process -Id $PID).Path
$Start.UseShellExecute = $false
foreach ($Arg in @('-NoProfile', '-NonInteractive', '-Command', '[System.Threading.Tasks.Task]::Delay(10000).GetAwaiter().GetResult()')) {
    $Start.ArgumentList.Add($Arg)
}
$PassThru = $true
Assert-True $PassThru 'Lifecycle test must start in transfer mode.'
$Result = @(& $Tail)
$Owned = $Result[0]
try {
    Assert-True ($Result.Count -eq 1 -and $Owned -is [System.Diagnostics.Process]) 'PassThru must emit only one Process.'
    Assert-True (-not $Owned.HasExited) 'PassThru waited for or killed the child.'
    $Owned.Kill($true)
    Assert-True ($Owned.WaitForExit(5000)) 'Caller could not stop transferred child.'
} finally { Stop-OwnedProcess $Owned }
$Start.ArgumentList.Clear()
foreach ($Arg in @('-NoProfile', '-NonInteractive', '-Command', 'exit 0')) { $Start.ArgumentList.Add($Arg) }
$PassThru = $false
Assert-True (@(& $Tail).Count -eq 0) 'Foreground success must not return a process.'
$Start.ArgumentList[$Start.ArgumentList.Count - 1] = 'exit 9'
Assert-Throws { & $Tail } 'exited with code 9'

# Test the exact dotenv/URL probe using the existing application interpreter.
# Temporary config only; do not touch the project's .env or environment.
$ProbeAssignment = $RunAst.Find({ param($Node)
    $Node -is [System.Management.Automation.Language.AssignmentStatementAst] -and $Node.Left.Extent.Text -eq '$Probe'
}, $false)
$Probe = $ProbeAssignment.Right.Expression.Value
$TempEnv = Join-Path ([IO.Path]::GetTempPath()) ("launch-test-" + [guid]::NewGuid() + '.env')
$OldUrl = [Environment]::GetEnvironmentVariable('COMFYUI_URL')
try {
    Remove-Item Env:COMFYUI_URL -ErrorAction SilentlyContinue
    [IO.File]::WriteAllText($TempEnv, "COMFYUI_URL=http://127.0.0.1:8199/`n")
    $Json = & $Python -I -c $Probe $TempEnv
    Assert-True ($LASTEXITCODE -eq 0) 'Dotenv URL probe failed.'
    $Config = $Json | ConvertFrom-Json
    Assert-True ($Config.port -eq 8199 -and $Config.origin -eq 'http://127.0.0.1:8199') 'Dotenv port/origin mismatch.'
    foreach ($Url in @('http://[::1]:8188', 'https://127.0.0.1:8443')) {
        [Environment]::SetEnvironmentVariable('COMFYUI_URL', $Url)
        $Json = & $Python -I -c $Probe $TempEnv
        Assert-True ($LASTEXITCODE -eq 0) "Valid origin rejected: $Url"
    }
    foreach ($Url in @('http://example.com', 'http://localhost:8188', 'http://127.0.0.1:8188/path',
        'http://user@127.0.0.1:8188', 'http://127.0.0.1:8188?x=1', 'http://127.0.0.1:8188#x',
        'file:///tmp/server', 'http://127.0.0.1:0', 'http://127.0.0.1:65536')) {
        [Environment]::SetEnvironmentVariable('COMFYUI_URL', $Url)
        & $Python -I -c $Probe $TempEnv 2>$null | Out-Null
        Assert-True ($LASTEXITCODE -ne 0) "Unsafe origin accepted: $Url"
    }
} finally {
    if ($null -eq $OldUrl) { Remove-Item Env:COMFYUI_URL -ErrorAction SilentlyContinue }
    else { [Environment]::SetEnvironmentVariable('COMFYUI_URL', $OldUrl) }
    Remove-Item -LiteralPath $TempEnv -ErrorAction SilentlyContinue
}
Write-Host 'PASS: launch script parsing, environment isolation, mocked readiness, process ownership, foreground behavior and dotenv URL validation.'