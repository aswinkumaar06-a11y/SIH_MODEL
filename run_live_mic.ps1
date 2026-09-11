# SIH 26172: Live Microphone Edge Voice Activator Runner
param (
    [string]$Keyword = "ZORA",
    [double]$Threshold = 0.87,
    [switch]$RecordUser,
    [double]$Duration = 0.0,
    [int]$Device = $null
)

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ScriptPath = Join-Path $RepoRoot "scripts\live_mic_activator.py"

$ArgsList = @(
    $ScriptPath,
    "--keyword", $Keyword,
    "--threshold", $Threshold.ToString("0.00")
)

if ($RecordUser) {
    $ArgsList += "--record-user"
}
if ($Duration -gt 0) {
    $ArgsList += @("--duration", $Duration.ToString())
}
if ($null -ne $Device) {
    $ArgsList += @("--device", $Device.ToString())
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "   SIH 26172: EDGE VOICE ACTIVATOR - LIVE LAPTOP MICROPHONE" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Target Keyword:  $Keyword"
Write-Host "Threshold:       $Threshold"
Write-Host "Python:          $PythonExe"
Write-Host "---------------------------------------------------------------------" -ForegroundColor Gray

& $PythonExe $ArgsList
