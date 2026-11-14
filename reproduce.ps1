# Valhalla one-click reproduction (Windows).
#   .\reproduce.ps1           full battery (~15 min)
#   .\reproduce.ps1 --quick   probes + smoke + 806 tier only
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Ai\Valhalla-main\.venv-llm\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py tests\reproduce_all.py @args
exit $LASTEXITCODE
