param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$workerPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$worker = Join-Path $projectRoot "hybrid_input\office_worker.py"
$source = (Get-ChildItem -LiteralPath (Join-Path $projectRoot "visual_index_sourse") -File -Filter "*.pptx" | Select-Object -First 1).FullName
$root = Join-Path $projectRoot "schema2-index-acceptance\ppt-com-diagnostic"
$output = Join-Path $root "output"
$result = Join-Path $root "result.json"
$log = Join-Path $root "worker.log"

if (-not (Test-Path -LiteralPath $workerPython -PathType Leaf)) {
    throw "Project Python was not found at $workerPython. Run setup.cmd first."
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue

Write-Host "Starting PowerPoint COM diagnostic for $source" -ForegroundColor Cyan
& $workerPython -X utf8 $worker --input $source --output $output --result $result --mode office-visuals *>> $log
$exitCode = $LASTEXITCODE
[PSCustomObject]@{
    status = if ($exitCode -eq 0) { "complete" } else { "failed" }
    exit_code = $exitCode
    source = $source
    log = $log
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root "report.json") -Encoding UTF8
Write-Host "Worker exited with code $exitCode. Log: $log" -ForegroundColor Yellow
Read-Host "Press Enter to finish"
