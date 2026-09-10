$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".build-env\Scripts\python.exe"
$document = (Get-ChildItem -LiteralPath (Join-Path $projectRoot "schema2-index-acceptance\ppt\artifacts") -Recurse -Filter "hybrid-document.json" | Select-Object -First 1).FullName
$source = (Get-ChildItem -LiteralPath (Join-Path $projectRoot "visual_index_sourse") -File -Filter "*.pptx" | Select-Object -First 1).FullName
$script = Join-Path $projectRoot "tools\run_schema2_snapshot_acceptance.py"
$index = Join-Path $projectRoot "schema2-index-acceptance\ppt\index"

Write-Host "Starting interactive strict PPT index build" -ForegroundColor Cyan
& $python -X utf8 $script --document $document --source $source --preserve-visual-assets --index-dir $index --model (Join-Path $projectRoot "model-cache\Qwen3-VL-Embedding-2B") --device cpu
$exitCode = $LASTEXITCODE
Write-Host "PPT index build exited with code $exitCode" -ForegroundColor Yellow
Read-Host "Press Enter to finish"
