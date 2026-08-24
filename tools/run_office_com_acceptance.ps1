param(
    [ValidateSet("word", "powerpoint", "excel")]
    [string[]]$Applications = @("word", "powerpoint", "excel")
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$desktopRoot = Split-Path -Parent $projectRoot
$workerPython = Get-ChildItem -LiteralPath $desktopRoot -Directory | ForEach-Object {
    Join-Path $_.FullName ".conda-env\python.exe"
} | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
$worker = Join-Path $projectRoot "hybrid_input\office_worker.py"
$sourceRoot = Join-Path $projectRoot "visual_index_sourse"
$outputRoot = Join-Path $projectRoot "schema2-index-acceptance\office-com-interactive"

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
trap {
    $failure = [PSCustomObject]@{
        status = "failed"
        user = [Environment]::UserName
        session_id = (Get-Process -Id $PID).SessionId
        error = $_.Exception.ToString()
    }
    $failure | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputRoot "office-com-report.json") -Encoding UTF8
    Write-Host $_.Exception.ToString() -ForegroundColor Red
    Read-Host "Office COM acceptance failed. Press Enter to close"
    exit 1
}

if (-not $workerPython) {
    throw "Office worker Python was not found under the desktop directories"
}

$cases = @(
    @{ Name = "word"; File = (Get-ChildItem -LiteralPath $sourceRoot -File -Filter "*.docx" | Select-Object -First 1).FullName },
    @{ Name = "powerpoint"; File = (Get-ChildItem -LiteralPath $sourceRoot -File -Filter "*.pptx" | Select-Object -First 1).FullName },
    @{ Name = "excel"; File = (Get-ChildItem -LiteralPath $sourceRoot -File -Filter "*.xlsx" | Select-Object -First 1).FullName }
) | Where-Object { $_.Name -in $Applications }

$summary = @()

foreach ($case in $cases) {
    $source = $case.File
    if (-not $source) {
        throw "No $($case.Name) source file was found"
    }
    $caseOutput = Join-Path $outputRoot $case.Name
    $result = Join-Path $outputRoot "$($case.Name)-result.json"
    New-Item -ItemType Directory -Force -Path $caseOutput | Out-Null
    Write-Host "Testing Office COM native export: $([IO.Path]::GetFileName($source))" -ForegroundColor Cyan
    & $workerPython -X utf8 $worker --input $source --output $caseOutput --result $result --mode office-visuals
    if ($LASTEXITCODE -ne 0) {
        throw "Office COM native export failed for $([IO.Path]::GetFileName($source))"
    }
    $payload = Get-Content -LiteralPath $result -Raw -Encoding UTF8 | ConvertFrom-Json
    $summary += [PSCustomObject]@{
        application = $case.Name
        source = $source
        exported_visuals = @($payload.visuals).Count
        status = "complete"
    }
}

$summaryPath = Join-Path $outputRoot "office-com-report.json"
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | Format-Table -AutoSize
Write-Host "Office COM acceptance passed: $summaryPath" -ForegroundColor Green
Write-Host "You may close this window." -ForegroundColor Green
Read-Host "Press Enter to finish"
