[CmdletBinding()]
param(
    [switch]$SkipUpgrade,
    [string]$PythonExecutable
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements.txt'

function Find-Python312 {
    if ($PythonExecutable) {
        $candidate = [System.IO.Path]::GetFullPath($PythonExecutable)
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Configured Python executable does not exist: $candidate"
        }
        & $candidate -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Configured Python must be 64-bit Python 3.12: $candidate"
        }
        return @($candidate)
    }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @($launcher.Source, '-3.12')
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @($python.Source)
        }
    }

    throw 'Python 3.12 64-bit was not found. Install it from https://www.python.org/downloads/windows/ and enable the Python launcher.'
}

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing dependency file: $Requirements"
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonCommand = @(Find-Python312)
    $PythonExecutable = $PythonCommand[0]
    $PythonArguments = @()
    if ($PythonCommand.Count -gt 1) {
        $PythonArguments += $PythonCommand[1..($PythonCommand.Count - 1)]
    }
    Write-Host 'Creating the project virtual environment in .venv ...'
    & $PythonExecutable @PythonArguments -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the virtual environment. Exit code: $LASTEXITCODE"
    }
}

& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv does not use 64-bit Python 3.12. Remove .venv and run setup again."
}

$env:PYTHONUTF8 = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

if (-not $SkipUpgrade) {
    Write-Host 'Updating pip, setuptools, and wheel ...'
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to update Python packaging tools. Exit code: $LASTEXITCODE"
    }
}

Write-Host 'Installing PixelRAG Studio dependencies. The first installation downloads several gigabytes and can take a long time ...'
& $VenvPython -m pip install --requirement $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. Exit code: $LASTEXITCODE"
}

Write-Host 'Checking the installed environment ...'
& $VenvPython (Join-Path $ProjectRoot 'verify_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw "Environment verification failed. Exit code: $LASTEXITCODE"
}

Write-Host ''
Write-Host 'Setup completed successfully.' -ForegroundColor Green
Write-Host 'Run start-studio.cmd to start PixelRAG Studio.'
