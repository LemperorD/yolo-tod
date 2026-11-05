[CmdletBinding()]
param(
    [string]$PythonVersion = "",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipChecks
)

# PowerShell 5.1 treats native stderr as errors under Stop; use exit codes instead.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"

$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_INPUT = "1"
$env:TMP = Join-Path $root ".cache\tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TMP | Out-Null

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red; exit 1 }

Step "1/6 select interpreter and create $venv"

if (-not $PythonVersion) { $PythonVersion = "python" }

$useLauncher = $false
$launchArgs = @()
if ($PythonVersion -match '^[0-9]+\.[0-9]+$') {
    $useLauncher = $true
    $launchArgs = @("-$PythonVersion")
    $probe = "py -$PythonVersion"
} else {
    $probe = $PythonVersion
}

function Invoke-Base([string[]]$extraArgs) {
    if ($useLauncher) { return (& py @launchArgs @extraArgs) }
    return (& $PythonVersion @extraArgs)
}

$baseVer = $null
try {
    $baseVer = (Invoke-Base @("-c", "import sys; print(sys.version)")) 2>&1 | Out-String
    $baseVer = $baseVer.Trim()
    if ($LASTEXITCODE -ne 0 -or -not $baseVer) { throw "exit $LASTEXITCODE" }
    Ok "base interpreter: $probe -> $baseVer"
} catch {
    Die "interpreter '$probe' is not usable ($($_.Exception.Message)). Use -PythonVersion, e.g. -PythonVersion 3.12 or -PythonVersion C:\Python314\python.exe"
}

if ($baseVer -match '^(\d+)\.(\d+)\.(\d+)') {
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]; $patch = [int]$Matches[3]
    if ($major -ne 3 -or $minor -lt 10) { Die "Python >= 3.10 required, got $baseVer" }
    if ($minor -eq 14 -and $patch -eq 1) {
        Die "torchvision excludes Python 3.14.1 exactly (see its requires-python). Use 3.12/3.13 or 3.14.2+"
    }
} else {
    Warn "could not parse the interpreter version; continuing"
}

function Test-Pip([string]$py) {
    if (-not (Test-Path $py)) { return $false }
    $null = & $py -m pip --version 2>$null
    return ($LASTEXITCODE -eq 0)
}

function New-Venv {
    Invoke-Base @("-m", "venv", $venv) | Out-Null
    return (Test-Path $venvPy)
}

if (Test-Path $venvPy) {
    if (Test-Pip $venvPy) {
        Ok ".venv already exists, skipping creation (delete $venv to rebuild)"
    } else {
        Warn ".venv exists but has no pip; recreating it"
        Remove-Item -Recurse -Force $venv
        if (-not (New-Venv)) {
            Warn "python -m venv failed (ensurepip); retrying with --without-pip"
            Invoke-Base @("-m", "venv", "--without-pip", $venv) | Out-Null
            if (-not (Test-Path $venvPy)) { Die "venv creation failed; no python.exe at $venvPy" }
        }
        Ok "created $venv"
    }
} else {
    if (-not (New-Venv)) {

        Warn "python -m venv failed (ensurepip); retrying with --without-pip"
        Invoke-Base @("-m", "venv", "--without-pip", $venv) | Out-Null
        if (-not (Test-Path $venvPy)) { Die "venv creation failed; no python.exe at $venvPy" }
    }
    Ok "created $venv"
}

Step "2/6 bootstrap pip in the venv"
if (-not (Test-Pip $venvPy)) {
    Warn "venv has no pip; installing it with the base interpreter"
    $null = Invoke-Base @("-m", "pip", "--python", $venvPy, "install", "--upgrade", "pip") 2>$null
    if (-not (Test-Pip $venvPy)) {
        Die "could not bootstrap pip in the venv. Run manually:`n  $probe -m venv $venv`n  then: $venvPy -m ensurepip --upgrade"
    }
}

$null = & $venvPy -m pip install --upgrade pip 2>$null
if ($LASTEXITCODE -ne 0) { Warn "pip self-upgrade failed (continuing with the bundled pip)" }
Ok ((& $venvPy -m pip --version 2>$null | Out-String).Trim())

Step "3/6 install torch stack from $TorchIndex"

$torchReq = Join-Path $root "requirement-torch.txt"
if (Test-Path $torchReq) {
    Ok "using $torchReq"
    & $venvPy -m pip install -r $torchReq --index-url $TorchIndex
} else {
    Ok "no requirement-torch.txt; resolving the newest wheels on the index"
    & $venvPy -m pip install torch torchvision --index-url $TorchIndex
}
if ($LASTEXITCODE -ne 0) {
    Die "torch install failed. Check that the index has wheels for this Python version. Use -TorchIndex to pick another CUDA build (cu118 / cu126 / cu128 / cu129 / cu130, or https://download.pytorch.org/whl/cpu)"
}
Ok "torch stack installed"

Step "4/6 install remaining dependencies (requirement.txt)"
$mainReq = Join-Path $root "requirement.txt"
if (-not (Test-Path $mainReq)) { Die "missing $mainReq" }
& $venvPy -m pip install -r $mainReq
if ($LASTEXITCODE -ne 0) { Die "requirement.txt install failed" }
Ok "dependencies installed"

Step "5/6 editable install of this repo (--no-deps; deps are already in place)"
Push-Location $root
try {
    & $venvPy -m pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { Die "editable install failed" }
} finally {
    Pop-Location
}
Ok "tod installed in editable mode"

if ($SkipChecks) {
    Write-Host ""
    Write-Host "Self-check skipped. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 0
}

Step "6/6 self-check"
Push-Location $root
try {
    $importOut = & $venvPy -c "import tod, torch, ultralytics; print(torch.__version__, ultralytics.__version__)" 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { Die "import tod/torch/ultralytics failed:`n$importOut" }
    Ok ("torch/ultralytics: " + $importOut.Trim())

    $cudaOut = & $venvPy -c "import torch; print(int(torch.cuda.is_available())); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')" 2>&1 | Out-String
    $cudaLines = $cudaOut.Trim() -split "`r?`n"
    if ($cudaLines[0] -eq "1") {
        Ok ("GPU available: " + $cudaLines[1])
    } else {
        Warn "torch cannot see a GPU ($cudaOut). If a CPU build was installed, rerun:"
        Warn "  $venvPy -m pip install --force-reinstall torch torchvision --index-url $TorchIndex"
    }

    & $venvPy "tests\smoke.py"
    if ($LASTEXITCODE -ne 0) { Die "tests/smoke.py did not pass" }
    Ok "architecture smoke test passed"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Next: python tests\test_pose.py   or   python tools\train_mav6d.py --model phantom4" -ForegroundColor Green
