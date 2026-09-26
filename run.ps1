# Native Windows entry point. PowerShell 5.1+; no Bash or administrator rights needed.
[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$NoSetup,
    [switch]$Tui,
    [switch]$DryRun,
    [string]$Plan,
    [string]$Experiment,
    [string]$Layers,
    [string]$Devices,
    [int]$NumGpus = 1,
    [string]$EnsembleRange = '0:9',
    [switch]$NoEcon,
    [switch]$GroupAblationOnly,
    [switch]$ListExperiments,
    [switch]$Help
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Confirm-Step([string]$Message) {
    while ($true) {
        $reply = Read-Host "$Message [Y/n]"
        if ($null -eq $reply) { return $false }
        if ($reply -match '^(y|yes)?$') { return $true }
        if ($reply -match '^(n|no)$') { return $false }
    }
}

function Invoke-Retry([string]$Label, [scriptblock]$Action) {
    while ($true) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "$Label failed. Check the output above, network/proxy settings and free disk space."
        if (!(Confirm-Step 'Retry this step?')) { throw 'Setup stopped. Run .\run.ps1 again to resume.' }
    }
}

function Find-Uv {
    $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    $candidates = @()
    if ($command) { $candidates += $command.Source }
    $candidates += @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    return $null
}

function Test-Environment([switch]$PythonOnly) {
    if (!$script:PythonBin -or !(Test-Path $script:PythonBin)) { return 10 }
    $probeArgs = @((Join-Path $ProjectRoot 'utils\check_environment.py'), '--project-root', $ProjectRoot, '--profile', $script:BootstrapProfile)
    if ($PythonOnly) { $probeArgs += @('--python-only', '--project-venv') }
    & $script:PythonBin @probeArgs | Out-Host
    return $LASTEXITCODE
}

function Install-Dependencies([switch]$Reinstall) {
    $baseArgs = @('pip', 'install', '--python', $ProjectPython)
    if ($Reinstall) { $baseArgs += '--reinstall' }
    & $script:UvBin @baseArgs -r "$ProjectRoot\docker\requirements\common.txt" -r "$ProjectRoot\docker\requirements\data.txt" -r "$ProjectRoot\docker\requirements\windows.txt"
    if ($LASTEXITCODE -ne 0) { return }
    if ($script:ForceCpu) { $baseArgs += @('--reinstall-package', 'torch') }
    if ($script:BootstrapProfile -eq 'cuda') {
        & $script:UvBin @baseArgs -r "$ProjectRoot\docker\requirements\torch.txt" --index-url https://download.pytorch.org/whl/cu128
    } else {
        & $script:UvBin @baseArgs -r "$ProjectRoot\docker\requirements\torch-cpu.txt" --index-url https://download.pytorch.org/whl/cpu
    }
}

try {
    if ($Help) {
        Write-Host @'
Native Windows launcher (PowerShell):
  .\run.ps1                         Open the TUI; offer environment setup if needed
  .\run.ps1 -Setup                  Set up the environment, then exit
  .\run.ps1 -NoSetup                Use an environment you manage yourself
  .\run.ps1 -Plan plan.json [-DryRun] [-NoEcon]
  .\run.ps1 -Experiment resnetps1 -Layers 1:3 -Devices cpu -EnsembleRange 0:0 -DryRun
  .\run.ps1 -ListExperiments
Optional experiment flags: -NumGpus N, -NoEcon, -GroupAblationOnly
Interpreter priority: PYTHON_BIN, VIRTUAL_ENV, project .venv, PATH python.
'@
        exit 0
    }
    if ($ListExperiments) {
        foreach ($family in @('resnetps', 'nnps', 'resnets', 'nns')) {
            foreach ($width in 1..4) { Write-Host "${family}${width}  hidden layers ${width}:20" }
        }
        exit 0
    }
    if ($env:OS -ne 'Windows_NT') { throw 'Use ./run.sh on macOS, Linux or WSL.' }
    if ($Setup -and ($NoSetup -or $Tui -or $DryRun -or $Plan -or $Experiment -or $GroupAblationOnly)) { throw 'Use -Setup on its own.' }
    if ($Plan -and ($Experiment -or $Tui -or $GroupAblationOnly)) { throw '-Plan cannot be combined with -Experiment, -Tui or -GroupAblationOnly.' }
    if ($Experiment -and (!$Layers -or $Tui)) { throw '-Experiment requires -Layers and cannot be combined with -Tui.' }
    if ($GroupAblationOnly -and !$Experiment) { throw '-GroupAblationOnly requires -Experiment and -Layers.' }
    if ($Plan -and !(Test-Path $Plan -PathType Leaf)) { throw "Plan not found: $Plan" }

    $architecture = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $architecture = $env:PROCESSOR_ARCHITEW6432 }
    $script:BootstrapProfile = 'cpu'
    $script:ForceCpu = $false
    $nvidia = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue
    if ($nvidia) {
        $gpuInfo = & $nvidia.Source --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuInfo) { $script:BootstrapProfile = 'cuda' }
    }
    if ($Devices -eq 'cpu') { $script:BootstrapProfile = 'cpu' }
    $backendFile = Join-Path $ProjectRoot '.venv\.launcher-backend'
    if (!$Devices -and $script:BootstrapProfile -eq 'cuda' -and (Test-Path $backendFile)) {
        if ((Get-Content $backendFile -Raw).Trim() -eq 'cpu') { $script:BootstrapProfile = 'cpu' }
    }
    Write-Host "System: Windows | Hardware: $architecture | Backend: $script:BootstrapProfile"
    if ($script:BootstrapProfile -eq 'cuda') { Write-Host "NVIDIA GPU: $gpuInfo" }

    $script:PythonBin = $null
    $candidates = @()
    if ($env:PYTHON_BIN) { $candidates = @($env:PYTHON_BIN) }
    else {
        if ($env:VIRTUAL_ENV) { $candidates += "$env:VIRTUAL_ENV\Scripts\python.exe" }
        $candidates += $ProjectPython
        $command = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
        # Do not open the Microsoft Store through the Python execution alias.
        if ($command -and $command.Source -notmatch '\\WindowsApps\\') { $candidates += $command.Source }
    }
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { $script:PythonBin = $command.Source; break }
    }

    if (!$NoSetup) {
        $status = Test-Environment
        if ($status -ne 0) {
            if (![Environment]::UserInteractive -or [Console]::IsInputRedirected -or [Console]::IsOutputRedirected) {
                throw 'Setup needs an interactive terminal. Run .\run.ps1 -Setup in Windows Terminal first. No packages were installed.'
            }
            if ($DryRun) { throw 'Run .\run.ps1 -Setup first; -DryRun never installs packages.' }
            if ($architecture -ne 'AMD64') { throw 'Automatic native Windows setup requires x64 for the pinned PyTorch wheels. Use a compatible environment with -NoSetup, or Linux/WSL.' }
            if ($script:BootstrapProfile -eq 'cuda') {
                Write-Host 'CUDA 12.8 PyTorch needs a compatible NVIDIA driver.'
                if (!(Confirm-Step 'Install the CUDA build? (n selects the smaller CPU build)')) {
                    $script:BootstrapProfile = 'cpu'
                    $script:ForceCpu = $true
                }
            }
            Write-Host "Setup will use $ProjectRoot\.venv and may download uv, Python 3.13.5, PyTorch and notebook/data dependencies."
            Write-Host 'An internet connection and several GB of free space are needed. System Python and other active environments are left unchanged.'
            if (!(Confirm-Step 'Set up or repair the project environment?')) { throw 'Setup cancelled.' }
            $script:UvBin = Find-Uv
            if (!$script:UvBin) {
                Write-Host "Install uv from https://astral.sh/uv/install.ps1 to $env:USERPROFILE\.local\bin; shell profiles will be left unchanged."
                if (!(Confirm-Step 'Install uv?')) { throw 'Setup cancelled.' }
                $installer = Join-Path ([IO.Path]::GetTempPath()) "resassetpricing-uv-$([Guid]::NewGuid()).ps1"
                $oldInstallDir = $env:UV_INSTALL_DIR
                $oldNoModify = $env:UV_NO_MODIFY_PATH
                try {
                    $env:UV_INSTALL_DIR = "$env:USERPROFILE\.local\bin"
                    $env:UV_NO_MODIFY_PATH = '1'
                    Invoke-Retry 'uv installation' {
                        try {
                            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                            Invoke-WebRequest https://astral.sh/uv/install.ps1 -UseBasicParsing -OutFile $installer
                            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
                        } catch { Write-Host $_; $global:LASTEXITCODE = 1 }
                    }
                } finally {
                    Remove-Item $installer -ErrorAction SilentlyContinue
                    $env:UV_INSTALL_DIR = $oldInstallDir
                    $env:UV_NO_MODIFY_PATH = $oldNoModify
                }
                $script:UvBin = Find-Uv
                if (!$script:UvBin) { throw 'uv was installed but could not be run.' }
            }
            $script:PythonBin = $ProjectPython
            $venv = Join-Path $ProjectRoot '.venv'
            $venvItem = Get-Item $venv -ErrorAction SilentlyContinue
            $isLink = $venvItem -and ($venvItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
            if ((Test-Environment -PythonOnly) -ne 0 -or $isLink) {
                if ($venvItem) {
                    $backup = "$venv.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss').$PID"
                    Write-Host "The existing environment will be preserved at $backup"
                    if (!(Confirm-Step 'Back up the existing environment and create a new one?')) { throw 'Setup cancelled.' }
                    Move-Item $venv $backup
                }
                Invoke-Retry 'Python environment creation' { & $script:UvBin venv --python 3.13.5 --managed-python $venv }
            }
            $status = Test-Environment
            if ($script:ForceCpu) { $status = 20 }
            $reinstall = $status -ne 20
            while ($status -ne 0) {
                Invoke-Retry 'Dependency installation' { Install-Dependencies -Reinstall:$reinstall }
                $status = Test-Environment
                if ($status -ne 0) {
                    if (!(Confirm-Step 'Verification failed. Reinstall dependencies and retry?')) { throw 'Setup stopped. See the diagnostic above.' }
                    $reinstall = $true
                }
            }
            Set-Content -Path $backendFile -Value $script:BootstrapProfile -Encoding Ascii
        }
    }
    if (!$script:PythonBin) { throw 'No Python found. Run .\run.ps1 -Setup.' }
    & $script:PythonBin -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ required. Run .\run.ps1 -Setup.' }
    if ($Setup) { Write-Host 'Environment ready. Run .\run.ps1 to open the launcher.'; exit 0 }

    $launchArgs = @((Join-Path $ProjectRoot 'utils\launch.py'))
    if ($Plan) { $launchArgs += @('--plan', (Resolve-Path $Plan).Path) }
    if ($Experiment) { $launchArgs += @('--experiment', $Experiment, '--layers', $Layers, '--ensemble-range', $EnsembleRange, '--num-gpus', "$NumGpus") }
    if ($Devices) { $launchArgs += @('--devices', $Devices) }
    if ($DryRun) { $launchArgs += '--dry-run' }
    if ($NoEcon) { $launchArgs += '--no-econ' }
    if ($GroupAblationOnly) { $launchArgs += '--group-ablation-only' }
    if ($ListExperiments) { $launchArgs += '--list-experiments' }
    & $script:PythonBin @launchArgs
    exit $LASTEXITCODE
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
