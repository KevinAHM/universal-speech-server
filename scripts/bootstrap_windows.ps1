param(
    [string]$Target = "auto",
    [switch]$Update,
    [switch]$RunServer
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$runtimeRoot = Join-Path $root "runtime"
$pythonDir = Join-Path $runtimeRoot "python"
$pythonExe = Join-Path $pythonDir "python.exe"
$payloadDir = Join-Path $root "vendor\python"
$bundleManifestPath = Join-Path $payloadDir "bundle-manifest.json"
$securityManifestPath = Join-Path $root "vendor\security-tools.json"
$sfwDir = Join-Path $root "bin"
$sfwExe = Join-Path $sfwDir "sfw.exe"
Set-Location -LiteralPath $root
Write-Host "Preparing Universal Speech Server..."

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$Quiet
    )
    # Windows PowerShell turns a native program's stderr into ErrorRecord
    # objects. With the script-wide Stop preference, an expected nonzero probe
    # (notably `python -m pip` before pip exists) would terminate before its exit
    # code could be handled. Keep Stop for PowerShell operations, but handle
    # native failures explicitly by exit code.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($Quiet) {
            & $FilePath @Arguments *> $null
        }
        else {
            # Keep stdout/stderr attached directly to the console. Piping here
            # makes Python block-buffer its output, leaving long downloads and
            # server startup looking completely silent.
            & $FilePath @Arguments
        }
        $script:NativeExitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Set-SelectedGpuEnvironment([string]$RequestedTarget, [string]$ManagedPython) {
    $manifestPath = if ($env:SPEECH_SERVER_RUNTIME_MANIFEST) {
        [System.IO.Path]::GetFullPath($env:SPEECH_SERVER_RUNTIME_MANIFEST)
    }
    else {
        Join-Path $runtimeRoot "crispasr\installed.json"
    }
    $previousPreference = $ErrorActionPreference
    try {
        # Keep the selector's prompts on the console while capturing its sole
        # stdout line (VARIABLE=value) for this parent process.
        $ErrorActionPreference = "Continue"
        $selection = @(
            & $ManagedPython -m speech_server.gpu_select `
                --target $RequestedTarget --manifest $manifestPath
        )
        $selectorExitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($selectorExitCode -ne 0) {
        throw "GPU selection failed."
    }
    $assignment = [string]($selection | Where-Object { $_ -ne "" } | Select-Object -Last 1)
    if (-not $assignment) {
        return
    }
    if ($assignment -notmatch '^(CUDA_VISIBLE_DEVICES|GGML_VK_VISIBLE_DEVICES)=(.+)$') {
        throw "GPU selector returned an invalid assignment."
    }
    [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
}

function Assert-FileHash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required bundled file is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path (expected $Expected, got $actual)"
    }
}

function Test-FileHash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $actual -eq $Expected.ToLowerInvariant()
}

function Resolve-PayloadFile([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name) -or [System.IO.Path]::GetFileName($Name) -ne $Name) {
        throw "Unsafe bundled payload filename: $Name"
    }
    return Join-Path $payloadDir $Name
}

function Ensure-SocketFirewall {
    if (-not (Test-Path -LiteralPath $securityManifestPath -PathType Leaf)) {
        throw "Socket Firewall provenance manifest is missing: $securityManifestPath"
    }
    $security = Get-Content -LiteralPath $securityManifestPath -Raw | ConvertFrom-Json
    if ($security.schemaVersion -ne 1) {
        throw "Unsupported Socket Firewall provenance manifest."
    }
    $sfw = $security.socketFirewall
    $platform = $sfw.platforms.'windows-x86_64'
    if (-not (Test-FileHash $sfwExe $platform.sha256)) {
        New-Item -ItemType Directory -Path $sfwDir -Force | Out-Null
        $url = "https://github.com/$($sfw.repository)/releases/download/$($sfw.tag)/$($platform.asset)"
        $temporary = Join-Path $sfwDir (".sfw-download-" + [guid]::NewGuid().ToString("N") + ".exe")
        try {
            Write-Host "Bundled Socket Firewall is missing or invalid; downloading pinned $($sfw.tag)..."
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $temporary
            Assert-FileHash $temporary $platform.sha256
            Move-Item -LiteralPath $temporary -Destination $sfwExe -Force
        }
        finally {
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Force
            }
        }
    }
    Assert-FileHash $sfwExe $platform.sha256
}

function Install-EmbeddedPython {
    if (-not (Test-Path -LiteralPath $bundleManifestPath -PathType Leaf)) {
        throw "The Windows release payload is incomplete: vendor\python\bundle-manifest.json is missing."
    }
    $bundle = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    if ($bundle.schemaVersion -ne 1 -or $bundle.platform -ne "windows-x86_64") {
        throw "Unsupported bundled Python manifest: $bundleManifestPath"
    }
    $pythonArchive = Resolve-PayloadFile $bundle.python.archive
    Assert-FileHash $pythonArchive $bundle.python.sha256
    $getPip = Resolve-PayloadFile $bundle.pipBootstrap.archive
    Assert-FileHash $getPip $bundle.pipBootstrap.sha256

    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    $staging = Join-Path $runtimeRoot (".python-install-" + [guid]::NewGuid().ToString("N"))
    try {
        New-Item -ItemType Directory -Path $staging -Force | Out-Null
        Expand-Archive -LiteralPath $pythonArchive -DestinationPath $staging
        $sitePackages = Join-Path $staging "Lib\site-packages"
        New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null

        $pth = Get-ChildItem -LiteralPath $staging -Filter "python*._pth" | Select-Object -First 1
        if ($null -eq $pth) {
            throw "Embedded Python archive contains no python*._pth file."
        }
        $stdlibZip = Get-ChildItem -LiteralPath $staging -Filter "python*.zip" |
            Where-Object { $_.Name -match '^python[0-9]+\.zip$' } |
            Select-Object -First 1
        if ($null -eq $stdlibZip) {
            throw "Embedded Python archive contains no standard-library ZIP."
        }
        @(
            $stdlibZip.Name
            "."
            "Lib\site-packages"
            "..\.."
            "import site"
        ) | Set-Content -LiteralPath $pth.FullName -Encoding Ascii
        Copy-Item -LiteralPath $bundleManifestPath -Destination (Join-Path $staging "bundle-manifest.json")

        $resolvedRuntime = [System.IO.Path]::GetFullPath($runtimeRoot)
        $resolvedPython = [System.IO.Path]::GetFullPath($pythonDir)
        if (-not $resolvedPython.StartsWith($resolvedRuntime + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to replace Python outside the runtime directory: $resolvedPython"
        }
        $backup = Join-Path $runtimeRoot ".python-previous"
        if (Test-Path -LiteralPath $backup) {
            if (Test-Path -LiteralPath $pythonDir) {
                Remove-Item -LiteralPath $backup -Recurse -Force
            }
            else {
                Move-Item -LiteralPath $backup -Destination $pythonDir
            }
        }
        if (Test-Path -LiteralPath $pythonDir) {
            Move-Item -LiteralPath $pythonDir -Destination $backup
        }
        try {
            Move-Item -LiteralPath $staging -Destination $pythonDir
        }
        catch {
            if ((Test-Path -LiteralPath $backup) -and -not (Test-Path -LiteralPath $pythonDir)) {
                Move-Item -LiteralPath $backup -Destination $pythonDir
            }
            throw
        }
        if (Test-Path -LiteralPath $backup) {
            Remove-Item -LiteralPath $backup -Recurse -Force
        }
    }
    finally {
        if (Test-Path -LiteralPath $staging) {
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

function Test-EmbeddedPythonCurrent {
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        return $false
    }
    $installedManifest = Join-Path $pythonDir "bundle-manifest.json"
    if (-not (Test-Path -LiteralPath $installedManifest -PathType Leaf)) {
        return $false
    }
    $desiredHash = (Get-FileHash -LiteralPath $bundleManifestPath -Algorithm SHA256).Hash
    $installedHash = (Get-FileHash -LiteralPath $installedManifest -Algorithm SHA256).Hash
    if ($desiredHash -ne $installedHash) {
        return $false
    }
    $bundle = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    Invoke-NativeCommand -FilePath $pythonExe -Arguments @(
        "-c",
        "import platform, sys; sys.exit(0 if platform.python_version() == sys.argv[1] else 1)",
        [string]$bundle.python.version
    ) -Quiet
    $exitCode = $script:NativeExitCode
    return $exitCode -eq 0
}

function Ensure-PythonDependencies([string]$ManagedPython) {
    $bundle = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    $getPip = Resolve-PayloadFile $bundle.pipBootstrap.archive
    Assert-FileHash $getPip $bundle.pipBootstrap.sha256

    Invoke-NativeCommand -FilePath $ManagedPython -Arguments @(
        "-m", "pip", "--version"
    ) -Quiet
    $exitCode = $script:NativeExitCode
    if ($exitCode -ne 0) {
        Write-Host "Bootstrapping pip from the bundled, verified get-pip.py..."
        Invoke-NativeCommand -FilePath $sfwExe -Arguments @(
            $ManagedPython,
            $getPip,
            "--no-warn-script-location",
            "--disable-pip-version-check"
        )
        $exitCode = $script:NativeExitCode
        if ($exitCode -ne 0) {
            throw "Failed to bootstrap pip in the embedded Python runtime."
        }
    }

    $requirements = Join-Path $root "requirements-runtime.txt"
    $requirementsHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash.ToLowerInvariant()
    $requirementsMarker = Join-Path $pythonDir ".requirements.sha256"
    $installedHash = ""
    if (Test-Path -LiteralPath $requirementsMarker -PathType Leaf) {
        $installedHash = (Get-Content -LiteralPath $requirementsMarker -Raw).Trim()
    }
    Invoke-NativeCommand -FilePath $ManagedPython -Arguments @(
        "-c",
        "import colorama, dotenv, fastapi, httptools, numpy, pydantic, uvicorn, watchfiles, websockets, yaml"
    ) -Quiet
    $exitCode = $script:NativeExitCode
    $importsWork = $exitCode -eq 0
    Invoke-NativeCommand -FilePath $ManagedPython -Arguments @(
        "-m", "pip", "check"
    ) -Quiet
    $exitCode = $script:NativeExitCode
    $pipCheckWorks = $exitCode -eq 0
    if (-not $importsWork -or -not $pipCheckWorks -or $installedHash -ne $requirementsHash) {
        Write-Host "Installing Python dependencies through Socket Firewall..."
        Invoke-NativeCommand -FilePath $sfwExe -Arguments @(
            $ManagedPython,
            "-m", "pip", "install",
            "--requirement", $requirements,
            "--no-warn-script-location",
            "--disable-pip-version-check"
        )
        $exitCode = $script:NativeExitCode
        if ($exitCode -ne 0) {
            throw "Failed to install speech-server dependencies through Socket Firewall."
        }
        Invoke-NativeCommand -FilePath $ManagedPython -Arguments @(
            "-c",
            "import colorama, dotenv, fastapi, httptools, numpy, pydantic, uvicorn, watchfiles, websockets, yaml"
        )
        $exitCode = $script:NativeExitCode
        if ($exitCode -ne 0) {
            throw "The installed dependency set is incomplete."
        }
        Invoke-NativeCommand -FilePath $ManagedPython -Arguments @(
            "-m", "pip", "check"
        )
        $exitCode = $script:NativeExitCode
        if ($exitCode -ne 0) {
            throw "The installed dependency set has incompatible requirements."
        }
        Set-Content -LiteralPath $requirementsMarker -Value $requirementsHash -Encoding Ascii
    }
}

$override = $env:SPEECH_SERVER_PYTHON
if ($env:SPEECH_SERVER_UPDATE -eq "1") {
    $Update = $true
}
Ensure-SocketFirewall
if ($override) {
    $pythonExe = [System.IO.Path]::GetFullPath($override)
}
elseif (Test-Path -LiteralPath $bundleManifestPath -PathType Leaf) {
    if (-not (Test-EmbeddedPythonCurrent)) {
        Install-EmbeddedPython
    }
}
else {
    $developerPython = Join-Path $root ".venv-torch\Scripts\python.exe"
    if (Test-Path -LiteralPath $developerPython -PathType Leaf) {
        $pythonExe = $developerPython
        Write-Host "Bundled Python payload absent; using the local developer environment."
    }
    else {
        throw "No bundled Windows Python payload is available. Build the release payload first."
    }
}

if ([System.IO.Path]::GetFullPath($pythonExe) -eq [System.IO.Path]::GetFullPath((Join-Path $pythonDir "python.exe"))) {
    Ensure-PythonDependencies $pythonExe
}

$env:PYTHONUTF8 = "1"
$bootstrapArgs = @("-m", "speech_server.bootstrap", "setup-native", "--target", $Target)
if ($Update) {
    $bootstrapArgs += "--update"
}
Invoke-NativeCommand -FilePath $pythonExe -Arguments $bootstrapArgs
$exitCode = $script:NativeExitCode
if ($exitCode -ne 0) {
    exit $exitCode
}
if ($RunServer) {
    Set-SelectedGpuEnvironment $Target $pythonExe
    Write-Host "Starting Universal Speech Server..."
    Invoke-NativeCommand -FilePath $pythonExe -Arguments @(
        "-m", "speech_server"
    )
    $exitCode = $script:NativeExitCode
    exit $exitCode
}
