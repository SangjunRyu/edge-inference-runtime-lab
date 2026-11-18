# Pinned local SDKs; build/deps is ignored by Git. No Python package changes.
[CmdletBinding()]
param(
    [string]$PythonEnvironment = '.venv-cuda',
    [string]$Configuration = 'Release'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$depsRoot = Join-Path $projectRoot 'cpp\build\deps'
$buildRoot = Join-Path $projectRoot 'cpp\build\ort-msvc'
$runtimeRoot = Join-Path $projectRoot "$PythonEnvironment\Lib\site-packages\onnxruntime\capi"
$nvidiaRoot = Join-Path $projectRoot "$PythonEnvironment\Lib\site-packages\nvidia"
if (!(Test-Path -LiteralPath (Join-Path $runtimeRoot 'onnxruntime.dll'))) {
    throw "ONNX Runtime DLL missing in $runtimeRoot"
}
New-Item -ItemType Directory -Force -Path $depsRoot | Out-Null
function Get-SdkArchive([string]$name, [string]$url) {
    $archivePath = Join-Path $depsRoot $name
    if (!(Test-Path -LiteralPath $archivePath)) {
        $temporaryPath = "$archivePath.download"
        & curl.exe --fail --location --retry 3 --output $temporaryPath $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
        Move-Item -LiteralPath $temporaryPath -Destination $archivePath
    }
    return $archivePath
}
$ortSdkRoot = Join-Path $depsRoot 'onnxruntime-win-x64-1.26.0'
if (!(Test-Path -LiteralPath (Join-Path $ortSdkRoot 'include\onnxruntime_cxx_api.h'))) {
    $archive = Get-SdkArchive 'onnxruntime-win-x64-1.26.0.zip' 'https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-win-x64-1.26.0.zip'
    Expand-Archive -LiteralPath $archive -DestinationPath $depsRoot -Force
}
$opencvDir = Join-Path $depsRoot 'opencv\build\x64\vc16\lib'
if (!(Test-Path -LiteralPath (Join-Path $opencvDir 'OpenCVConfig.cmake'))) {
    $archive = Get-SdkArchive 'opencv-4.12.0-windows.exe' 'https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-windows.exe'
    $process = Start-Process -FilePath $archive -ArgumentList @('-y', ('-o"{0}"' -f $depsRoot)) -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw 'OpenCV extraction failed' }
}
& cmake -S (Join-Path $projectRoot 'cpp') -B $buildRoot -G 'Visual Studio 17 2022' -A x64 `
    -DEIRL_BUILD_OPENVINO=OFF -DEIRL_BUILD_ORT=ON "-DORT_ROOT=$ortSdkRoot" `
    "-DORT_RUNTIME_DIR=$runtimeRoot" "-DNVIDIA_ROOT=$nvidiaRoot" "-DOpenCV_DIR=$opencvDir"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& cmake --build $buildRoot --config $Configuration --target run_sync_ort run_async_ort -j 4
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
Write-Output "Executables: $buildRoot\$Configuration"
