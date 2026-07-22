@echo off
setlocal
set "SPEECH_TARGET=auto"
set "SPEECH_TARGET_SET="
set "SPEECH_GPU="

:parse_args
if "%~1"=="" goto launch
if /I "%~1"=="--gpu" goto parse_gpu
if /I "%~1"=="--help" goto usage
if defined SPEECH_TARGET_SET (
    echo Unexpected argument: %~1 1>&2
    goto usage_error
)
set "SPEECH_TARGET=%~1"
set "SPEECH_TARGET_SET=1"
shift
goto parse_args

:parse_gpu
if "%~2"=="" (
    echo --gpu requires a device index. 1>&2
    goto usage_error
)
set "SPEECH_GPU=%~2"
shift
shift
goto parse_args

:launch
if defined SPEECH_GPU (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1" -Target "%SPEECH_TARGET%" -GpuDevice "%SPEECH_GPU%" -RunServer
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1" -Target "%SPEECH_TARGET%" -RunServer
)
exit /b %ERRORLEVEL%

:usage
echo Usage: start_server.bat [target] [--gpu INDEX]
echo Example: start_server.bat nvidia --gpu 0
exit /b 0

:usage_error
echo Usage: start_server.bat [target] [--gpu INDEX] 1>&2
exit /b 2
