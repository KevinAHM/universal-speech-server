@echo off
setlocal
set "SPEECH_TARGET=%~1"
if "%SPEECH_TARGET%"=="" set "SPEECH_TARGET=auto"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1" -Target "%SPEECH_TARGET%" -RunServer
exit /b %ERRORLEVEL%
