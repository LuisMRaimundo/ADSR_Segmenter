@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ADSR_Segmenter

if not exist "split_audio_segments.py" (
    echo Keep run.bat in the ADSR_Segmenter folder ^(next to split_audio_segments.py^).
    pause
    exit /b 1
)

echo Starting ADSR_Segmenter...

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 "%~dp0split_audio_segments.py"
    goto :done
)

python "%~dp0split_audio_segments.py"

:done
set "RC=%ERRORLEVEL%"
if %RC% neq 0 (
    echo.
    echo Could not start. If libraries are missing, run once:
    echo   pip install -r requirements.txt
    echo.
    pause
)
exit /b %RC%
