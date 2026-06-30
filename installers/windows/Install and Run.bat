@echo off
setlocal EnableExtensions
title ADSR_Segmenter
cd /d "%~dp0"
call "%~dp0INSTALL.bat"
exit /b %ERRORLEVEL%
