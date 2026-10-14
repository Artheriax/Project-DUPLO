@echo off
setlocal

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Kan de projectmap niet openen.
    exit /b 1
)

set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: De virtuele omgeving ontbreekt.
    echo Start eerst run_yolo_test.bat om de omgeving aan te maken.
    exit /b 1
)

echo [1/3] Python packages controleren...
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -U -r requirements.txt
if errorlevel 1 (
    echo ERROR: Een of meer packages konden niet worden geinstalleerd.
    exit /b 1
)

echo [2/3] Imports verifieren...
"%PYTHON%" -c "import cv2, pandas, yaml, tqdm, ultralytics; print('Alle imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

echo [3/3] ByteTrack-tracking uitvoeren...
"%PYTHON%" -u src\run_tracking.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. De trackgeschiedenis staat in outputs\Disassembly1
) else (
    echo.
    echo ERROR: De ByteTrack-test is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%
