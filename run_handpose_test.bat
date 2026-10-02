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
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Een of meer packages konden niet worden geinstalleerd.
    exit /b 1
)

echo [2/3] Imports verifieren...
"%PYTHON%" -c "import cv2, mediapipe, pandas, yaml; print('Alle MediaPipe imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

if not exist "data\Disassembly1.mp4" (
    echo ERROR: Video ontbreekt: data\Disassembly1.mp4
    exit /b 1
)

echo [3/3] MediaPipe handpose-test uitvoeren...
"%PYTHON%" src\run_handpose.py --config configs\baseline.yaml
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Handlandmarks staan in outputs\Disassembly1
) else (
    echo.
    echo ERROR: De MediaPipe-test is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%