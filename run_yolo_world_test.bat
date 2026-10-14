@echo off
setlocal

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Kan de projectmap niet openen.
    exit /b 1
)

set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher ^(py^) is niet gevonden.
    echo Installeer Python 3.10 of nieuwer en probeer opnieuw.
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [1/4] Virtuele omgeving aanmaken...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Aanmaken van de virtuele omgeving is mislukt.
        exit /b 1
    )
) else (
    echo [1/4] Bestaande virtuele omgeving gebruiken.
)

echo [2/4] Python packages controleren en installeren...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip kon niet worden bijgewerkt.
    exit /b 1
)
"%PYTHON%" -m pip install -U -r requirements.txt
if errorlevel 1 (
    echo ERROR: Een of meer packages konden niet worden geinstalleerd.
    exit /b 1
)

echo [3/4] Imports verifieren...
"%PYTHON%" -c "import cv2, pandas, yaml, tqdm, ultralytics; print('Alle vereiste imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

echo [4/4] YOLO-World-test uitvoeren...
"%PYTHON%" -u src\run_yolo_world_video.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Resultaten staan in outputs\Disassembly1\yolo_world
) else (
    echo.
    echo ERROR: De YOLO-World-test is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%