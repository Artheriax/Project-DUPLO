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
    echo ERROR: De virtuele omgeving ontbreekt; draai eerst een run_*_test.bat script.
    exit /b 1
)

echo [1/2] Imports verifieren...
"%PYTHON%" -c "import cv2, pandas, yaml; print('Alle vereiste imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

echo [2/2] Modelvergelijking uitvoeren...
"%PYTHON%" -u src\compare_models.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Het rapport staat in outputs\Disassembly1\comparison
) else (
    echo.
    echo ERROR: De modelvergelijking is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%