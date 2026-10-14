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

echo [3/4] SAM3-omgeving controleren...
"%PYTHON%" -c "import sam3, triton" 2>nul
if errorlevel 1 (
    echo ERROR: sam3 of triton ontbreekt terwijl requirements.txt die installeert.
    echo Controleer of 'pip install -U -r requirements.txt' goed is gegaan.
    echo Op Windows is triton-windows nodig; het officiele triton heeft geen Windows-wheels.
    exit /b 1
)

echo Let op: de SAM3-gewichten van facebook/sam3 vereisen een goedgekeurde
echo Hugging Face-toegangsaanvraag en een login via 'hf auth login'.

echo [4/4] SAM3-test uitvoeren...
"%PYTHON%" -u src\run_sam3_video.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Resultaten staan in outputs\Disassembly1\sam3
) else (
    echo.
    echo ERROR: De SAM3-test is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%