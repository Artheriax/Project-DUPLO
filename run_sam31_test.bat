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

echo [3/4] SAM 3.1-omgeving controleren...
"%PYTHON%" -c "import ultralytics; from ultralytics.models.sam import SAM3SemanticPredictor" 2>nul
if errorlevel 1 (
    echo ERROR: ultralytics met SAM3-ondersteuning ontbreekt terwijl requirements.txt die installeert.
    echo Controleer of 'pip install -U -r requirements.txt' goed is gegaan.
    exit /b 1
)

if exist "models\sam3.1_multiplex.pt" (
    echo Lokaal SAM 3.1-checkpoint gevonden: models\sam3.1_multiplex.pt ^(geen Hugging Face-login nodig^).
) else (
    echo Let op: models\sam3.1_multiplex.pt ontbreekt; de SAM 3.1-gewichten van facebook/sam3.1 ^(sam3.1_multiplex.pt^) vereisen een goedgekeurde
    echo Hugging Face-toegangsaanvraag en een login via 'huggingface-cli login' ^(of 'hf auth login'^).
    echo Zonder login kan Ultralytics de gated gewichten niet downloaden; plaats het bestand handmatig in models\ of zet 'sam31_model' in de config.
)

echo [4/4] SAM 3.1-test uitvoeren...
"%PYTHON%" -u src\run_sam31_video.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Resultaten staan in outputs\Disassembly1\sam31
) else (
    echo.
    echo ERROR: De SAM 3.1-test is gestopt met code %EXIT_CODE%.
)

popd
exit /b %EXIT_CODE%
