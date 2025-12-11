@echo off
echo ============================================
echo   HANAFISHIELD - Installation Script
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python n'est pas installe!
    echo Telechargez Python depuis: https://python.org
    pause
    exit /b 1
)

echo [OK] Python trouve
echo.

REM Create virtual environment
echo Creation de l'environnement virtuel...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Impossible de creer l'environnement virtuel
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Mise a jour de pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installation des dependances...
echo Cela peut prendre plusieurs minutes...
echo.

pip install opencv-python numpy Pillow
pip install customtkinter
pip install sounddevice scipy
pip install mediapipe

echo.
echo [OPTIONNEL] Voulez-vous installer YOLOv8 pour une detection plus precise?
echo (Necessite ~2GB d'espace disque)
set /p install_yolo="Installer YOLOv8? (o/n): "
if /i "%install_yolo%"=="o" (
    echo Installation de PyTorch et YOLOv8...
    pip install torch torchvision
    pip install ultralytics
)

echo.
echo ============================================
echo   Installation terminee!
echo ============================================
echo.
echo Pour lancer HANAFISHIELD:
echo   1. Double-cliquez sur 'run.bat'
echo   OU
echo   2. Executez: python main.py
echo.
pause
