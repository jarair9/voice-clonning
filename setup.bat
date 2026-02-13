@echo off
REM Voice Cloning Studio - Setup Script for Windows
REM This script sets up the environment and installs dependencies

echo 🎙️ Voice Cloning Studio - Setup Script
echo ========================================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% found

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "voice_cloning_env" (
    python -m venv voice_cloning_env
    echo ✅ Virtual environment created
) else (
    echo ℹ️ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call voice_cloning_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if installation was successful
echo 🔍 Verifying installation...
python -c "import streamlit; print('✅ Streamlit installed')" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit installation failed
    pause
    exit /b 1
)

python -c "import torch; print('✅ PyTorch installed')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch installation failed
    pause
    exit /b 1
)

python -c "import torchaudio; print('✅ TorchAudio installed')" 2>nul
if errorlevel 1 (
    echo ❌ TorchAudio installation failed
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "temp_outputs" mkdir temp_outputs
if not exist "cloned_voices" mkdir cloned_voices
if not exist "uploads" mkdir uploads

echo ✅ Setup complete!
echo.
echo 🚀 To run the application:
echo    voice_cloning_env\Scripts\activate.bat
echo    streamlit run app.py
echo.
echo 🌐 The app will be available at: http://localhost:8501
echo.
pause