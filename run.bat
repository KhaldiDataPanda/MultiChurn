@echo off
echo Multi-Churn Prediction Platform - Quick Start
echo ============================================
echo.

REM Create and activate virtual environment if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing/updating dependencies...
pip install -r requirements.txt

REM Check if models exist, if not train them
echo Checking for existing models...
python train_models.py
echo Models are ready.

REM Start the services
echo.
echo Starting FastAPI server...
start "FastAPI Server" cmd /k "python api.py"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

echo Starting Streamlit interface...
streamlit run streamlit_app.py

pause