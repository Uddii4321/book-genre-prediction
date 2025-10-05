@echo off
echo 🚀 Starting Book Genre Classification App...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment and run the app
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run app.py --server.port 8501
