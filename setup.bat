@echo off
echo 📚 Book Genre Classification Project Setup
echo ==========================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

echo.
echo ✅ Setup completed successfully!
echo.
echo Next steps:
echo 1. Run the Jupyter notebook: jupyter notebook book_genre_classification.ipynb
echo 2. Or run the Streamlit app: streamlit run app.py
echo 3. Make sure to train the models first by running the notebook!
echo.
pause
