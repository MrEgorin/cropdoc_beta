@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -3 -m venv .venv || python -m venv .venv || goto err
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt || goto err
pip install pillow-heif
python app.py
pause
exit /b
:err
echo.
echo Something failed above. Install Python 3.10-3.12 from python.org (tick "Add to PATH") and run again.
pause
