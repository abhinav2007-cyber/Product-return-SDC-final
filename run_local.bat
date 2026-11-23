@echo off
echo Starting Product Return & Cancellation Analysis Server...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python -m uvicorn main:app --reload --port 8000
