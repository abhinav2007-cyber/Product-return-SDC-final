#!/usr/bin/env bash
echo "Starting Product Return & Cancellation Analysis Server..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
python -m uvicorn main:app --reload --port 8000
