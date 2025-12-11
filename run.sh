#!/bin/bash

echo "============================================"
echo "  HANAFISHIELD - Violence Detection System"
echo "============================================"
echo ""

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run the application
python3 main.py
