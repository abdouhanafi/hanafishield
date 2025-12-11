#!/bin/bash

echo "============================================"
echo "  HANAFISHIELD - Installation Script"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 n'est pas installé!"
    echo "Installez Python avec: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "[OK] Python3 trouvé: $(python3 --version)"
echo ""

# Install system dependencies
echo "Installation des dépendances système..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-tk portaudio19-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install portaudio
fi

# Create virtual environment
echo ""
echo "Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installation des dépendances Python..."
echo "Cela peut prendre plusieurs minutes..."
echo ""

pip install opencv-python numpy Pillow
pip install customtkinter
pip install sounddevice scipy librosa
pip install mediapipe

# Optional: Install YOLOv8
echo ""
read -p "Installer YOLOv8 pour une détection plus précise? (o/n): " install_yolo
if [[ "$install_yolo" == "o" || "$install_yolo" == "O" ]]; then
    echo "Installation de PyTorch et YOLOv8..."
    pip install torch torchvision
    pip install ultralytics
fi

echo ""
echo "============================================"
echo "  Installation terminée!"
echo "============================================"
echo ""
echo "Pour lancer HANAFISHIELD:"
echo "  1. ./run.sh"
echo "  OU"
echo "  2. source venv/bin/activate && python main.py"
echo ""
