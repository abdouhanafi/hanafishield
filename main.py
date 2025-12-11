#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                        HANAFISHIELD                                ║
║         Real-Time Violence Detection System                        ║
║                                                                    ║
║  Multimodal AI-powered violence detection using:                   ║
║  - Computer Vision (YOLOv8 Pose / MediaPipe)                       ║
║  - Audio Analysis (Spectral Features / ML Classification)          ║
║                                                                    ║
║  Detects: Punches, Kicks, Defensive postures, Screams,            ║
║           Impacts, Glass breaking, Sustained shouting              ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import sys
import os

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

def check_dependencies():
    """Check and report on available dependencies"""
    print("\n🔍 Vérification des dépendances...\n")
    
    dependencies = {
        'cv2': ('opencv-python', 'Core video processing'),
        'numpy': ('numpy', 'Numerical operations'),
        'PIL': ('Pillow', 'Image processing'),
        'tkinter': ('tkinter', 'GUI framework'),
    }
    
    optional_deps = {
        'customtkinter': ('customtkinter', 'Modern GUI'),
        'ultralytics': ('ultralytics', 'YOLOv8 pose detection'),
        'mediapipe': ('mediapipe', 'Pose estimation'),
        'sounddevice': ('sounddevice', 'Audio capture'),
        'librosa': ('librosa', 'Advanced audio analysis'),
        'scipy': ('scipy', 'Signal processing'),
        'torch': ('torch', 'Deep learning'),
    }
    
    all_ok = True
    
    print("📦 Dépendances requises:")
    for module, (package, desc) in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {package}: {desc}")
        except ImportError:
            print(f"  ❌ {package}: {desc} - MANQUANT")
            all_ok = False
    
    print("\n📦 Dépendances optionnelles (pour fonctionnalités avancées):")
    for module, (package, desc) in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✅ {package}: {desc}")
        except ImportError:
            print(f"  ⚠️  {package}: {desc} - Non installé")
    
    print()
    return all_ok


def install_dependencies():
    """Show installation instructions"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   INSTALLATION DES DÉPENDANCES                     ║
╚═══════════════════════════════════════════════════════════════════╝

Pour installer toutes les dépendances, exécutez:

    pip install -r requirements.txt

Ou installez individuellement:

    # Core (requis)
    pip install opencv-python numpy Pillow

    # GUI moderne
    pip install customtkinter

    # Détection de poses (choisir un ou les deux)
    pip install ultralytics      # YOLOv8 - plus précis
    pip install mediapipe        # MediaPipe - plus rapide

    # Audio
    pip install sounddevice librosa scipy

    # Deep Learning (pour YOLOv8)
    pip install torch torchvision

Note: Sur Linux, vous pourriez avoir besoin de:
    sudo apt-get install python3-tk portaudio19-dev

Sur macOS:
    brew install portaudio
""")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     🛡️  HANAFISHIELD - Violence Detection System  🛡️         ║
    ║                                                                ║
    ║         Système de Détection de Violence en Temps Réel        ║
    ║                                                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Certaines dépendances requises sont manquantes.")
        install_dependencies()
        sys.exit(1)
    
    print("🚀 Démarrage de HANAFISHIELD...\n")
    
    try:
        # Import and run GUI
        from gui import HanafiShieldApp
        
        app = HanafiShieldApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        install_dependencies()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
