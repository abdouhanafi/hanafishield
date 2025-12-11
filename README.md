# 🛡️ HANAFISHIELD - Violence Detection System

Système de détection de violence en temps réel utilisant l'IA multimodale (vision + audio).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-purple.svg)

## 🎯 Fonctionnalités

### Détection Visuelle (Caméra)
- **Coups de poing** : Détecte les mouvements de frappe avec le bras
- **Coups de pied** : Identifie les mouvements de jambe violents
- **Postures défensives** : Reconnaît quand quelqu'un se protège (victime potentielle)
- **Mouvements brusques** : Analyse les changements rapides de position
- **Patterns de lutte** : Détecte les séquences de mouvements caractéristiques d'une altercation

### Détection Audio (Microphone)
- **Cris** : Détecte les cris et hurlements (hautes fréquences)
- **Impacts** : Identifie les bruits de coups/chocs (basses fréquences soudaines)
- **Bris de verre** : Reconnaît le son caractéristique du verre qui se brise
- **Altercations** : Détecte les cris soutenus/disputes

### Interface Graphique
- Dashboard moderne et intuitif
- Visualisation en temps réel du flux vidéo
- Spectre audio avec indicateurs de fréquences
- Jauge de niveau de menace
- Historique des alertes
- Statistiques (temps actif, nombre de détections)

## 📋 Prérequis

- Python 3.8 ou supérieur
- Webcam
- Microphone
- GPU NVIDIA (optionnel, pour de meilleures performances)

## 🚀 Installation

### 1. Cloner ou télécharger le projet

```bash
cd hanafishield
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Installer les dépendances

**Installation complète (recommandée) :**
```bash
pip install -r requirements.txt
```

**Installation minimale (sans ML avancé) :**
```bash
pip install opencv-python numpy Pillow sounddevice
```

**Installation avec GPU NVIDIA :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics mediapipe
```

### 4. Dépendances système

**Linux (Ubuntu/Debian) :**
```bash
sudo apt-get update
sudo apt-get install python3-tk portaudio19-dev
```

**macOS :**
```bash
brew install portaudio
```

**Windows :**
Les dépendances sont généralement incluses avec Python.

## 🎮 Utilisation

### Lancer l'application

```bash
python main.py
```

### Interface

1. **Activer la surveillance** : Cliquez sur le bouton vert "Activer la surveillance"
2. **Autoriser l'accès** : Acceptez les permissions pour la caméra et le microphone
3. **Surveiller** : Le système analyse en temps réel
4. **Alertes** : Les détections apparaissent dans le panneau de droite
5. **Désactiver** : Cliquez sur le bouton rouge pour arrêter

## 🔧 Configuration

### Ajuster les seuils de détection

Dans `src/video_detector.py` :
```python
VIOLENCE_KEYPOINTS = {
    'punch': {
        'velocity_threshold': 150,  # Sensibilité des coups de poing
        'extension_threshold': 0.7
    },
    'kick': {
        'velocity_threshold': 200,  # Sensibilité des coups de pied
        'extension_threshold': 0.8
    }
}
```

Dans `src/audio_detector.py` :
```python
THRESHOLDS = {
    'scream': {
        'rms_min': 0.15,  # Volume minimum pour un cri
        'high_freq_ratio_min': 0.3
    },
    'impact': {
        'peak_min': 0.6,  # Seuil pour les impacts
    }
}
```

## 📊 Architecture

```
hanafishield/
├── main.py                 # Point d'entrée
├── requirements.txt        # Dépendances
├── README.md              # Documentation
├── src/
│   ├── __init__.py
│   ├── video_detector.py   # Détection visuelle (YOLO/MediaPipe)
│   ├── audio_detector.py   # Détection audio
│   └── gui.py              # Interface graphique
├── models/                 # Modèles ML (auto-téléchargés)
├── data/                   # Données locales
└── assets/                 # Ressources graphiques
```

## 🤖 Modèles IA utilisés

### Détection de pose
- **YOLOv8-Pose** : Modèle de détection de poses humaines ultra-rapide
- **MediaPipe Pose** : Alternative plus légère de Google

### Analyse audio
- **Analyse spectrale** : FFT pour extraction de fréquences
- **Classification ML** : Détection basée sur les caractéristiques audio (RMS, ZCR, MFCC)

## ⚠️ Niveaux d'alerte

| Niveau | Couleur | Description |
|--------|---------|-------------|
| Normal | 🟢 Vert | Aucune activité suspecte |
| Medium | 🟡 Jaune | Mouvement brusque ou son élevé isolé |
| High | 🟠 Orange | Mouvements répétés ou cris détectés |
| Critical | 🔴 Rouge | Violence confirmée (combinaison audio/vidéo) |

## 🔒 Confidentialité

- Tout le traitement est effectué **localement** sur votre machine
- Aucune donnée n'est envoyée à des serveurs externes
- Les flux vidéo/audio ne sont pas enregistrés par défaut

## 🐛 Dépannage

### La caméra ne s'ouvre pas
```bash
# Vérifier les permissions
ls -la /dev/video*  # Linux
```

### Erreur audio
```bash
# Installer portaudio
sudo apt-get install portaudio19-dev  # Linux
brew install portaudio  # macOS
```

### Performance lente
- Utilisez un GPU NVIDIA avec CUDA
- Réduisez la résolution de la caméra dans `gui.py`
- Utilisez MediaPipe au lieu de YOLO (plus léger)

## 📝 Licence

Ce projet est fourni à des fins éducatives et de recherche.

## 🙏 Crédits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

---

**⚠️ Avertissement** : Ce système est conçu pour la détection préventive et ne remplace pas les services de sécurité professionnels. Utilisez de manière responsable et éthique.
