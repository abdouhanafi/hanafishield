"""
HANAFISHIELD - Modern GUI Interface
Beautiful dashboard for real-time violence detection
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
from datetime import datetime
from collections import deque
from typing import Optional, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    USE_CTK = True
except ImportError:
    USE_CTK = False
    print("⚠️ customtkinter not available, using standard tkinter")

from video_detector import create_detector, ViolenceEvent
from audio_detector import create_audio_detector, AudioEvent


# Color scheme
class Colors:
    BG_DARK = "#0a0a0f"
    BG_CARD = "#1a1a2e"
    BG_LIGHTER = "#252540"
    
    ACCENT_PRIMARY = "#ff0080"
    ACCENT_SECONDARY = "#7928ca"
    
    SUCCESS = "#00ff88"
    WARNING = "#ffc800"
    DANGER = "#ff6400"
    CRITICAL = "#ff0000"
    
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#888888"
    TEXT_MUTED = "#555555"
    
    BORDER = "#333344"


# Fonts (with fallbacks for Windows)
class Fonts:
    # Use system fonts that work across platforms
    FAMILY = "Segoe UI"  # Windows default, falls back gracefully
    MONO = "Consolas"    # Windows monospace
    
    @staticmethod
    def get(size=11, weight="normal"):
        return (Fonts.FAMILY, size, weight)
    
    @staticmethod
    def mono(size=11):
        return (Fonts.MONO, size)


class GradientFrame(tk.Canvas):
    """A frame with gradient background"""
    def __init__(self, parent, color1, color2, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.color1 = color1
        self.color2 = color2
        self.bind("<Configure>", self._draw_gradient)
    
    def _draw_gradient(self, event=None):
        self.delete("gradient")
        width = self.winfo_width()
        height = self.winfo_height()
        
        r1, g1, b1 = self._hex_to_rgb(self.color1)
        r2, g2, b2 = self._hex_to_rgb(self.color2)
        
        for i in range(height):
            ratio = i / height
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.create_line(0, i, width, i, fill=color, tags="gradient")
    
    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class CircularProgress(tk.Canvas):
    """Circular progress indicator"""
    def __init__(self, parent, size=100, thickness=10, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        bg=Colors.BG_CARD, highlightthickness=0, **kwargs)
        self.size = size
        self.thickness = thickness
        self.value = 0
    
    def set_value(self, value: float, color: str = None):
        """Set progress value (0-1)"""
        self.value = max(0, min(1, value))
        self.delete("all")
        
        padding = self.thickness
        x0, y0 = padding, padding
        x1, y1 = self.size - padding, self.size - padding
        
        # Background arc
        self.create_arc(x0, y0, x1, y1, start=90, extent=-360,
                       outline=Colors.BG_LIGHTER, width=self.thickness, style="arc")
        
        # Progress arc
        if color is None:
            if self.value > 0.7:
                color = Colors.CRITICAL
            elif self.value > 0.4:
                color = Colors.WARNING
            else:
                color = Colors.SUCCESS
        
        extent = -360 * self.value
        self.create_arc(x0, y0, x1, y1, start=90, extent=extent,
                       outline=color, width=self.thickness, style="arc")
        
        # Center text
        text = f"{int(self.value * 100)}%"
        self.create_text(self.size/2, self.size/2, text=text,
                        fill=color, font=("Segoe UI", 16, "bold"))


class AudioVisualizer(tk.Canvas):
    """Audio frequency visualizer"""
    def __init__(self, parent, width=300, height=80, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=Colors.BG_DARK, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.bars = 32
        self.bar_width = (width - 20) / self.bars - 2
        self.data = np.zeros(self.bars)
    
    def update_data(self, data: np.ndarray, level: float = 0):
        """Update visualization with new frequency data"""
        self.delete("all")
        
        # Determine color based on level
        if level > 0.7:
            color1, color2 = Colors.CRITICAL, Colors.DANGER
        elif level > 0.4:
            color1, color2 = Colors.DANGER, Colors.WARNING
        else:
            color1, color2 = Colors.SUCCESS, "#00ffcc"
        
        # Draw bars
        for i, val in enumerate(data[:self.bars]):
            x = 10 + i * (self.bar_width + 2)
            bar_height = max(4, val * (self.height - 20))
            y = self.height - 10 - bar_height
            
            # Gradient effect
            ratio = i / self.bars
            r1, g1, b1 = self._hex_to_rgb(color1)
            r2, g2, b2 = self._hex_to_rgb(color2)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.create_rectangle(x, y, x + self.bar_width, self.height - 10,
                                 fill=color, outline="")
    
    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class ThreatBar(tk.Canvas):
    """Horizontal threat level bar"""
    def __init__(self, parent, width=300, height=24, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=Colors.BG_DARK, highlightthickness=0, **kwargs)
        self.bar_width = width
        self.bar_height = height
        self.value = 0
    
    def set_value(self, value: float):
        self.value = max(0, min(1, value))
        self.delete("all")
        
        # Background
        self.create_rectangle(0, 4, self.bar_width, self.bar_height - 4,
                             fill=Colors.BG_LIGHTER, outline="")
        
        # Fill
        fill_width = int(self.bar_width * self.value)
        if self.value > 0.7:
            color = Colors.CRITICAL
        elif self.value > 0.4:
            color = Colors.DANGER
        else:
            color = Colors.SUCCESS
        
        if fill_width > 0:
            self.create_rectangle(0, 4, fill_width, self.bar_height - 4,
                                 fill=color, outline="")
        
        # Border
        self.create_rectangle(0, 4, self.bar_width, self.bar_height - 4,
                             outline=Colors.BORDER, width=1)


class AlertCard(tk.Frame):
    """Individual alert display card"""
    def __init__(self, parent, event, **kwargs):
        super().__init__(parent, bg=Colors.BG_CARD, **kwargs)
        
        # Determine color based on severity
        if event.severity == 'critical':
            border_color = Colors.CRITICAL
            bg_color = "#1a0000"
        elif event.severity == 'high':
            border_color = Colors.DANGER
            bg_color = "#1a0a00"
        else:
            border_color = Colors.WARNING
            bg_color = "#1a1a00"
        
        self.configure(highlightbackground=border_color, highlightthickness=2)
        
        # Content frame
        content = tk.Frame(self, bg=bg_color, padx=10, pady=8)
        content.pack(fill="both", expand=True)
        
        # Icon and message
        msg_label = tk.Label(content, text=event.description,
                            font=("Segoe UI", 11, "bold"),
                            fg=Colors.TEXT_PRIMARY, bg=bg_color,
                            anchor="w")
        msg_label.pack(fill="x")
        
        # Time and source
        time_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        source = getattr(event, 'event_type', 'Unknown')
        meta_label = tk.Label(content, text=f"{time_str} • {source}",
                             font=("Segoe UI", 9),
                             fg=Colors.TEXT_MUTED, bg=bg_color,
                             anchor="w")
        meta_label.pack(fill="x")


class HanafiShieldApp:
    """Main application class"""
    
    def __init__(self):
        # Create main window
        if USE_CTK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        
        self.root.title("HANAFISHIELD - Violence Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg=Colors.BG_DARK)
        self.root.minsize(1200, 800)
        
        # State
        self.is_running = False
        self.cap = None
        self.video_detector = None
        self.audio_detector = None
        
        # Metrics
        self.detection_count = 0
        self.uptime_seconds = 0
        self.threat_level = 0.0
        self.alert_history = deque(maxlen=50)
        
        # Build UI
        self._build_ui()
        
        # Start update loops
        self._start_update_loops()
    
    def _build_ui(self):
        """Build the user interface"""
        # Main container
        main_container = tk.Frame(self.root, bg=Colors.BG_DARK)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        self._build_header(main_container)
        
        # Content area
        content = tk.Frame(main_container, bg=Colors.BG_DARK)
        content.pack(fill="both", expand=True, pady=(20, 0))
        
        # Left panel (video)
        left_panel = tk.Frame(content, bg=Colors.BG_DARK)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        self._build_video_section(left_panel)
        self._build_control_section(left_panel)
        
        # Right panel (analytics)
        right_panel = tk.Frame(content, bg=Colors.BG_DARK, width=380)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        self._build_status_card(right_panel)
        self._build_audio_card(right_panel)
        self._build_motion_card(right_panel)
        self._build_alerts_card(right_panel)
    
    def _build_header(self, parent):
        """Build header section"""
        header = tk.Frame(parent, bg=Colors.BG_DARK)
        header.pack(fill="x")
        
        # Left side - Logo and title
        left = tk.Frame(header, bg=Colors.BG_DARK)
        left.pack(side="left")
        
        # Logo
        logo_frame = tk.Frame(left, bg=Colors.ACCENT_PRIMARY, width=48, height=48)
        logo_frame.pack(side="left", padx=(0, 15))
        logo_frame.pack_propagate(False)
        logo_label = tk.Label(logo_frame, text="🛡️", font=("", 24), bg=Colors.ACCENT_PRIMARY)
        logo_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        title_frame = tk.Frame(left, bg=Colors.BG_DARK)
        title_frame.pack(side="left")
        
        title = tk.Label(title_frame, text="HANAFISHIELD",
                        font=("Segoe UI", 24, "bold"),
                        fg=Colors.TEXT_PRIMARY, bg=Colors.BG_DARK)
        title.pack(anchor="w")
        
        subtitle = tk.Label(title_frame, text="SYSTÈME DE DÉTECTION DE VIOLENCE",
                           font=("Segoe UI", 10),
                           fg=Colors.TEXT_MUTED, bg=Colors.BG_DARK)
        subtitle.pack(anchor="w")
        
        # Right side - Stats
        right = tk.Frame(header, bg=Colors.BG_DARK)
        right.pack(side="right")
        
        # Uptime
        uptime_frame = tk.Frame(right, bg=Colors.BG_DARK)
        uptime_frame.pack(side="left", padx=20)
        
        tk.Label(uptime_frame, text="TEMPS ACTIF",
                font=("Segoe UI", 9), fg=Colors.TEXT_MUTED,
                bg=Colors.BG_DARK).pack()
        self.uptime_label = tk.Label(uptime_frame, text="00:00:00",
                                    font=("Consolas", 18), fg=Colors.SUCCESS,
                                    bg=Colors.BG_DARK)
        self.uptime_label.pack()
        
        # Detections
        detect_frame = tk.Frame(right, bg=Colors.BG_DARK)
        detect_frame.pack(side="left", padx=20)
        
        tk.Label(detect_frame, text="DÉTECTIONS",
                font=("Segoe UI", 9), fg=Colors.TEXT_MUTED,
                bg=Colors.BG_DARK).pack()
        self.detection_label = tk.Label(detect_frame, text="0",
                                       font=("Consolas", 18), fg=Colors.DANGER,
                                       bg=Colors.BG_DARK)
        self.detection_label.pack()
    
    def _build_video_section(self, parent):
        """Build video display section"""
        # Container
        video_container = tk.Frame(parent, bg=Colors.BG_CARD,
                                  highlightbackground=Colors.BORDER,
                                  highlightthickness=1)
        video_container.pack(fill="both", expand=True)
        
        # Video label
        self.video_label = tk.Label(video_container, bg=Colors.BG_DARK)
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Placeholder
        self._show_video_placeholder()
        
        # Threat bar overlay (at bottom)
        self.threat_overlay = tk.Frame(video_container, bg=Colors.BG_CARD)
        self.threat_overlay.place(relx=0, rely=1.0, relwidth=1.0, anchor="sw", height=60)
        
        threat_inner = tk.Frame(self.threat_overlay, bg=Colors.BG_CARD)
        threat_inner.pack(fill="both", expand=True, padx=15, pady=10)
        
        tk.Label(threat_inner, text="Niveau de menace",
                font=("Segoe UI", 10), fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_CARD).pack(anchor="w")
        
        self.video_threat_bar = ThreatBar(threat_inner, width=400, height=20)
        self.video_threat_bar.pack(fill="x", pady=(5, 0))
    
    def _show_video_placeholder(self):
        """Show placeholder when camera is off"""
        # Create placeholder image
        img = Image.new('RGB', (854, 480), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Draw camera icon placeholder
        cx, cy = 427, 200
        draw.ellipse([cx-60, cy-60, cx+60, cy+60], outline=(50, 50, 60), width=3)
        draw.text((cx-10, cy-15), "📹", fill=(80, 80, 90))
        
        # Text
        draw.text((340, 300), "Surveillance inactive", fill=(100, 100, 110))
        draw.text((300, 330), "Cliquez sur 'Activer' pour démarrer", fill=(70, 70, 80))
        
        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def _build_control_section(self, parent):
        """Build control buttons"""
        control_frame = tk.Frame(parent, bg=Colors.BG_DARK)
        control_frame.pack(fill="x", pady=(15, 0))
        
        if USE_CTK:
            self.start_button = ctk.CTkButton(
                control_frame,
                text="▶ Activer la surveillance",
                font=("Segoe UI", 14, "bold"),
                fg_color=Colors.SUCCESS,
                text_color="#000000",
                hover_color="#00cc6a",
                height=50,
                command=self._toggle_surveillance
            )
        else:
            self.start_button = tk.Button(
                control_frame,
                text="▶ Activer la surveillance",
                font=("Segoe UI", 12, "bold"),
                bg=Colors.SUCCESS,
                fg="#000000",
                activebackground="#00cc6a",
                relief="flat",
                padx=20, pady=10,
                command=self._toggle_surveillance
            )
        self.start_button.pack(fill="x")
    
    def _build_status_card(self, parent):
        """Build system status card"""
        card = tk.Frame(parent, bg=Colors.BG_CARD,
                       highlightbackground=Colors.BORDER,
                       highlightthickness=1)
        card.pack(fill="x", pady=(0, 15))
        
        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=20)
        inner.pack(fill="x")
        
        tk.Label(inner, text="STATUT SYSTÈME",
                font=("Segoe UI", 10),
                fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD).pack(anchor="w")
        
        status_row = tk.Frame(inner, bg=Colors.BG_CARD)
        status_row.pack(fill="x", pady=(15, 0))
        
        # Status indicator
        self.status_indicator = tk.Canvas(status_row, width=48, height=48,
                                         bg=Colors.BG_CARD, highlightthickness=0)
        self.status_indicator.pack(side="left", padx=(0, 15))
        self._update_status_indicator(False)
        
        # Status text
        status_text = tk.Frame(status_row, bg=Colors.BG_CARD)
        status_text.pack(side="left")
        
        self.status_title = tk.Label(status_text, text="Système inactif",
                                    font=("Segoe UI", 14, "bold"),
                                    fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.status_title.pack(anchor="w")
        
        self.status_subtitle = tk.Label(status_text, text="En attente d'activation",
                                       font=("Segoe UI", 10),
                                       fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.status_subtitle.pack(anchor="w")
    
    def _update_status_indicator(self, active: bool):
        """Update status indicator"""
        self.status_indicator.delete("all")
        color = Colors.SUCCESS if active else Colors.TEXT_MUTED
        self.status_indicator.create_oval(4, 4, 44, 44, fill=color, outline="")
        symbol = "✓" if active else "○"
        self.status_indicator.create_text(24, 24, text=symbol,
                                         font=("", 18), fill="#000" if active else Colors.TEXT_MUTED)
    
    def _build_audio_card(self, parent):
        """Build audio analysis card"""
        card = tk.Frame(parent, bg=Colors.BG_CARD,
                       highlightbackground=Colors.BORDER,
                       highlightthickness=1)
        card.pack(fill="x", pady=(0, 15))
        
        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=20)
        inner.pack(fill="x")
        
        # Header
        header = tk.Frame(inner, bg=Colors.BG_CARD)
        header.pack(fill="x")
        
        tk.Label(header, text="ANALYSE AUDIO",
                font=("Segoe UI", 10),
                fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD).pack(side="left")
        
        self.audio_status = tk.Label(header, text="INACTIF",
                                    font=("Segoe UI", 9, "bold"),
                                    fg=Colors.TEXT_MUTED, bg=Colors.BG_LIGHTER,
                                    padx=8, pady=2)
        self.audio_status.pack(side="right")
        
        # Visualizer
        self.audio_viz = AudioVisualizer(inner, width=340, height=70)
        self.audio_viz.pack(fill="x", pady=(15, 10))
        
        # Level indicator
        self.audio_level_label = tk.Label(inner, text="Niveau: 0%",
                                         font=("Segoe UI", 10),
                                         fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.audio_level_label.pack(anchor="w")
    
    def _build_motion_card(self, parent):
        """Build motion analysis card"""
        card = tk.Frame(parent, bg=Colors.BG_CARD,
                       highlightbackground=Colors.BORDER,
                       highlightthickness=1)
        card.pack(fill="x", pady=(0, 15))
        
        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=20)
        inner.pack(fill="x")
        
        # Header
        header = tk.Frame(inner, bg=Colors.BG_CARD)
        header.pack(fill="x")
        
        tk.Label(header, text="ANALYSE MOUVEMENT",
                font=("Segoe UI", 10),
                fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD).pack(side="left")
        
        self.motion_status = tk.Label(header, text="INACTIF",
                                     font=("Segoe UI", 9, "bold"),
                                     fg=Colors.TEXT_MUTED, bg=Colors.BG_LIGHTER,
                                     padx=8, pady=2)
        self.motion_status.pack(side="right")
        
        # Threat circle
        self.motion_circle = CircularProgress(inner, size=100, thickness=8)
        self.motion_circle.pack(pady=15)
        
        # Level indicator
        self.motion_level_label = tk.Label(inner, text="Intensité: 0%",
                                          font=("Segoe UI", 10),
                                          fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.motion_level_label.pack()
    
    def _build_alerts_card(self, parent):
        """Build alerts history card"""
        card = tk.Frame(parent, bg=Colors.BG_CARD,
                       highlightbackground=Colors.BORDER,
                       highlightthickness=1)
        card.pack(fill="both", expand=True)
        
        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=20)
        inner.pack(fill="both", expand=True)
        
        tk.Label(inner, text="HISTORIQUE DES ALERTES",
                font=("Segoe UI", 10),
                fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD).pack(anchor="w")
        
        # Scrollable alert list
        self.alerts_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        self.alerts_frame.pack(fill="both", expand=True, pady=(15, 0))
        
        self.no_alerts_label = tk.Label(self.alerts_frame, text="Aucune alerte",
                                       font=("Segoe UI", 11),
                                       fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
        self.no_alerts_label.pack(pady=20)
    
    def _toggle_surveillance(self):
        """Toggle surveillance on/off"""
        if self.is_running:
            self._stop_surveillance()
        else:
            self._start_surveillance()
    
    def _start_surveillance(self):
        """Start video and audio surveillance"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Cannot open camera")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize detectors
            print("🔄 Initializing detectors...")
            self.video_detector = create_detector(use_ml=True)
            self.audio_detector = create_audio_detector(use_advanced=True)
            
            # Start audio
            if self.audio_detector:
                self.audio_detector.start()
            
            self.is_running = True
            
            # Update UI
            self._update_ui_running(True)
            
            # Start video thread
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
            
            print("✅ Surveillance started")
            
        except Exception as e:
            print(f"❌ Error starting surveillance: {e}")
            self._stop_surveillance()
    
    def _stop_surveillance(self):
        """Stop surveillance"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.audio_detector:
            self.audio_detector.stop()
        
        if self.video_detector and hasattr(self.video_detector, 'release'):
            self.video_detector.release()
        
        # Update UI
        self._update_ui_running(False)
        self._show_video_placeholder()
        
        print("🛑 Surveillance stopped")
    
    def _update_ui_running(self, running: bool):
        """Update UI elements based on running state"""
        if running:
            if USE_CTK:
                self.start_button.configure(
                    text="⏹ Désactiver",
                    fg_color=Colors.CRITICAL,
                    hover_color="#cc0000"
                )
            else:
                self.start_button.configure(
                    text="⏹ Désactiver",
                    bg=Colors.CRITICAL
                )
            
            self.status_title.configure(text="Protection active", fg=Colors.SUCCESS)
            self.status_subtitle.configure(text="Surveillance en cours...")
            self._update_status_indicator(True)
            
            self.audio_status.configure(text="ACTIF", fg=Colors.SUCCESS, bg="#002200")
            self.motion_status.configure(text="ACTIF", fg=Colors.SUCCESS, bg="#002200")
        else:
            if USE_CTK:
                self.start_button.configure(
                    text="▶ Activer la surveillance",
                    fg_color=Colors.SUCCESS,
                    hover_color="#00cc6a"
                )
            else:
                self.start_button.configure(
                    text="▶ Activer la surveillance",
                    bg=Colors.SUCCESS
                )
            
            self.status_title.configure(text="Système inactif", fg=Colors.TEXT_MUTED)
            self.status_subtitle.configure(text="En attente d'activation")
            self._update_status_indicator(False)
            
            self.audio_status.configure(text="INACTIF", fg=Colors.TEXT_MUTED, bg=Colors.BG_LIGHTER)
            self.motion_status.configure(text="INACTIF", fg=Colors.TEXT_MUTED, bg=Colors.BG_LIGHTER)
    
    def _video_loop(self):
        """Main video processing loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process frame
                annotated_frame, events, video_threat = self.video_detector.process_frame(frame)
                
                # Get audio data
                audio_threat = 0.0
                audio_events = []
                if self.audio_detector:
                    audio_threat = self.audio_detector.get_threat_level()
                    audio_events = self.audio_detector.current_events
                
                # Combined threat level
                self.threat_level = max(video_threat, audio_threat)
                
                # Process events
                all_events = events + audio_events
                for event in all_events:
                    if event not in list(self.alert_history)[-5:]:
                        self.alert_history.append(event)
                        self.detection_count += 1
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((854, 480), Image.Resampling.LANCZOS)
                
                # Add threat border if needed
                if self.threat_level > 0.7:
                    img = self._add_threat_border(img, Colors.CRITICAL)
                elif self.threat_level > 0.4:
                    img = self._add_threat_border(img, Colors.DANGER)
                
                photo = ImageTk.PhotoImage(img)
                
                # Update UI (thread-safe)
                self.root.after(0, self._update_video_display, photo, video_threat)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Video loop error: {e}")
                time.sleep(0.1)
    
    def _add_threat_border(self, img: Image.Image, color: str) -> Image.Image:
        """Add colored border to image"""
        draw = ImageDraw.Draw(img)
        w, h = img.size
        border_width = 4
        
        # Convert hex to RGB
        color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        draw.rectangle([0, 0, w-1, border_width], fill=color_rgb)
        draw.rectangle([0, h-border_width, w-1, h-1], fill=color_rgb)
        draw.rectangle([0, 0, border_width, h-1], fill=color_rgb)
        draw.rectangle([w-border_width, 0, w-1, h-1], fill=color_rgb)
        
        return img
    
    def _update_video_display(self, photo, threat_level):
        """Update video display (must be called from main thread)"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        self.video_threat_bar.set_value(threat_level)
    
    def _start_update_loops(self):
        """Start periodic UI update loops"""
        self._update_audio_ui()
        self._update_stats_ui()
        self._update_alerts_ui()
    
    def _update_audio_ui(self):
        """Update audio visualization"""
        if self.is_running and self.audio_detector:
            freq_data = self.audio_detector.get_frequency_data()
            level = self.audio_detector.get_threat_level()
            
            self.audio_viz.update_data(freq_data, level)
            
            level_color = Colors.CRITICAL if level > 0.7 else Colors.WARNING if level > 0.4 else Colors.SUCCESS
            self.audio_level_label.configure(
                text=f"Niveau: {int(level * 100)}%",
                fg=level_color
            )
            
            # Update status badge
            if self.audio_detector.current_events:
                self.audio_status.configure(text="ALERTE", fg=Colors.DANGER, bg="#220000")
            else:
                self.audio_status.configure(text="NORMAL", fg=Colors.SUCCESS, bg="#002200")
        
        self.root.after(50, self._update_audio_ui)
    
    def _update_stats_ui(self):
        """Update statistics display"""
        if self.is_running:
            self.uptime_seconds += 1
        
        # Format uptime
        hours = self.uptime_seconds // 3600
        minutes = (self.uptime_seconds % 3600) // 60
        seconds = self.uptime_seconds % 60
        self.uptime_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update detection count
        self.detection_label.configure(text=str(self.detection_count))
        
        # Update motion circle
        if self.is_running and self.video_detector:
            self.motion_circle.set_value(self.threat_level)
            
            level_color = Colors.CRITICAL if self.threat_level > 0.7 else Colors.WARNING if self.threat_level > 0.4 else Colors.SUCCESS
            self.motion_level_label.configure(
                text=f"Intensité: {int(self.threat_level * 100)}%",
                fg=level_color
            )
            
            # Update motion status
            if self.threat_level > 0.5:
                self.motion_status.configure(text="ALERTE", fg=Colors.DANGER, bg="#220000")
            else:
                self.motion_status.configure(text="NORMAL", fg=Colors.SUCCESS, bg="#002200")
        
        self.root.after(1000, self._update_stats_ui)
    
    def _update_alerts_ui(self):
        """Update alerts display"""
        # Clear existing alerts
        for widget in self.alerts_frame.winfo_children():
            widget.destroy()
        
        recent_alerts = list(self.alert_history)[-5:]
        
        if not recent_alerts:
            self.no_alerts_label = tk.Label(self.alerts_frame, text="Aucune alerte",
                                           font=("Segoe UI", 11),
                                           fg=Colors.TEXT_MUTED, bg=Colors.BG_CARD)
            self.no_alerts_label.pack(pady=20)
        else:
            for event in reversed(recent_alerts):
                card = AlertCard(self.alerts_frame, event)
                card.pack(fill="x", pady=(0, 8))
        
        self.root.after(500, self._update_alerts_ui)
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
    
    def _on_close(self):
        """Handle window close"""
        self._stop_surveillance()
        self.root.destroy()


def main():
    """Entry point"""
    print("=" * 50)
    print("  HANAFISHIELD - Violence Detection System")
    print("=" * 50)
    print()
    
    app = HanafiShieldApp()
    app.run()


if __name__ == "__main__":
    main()
