"""
HANAFISHIELD - Audio Violence Detection Module
Detects violent sounds: screams, impacts, glass breaking, shouting
"""

import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Optional, Callable
import threading
import queue

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available. Install with: pip install sounddevice")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available. Install with: pip install librosa")

try:
    from scipy import signal
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available. Install with: pip install scipy")


@dataclass
class AudioEvent:
    """Represents a detected audio event"""
    timestamp: float
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    audio_level: float


class AudioFeatureExtractor:
    """Extracts audio features for violence detection"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_data: np.ndarray) -> dict:
        """Extract various audio features"""
        features = {}
        
        # Basic features
        features['rms'] = np.sqrt(np.mean(audio_data ** 2))
        features['peak'] = np.max(np.abs(audio_data))
        features['zero_crossing_rate'] = np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2
        
        # Frequency features using FFT
        if len(audio_data) > 0:
            fft_data = np.abs(fft(audio_data))[:len(audio_data)//2]
            freqs = np.linspace(0, self.sample_rate/2, len(fft_data))
            
            # Spectral centroid (brightness)
            features['spectral_centroid'] = np.sum(freqs * fft_data) / (np.sum(fft_data) + 1e-10)
            
            # High frequency energy ratio (screams have high frequencies)
            high_freq_threshold = 2000  # Hz
            high_freq_idx = int(high_freq_threshold * len(fft_data) / (self.sample_rate / 2))
            high_freq_energy = np.sum(fft_data[high_freq_idx:] ** 2)
            total_energy = np.sum(fft_data ** 2) + 1e-10
            features['high_freq_ratio'] = high_freq_energy / total_energy
            
            # Low frequency energy (impacts/thuds)
            low_freq_threshold = 500  # Hz
            low_freq_idx = int(low_freq_threshold * len(fft_data) / (self.sample_rate / 2))
            low_freq_energy = np.sum(fft_data[:low_freq_idx] ** 2)
            features['low_freq_ratio'] = low_freq_energy / total_energy
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum = np.cumsum(fft_data ** 2)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            features['spectral_rolloff'] = freqs[min(rolloff_idx, len(freqs)-1)]
        else:
            features['spectral_centroid'] = 0
            features['high_freq_ratio'] = 0
            features['low_freq_ratio'] = 0
            features['spectral_rolloff'] = 0
        
        return features
    
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features (if librosa available)"""
        if not LIBROSA_AVAILABLE or len(audio_data) < 512:
            return np.zeros(n_mfcc)
        
        try:
            mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=self.sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs, axis=1)
        except:
            return np.zeros(n_mfcc)


class ViolentSoundClassifier:
    """Classifies audio as violent or non-violent based on features"""
    
    # Thresholds for violence detection
    THRESHOLDS = {
        'scream': {
            'rms_min': 0.15,
            'high_freq_ratio_min': 0.3,
            'spectral_centroid_min': 1500,
            'zcr_min': 0.1
        },
        'impact': {
            'peak_min': 0.6,
            'low_freq_ratio_min': 0.4,
            'rms_spike_ratio': 3.0  # Sudden increase
        },
        'shouting': {
            'rms_min': 0.2,
            'sustained_duration': 1.0,  # seconds
            'spectral_centroid_range': (500, 2000)
        },
        'glass_break': {
            'high_freq_ratio_min': 0.5,
            'spectral_rolloff_min': 4000,
            'peak_min': 0.4
        }
    }
    
    def __init__(self):
        self.rms_history = deque(maxlen=50)
        self.sustained_loud_count = 0
        self.last_detection_time = {}
    
    def classify(self, features: dict) -> List[AudioEvent]:
        """Classify audio features and detect violent sounds"""
        events = []
        current_time = time.time()
        
        rms = features.get('rms', 0)
        self.rms_history.append(rms)
        
        # Calculate average RMS for spike detection
        avg_rms = np.mean(list(self.rms_history)) if self.rms_history else 0
        
        # 1. Scream detection
        if self._detect_scream(features):
            if self._cooldown_passed('scream', current_time, 0.5):
                confidence = self._calculate_scream_confidence(features)
                events.append(AudioEvent(
                    timestamp=current_time,
                    event_type='scream',
                    severity='high' if confidence > 0.7 else 'medium',
                    confidence=confidence,
                    description='🔊 Cri détecté!',
                    audio_level=rms
                ))
                self.last_detection_time['scream'] = current_time
        
        # 2. Impact detection (sudden loud sound)
        if self._detect_impact(features, avg_rms):
            if self._cooldown_passed('impact', current_time, 0.3):
                confidence = self._calculate_impact_confidence(features, avg_rms)
                events.append(AudioEvent(
                    timestamp=current_time,
                    event_type='impact',
                    severity='high' if confidence > 0.8 else 'medium',
                    confidence=confidence,
                    description='💥 Impact sonore!',
                    audio_level=rms
                ))
                self.last_detection_time['impact'] = current_time
        
        # 3. Sustained shouting
        if rms > self.THRESHOLDS['shouting']['rms_min']:
            self.sustained_loud_count += 1
        else:
            self.sustained_loud_count = max(0, self.sustained_loud_count - 2)
        
        if self.sustained_loud_count > 20:  # About 1 second at 20 FPS
            if self._cooldown_passed('shouting', current_time, 2.0):
                events.append(AudioEvent(
                    timestamp=current_time,
                    event_type='shouting',
                    severity='medium',
                    confidence=min(1.0, self.sustained_loud_count / 30),
                    description='📢 Cris/Altercation en cours',
                    audio_level=rms
                ))
                self.last_detection_time['shouting'] = current_time
        
        # 4. Glass breaking detection
        if self._detect_glass_break(features):
            if self._cooldown_passed('glass_break', current_time, 1.0):
                events.append(AudioEvent(
                    timestamp=current_time,
                    event_type='glass_break',
                    severity='high',
                    confidence=0.7,
                    description='🔨 Bris de verre détecté!',
                    audio_level=rms
                ))
                self.last_detection_time['glass_break'] = current_time
        
        return events
    
    def _cooldown_passed(self, event_type: str, current_time: float, cooldown: float) -> bool:
        """Check if enough time has passed since last detection"""
        last_time = self.last_detection_time.get(event_type, 0)
        return current_time - last_time > cooldown
    
    def _detect_scream(self, features: dict) -> bool:
        """Detect scream based on features"""
        thresh = self.THRESHOLDS['scream']
        return (
            features.get('rms', 0) > thresh['rms_min'] and
            features.get('high_freq_ratio', 0) > thresh['high_freq_ratio_min'] and
            features.get('spectral_centroid', 0) > thresh['spectral_centroid_min'] and
            features.get('zero_crossing_rate', 0) > thresh['zcr_min']
        )
    
    def _detect_impact(self, features: dict, avg_rms: float) -> bool:
        """Detect impact/hit sound"""
        thresh = self.THRESHOLDS['impact']
        rms = features.get('rms', 0)
        return (
            features.get('peak', 0) > thresh['peak_min'] and
            features.get('low_freq_ratio', 0) > thresh['low_freq_ratio_min'] and
            (avg_rms > 0 and rms / (avg_rms + 0.01) > thresh['rms_spike_ratio'])
        )
    
    def _detect_glass_break(self, features: dict) -> bool:
        """Detect glass breaking sound"""
        thresh = self.THRESHOLDS['glass_break']
        return (
            features.get('high_freq_ratio', 0) > thresh['high_freq_ratio_min'] and
            features.get('spectral_rolloff', 0) > thresh['spectral_rolloff_min'] and
            features.get('peak', 0) > thresh['peak_min']
        )
    
    def _calculate_scream_confidence(self, features: dict) -> float:
        """Calculate confidence for scream detection"""
        confidence = 0.0
        
        # RMS contribution
        rms = features.get('rms', 0)
        confidence += min(0.3, rms / 0.5 * 0.3)
        
        # High frequency contribution
        hfr = features.get('high_freq_ratio', 0)
        confidence += min(0.3, hfr / 0.6 * 0.3)
        
        # Spectral centroid contribution
        sc = features.get('spectral_centroid', 0)
        confidence += min(0.4, (sc - 1000) / 3000 * 0.4)
        
        return min(1.0, confidence)
    
    def _calculate_impact_confidence(self, features: dict, avg_rms: float) -> float:
        """Calculate confidence for impact detection"""
        confidence = 0.0
        
        # Peak contribution
        peak = features.get('peak', 0)
        confidence += min(0.4, peak / 0.8 * 0.4)
        
        # Spike ratio contribution
        rms = features.get('rms', 0)
        spike_ratio = rms / (avg_rms + 0.01)
        confidence += min(0.3, spike_ratio / 5 * 0.3)
        
        # Low frequency contribution
        lfr = features.get('low_freq_ratio', 0)
        confidence += min(0.3, lfr / 0.6 * 0.3)
        
        return min(1.0, confidence)


class AudioViolenceDetector:
    """Main audio violence detection class with real-time processing"""
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024, 
                 callback: Optional[Callable] = None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.callback = callback
        
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.classifier = ViolentSoundClassifier()
        
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
        # Current state
        self.current_level = 0.0
        self.current_events = []
        self.frequency_data = np.zeros(32)
        
        # History
        self.level_history = deque(maxlen=100)
        self.event_history = deque(maxlen=50)
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start(self):
        """Start audio capture and processing"""
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ Cannot start audio: sounddevice not available")
            return False
        
        try:
            self.is_running = True
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.stream.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            
            print("✅ Audio capture started")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop audio capture"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("🛑 Audio capture stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_data = audio_data.flatten()
                
                # Extract features
                features = self.feature_extractor.extract_features(audio_data)
                
                # Update current level
                self.current_level = features.get('rms', 0)
                self.level_history.append(self.current_level)
                
                # Calculate frequency data for visualization
                self._update_frequency_data(audio_data)
                
                # Classify for violence
                events = self.classifier.classify(features)
                self.current_events = events
                
                # Store events
                for event in events:
                    self.event_history.append(event)
                
                # Callback
                if self.callback and events:
                    self.callback(events)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    def _update_frequency_data(self, audio_data: np.ndarray):
        """Update frequency data for visualization"""
        if len(audio_data) < 64:
            return
        
        fft_data = np.abs(fft(audio_data))[:len(audio_data)//2]
        
        # Bin into 32 frequency bands
        n_bins = 32
        bin_size = len(fft_data) // n_bins
        
        if bin_size > 0:
            freq_bins = []
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size
                freq_bins.append(np.mean(fft_data[start:end]))
            
            # Normalize
            max_val = max(freq_bins) if max(freq_bins) > 0 else 1
            self.frequency_data = np.array(freq_bins) / max_val
    
    def get_threat_level(self) -> float:
        """Get current audio threat level"""
        base_level = min(1.0, self.current_level * 3)
        
        if self.current_events:
            severity_boost = {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.5,
                'critical': 0.7
            }
            max_boost = max(severity_boost.get(e.severity, 0) for e in self.current_events)
            base_level = min(1.0, base_level + max_boost)
        
        return base_level
    
    def get_frequency_data(self) -> np.ndarray:
        """Get current frequency data for visualization"""
        return self.frequency_data.copy()
    
    def get_recent_events(self, n: int = 10) -> List[AudioEvent]:
        """Get recent events"""
        return list(self.event_history)[-n:]


# Simple fallback audio detector (minimal dependencies)
class SimpleAudioDetector:
    """Simple audio detector using only numpy"""
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        self.is_running = False
        self.current_level = 0.0
        self.frequency_data = np.zeros(32)
        self.level_history = deque(maxlen=50)
        self.current_events = []
        self.event_history = deque(maxlen=50)
        
        self.stream = None
        self.audio_queue = queue.Queue()
        self.high_level_count = 0
    
    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())
    
    def start(self):
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ sounddevice not available")
            return False
        
        try:
            self.is_running = True
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.stream.start()
            
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            
            print("✅ Simple audio detector started")
            return True
        except Exception as e:
            print(f"❌ Audio error: {e}")
            return False
    
    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def _process_loop(self):
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1).flatten()
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data ** 2))
                self.current_level = rms
                self.level_history.append(rms)
                
                # Simple FFT for visualization
                if len(audio_data) >= 64:
                    fft_data = np.abs(np.fft.fft(audio_data))[:len(audio_data)//2]
                    n_bins = 32
                    bin_size = max(1, len(fft_data) // n_bins)
                    freq_bins = [np.mean(fft_data[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]
                    max_val = max(freq_bins) if max(freq_bins) > 0 else 1
                    self.frequency_data = np.array(freq_bins) / max_val
                
                # Simple detection
                self.current_events = []
                avg_level = np.mean(list(self.level_history)) if len(self.level_history) > 5 else 0.05
                
                # Sudden loud sound
                if rms > 0.3 and rms > avg_level * 3:
                    self.current_events.append(AudioEvent(
                        timestamp=time.time(),
                        event_type='impact',
                        severity='high',
                        confidence=min(1.0, rms),
                        description='💥 Son fort détecté!',
                        audio_level=rms
                    ))
                
                # Sustained loud
                if rms > 0.15:
                    self.high_level_count += 1
                else:
                    self.high_level_count = max(0, self.high_level_count - 1)
                
                if self.high_level_count > 20:
                    self.current_events.append(AudioEvent(
                        timestamp=time.time(),
                        event_type='shouting',
                        severity='medium',
                        confidence=min(1.0, self.high_level_count / 30),
                        description='📢 Bruit soutenu',
                        audio_level=rms
                    ))
                
                for event in self.current_events:
                    self.event_history.append(event)
                    
            except queue.Empty:
                continue
            except Exception as e:
                pass
    
    def get_threat_level(self) -> float:
        level = min(1.0, self.current_level * 3)
        if self.current_events:
            level = min(1.0, level + 0.3)
        return level
    
    def get_frequency_data(self) -> np.ndarray:
        return self.frequency_data.copy()
    
    def get_recent_events(self, n: int = 10) -> List[AudioEvent]:
        return list(self.event_history)[-n:]


def create_audio_detector(use_advanced: bool = True) -> AudioViolenceDetector:
    """Factory function to create audio detector"""
    if use_advanced and LIBROSA_AVAILABLE and SCIPY_AVAILABLE:
        return AudioViolenceDetector()
    else:
        return SimpleAudioDetector()
