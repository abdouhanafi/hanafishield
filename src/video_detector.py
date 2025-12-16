"""
HANAFISHIELD - Video Violence Detection Module
Uses YOLOv8 Pose + MediaPipe for real-time violence detection
"""

import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 not available. Install with: pip install ultralytics")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available. Install with: pip install mediapipe")


@dataclass
class ViolenceEvent:
    """Represents a detected violence event"""
    timestamp: float
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    bbox: Optional[Tuple[int, int, int, int]] = None


class PoseAnalyzer:
    """Analyzes body poses to detect violent movements"""
    
    # Key pose landmarks for violence detection
    VIOLENCE_KEYPOINTS = {
        'punch': {
            'joints': ['right_wrist', 'right_elbow', 'right_shoulder'],
            'velocity_threshold': 150,
            'extension_threshold': 0.7
        },
        'kick': {
            'joints': ['right_ankle', 'right_knee', 'right_hip'],
            'velocity_threshold': 200,
            'extension_threshold': 0.8
        },
        'defensive': {
            'joints': ['left_wrist', 'right_wrist', 'nose'],
            'proximity_threshold': 100
        }
    }
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.pose_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        
        # MediaPipe pose indices
        self.mp_indices = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
    
    def calculate_velocity(self, current_pose: dict, prev_pose: dict, dt: float) -> dict:
        """Calculate velocity of each keypoint"""
        velocities = {}
        for key in current_pose:
            if key in prev_pose and dt > 0:
                dx = current_pose[key][0] - prev_pose[key][0]
                dy = current_pose[key][1] - prev_pose[key][1]
                velocities[key] = np.sqrt(dx**2 + dy**2) / dt
            else:
                velocities[key] = 0
        return velocities
    
    def calculate_limb_extension(self, pose: dict, joint1: str, joint2: str, joint3: str) -> float:
        """Calculate how extended a limb is (0 = bent, 1 = fully extended)"""
        if not all(j in pose for j in [joint1, joint2, joint3]):
            return 0.0
        
        p1 = np.array(pose[joint1])
        p2 = np.array(pose[joint2])
        p3 = np.array(pose[joint3])
        
        # Calculate angle at middle joint
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Convert to extension ratio (180 degrees = fully extended)
        return angle / np.pi
    
    def detect_punch(self, pose: dict, velocities: dict) -> Optional[ViolenceEvent]:
        """Detect punching motion"""
        for side in ['right', 'left']:
            wrist = f'{side}_wrist'
            elbow = f'{side}_elbow'
            shoulder = f'{side}_shoulder'
            
            if wrist not in velocities:
                continue
            
            wrist_velocity = velocities.get(wrist, 0)
            extension = self.calculate_limb_extension(pose, wrist, elbow, shoulder)
            
            # Check for punch: high wrist velocity + arm extension
            if wrist_velocity > self.VIOLENCE_KEYPOINTS['punch']['velocity_threshold']:
                if extension > self.VIOLENCE_KEYPOINTS['punch']['extension_threshold']:
                    confidence = min(1.0, wrist_velocity / 300) * extension
                    severity = 'high' if confidence > 0.7 else 'medium'
                    return ViolenceEvent(
                        timestamp=time.time(),
                        event_type='punch',
                        severity=severity,
                        confidence=confidence,
                        description=f'Coup de poing détecté ({side})'
                    )
        return None
    
    def detect_kick(self, pose: dict, velocities: dict) -> Optional[ViolenceEvent]:
        """Detect kicking motion"""
        for side in ['right', 'left']:
            ankle = f'{side}_ankle'
            knee = f'{side}_knee'
            hip = f'{side}_hip'
            
            if ankle not in velocities:
                continue
            
            ankle_velocity = velocities.get(ankle, 0)
            extension = self.calculate_limb_extension(pose, ankle, knee, hip)
            
            if ankle_velocity > self.VIOLENCE_KEYPOINTS['kick']['velocity_threshold']:
                if extension > self.VIOLENCE_KEYPOINTS['kick']['extension_threshold']:
                    confidence = min(1.0, ankle_velocity / 400) * extension
                    severity = 'critical' if confidence > 0.8 else 'high'
                    return ViolenceEvent(
                        timestamp=time.time(),
                        event_type='kick',
                        severity=severity,
                        confidence=confidence,
                        description=f'Coup de pied détecté ({side})'
                    )
        return None
    
    def detect_defensive_posture(self, pose: dict) -> Optional[ViolenceEvent]:
        """Detect defensive posture (hands near face)"""
        if not all(k in pose for k in ['left_wrist', 'right_wrist', 'nose']):
            return None
        
        nose = np.array(pose['nose'])
        left_wrist = np.array(pose['left_wrist'])
        right_wrist = np.array(pose['right_wrist'])
        
        # Check if both hands are near face
        left_dist = np.linalg.norm(left_wrist - nose)
        right_dist = np.linalg.norm(right_wrist - nose)
        
        threshold = self.VIOLENCE_KEYPOINTS['defensive']['proximity_threshold']
        
        if left_dist < threshold and right_dist < threshold:
            confidence = 1.0 - (left_dist + right_dist) / (2 * threshold)
            return ViolenceEvent(
                timestamp=time.time(),
                event_type='defensive',
                severity='medium',
                confidence=confidence,
                description='Posture défensive détectée (victime potentielle)'
            )
        return None
    
    def analyze(self, pose: dict, dt: float) -> List[ViolenceEvent]:
        """Analyze pose for violent movements"""
        events = []
        
        if not pose:
            return events
        
        # Calculate velocities
        velocities = {}
        if self.pose_history:
            prev_pose = self.pose_history[-1]
            velocities = self.calculate_velocity(pose, prev_pose, dt)
        
        # Store history
        self.pose_history.append(pose)
        self.velocity_history.append(velocities)
        
        # Detect various violence types
        punch_event = self.detect_punch(pose, velocities)
        if punch_event:
            events.append(punch_event)
        
        kick_event = self.detect_kick(pose, velocities)
        if kick_event:
            events.append(kick_event)
        
        defensive_event = self.detect_defensive_posture(pose)
        if defensive_event:
            events.append(defensive_event)
        
        return events


class VideoViolenceDetector:
    """Main video violence detection class"""
    
    def __init__(self, use_yolo: bool = True, use_mediapipe: bool = True):
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        # Initialize YOLO
        self.yolo_model = None
        if self.use_yolo:
            try:
                # Use YOLOv8 pose model
                self.yolo_model = YOLO('yolov8n-pose.pt')
                print("✅ YOLOv8 Pose model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load YOLO: {e}")
                self.use_yolo = False
        
        # Initialize MediaPipe
        self.mp_pose = None
        self.pose_detector = None
        if self.use_mediapipe:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
                print("✅ MediaPipe Pose loaded")
            except Exception as e:
                print(f"⚠️ Failed to load MediaPipe: {e}")
                self.use_mediapipe = False
        
        # Pose analyzer
        self.pose_analyzer = PoseAnalyzer()
        
        # Motion detection (fallback)
        self.prev_frame = None
        self.motion_history = deque(maxlen=30)
        
        # Timing
        self.last_time = time.time()
        
        # Detection state
        self.current_threat_level = 0.0
        self.events = []
    
    def extract_pose_from_mediapipe(self, results, frame_shape) -> dict:
        """Extract pose keypoints from MediaPipe results"""
        if not results.pose_landmarks:
            return {}
        
        h, w = frame_shape[:2]
        pose = {}
        
        landmarks = results.pose_landmarks.landmark
        keypoint_map = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        for name, idx in keypoint_map.items():
            lm = landmarks[idx]
            pose[name] = (lm.x * w, lm.y * h)
        
        return pose
    
    def extract_pose_from_yolo(self, results, frame_shape) -> dict:
        """Extract pose keypoints from YOLO results"""
        if not results or len(results) == 0:
            return {}
        
        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return {}
        
        keypoints = result.keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return {}
        
        # Get first person's keypoints
        kpts = keypoints.xy[0].cpu().numpy()
        
        # YOLO pose keypoint indices
        keypoint_map = {
            'nose': 0,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        pose = {}
        for name, idx in keypoint_map.items():
            if idx < len(kpts):
                pose[name] = (kpts[idx][0], kpts[idx][1])
        
        return pose
    
    def detect_motion(self, frame: np.ndarray) -> float:
        """Fallback motion detection using frame differencing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate motion ratio
        motion_ratio = np.sum(thresh > 0) / thresh.size
        
        self.prev_frame = gray
        self.motion_history.append(motion_ratio)
        
        return motion_ratio
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[ViolenceEvent], float]:
        """
        Process a single frame for violence detection
        Returns: (annotated_frame, events, threat_level)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        events = []
        pose = {}
        annotated_frame = frame.copy()
        
        # Method 1: MediaPipe Pose
        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw pose
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                pose = self.extract_pose_from_mediapipe(results, frame.shape)
        
        # Method 2: YOLO Pose (if MediaPipe didn't detect)
        elif self.use_yolo and not pose:
            results = self.yolo_model(frame, verbose=False)
            if results:
                annotated_frame = results[0].plot()
                pose = self.extract_pose_from_yolo(results, frame.shape)
        
        # Analyze pose for violence
        if pose:
            pose_events = self.pose_analyzer.analyze(pose, dt)
            events.extend(pose_events)
        
        # Fallback: Motion detection
        motion_level = self.detect_motion(frame)
        
        # Sudden high motion detection
        if len(self.motion_history) > 5:
            avg_motion = np.mean(list(self.motion_history)[-5:])
            if motion_level > 0.15 and motion_level > avg_motion * 3:
                events.append(ViolenceEvent(
                    timestamp=current_time,
                    event_type='sudden_motion',
                    severity='medium',
                    confidence=min(1.0, motion_level * 5),
                    description='Mouvement brusque détecté'
                ))
        
        # Calculate overall threat level
        threat_level = self._calculate_threat_level(events, motion_level)
        self.current_threat_level = threat_level
        self.events = events
        
        # Draw threat indicator
        annotated_frame = self._draw_threat_indicator(annotated_frame, threat_level, events)
        
        return annotated_frame, events, threat_level
    
    def _calculate_threat_level(self, events: List[ViolenceEvent], motion_level: float) -> float:
        """Calculate overall threat level from 0 to 1"""
        if not events:
            return min(0.3, motion_level)
        
        severity_weights = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.75,
            'critical': 1.0
        }
        
        max_threat = 0
        for event in events:
            threat = severity_weights.get(event.severity, 0.5) * event.confidence
            max_threat = max(max_threat, threat)
        
        return max_threat
    
    def _draw_threat_indicator(self, frame: np.ndarray, threat_level: float, events: List[ViolenceEvent]) -> np.ndarray:
        """Draw threat level indicator on frame"""
        h, w = frame.shape[:2]
        
        # Threat level bar
        bar_width = int(w * 0.3)
        bar_height = 20
        bar_x = 20
        bar_y = h - 40
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Threat level fill
        fill_width = int(bar_width * threat_level)
        if threat_level > 0.7:
            color = (0, 0, 255)  # Red
        elif threat_level > 0.4:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Label
        cv2.putText(frame, f'Menace: {int(threat_level * 100)}%', 
                   (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Event labels
        y_offset = 30
        for event in events[:3]:  # Show max 3 events
            color = (0, 0, 255) if event.severity in ['high', 'critical'] else (0, 165, 255)
            cv2.putText(frame, f'⚠ {event.description}', 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return frame
    
    def release(self):
        """Release resources"""
        if self.pose_detector:
            self.pose_detector.close()


# Simplified fallback detector (works without ML libraries)
class SimpleMotionDetector:
    """Simple motion-based violence detector (no ML required)"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_history = deque(maxlen=30)
        self.last_time = time.time()
        self.high_motion_streak = 0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[ViolenceEvent], float]:
        """Process frame using motion analysis"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        events = []
        annotated_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return annotated_frame, events, 0.0
        
        # Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_ratio = np.sum(thresh > 0) / thresh.size
        self.motion_history.append(motion_ratio)
        
        # Analyze motion patterns
        large_movements = 0
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            
            large_movements += 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect violence patterns
        threat_level = 0.0
        
        # Sudden motion spike
        if len(self.motion_history) > 5:
            avg_motion = np.mean(list(self.motion_history)[-10:-1]) if len(self.motion_history) > 10 else 0.01
            
            if motion_ratio > 0.1 and motion_ratio > avg_motion * 2.5:
                events.append(ViolenceEvent(
                    timestamp=current_time,
                    event_type='sudden_motion',
                    severity='medium',
                    confidence=min(1.0, motion_ratio * 5),
                    description='Mouvement brusque détecté!'
                ))
                self.high_motion_streak += 1
            else:
                self.high_motion_streak = max(0, self.high_motion_streak - 1)
        
        # Sustained violent motion
        if self.high_motion_streak > 5:
            events.append(ViolenceEvent(
                timestamp=current_time,
                event_type='sustained_violence',
                severity='high',
                confidence=min(1.0, self.high_motion_streak / 10),
                description='⚠️ Activité violente soutenue!'
            ))
        
        # Multiple rapid movements
        if large_movements > 3 and motion_ratio > 0.08:
            events.append(ViolenceEvent(
                timestamp=current_time,
                event_type='multiple_movements',
                severity='medium',
                confidence=min(1.0, large_movements / 5),
                description='Mouvements multiples rapides'
            ))
        
        # Calculate threat level
        threat_level = min(1.0, motion_ratio * 3 + (self.high_motion_streak * 0.1))
        
        if events:
            max_severity = max(e.severity for e in events)
            if max_severity == 'critical':
                threat_level = max(threat_level, 0.9)
            elif max_severity == 'high':
                threat_level = max(threat_level, 0.7)
            elif max_severity == 'medium':
                threat_level = max(threat_level, 0.5)
        
        # Draw indicators
        annotated_frame = self._draw_indicators(annotated_frame, threat_level, events, motion_ratio)
        
        self.prev_frame = gray
        return annotated_frame, events, threat_level
    
    def _draw_indicators(self, frame, threat_level, events, motion_ratio):
        """Draw threat indicators on frame"""
        h, w = frame.shape[:2]
        
        # Threat level bar
        bar_width = int(w * 0.3)
        bar_height = 20
        bar_x = 20
        bar_y = h - 40
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        fill_width = int(bar_width * threat_level)
        color = (0, 0, 255) if threat_level > 0.7 else (0, 165, 255) if threat_level > 0.4 else (0, 255, 0)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        cv2.putText(frame, f'Menace: {int(threat_level * 100)}%', 
                   (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Motion ratio
        cv2.putText(frame, f'Motion: {motion_ratio:.2%}', 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Events
        y_offset = 30
        for event in events[:3]:
            color = (0, 0, 255) if event.severity in ['high', 'critical'] else (0, 165, 255)
            cv2.putText(frame, event.description, 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return frame


def create_detector(use_ml: bool = True):
    """Factory function to create appropriate detector"""
    if use_ml and (YOLO_AVAILABLE or MEDIAPIPE_AVAILABLE):
        return VideoViolenceDetector(use_yolo=YOLO_AVAILABLE, use_mediapipe=MEDIAPIPE_AVAILABLE)
    else:
        print("⚠️ Using simple motion detector (ML libraries not available)")
        return SimpleMotionDetector()
