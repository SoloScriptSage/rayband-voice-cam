"""
PyQt-based camera viewer with modern, professional UI design.
Features glassmorphism effects, smooth animations, and contemporary styling.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QFont, QColor, QPen, 
                        QLinearGradient, QRadialGradient, QPainterPath, QBrush)
import logging

logger = logging.getLogger(__name__)


class ModernCameraWidget(QLabel):
    """Modern camera widget with glassmorphism and smooth animations."""
    
    def __init__(self):
        super().__init__()
        self.frame = None
        self.transcript_text = ""
        self.is_recording = False
        self.current_language = "english"
        self.fps = 0
        self.sign_text = ""
        self.sign_confidence = 0.0
        
        # Animation properties
        self._recording_opacity = 0
        self._rec_pulse_direction = 1
        self._sign_box_opacity = 0
        
        # Widget settings
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0a0e27;")
        
        # Setup animation timer
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._update_animations)
        self.anim_timer.start(50)  # 20 FPS for animations
    
    def _update_animations(self):
        """Update animation states."""
        if self.is_recording:
            # Pulse effect for recording indicator
            self._recording_opacity += 0.1 * self._rec_pulse_direction
            if self._recording_opacity >= 1.0:
                self._recording_opacity = 1.0
                self._rec_pulse_direction = -1
            elif self._recording_opacity <= 0.3:
                self._recording_opacity = 0.3
                self._rec_pulse_direction = 1
        
        # Fade in/out sign language box
        if self.sign_text and self.sign_confidence > 0.7:
            self._sign_box_opacity = min(1.0, self._sign_box_opacity + 0.1)
        else:
            self._sign_box_opacity = max(0.0, self._sign_box_opacity - 0.1)
        
        self.update()
    
    def set_frame(self, frame: np.ndarray):
        """Update the displayed frame."""
        self.frame = frame
        self.update()
    
    def set_transcript(self, text: str):
        """Update transcript text."""
        self.transcript_text = text
        self.update()
    
    def set_recording(self, recording: bool):
        """Update recording status."""
        if recording and not self.is_recording:
            self._recording_opacity = 0.3
        self.is_recording = recording
        self.update()
    
    def set_language(self, language: str):
        """Update current language."""
        self.current_language = language
        self.update()
    
    def set_fps(self, fps: int):
        """Update FPS counter."""
        self.fps = fps
        self.update()
    
    def set_sign_info(self, sign_text: str, confidence: float):
        """Update sign language detection info."""
        self.sign_text = sign_text
        self.sign_confidence = confidence
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event with modern design."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Draw gradient background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(10, 14, 39))
        gradient.setColorAt(1, QColor(20, 25, 55))
        painter.fillRect(self.rect(), gradient)
        
        if self.frame is None:
            self._draw_no_camera(painter)
            return
        
        # Convert and display frame with rounded corners
        height, width, channel = self.frame.shape
        bytes_per_line = 3 * width
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit with padding
        padding = 20
        available_width = self.width() - (padding * 2)
        available_height = self.height() - (padding * 2)
        
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            available_width, available_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Center the frame
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        
        # Draw frame with rounded corners and glow effect
        self._draw_frame_with_effects(painter, scaled_pixmap, x, y)
        
        # Draw overlays
        self._draw_recording_indicator(painter, x, y)
        self._draw_sign_language_box(painter, x, y, scaled_pixmap.width())
        self._draw_transcript_box(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
        self._draw_status_bar(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
    
    def _draw_no_camera(self, painter):
        """Draw 'no camera' screen."""
        painter.setPen(QColor(100, 120, 180))
        font = QFont("Segoe UI", 20, QFont.Light)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, "📹 Initializing Camera...")
    
    def _draw_frame_with_effects(self, painter, pixmap, x, y):
        """Draw camera frame with rounded corners and glow."""
        # Create rounded rectangle path
        path = QPainterPath()
        radius = 12
        path.addRoundedRect(x, y, pixmap.width(), pixmap.height(), radius, radius)
        
        # Draw subtle outer glow
        painter.setPen(Qt.NoPen)
        glow_gradient = QRadialGradient(
            x + pixmap.width() / 2, 
            y + pixmap.height() / 2,
            max(pixmap.width(), pixmap.height()) / 2 + 10
        )
        glow_gradient.setColorAt(0.9, QColor(100, 150, 255, 30))
        glow_gradient.setColorAt(1.0, QColor(100, 150, 255, 0))
        painter.setBrush(glow_gradient)
        painter.drawPath(path)
        
        # Clip and draw pixmap
        painter.setClipPath(path)
        painter.drawPixmap(x, y, pixmap)
        painter.setClipping(False)
        
        # Draw subtle border
        painter.setPen(QPen(QColor(100, 120, 180, 60), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
    
    def _draw_recording_indicator(self, painter, offset_x, offset_y):
        """Draw modern recording indicator with pulse effect."""
        if not self.is_recording:
            return
        
        # Pulsing circle
        painter.setPen(Qt.NoPen)
        
        # Outer glow
        glow_color = QColor(255, 50, 80, int(100 * self._recording_opacity))
        painter.setBrush(glow_color)
        painter.drawEllipse(offset_x + 15, offset_y + 15, 28, 28)
        
        # Inner circle
        inner_color = QColor(255, 50, 80, int(255 * self._recording_opacity))
        painter.setBrush(inner_color)
        painter.drawEllipse(offset_x + 20, offset_y + 20, 18, 18)
        
        # REC text with glassmorphism background
        text_x = offset_x + 50
        text_y = offset_y + 15
        
        # Glass background
        self._draw_glass_rect(painter, text_x, text_y, 60, 28, QColor(255, 50, 80, 150))
        
        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(text_x + 5, text_y + 20, "REC")
    
    def _draw_sign_language_box(self, painter, offset_x, offset_y, width):
        """Draw modern sign language detection box."""
        if self._sign_box_opacity <= 0:
            return
        
        box_width = min(500, width - 40)
        box_height = 100
        box_x = offset_x + (width - box_width) // 2
        box_y = offset_y + 80
        
        painter.setOpacity(self._sign_box_opacity)
        
        # Glassmorphism background
        self._draw_glass_rect(painter, box_x, box_y, box_width, box_height, 
                             QColor(30, 40, 80, 200))
        
        # Icon and title
        painter.setPen(QColor(100, 200, 255))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(box_x + 20, box_y + 30, "🤟 SIGN DETECTED")
        
        # Sign description
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI Emoji", 16)
        painter.setFont(font)
        text = self.sign_text
        painter.drawText(box_x + 20, box_y + 60, text)
        
        # Modern confidence bar
        bar_x = box_x + 20
        bar_y = box_y + 75
        bar_width = box_width - 40
        bar_height = 8
        
        # Background track
        track_path = QPainterPath()
        track_path.addRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)
        painter.fillPath(track_path, QColor(50, 60, 100, 150))
        
        # Confidence fill with gradient
        conf_width = int(bar_width * self.sign_confidence)
        if conf_width > 0:
            fill_path = QPainterPath()
            fill_path.addRoundedRect(bar_x, bar_y, conf_width, bar_height, 4, 4)
            
            gradient = QLinearGradient(bar_x, bar_y, bar_x + conf_width, bar_y)
            gradient.setColorAt(0, QColor(100, 200, 255))
            gradient.setColorAt(1, QColor(150, 100, 255))
            painter.fillPath(fill_path, gradient)
        
        painter.setOpacity(1.0)
    
    def _draw_transcript_box(self, painter, offset_x, offset_y, width, height):
        """Draw modern transcript box."""
        if not self.transcript_text:
            return
        
        box_height = 120
        box_y = offset_y + height - box_height - 60
        padding = 20
        
        # Glassmorphism background
        self._draw_glass_rect(painter, offset_x, box_y, width, box_height,
                             QColor(20, 30, 60, 220))
        
        # Icon and label
        painter.setPen(QColor(100, 200, 255))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(offset_x + padding, box_y + 25, "🎤 VOICE TRANSCRIPT")
        
        # Transcript text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 14)
        painter.setFont(font)
        
        fm = painter.fontMetrics()
        text_width = width - (padding * 2)
        lines = self._wrap_text(self.transcript_text, text_width, fm)
        
        y = box_y + 55
        for line in lines[:3]:  # Max 3 lines
            painter.drawText(offset_x + padding, y, line)
            y += fm.height() + 5
    
    def _draw_status_bar(self, painter, offset_x, offset_y, width, height):
        """Draw modern status bar."""
        bar_height = 50
        bar_y = offset_y + height - bar_height
        
        # Glassmorphism background
        self._draw_glass_rect(painter, offset_x, bar_y, width, bar_height,
                             QColor(15, 20, 40, 230))
        
        # Left side: FPS with icon
        painter.setPen(QColor(100, 200, 255))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        fps_text = f"⚡ {self.fps} FPS"
        painter.drawText(offset_x + 20, bar_y + 32, fps_text)
        
        # Center: App title
        painter.setPen(QColor(255, 255, 255, 180))
        painter.setFont(QFont("Segoe UI", 10))
        title = "RayBand Voice Camera"
        title_width = painter.fontMetrics().horizontalAdvance(title)
        painter.drawText(offset_x + (width - title_width) // 2, bar_y + 32, title)
        
        # Right side: Language with gradient
        lang_text = f"🌐 {self.current_language.upper()}"
        text_width = painter.fontMetrics().horizontalAdvance(lang_text)
        
        # Language badge background
        badge_x = offset_x + width - text_width - 50
        badge_y = bar_y + 12
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_x, badge_y, text_width + 30, 26, 13, 13)
        
        gradient = QLinearGradient(badge_x, badge_y, badge_x + text_width + 30, badge_y)
        gradient.setColorAt(0, QColor(100, 150, 255, 100))
        gradient.setColorAt(1, QColor(150, 100, 255, 100))
        painter.fillPath(badge_path, gradient)
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(badge_x + 15, bar_y + 32, lang_text)
    
    def _draw_glass_rect(self, painter, x, y, width, height, color):
        """Draw glassmorphism rectangle with blur effect simulation."""
        # Create rounded rectangle
        path = QPainterPath()
        path.addRoundedRect(x, y, width, height, 10, 10)
        
        # Fill with semi-transparent color
        painter.fillPath(path, color)
        
        # Add subtle gradient overlay for glass effect
        gradient = QLinearGradient(x, y, x, y + height)
        gradient.setColorAt(0, QColor(255, 255, 255, 30))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 5))
        gradient.setColorAt(1, QColor(0, 0, 0, 20))
        painter.fillPath(path, gradient)
        
        # Border
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
    
    def _wrap_text(self, text, max_width, fm):
        """Wrap text to fit within width."""
        if not text:
            return []
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = word if not current_line else current_line + " " + word
            if fm.horizontalAdvance(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Q:
            QApplication.quit()


class ModernCameraViewer(QMainWindow):
    """Modern main window with professional styling."""
    
    def __init__(self, camera_controller):
        super().__init__()
        self.camera_controller = camera_controller
        
        # Setup window
        self.setWindowTitle("RayBand Voice Camera")
        self.setGeometry(100, 100, 1280, 720)
        
        # Modern dark stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0e27, stop:1 #14182e
                );
            }
        """)
        
        # Create camera widget
        self.camera_widget = ModernCameraWidget()
        self.setCentralWidget(self.camera_widget)
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        logger.info("✓ Modern PyQt camera viewer initialized")
    
    def update_frame(self):
        """Update camera frame."""
        pass
    
    def set_frame(self, frame):
        """Update the displayed frame."""
        self.camera_widget.set_frame(frame)
    
    def set_transcript(self, text):
        """Update transcript text."""
        self.camera_widget.set_transcript(text)
    
    def set_recording(self, recording):
        """Update recording status."""
        self.camera_widget.set_recording(recording)
    
    def set_language(self, language):
        """Update current language."""
        self.camera_widget.set_language(language)
    
    def set_fps(self, fps):
        """Update FPS counter."""
        self.camera_widget.set_fps(fps)
    
    def set_sign_info(self, sign_text, confidence):
        """Update sign language info."""
        self.camera_widget.set_sign_info(sign_text, confidence)
    
    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Closing modern viewer...")
        event.accept()


def create_pyqt_viewer(camera_controller):
    """
    Create modern PyQt viewer instance.
    
    Replace the existing create_pyqt_viewer function in pyqt_viewer.py
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    viewer = ModernCameraViewer(camera_controller)
    return app, viewer