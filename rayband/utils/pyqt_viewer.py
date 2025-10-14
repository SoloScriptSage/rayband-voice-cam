"""
PyQt-based camera viewer with full Unicode support.
Replace OpenCV window with PyQt for proper Ukrainian text display.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen, QIcon
import logging

logger = logging.getLogger(__name__)


class CameraWidget(QLabel):
    """Custom widget for displaying camera feed with overlays."""
    
    def __init__(self):
        super().__init__()
        self.frame = None
        self.transcript_text = ""
        self.is_recording = False
        self.current_language = "english"
        self.fps = 0
        self.faces = []
        self.hand_results = None
        self.sign_text = ""
        self.sign_confidence = 0.0
        
        # Widget settings
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
    
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
        """Custom paint event for drawing frame and overlays."""
        painter = QPainter(self)
        
        if self.frame is None:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "No camera feed")
            return
        
        # Convert OpenCV frame to QPixmap
        height, width, channel = self.frame.shape
        bytes_per_line = 3 * width
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Center the pixmap
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Calculate scaling factor for overlay positioning
        scale_x = scaled_pixmap.width() / width
        scale_y = scaled_pixmap.height() / height
        
        # Draw transcript overlay (with Ukrainian support!)
        if self.transcript_text:
            self._draw_transcript(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Draw sign language info (with emoji support!)
        if self.sign_text and self.sign_confidence > 0.7:
            self._draw_sign_info(painter, x, y)
        
        # Draw recording indicator
        if self.is_recording:
            self._draw_recording_indicator(painter, x, y)
        
        # Draw status bar
        self._draw_status_bar(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
    
    def _draw_transcript(self, painter, offset_x, offset_y, width, height):
        """Draw transcript text with semi-transparent background."""
        # Setup font with emoji support
        font = QFont("Segoe UI Emoji", 16)  # Windows emoji font
        if font.family() == "":  # Fallback if not available
            font = QFont("Arial", 16)
        font.setBold(True)
        painter.setFont(font)
        
        # Calculate text metrics
        fm = painter.fontMetrics()
        lines = self._wrap_text(self.transcript_text, width - 40, fm)
        
        if not lines:
            return
        
        # Calculate background dimensions
        line_height = fm.height()
        total_height = len(lines) * line_height + 20
        status_bar_height = 40
        
        # Draw semi-transparent background
        bg_rect = painter.viewport()
        bg_y = offset_y + height - total_height - status_bar_height - 10
        painter.fillRect(
            offset_x, bg_y, width, total_height,
            QColor(0, 0, 0, 180)  # Semi-transparent black
        )
        
        # Draw text lines
        painter.setPen(QColor(0, 255, 0))  # Green text
        y = bg_y + 20
        for line in lines:
            painter.drawText(offset_x + 20, y, line)
            y += line_height
    
    def _draw_recording_indicator(self, painter, offset_x, offset_y):
        """Draw REC indicator."""
        # Red circle
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(offset_x + 20, offset_y + 20, 16, 16)
        
        # REC text
        painter.setPen(QColor(255, 0, 0))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(offset_x + 45, offset_y + 33, "REC")
    
    def _draw_sign_info(self, painter, offset_x, offset_y):
        """Draw sign language detection info with emoji support."""
        box_width = 400
        box_height = 60
        box_x = offset_x + 10
        box_y = offset_y + 80
        
        # Draw semi-transparent background
        painter.fillRect(
            box_x, box_y, box_width, box_height,
            QColor(0, 0, 0, 180)
        )
        
        # Draw sign text with emoji support
        painter.setPen(QColor(0, 255, 0))
        font = QFont("Segoe UI Emoji", 14)
        if font.family() == "":
            font = QFont("Arial", 14)
        font.setBold(True)
        painter.setFont(font)
        
        text = f"Sign: {self.sign_text}"
        painter.drawText(box_x + 10, box_y + 30, text)
        
        # Draw confidence bar
        bar_x = box_x + 10
        bar_y = box_y + 40
        bar_width = 200
        bar_height = 10
        
        # Background bar
        painter.fillRect(bar_x, bar_y, bar_width, bar_height, QColor(50, 50, 50))
        
        # Confidence bar
        conf_width = int(bar_width * self.sign_confidence)
        painter.fillRect(bar_x, bar_y, conf_width, bar_height, QColor(0, 255, 0))
    
    def _draw_status_bar(self, painter, offset_x, offset_y, width, height):
        """Draw status bar with FPS and language info."""
        bar_height = 40
        bar_y = offset_y + height - bar_height
        
        # Draw semi-transparent background
        painter.fillRect(
            offset_x, bar_y, width, bar_height,
            QColor(40, 40, 40, 200)
        )
        
        # Left side: FPS and language
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Segoe UI Emoji", 12)  # Emoji support
        if font.family() == "":
            font = QFont("Arial", 12)
        painter.setFont(font)
        status_text = f"FPS: {self.fps} | {self.current_language.upper()}"
        painter.drawText(offset_x + 10, bar_y + 25, status_text)
        
        # Right side: Quit instruction
        quit_text = "Press Q to quit"
        text_width = painter.fontMetrics().horizontalAdvance(quit_text)
        painter.drawText(offset_x + width - text_width - 10, bar_y + 25, quit_text)
    
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
                if len(lines) >= 3:  # Max 3 lines
                    break
        
        if current_line and len(lines) < 3:
            lines.append(current_line)
        
        # Add ellipsis if needed
        if len(words) > len(" ".join(lines).split()) and lines:
            lines[-1] = lines[-1] + "..."
        
        return lines
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Q:
            QApplication.quit()


class PyQtCameraViewer(QMainWindow):
    """Main window for PyQt camera viewer."""
    
    def __init__(self, camera_controller):
        super().__init__()
        self.camera_controller = camera_controller
        
        # Setup window
        self.setWindowTitle("RayBand Voice Camera")
        self.setGeometry(100, 100, 1280, 720)
        
        # Create camera widget
        self.camera_widget = CameraWidget()
        self.setCentralWidget(self.camera_widget)
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS update rate
        
        logger.info("✓ PyQt camera viewer initialized")
    
    def update_frame(self):
        """Update camera frame and overlays."""
        # This will be called by the camera controller
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
        logger.info("Closing PyQt viewer...")
        event.accept()


def create_pyqt_viewer(camera_controller):
    """
    Create and return PyQt viewer instance.
    
    Usage in camera_controller:
        from rayband.utils.pyqt_viewer import create_pyqt_viewer
        
        # In start() method, replace cv2.imshow with:
        app, viewer = create_pyqt_viewer(self)
        viewer.show()
        
        # In main loop:
        viewer.set_frame(frame)
        viewer.set_transcript(current_text)
        viewer.set_recording(self.is_recording)
        viewer.set_language(self.current_language)
        viewer.set_fps(int(self._current_fps))
        viewer.set_sign_info(sign_text, confidence)
        QApplication.processEvents()  # Process Qt events
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    viewer = PyQtCameraViewer(camera_controller)
    return app, viewer