"""
IMPAIR - Splash/Loading Screen
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QFont, QTransform, QPainter
from styles import COLORS


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        self.setup_ui()
        self.start_animations()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        
        # Logo container
        logo_container = QWidget()
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.setSpacing(20)
        
        # Logo icon
        self.logo_icon = QLabel()
        self.logo_icon.setFixedSize(450, 450)
        self.logo_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap("assets/logo.png")
        if not pixmap.isNull():
            self.logo_icon.setPixmap(pixmap.scaled(
                540, 540,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.logo_icon.setStyleSheet(f"background-color: {COLORS['primary']}; border-radius: 300px;")
        
        logo_layout.addWidget(self.logo_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # App name
        self.app_name = QLabel("Impair")
        self.app_name.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        self.app_name.setStyleSheet(f"color: {COLORS['text']};")
        self.app_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(self.app_name)
        
        # Tagline
        self.tagline = QLabel("Sign Language Translator")
        self.tagline.setFont(QFont("Segoe UI", 16))
        self.tagline.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(self.tagline)
        
        layout.addWidget(logo_container)
        
        # Spinner loading animation
        self.spinner = QProgressBar()
        self.spinner.setFixedSize(200, 4)
        self.spinner.setTextVisible(False)
        self.spinner.setRange(0, 0)
        self.spinner.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['surface']};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Loading text
        self.loading_label = QLabel("Loading")
        self.loading_label.setFont(QFont("Segoe UI", 12))
        self.loading_label.setStyleSheet(f"color: {COLORS['primary']};")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
    

    
    def start_animations(self):
        # Fade in effect for logo
        self.opacity_effect = QGraphicsOpacityEffect(self.logo_icon)
        self.logo_icon.setGraphicsEffect(self.opacity_effect)
        
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_anim.start()
