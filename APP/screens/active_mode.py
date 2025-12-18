"""
IMPAIR - Active Mode Screen (Detection/Upload)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsDropShadowEffect, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon, QTransform

from styles import COLORS, RADIUS, MENU_BUTTON_STYLE, TRANSLATION_BOX_STYLE, MODE_BUTTON_STYLE
from components.slide_menu import SlideMenu
from components.video_frame import VideoFrame


class ActiveMode(QWidget):
    go_home = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.current_mode = "live"
        self.menu_visible = False
        self.setup_ui()
    
    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Slide menu (hidden by default)
        self.slide_menu = SlideMenu()
        self.slide_menu.home_clicked.connect(self.go_home.emit)
        self.slide_menu.setMaximumHeight(0)
        self.main_layout.addWidget(self.slide_menu)
        
        # Menu toggle button at top
        toggle_container = QWidget()
        toggle_container.setFixedHeight(40)
        toggle_layout = QHBoxLayout(toggle_container)
        toggle_layout.setContentsMargins(0, 5, 0, 5)
        toggle_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.toggle_btn = QPushButton()
        self.toggle_btn.setFixedSize(60, 30)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                border-radius: 15px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_alt']};
            }}
        """)
        self.chevron_icon = QIcon("assets/chevron_down.png")
        if not self.chevron_icon.isNull():
            self.toggle_btn.setIcon(self.chevron_icon)
            self.toggle_btn.setIconSize(QSize(20, 20))
        else:
            self.toggle_btn.setText("▼")
        self.toggle_btn.clicked.connect(self.toggle_menu)
        toggle_layout.addWidget(self.toggle_btn)
        
        self.main_layout.addWidget(toggle_container)
        
        # Content area
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 10, 30, 30)
        content_layout.setSpacing(20)
        
        # Mode title
        self.mode_title = QLabel("Live Detection")
        title_font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        self.mode_title.setFont(title_font)
        self.mode_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_title.setStyleSheet(f"color: {COLORS['primary']};")
        content_layout.addWidget(self.mode_title)
        
        # Video frame area
        self.video_frame = VideoFrame()
        content_layout.addWidget(self.video_frame, stretch=1)
        
        # Translation output box
        translation_container = QWidget()
        trans_layout = QHBoxLayout(translation_container)
        trans_layout.setContentsMargins(50, 0, 50, 0)
        
        self.translation_box = QLabel("Waiting for input...")
        self.translation_box.setStyleSheet(TRANSLATION_BOX_STYLE)
        self.translation_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translation_box.setMinimumHeight(70)
        self.translation_box.setWordWrap(True)
        
        # Shadow for translation box
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.translation_box.setGraphicsEffect(shadow)
        
        trans_layout.addWidget(self.translation_box)
        content_layout.addWidget(translation_container)
        
        # Mode selection buttons (for word modes)
        self.mode_buttons = QWidget()
        mode_btn_layout = QHBoxLayout(self.mode_buttons)
        mode_btn_layout.setSpacing(15)
        mode_btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.letters_btn = QPushButton("Letters")
        self.letters_btn.setStyleSheet(MODE_BUTTON_STYLE)
        self.letters_btn.setCheckable(True)
        self.letters_btn.setChecked(True)
        self.letters_btn.setFixedWidth(120)
        icon = QIcon("assets/letters_mode.png")
        if not icon.isNull():
            self.letters_btn.setIcon(icon)
            self.letters_btn.setIconSize(QSize(28, 28))
        mode_btn_layout.addWidget(self.letters_btn)
        
        self.words5_btn = QPushButton("5 Words")
        self.words5_btn.setStyleSheet(MODE_BUTTON_STYLE)
        self.words5_btn.setCheckable(True)
        self.words5_btn.setFixedWidth(120)
        icon = QIcon("assets/words_5_mode.png")
        if not icon.isNull():
            self.words5_btn.setIcon(icon)
            self.words5_btn.setIconSize(QSize(28, 28))
        mode_btn_layout.addWidget(self.words5_btn)
        
        self.words80_btn = QPushButton("80 Words")
        self.words80_btn.setStyleSheet(MODE_BUTTON_STYLE)
        self.words80_btn.setCheckable(True)
        self.words80_btn.setFixedWidth(120)
        icon = QIcon("assets/words_80_mode.png")
        if not icon.isNull():
            self.words80_btn.setIcon(icon)
            self.words80_btn.setIconSize(QSize(28, 28))
        mode_btn_layout.addWidget(self.words80_btn)
        
        # Connect mode buttons
        self.letters_btn.clicked.connect(lambda: self.select_detection_mode("letters"))
        self.words5_btn.clicked.connect(lambda: self.select_detection_mode("words5"))
        self.words80_btn.clicked.connect(lambda: self.select_detection_mode("words80"))
        
        content_layout.addWidget(self.mode_buttons)
        
        self.main_layout.addWidget(content, stretch=1)
        
        # Animation for menu
        self.menu_animation = QPropertyAnimation(self.slide_menu, b"maximumHeight")
        self.menu_animation.setDuration(300)
        self.menu_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def set_mode(self, mode: str):
        self.current_mode = mode
        
        if mode == "live":
            self.mode_title.setText("Live Detection")
            self.video_frame.set_mode("webcam")
            self.mode_buttons.show()
            self.translation_box.setText("Position your hands in front of the camera...")
        elif mode == "upload":
            self.mode_title.setText("Upload Video")
            self.video_frame.set_mode("upload")
            self.mode_buttons.show()
            self.translation_box.setText("Upload a video to begin translation...")
        elif mode == "about":
            self.mode_title.setText("About IMPAIR")
            self.video_frame.set_mode("about")
            self.mode_buttons.hide()
            self.translation_box.setText("IMPAIR uses AI to translate sign language in real-time")
    
    def toggle_menu(self):
        if self.menu_visible:
            self.menu_animation.setStartValue(150)
            self.menu_animation.setEndValue(0)
            self.menu_visible = False
            # Rotate chevron down
            if not self.chevron_icon.isNull():
                pixmap = self.chevron_icon.pixmap(QSize(20, 20))
                self.toggle_btn.setIcon(QIcon(pixmap))
        else:
            self.menu_animation.setStartValue(0)
            self.menu_animation.setEndValue(150)
            self.menu_visible = True
            # Rotate chevron up (flip vertically)
            if not self.chevron_icon.isNull():
                pixmap = self.chevron_icon.pixmap(QSize(20, 20))
                transform = pixmap.transformed(QTransform().scale(1, -1))
                self.toggle_btn.setIcon(QIcon(transform))
        self.menu_animation.start()
    
    def select_detection_mode(self, mode: str):
        self.letters_btn.setChecked(mode == "letters")
        self.words5_btn.setChecked(mode == "words5")
        self.words80_btn.setChecked(mode == "words80")
