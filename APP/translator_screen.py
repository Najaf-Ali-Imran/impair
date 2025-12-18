"""
IMPAIR - Main Translator Screen
Matches the design screenshot exactly
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect, QProgressBar
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon
from styles import COLORS, RADIUS


class TranslatorScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        self.current_mode = "Phrase"
        self.is_recording = False
        self.elapsed_time = 0
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)
        
        # Header
        main_layout.addWidget(self.create_header())
        
        # Title and Mode Toggle Row
        title_row = QHBoxLayout()
        title_row.addWidget(self.create_title_section(), 1)
        title_row.addWidget(self.create_mode_toggle())
        main_layout.addLayout(title_row)
        
        # Main Content Area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # Left: Video Feed
        content_layout.addWidget(self.create_video_section(), 2)
        
        # Right: Translation Output
        content_layout.addWidget(self.create_translation_section(), 1)
        
        main_layout.addLayout(content_layout, 1)
        
        # Footer Tips
        main_layout.addWidget(self.create_footer())
    
    def create_header(self):
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        # Logo and App Name
        logo_section = QHBoxLayout()
        logo_section.setSpacing(12)
        
        logo_icon = QLabel()
        logo_icon.setFixedSize(48, 48)
        pixmap = QPixmap("assets/logo.png")
        if not pixmap.isNull():
            logo_icon.setPixmap(pixmap.scaled(
                55, 55,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            logo_icon.setStyleSheet(f"""
                background-color: {COLORS['primary']};
                border-radius: 12px;
            """)
        logo_section.addWidget(logo_icon)
        
        app_name = QLabel("Impair")
        app_name.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        app_name.setStyleSheet(f"color: {COLORS['text']};")
        logo_section.addWidget(app_name)
        logo_section.addStretch()
        
        header_layout.addLayout(logo_section, 1)
        
        # Right side buttons
        right_buttons = QHBoxLayout()
        right_buttons.setSpacing(15)
        
        # History button
        history_btn = QPushButton("History")
        history_btn.setFont(QFont("Segoe UI", 11))
        history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        history_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface_light']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['medium']}px;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        self.add_icon_to_button(history_btn, "assets/history_icon.png")
        right_buttons.addWidget(history_btn)
        
        # Fullscreen button
        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setFont(QFont("Segoe UI", 16))
        self.fullscreen_btn.setFixedSize(40, 40)
        self.fullscreen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface_light']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['medium']}px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        right_buttons.addWidget(self.fullscreen_btn)
        
        header_layout.addLayout(right_buttons)
        
        return header
    
    def create_title_section(self):
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        
        title = QLabel("Sign Language Translator")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Real-time AI translation powered by computer vision")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet(f"color: {COLORS['text_muted']};")
        title_layout.addWidget(subtitle)
        
        return title_widget
    
    def create_mode_toggle(self):
        toggle_widget = QWidget()
        toggle_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['surface']};
                border-radius: 24px;
            }}
        """)
        toggle_layout = QHBoxLayout(toggle_widget)
        toggle_layout.setContentsMargins(6, 6, 6, 6)
        toggle_layout.setSpacing(4)
        
        modes = ["Letters", "Phrase", "Full Sentences"]
        self.mode_buttons = []
        
        for mode in modes:
            btn = QPushButton(mode)
            btn.setFont(QFont("Segoe UI", 11))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setChecked(mode == self.current_mode)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode(m))
            
            self.update_mode_button_style(btn, mode == self.current_mode)
            self.mode_buttons.append((btn, mode))
            toggle_layout.addWidget(btn)
        
        return toggle_widget
    
    def update_mode_button_style(self, btn, is_active):
        if is_active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: {COLORS['text']};
                    border: none;
                    border-radius: 20px;
                    padding: 10px 24px;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {COLORS['text']};
                    border: none;
                    border-radius: 20px;
                    padding: 10px 24px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['surface_light']};
                }}
            """)
    
    def set_mode(self, mode):
        self.current_mode = mode
        for btn, btn_mode in self.mode_buttons:
            btn.setChecked(btn_mode == mode)
            self.update_mode_button_style(btn, btn_mode == mode)
    
    def create_video_section(self):
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(12)
        
        # Video Frame
        self.video_frame = QFrame()
        self.video_frame.setMinimumHeight(450)
        self.video_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 2px dashed {COLORS['border']};
                border-radius: {RADIUS['xl']}px;
            }}
        """)
        
        video_inner_layout = QVBoxLayout(self.video_frame)
        video_inner_layout.setContentsMargins(20, 15, 20, 15)
        
        # Top bar with LIVE indicator
        top_bar = QHBoxLayout()
        
        live_indicator = QWidget()
        live_indicator.setStyleSheet("background-color: transparent;")
        live_layout = QHBoxLayout(live_indicator)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(8)
        
        # Red dot
        red_dot = QLabel("●")
        red_dot.setStyleSheet(f"color: {COLORS['primary']}; font-size: 22px; background: transparent;")
        live_layout.addWidget(red_dot)
        
        live_text = QLabel("LIVE FEED")
        live_text.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        live_text.setStyleSheet(f"color: {COLORS['text']}; background: transparent;")
        live_layout.addWidget(live_text)
        
        top_bar.addWidget(live_indicator)
        top_bar.addStretch()
        video_inner_layout.addLayout(top_bar)
        
        # Center - Camera placeholder
        video_inner_layout.addStretch()
        
        self.camera_placeholder = QLabel()
        self.camera_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_placeholder.setStyleSheet(f"color: {COLORS['text_muted']}; background: transparent; border: none;")
        pixmap = QPixmap("assets/no_camera.png")
        if not pixmap.isNull():
            self.camera_placeholder.setPixmap(pixmap.scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.camera_placeholder.setText("Camera Feed")
            self.camera_placeholder.setFont(QFont("Segoe UI", 14))
        video_inner_layout.addWidget(self.camera_placeholder)
        
        video_inner_layout.addStretch()
        
        # Bottom controls
        bottom_bar = QHBoxLayout()
        
        # Left controls (mute, camera)
        left_controls = QHBoxLayout()
        left_controls.setSpacing(10)
        
        for icon_name in ["no_hand.png", "hand_detected.png"]:
            ctrl_btn = QPushButton()
            ctrl_btn.setFixedSize(44, 44)
            ctrl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ctrl_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(0, 0, 0, 0.6);
                    border: none;
                    border-radius: 22px;
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 0, 0, 0.8);
                }}
            """)
            self.set_button_icon(ctrl_btn, f"assets/{icon_name}", 20)
            left_controls.addWidget(ctrl_btn)
        
        bottom_bar.addLayout(left_controls)
        bottom_bar.addStretch()
        
        # Stop/Record button
        self.record_btn = QPushButton()
        self.record_btn.setFixedSize(56, 56)
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.update_record_button()
        bottom_bar.addWidget(self.record_btn)
        
        video_inner_layout.addLayout(bottom_bar)
        video_layout.addWidget(self.video_frame)
        
        # Video info bar
        info_bar = QHBoxLayout()
        
        # Left info
        left_info = QHBoxLayout()
        left_info.setSpacing(20)
        
        # Resolution
        res_label = QLabel("  HD 1080p")
        res_label.setFont(QFont("Segoe UI", 11))
        res_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        left_info.addWidget(res_label)
        
        # Timer
        self.timer_label = QLabel("  00:00:00")
        self.timer_label.setFont(QFont("Segoe UI", 11))
        self.timer_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        left_info.addWidget(self.timer_label)
        
        info_bar.addLayout(left_info)
        info_bar.addStretch()
        
        # Camera settings link
        settings_link = QPushButton("Camera Settings")
        settings_link.setFont(QFont("Segoe UI", 11))
        settings_link.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_link.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: none;
                padding: 5px;
            }}
            QPushButton:hover {{
                text-decoration: underline;
            }}
        """)
        info_bar.addWidget(settings_link)
        
        video_layout.addLayout(info_bar)
        
        return video_container
    
    def update_record_button(self):
        if self.is_recording:
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    border: none;
                    border-radius: 28px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary_hover']};
                }}
            """)
            self.record_btn.setText("■")
            self.record_btn.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        else:
            icon = QIcon("assets/rec-button.png")
            if not icon.isNull():
                self.record_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        border: none;
                        border-radius: 28px;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(255, 255, 255, 0.05);
                    }}
                """)
                self.record_btn.setIcon(icon)
                self.record_btn.setIconSize(QSize(56, 56))
                self.record_btn.setText("")
            else:
                self.record_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS['primary']};
                        border: none;
                        border-radius: 28px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLORS['primary_hover']};
                    }}
                """)
                self.record_btn.setText("●")
                self.record_btn.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
    
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.update_record_button()
        if self.is_recording:
            self.elapsed_time = 0
            self.recording_timer.start(1000)
        else:
            self.recording_timer.stop()
    
    def setup_timer(self):
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.update_timer)
    
    def update_timer(self):
        self.elapsed_time += 1
        hours = self.elapsed_time // 3600
        minutes = (self.elapsed_time % 3600) // 60
        seconds = self.elapsed_time % 60
        self.timer_label.setText(f"  {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def create_translation_section(self):
        translation_container = QFrame()
        translation_container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['xl']}px;
            }}
        """)
        
        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        translation_container.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(translation_container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        
        header_left = QHBoxLayout()
        header_left.setSpacing(10)
        
        # Translation icon
        trans_icon = QLabel("文A")
        trans_icon.setFont(QFont("Segoe UI", 14))
        trans_icon.setStyleSheet(f"color: {COLORS['primary']}; background: transparent;border: none;")
        header_left.addWidget(trans_icon)
        
        header_title = QLabel("Translation Output")
        header_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header_title.setStyleSheet(f"color: {COLORS['text']}; background: transparent; border: none;")
        header_left.addWidget(header_title)
        
        header.addLayout(header_left)
        header.addStretch()
        
        # Copy and speaker buttons
        for icon_path in ["assets/copy_icon.png", "assets/spaeker_icon.png"]:
            action_btn = QPushButton()
            action_btn.setFixedSize(32, 32)
            action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            action_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['surface_light']};
                    border-radius: 16px;
                }}
            """)
            self.set_button_icon(action_btn, icon_path, 18)
            header.addWidget(action_btn)
        
        layout.addLayout(header)
        
        # Translation content
        content_frame = QFrame()
        content_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
                border-radius: {RADIUS['large']}px;
                border: none;
            }}
        """)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(10)
        
        # Previous text (grayed out)
        prev_text = QLabel("Hello, how are you today?")
        prev_text.setFont(QFont("Segoe UI", 12))
        prev_text.setStyleSheet(f"color: {COLORS['text_muted']};")
        prev_text.setWordWrap(True)
        content_layout.addWidget(prev_text)
        
        # Current translation
        current_text = QLabel("I am looking for the")
        current_text.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        current_text.setStyleSheet(f"color: {COLORS['text']};")
        current_text.setWordWrap(True)
        content_layout.addWidget(current_text)
        
        # Highlighted word (green)
        highlight_text = QLabel("train station")
        highlight_text.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        highlight_text.setStyleSheet(f"color: {COLORS['success']};")
        content_layout.addWidget(highlight_text)
        
        # Cursor blink
        cursor = QLabel("▌")
        cursor.setFont(QFont("Segoe UI", 16))
        cursor.setStyleSheet(f"color: {COLORS['primary']};")
        content_layout.addWidget(cursor)
        
        content_layout.addStretch()
        layout.addWidget(content_frame, 1)
        
        # Confidence meter
        confidence_frame = QFrame()
        confidence_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
                border-radius: {RADIUS['medium']}px;
            }}
        """)
        conf_layout = QVBoxLayout(confidence_frame)
        conf_layout.setContentsMargins(15, 12, 15, 12)
        conf_layout.setSpacing(8)
        
        conf_label = QLabel("CONFIDENCE: 94%")
        conf_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        conf_label.setStyleSheet(f"color: {COLORS['text_muted']}; background: transparent;border: none;")
        conf_layout.addWidget(conf_label)
        
        # Progress bar
        progress = QProgressBar()
        progress.setValue(94)
        progress.setTextVisible(False)
        progress.setFixedHeight(6)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['surface_light']};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['success']};
                border-radius: 3px;
            }}
        """)
        conf_layout.addWidget(progress)
        
        layout.addWidget(confidence_frame)
        
        layout.addStretch()
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        # Clear button
        clear_btn = QPushButton("  Clear")
        clear_btn.setFont(QFont("Segoe UI", 12))
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface_light']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['large']}px;
                padding: 12px 30px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        icon = QIcon("assets/clear_icon.png")
        if not icon.isNull():
            clear_btn.setIcon(icon)
            clear_btn.setIconSize(QSize(18, 18))
        btn_layout.addWidget(clear_btn)
        
        # Save button
        save_btn = QPushButton("  Save")
        save_btn.setFont(QFont("Segoe UI", 12))
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text']};
                border: none;
                border-radius: {RADIUS['large']}px;
                padding: 12px 30px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        icon = QIcon("assets/save_icon.png")
        if not icon.isNull():
            save_btn.setIcon(icon)
            save_btn.setIconSize(QSize(18, 18))
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        return translation_container
    
    def create_footer(self):
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 10, 0, 0)
        
        # Tips
        tips = [
            ("1", "Ensure good lighting"),
            ("2", "Keep hands visible"),
            ("3", "Face the camera"),
        ]
        
        tips_layout = QHBoxLayout()
        tips_layout.setSpacing(30)
        
        for num, tip in tips:
            tip_widget = QHBoxLayout()
            tip_widget.setSpacing(10)
            
            num_label = QLabel(num)
            num_label.setFixedSize(24, 24)
            num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            num_label.setStyleSheet(f"""
                background-color: {COLORS['surface_light']};
                color: {COLORS['text']};
                border-radius: 12px;
            """)
            tip_widget.addWidget(num_label)
            
            tip_text = QLabel(tip)
            tip_text.setFont(QFont("Segoe UI", 11))
            tip_text.setStyleSheet(f"color: {COLORS['text_muted']};")
            tip_widget.addWidget(tip_text)
            
            tips_layout.addLayout(tip_widget)
        
        footer_layout.addLayout(tips_layout)
        footer_layout.addStretch()
        
        # Help link
        help_link = QPushButton("?  Need help?")
        help_link.setFont(QFont("Segoe UI", 11))
        help_link.setCursor(Qt.CursorShape.PointingHandCursor)
        help_link.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: none;
            }}
            QPushButton:hover {{
                color: {COLORS['text']};
            }}
        """)
        footer_layout.addWidget(help_link)
        
        return footer
    
    def add_icon_to_button(self, button, icon_path):
        try:
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(16, 16))
        except:
            pass
    
    def set_button_icon(self, button, icon_path, size):
        try:
            icon = QIcon(icon_path)
            button.setIcon(icon)
            button.setIconSize(QSize(size, size))
        except:
            pass
    
    def toggle_fullscreen(self):
        window = self.window()
        if window.isFullScreen():
            window.showNormal()
            self.fullscreen_btn.setText("⛶")
        else:
            window.showFullScreen()
            self.fullscreen_btn.setText("✖")
    
    def toggle_fullscreen(self):
        window = self.window()
        if window.isFullScreen():
            window.showNormal()
            self.fullscreen_btn.setText("⛶")
        else:
            window.showFullScreen()
            self.fullscreen_btn.setText("⛶")
