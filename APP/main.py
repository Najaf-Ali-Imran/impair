"""
IMPAIR - Sign Language Translation App
Main Entry Point
"""

import sys
from PyQt6.QtWidgets import QApplication, QStackedWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from splash_screen import SplashScreen
from translator_screen import TranslatorScreen


class ImpairApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMPAIR - Sign Language Translator")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #0D0A09;")
        
        # Create screens
        self.splash = SplashScreen()
        self.translator = TranslatorScreen()
        
        # Add screens to stack
        self.addWidget(self.splash)
        self.addWidget(self.translator)
        
        # Start with splash screen
        self.setCurrentIndex(0)
        
        # Transition to main screen after 3 seconds
        QTimer.singleShot(3000, self.show_translator)
    
    def show_translator(self):
        self.setCurrentIndex(1)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ImpairApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
