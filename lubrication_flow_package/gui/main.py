"""
Main entry point for the Lubrication Flow Network GUI application.
"""
import sys
from PyQt5.QtWidgets import QApplication
from .app import App

def main():
    """
    Initializes and runs the PyQt5 application.
    """
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()