from PyQt6.QtWidgets import QApplication

from ui.OptimizationWindow import OptimizationWindow

def main():
    import sys
    app = QApplication(sys.argv)
    window = OptimizationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()