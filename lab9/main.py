import sys
from PySide6.QtWidgets import QApplication, QDialog
from form_ import Ui_Form


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QDialog()
    ui = Ui_Form()
    ui.setupUi(window)

    window.show()
    sys.exit(app.exec())
