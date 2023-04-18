
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt)
from PySide6.QtWidgets import (QApplication, QLabel, QPlainTextEdit, QPushButton,
                               QScrollArea, QSizePolicy, QTextEdit, QWidget)
from PySide6.QtWidgets import QLabel, QFormLayout, QGroupBox, QVBoxLayout
from PySide6 import QtCore, QtWidgets, QtGui

from k_medians import ClusteriserKMedians
from PIL import Image


class Ui_Form(object):
    def setupUi(self, Form):
        self.clusteriser = ClusteriserKMedians()

        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(749, 568)

        self.cat = QPushButton(Form)
        self.cat.setObjectName(u"cat")
        self.cat.setGeometry(QRect(100, 480, 89, 51))
        self.cat.clicked.connect(self.cat_click)

        self.manhattan_button = QPushButton(Form)
        self.manhattan_button.setObjectName(u"manhattan_button")
        self.manhattan_button.setGeometry(QRect(20, 140, 89, 25))
        self.manhattan_button.clicked.connect(self.manthattan_button_click)

        self.chebyshev_button = QPushButton(Form)
        self.chebyshev_button.setObjectName(u"chebyshev_button")
        self.chebyshev_button.setGeometry(QRect(130, 140, 89, 25))
        self.chebyshev_button.clicked.connect(self.chebyshev_button_click)

        self.solution_area = QScrollArea(Form)
        self.solution_area.setObjectName(u"solution_area")
        self.solution_area.setGeometry(QRect(30, 180, 231, 251))
        self.solution_area.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 229, 249))

        self.solution_area.setWidget(self.scrollAreaWidgetContents)

        self.plots_area = QScrollArea(Form)
        self.plots_area.setObjectName(u"plots_area")
        # self.plots_area.setGeometry(QRect(300, 40, 331, 431))
        self.plots_area.setGeometry(QtCore.QRect(340, 40, 421, 521))
        self.plots_area.setWidgetResizable(True)

        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 329, 429))

        self.plots_area.setWidget(self.scrollAreaWidgetContents_2)

        self.clusters_count = QtWidgets.QLineEdit(Form)
        self.clusters_count.setObjectName(u"clusters_count")
        self.clusters_count.setGeometry(QRect(140, 100, 151, 31))

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 80, 131, 61))

        self.x = QtWidgets.QLineEdit(Form)
        self.x.setObjectName(u"x")
        self.x.setGeometry(QRect(130, 20, 131, 21))

        self.y = QtWidgets.QLineEdit(Form)
        self.y.setObjectName(u"y")
        self.y.setGeometry(QRect(130, 50, 131, 21))

        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(0, -20, 131, 61))

        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(50, 30, 21, 41))

        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(50, 10, 21, 41))

        self.add = QPushButton(Form)
        self.add.setObjectName(u"add")
        self.add.setGeometry(QRect(20, 70, 89, 25))
        self.add.clicked.connect(self.add_point)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.manhattan_button.setText(QCoreApplication.translate("Form", u"\u041c\u0430\u043d\u0445\u044d\u0442\u0442\u0435\u043d", None))
        self.chebyshev_button.setText(QCoreApplication.translate("Form", u"\u0427\u0435\u0431\u044b\u0448\u0435\u0432", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u0427\u0438\u0441\u043b\u043e \u043a\u043b\u0430\u0441\u0441\u0442\u0435\u0440\u043e\u0432:", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b \u0442\u043e\u0447\u043a\u0438", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"y", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"x", None))
        self.add.setText(QCoreApplication.translate("Form", u"\u0434\u043e\u0431\u0430\u0432\u0438\u0442\u044c", None))
        self.cat.setText(QCoreApplication.translate("Form", u"\u043a\u043e\u0442\u0438\u043a", None))
    # retranslateUi

    def add_point(self):
        x = self.x.text()
        y = self.y.text()
        self.clusteriser.add_point(int(x), int(y))
        self.x.clear()
        self.y.clear()

    def show_clusters(self, img_count, output):
        formLayout = QFormLayout()
        groupBox = QGroupBox()

        for i in range(img_count + 1):
            label2 = QLabel()
            label2.setPixmap(QtGui.QPixmap('pics/report' + str(i) + '.png'))
            formLayout.addRow(label2)

        groupBox.setLayout(formLayout)

        self.plots_area.setWidget(groupBox)
        self.plots_area.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(self.plots_area)

        container = QtWidgets.QWidget()
        self.solution_area.setWidget(container)

        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 0, 0)

        label = QtWidgets.QLabel(output)
        lay.addWidget(label)
        lay.addStretch()

    def manthattan_button_click(self):
        cluster_number = self.clusters_count.text()
        img_count, output = self.clusteriser.get_clusters(int(cluster_number), 'manthattan')
        self.clusteriser.clear_clusters()

        self.show_clusters(img_count, output)

    def chebyshev_button_click(self):
        cluster_number = self.clusters_count.text()
        img_count, output = self.clusteriser.get_clusters(int(cluster_number), 'chebyshev')
        self.clusteriser.clear_clusters()

        self.show_clusters(img_count, output)

    def cat_click(self):
        img = Image.open(r'pics/cat.jpg')
        img.show()
