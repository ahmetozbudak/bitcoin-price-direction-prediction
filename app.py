# -*- coding: utf-8 -*-
"""
BTC Trend Tahmin — masaüstü arayüz
Ahmet Özbudak — bitirme tezi
"""

import sys
import os
import time
import traceback
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QLineEdit, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QTextEdit, QGridLayout, QFrame, QMessageBox,
    QSizePolicy, QScrollArea, QHeaderView, QSpacerItem, QFileDialog,
    QAction,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
from src.modeling.utils import (
    DATA_DIR, load_data, prepare_data, split_data, scale_data, scale_data_fit,
    calc_metrics, probability_threshold_predict,
)
from src.modeling.target_definitions import TARGET_MODES
from src.modeling.model_bundle import save_bundle

DATA_DIR_PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")

DATASETS = {
    "4H — v2 (geliştirilmiş)": os.path.join(DATA_DIR_PROCESSED, "btc_4h_features.csv"),
}


def _artifact_base(main_window, data, model_label, model_type):
    return {
        "feature_cols": list(data["feature_cols"]),
        "scaler": None,
        "model_label": model_label,
        "model_type": model_type,
        "target_meta": dict(data.get("target_meta") or {}),
        "csv_path": getattr(main_window, "current_csv_path", None),
        "dataset_key": getattr(main_window, "dataset_key", "") or "",
        "prepare_kwargs": dict(getattr(main_window, "prepare_kwargs", {}) or {}),
    }

# ──────────────────────────────────────────────────────────────
#  Soft dark theme (dusuk kontrast, yumusak yuzeyler)
# ──────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1b24;
    color: #e8e9ef;
    font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: transparent;
    color: #b4b8cc;
    padding: 4px 2px;
    spacing: 6px;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background-color: #2c2d3a;
    color: #e8e9ef;
}
QMenu {
    background-color: #252631;
    color: #e8e9ef;
    border: 1px solid #3a3b4d;
    border-radius: 8px;
    padding: 6px 0;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
}
QMenu::item:selected {
    background-color: #35364a;
}
QToolTip {
    background-color: #2c2d3a;
    color: #dcdfe8;
    border: 1px solid #45465c;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
}

/* Ust veri / hedef karti */
QFrame#topSurface {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #22232f, stop:1 #1e1f2a);
    border: 1px solid #343548;
    border-radius: 14px;
}
QLabel#fieldLabel {
    color: #a8adbf;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.02em;
}

/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #343548;
    background-color: #22232f;
    border-radius: 12px;
    top: -1px;
    padding: 4px;
}
QTabBar::tab {
    background-color: transparent;
    color: #8f95a8;
    padding: 10px 20px;
    margin-right: 4px;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    min-height: 20px;
}
QTabBar::tab:selected {
    background-color: #2e3040;
    color: #c5c9f5;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background-color: #282a38;
    color: #b8bdd4;
}

/* ── Group Box ── */
QGroupBox {
    background-color: #24252f;
    border: 1px solid #35364a;
    border-radius: 12px;
    margin-top: 22px;
    padding: 22px 18px 16px 18px;
    font-weight: 600;
    color: #c5c9f5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 10px;
    color: #a8adf0;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── Inputs ── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #2a2b38;
    color: #e8e9ef;
    border: 1px solid #3d3e52;
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 20px;
    selection-background-color: #6e76c4;
    selection-color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid #7c84d4;
    background-color: #2f3040;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #3d3e52;
    border-top-right-radius: 8px;
    background-color: #323344;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #3d3e52;
    border-bottom-right-radius: 8px;
    background-color: #323344;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px; height: 8px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px; height: 8px;
}
QComboBox::drop-down {
    border: none;
    padding-right: 10px;
    width: 26px;
}
QComboBox QAbstractItemView {
    background-color: #2a2b38;
    color: #e8e9ef;
    selection-background-color: #6e76c4;
    border: 1px solid #3d3e52;
    border-radius: 8px;
    padding: 4px;
    outline: 0;
}

/* ── Buttons ── */
QPushButton#trainBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #6b74c9, stop:1 #8b93eb);
    color: #fafbff;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
QPushButton#trainBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #7a83d6, stop:1 #9aa2f0);
}
QPushButton#trainBtn:pressed {
    background-color: #5a62b0;
}
QPushButton#trainBtn:disabled {
    background-color: #3a3b4a;
    color: #7a7d8c;
}
QPushButton#resetBtn {
    background-color: #2a2b38;
    color: #a8adbf;
    border: 1px solid #3d3e52;
    border-radius: 10px;
    padding: 12px 22px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton#resetBtn:hover {
    background-color: #323344;
    color: #e8e9ef;
    border-color: #6e76c4;
}
QPushButton#loadBtn {
    background-color: #6b74c9;
    color: #fafbff;
    border: none;
    border-radius: 8px;
    padding: 9px 22px;
    font-weight: 600;
}
QPushButton#loadBtn:hover {
    background-color: #848cdf;
}
QPushButton#saveBtn {
    background-color: #2a2b38;
    color: #e4e6f5;
    border: 1px solid #4a4d66;
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
}
QPushButton#saveBtn:hover {
    background-color: #34354a;
    border-color: #7c84d4;
    color: #ffffff;
}
QPushButton#saveBtn:disabled {
    color: #6b6d7e;
    border-color: #35364a;
    background-color: #23242e;
}

/* ── Progress Bar ── */
QProgressBar {
    background-color: #2a2b38;
    border: none;
    border-radius: 5px;
    max-height: 7px;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6b74c9, stop:1 #a89fd4);
    border-radius: 5px;
}

/* ── Metric Cards ── */
QFrame#metricCard {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #282936, stop:1 #23242f);
    border: 1px solid #38394a;
    border-radius: 12px;
    padding: 12px 8px;
    min-width: 100px;
}
QLabel#metricTitle {
    color: #8f95a8;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
}
QLabel#metricValue {
    color: #f0f1f5;
    font-size: 21px;
    font-weight: 600;
}

/* ── Section Title ── */
QLabel#sectionTitle {
    color: #b4b9ea;
    font-size: 13px;
    font-weight: 600;
    padding-bottom: 4px;
    letter-spacing: 0.02em;
}

/* ── Tables ── */
QTableWidget {
    background-color: #22232f;
    gridline-color: #343548;
    border: 1px solid #343548;
    border-radius: 10px;
    color: #e8e9ef;
    font-size: 13px;
}
QTableWidget::item {
    padding: 10px;
}
QTableWidget::item:selected {
    background-color: #3a3d5a;
}
QHeaderView::section {
    background-color: #2a2b38;
    color: #a8adbf;
    border: none;
    border-bottom: 1px solid #343548;
    padding: 10px;
    font-weight: 600;
    font-size: 12px;
}

/* ── Text Edit ── */
QTextEdit {
    background-color: #22232f;
    color: #dcdfe8;
    border: 1px solid #343548;
    border-radius: 10px;
    font-family: 'Cascadia Code', 'Consolas', 'Segoe UI Mono', monospace;
    font-size: 12px;
    padding: 12px;
    selection-background-color: #6e76c4;
}

/* ── Info Bar ── */
QFrame#infoBar {
    background-color: #22232f;
    border: 1px solid #343548;
    border-radius: 12px;
}
QLabel#infoLabel {
    color: #989daf;
    font-size: 12px;
    line-height: 1.45;
}

/* ── App Header ── */
QLabel#appTitle {
    color: #f4f5f8;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
}
QLabel#appSubtitle {
    color: #8b90a3;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Range Label ── */
QLabel#rangeLabel {
    color: #7a8094;
    font-size: 11px;
    font-style: italic;
}
QLabel#paramLabel {
    color: #c4c8d8;
    font-weight: 500;
    min-width: 130px;
}

/* ── Scroll Area ── */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #45465c;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background-color: #5a5c78;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Check Box ── */
QCheckBox {
    color: #dcdfe8;
    spacing: 10px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #3d3e52;
    background-color: #2a2b38;
}
QCheckBox::indicator:checked {
    background-color: #6b74c9;
    border: 1px solid #7c84d4;
}

/* ── Status Bar ── */
QStatusBar {
    background-color: #16171f;
    color: #7a8094;
    border-top: 1px solid #2c2d3a;
    font-size: 12px;
    padding: 6px 14px;
}
"""


# ──────────────────────────────────────────────────────────────
#  Training Worker Thread
# ──────────────────────────────────────────────────────────────
class TrainWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, train_fn, data, params):
        super().__init__()
        self.train_fn = train_fn
        self.data = data
        self.params = params

    def run(self):
        try:
            t0 = time.time()
            result = self.train_fn(self.data, self.params)
            result["time"] = time.time() - t0
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────
#  Metric Card Widget
# ──────────────────────────────────────────────────────────────
class MetricCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 16, 14, 16)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(value)

    def reset(self):
        self.value_label.setText("--")


# ──────────────────────────────────────────────────────────────
#  Base Model Tab
# ──────────────────────────────────────────────────────────────
class BaseModelTab(QWidget):
    def __init__(self, model_name, main_window, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.main_window = main_window
        self.defaults = {}
        self.param_widgets = {}
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 16)
        content_layout.setSpacing(14)

        # --- Parameters ---
        params_group = QGroupBox("Parametreler")
        self.params_grid = QGridLayout(params_group)
        self.params_grid.setSpacing(10)
        self.params_grid.setColumnMinimumWidth(3, 32)
        self.params_grid.setColumnStretch(1, 1)
        self.params_grid.setColumnStretch(5, 1)
        self.setup_params()
        content_layout.addWidget(params_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.train_btn = QPushButton("Modeli eğit")
        self.train_btn.setObjectName("trainBtn")
        self.train_btn.setMinimumHeight(46)
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.clicked.connect(self.start_training)

        self.reset_btn = QPushButton("Varsayılanlara sıfırla")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setMinimumHeight(46)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self.reset_params)

        btn_layout.addWidget(self.train_btn, stretch=3)
        btn_layout.addWidget(self.reset_btn, stretch=1)
        content_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        content_layout.addWidget(self.progress_bar)

        # --- Results ---
        self.results_group = QGroupBox("Sonuçlar")
        results_layout = QVBoxLayout(self.results_group)
        results_layout.setSpacing(14)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)
        self.card_accuracy = MetricCard("ACCURACY")
        self.card_precision = MetricCard("PRECISION")
        self.card_recall = MetricCard("RECALL")
        self.card_f1 = MetricCard("F1-SCORE")
        self.card_time = MetricCard("SÜRE")

        for card in [self.card_accuracy, self.card_precision,
                     self.card_recall, self.card_f1, self.card_time]:
            cards_layout.addWidget(card)
        results_layout.addLayout(cards_layout)

        detail_layout = QHBoxLayout()
        detail_layout.setSpacing(14)

        cm_box = QVBoxLayout()
        cm_title = QLabel("Karmaşıklık matrisi")
        cm_title.setObjectName("sectionTitle")
        cm_box.addWidget(cm_title)

        self.cm_table = QTableWidget(2, 2)
        self.cm_table.setHorizontalHeaderLabels(
            ["Tahmin: aşağı", "Tahmin: yukarı"])
        self.cm_table.setVerticalHeaderLabels(
            ["Gerçek: aşağı", "Gerçek: yukarı"])
        self.cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.setMinimumHeight(120)
        self.cm_table.setMaximumHeight(160)
        self.cm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cm_table.setSelectionMode(QTableWidget.NoSelection)
        cm_box.addWidget(self.cm_table)
        detail_layout.addLayout(cm_box, stretch=1)

        report_box = QVBoxLayout()
        report_title = QLabel("Sınıflandırma raporu")
        report_title.setObjectName("sectionTitle")
        report_box.addWidget(report_title)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(120)
        self.report_text.setMaximumHeight(160)
        report_box.addWidget(self.report_text)
        detail_layout.addLayout(report_box, stretch=1)

        results_layout.addLayout(detail_layout)
        self.results_group.setVisible(False)
        content_layout.addWidget(self.results_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # ── Parameter helpers ──

    def setup_params(self):
        raise NotImplementedError

    def _add_row(self, row, col, name, label_text, widget, range_text):
        base = col * 4
        lbl = QLabel(label_text)
        lbl.setObjectName("paramLabel")
        rng = QLabel(range_text)
        rng.setObjectName("rangeLabel")

        self.params_grid.addWidget(lbl, row, base)
        self.params_grid.addWidget(widget, row, base + 1)
        self.params_grid.addWidget(rng, row, base + 2)
        self.param_widgets[name] = widget

    def add_int(self, row, col, name, label, default, lo, hi, step=1):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(default)
        w.setMinimumWidth(130)
        self.defaults[name] = default
        self._add_row(row, col, name, label, w, f"({lo} - {hi})")

    def add_float(self, row, col, name, label, default, lo, hi,
                  step=0.01, decimals=3):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(default)
        w.setMinimumWidth(130)
        self.defaults[name] = default
        self._add_row(row, col, name, label, w, f"({lo} - {hi})")

    def add_combo(self, row, col, name, label, options, default):
        w = QComboBox()
        w.addItems(options)
        w.setCurrentText(default)
        w.setMinimumWidth(130)
        self.defaults[name] = default
        self._add_row(row, col, name, label, w, "")

    def add_text(self, row, col, name, label, default, hint=""):
        w = QLineEdit(default)
        w.setMinimumWidth(130)
        self.defaults[name] = default
        self._add_row(row, col, name, label, w, hint)

    def add_bool(self, row, col, name, label, default):
        w = QCheckBox()
        w.setChecked(default)
        self.defaults[name] = default
        self._add_row(row, col, name, label, w, "")

    # ── Read / Reset ──

    def get_params(self):
        p = {}
        for name, w in self.param_widgets.items():
            if isinstance(w, QSpinBox):
                p[name] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                p[name] = w.value()
            elif isinstance(w, QComboBox):
                p[name] = w.currentText()
            elif isinstance(w, QLineEdit):
                p[name] = w.text()
            elif isinstance(w, QCheckBox):
                p[name] = w.isChecked()
        return p

    def reset_params(self):
        for name, default in self.defaults.items():
            w = self.param_widgets[name]
            if isinstance(w, QSpinBox):
                w.setValue(default)
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(default)
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(default))
            elif isinstance(w, QLineEdit):
                w.setText(str(default))
            elif isinstance(w, QCheckBox):
                w.setChecked(default)

    # ── Training ──

    def start_training(self):
        data = self.main_window.get_data()
        if data is None:
            QMessageBox.warning(
                self, "Uyarı",
                "Lütfen önce bir veri seti seçin ve «Veri yükle» düğmesine tıklayın!")
            return

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.main_window.statusBar().showMessage(
            f"{self.model_name} eğitiliyor…")

        params = self.get_params()
        self.worker = TrainWorker(self.train_model, data, params)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def train_model(self, data, params):
        raise NotImplementedError

    def _on_finished(self, result):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results_group.setVisible(True)

        art = result.pop("artifact", None)
        if art is not None:
            self.main_window.set_last_bundle(art)

        self.card_accuracy.set_value(f"{result['accuracy']:.4f}")
        self.card_precision.set_value(f"{result['precision']:.4f}")
        self.card_recall.set_value(f"{result['recall']:.4f}")
        self.card_f1.set_value(f"{result['f1']:.4f}")
        self.card_time.set_value(f"{result['time']:.2f}s")

        cm = result.get("cm")
        if cm is not None:
            for i in range(2):
                for j in range(2):
                    item = QTableWidgetItem(f"{cm[i][j]:,}")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFont(QFont("Segoe UI", 13, QFont.Bold))
                    if i == j:
                        item.setBackground(QColor("#162e23"))
                        item.setForeground(QColor("#4ade80"))
                    else:
                        item.setBackground(QColor("#2e1620"))
                        item.setForeground(QColor("#f87171"))
                    self.cm_table.setItem(i, j, item)

        self.report_text.setPlainText(result.get("report", ""))
        self.main_window.statusBar().showMessage(
            f"{self.model_name} tamamlandı  |  "
            f"Accuracy: {result['accuracy']:.4f}  |  "
            f"F1: {result['f1']:.4f}  |  "
            f"Süre: {result['time']:.2f}s",
            15000,
        )

    def _on_error(self, msg):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
            self, "Eğitim hatası",
            f"{self.model_name} eğitimi sırasında hata oluştu:\n\n{msg}")
        self.main_window.statusBar().showMessage(
            f"{self.model_name} eğitimi başarısız!", 10000)


# ──────────────────────────────────────────────────────────────
#  Individual Model Tabs
# ──────────────────────────────────────────────────────────────
class XGBoostTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("XGBoost", main_window)

    def setup_params(self):
        self.add_int(0, 0, "n_estimators", "n_estimators", 200, 50, 2000, 50)
        self.add_int(0, 1, "max_depth", "max_depth", 6, 1, 20)
        self.add_float(1, 0, "learning_rate", "learning_rate", 0.1, 0.001, 1.0, 0.005, 3)
        self.add_float(1, 1, "subsample", "subsample", 0.8, 0.1, 1.0, 0.05, 2)
        self.add_float(2, 0, "colsample_bytree", "colsample_bytree", 0.8, 0.1, 1.0, 0.05, 2)
        self.add_float(2, 1, "min_child_weight", "min_child_weight", 1.0, 0.1, 20.0, 0.5, 1)
        self.add_float(3, 0, "gamma", "gamma (reg.)", 0.0, 0.0, 10.0, 0.1, 2)
        self.add_float(3, 1, "reg_alpha", "reg_alpha (L1)", 0.0, 0.0, 10.0, 0.1, 2)
        self.add_bool(4, 0, "auto_optimize", "Otomatik optimizasyon", False)
        self.add_bool(4, 1, "use_prob_threshold", "Val. olasılık eşiği", False)

    def train_model(self, data, params):
        from xgboost import XGBClassifier

        if params.get("auto_optimize", False):
            return self._run_optimization(data)

        mw = self.main_window
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        base_art = _artifact_base(mw, data, "XGBoost", "xgboost")

        if params.get("use_prob_threshold", False):
            y_pred, thr_meta, fitted = probability_threshold_predict(
                model,
                data["X_train"],
                data["y_train"],
                data["X_test"],
            )
            result = calc_metrics(
                data["y_test"], y_pred, "XGBoost (olasılık eşiği)")
            result["report"] += (
                f"\n\nDoğrulama tabanlı olasılık eşiği:\n"
                f"  Seçilen eşik (P≥t → yukarı): {thr_meta['threshold']:.4f}\n"
                f"  Doğrulama F1 (bu eşikle): {thr_meta['val_f1_tuned']:.4f}\n"
                f"  Doğrulama örnek sayısı: {thr_meta['val_samples']}\n"
                f"  Not: Test etiketi eşik seçiminde kullanılmadı.\n"
            )
            result["artifact"] = {
                **base_art,
                "model": fitted,
                "prob_threshold": thr_meta["threshold"],
            }
            return result

        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])
        result = calc_metrics(data["y_test"], y_pred, "XGBoost")
        result["artifact"] = {
            **base_art,
            "model": model,
            "prob_threshold": None,
        }
        return result

    def _run_optimization(self, data):
        from xgboost import XGBClassifier
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

        search_space = {
            "n_estimators": [100, 200, 300, 500, 700, 1000],
            "max_depth": [3, 4, 5, 6, 7, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": [0, 0.1, 0.3, 0.5, 1.0, 2.0],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0, 3.0],
        }

        base = XGBClassifier(
            random_state=42, eval_metric="logloss", verbosity=0)
        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            base, search_space, n_iter=40, scoring="f1_weighted",
            cv=cv, random_state=42, n_jobs=-1, verbose=0)
        search.fit(data["X_train"], data["y_train"])

        y_pred = search.predict(data["X_test"])
        result = calc_metrics(data["y_test"], y_pred, "XGBoost (Optimized)")

        bp = search.best_params_
        extra = (
            f"\n\nOtomatik optimizasyon sonuçları:\n"
            f"  CV F1 (en iyi): {search.best_score_:.4f}\n"
            f"  En iyi parametreler:\n"
        )
        for k, v in sorted(bp.items()):
            extra += f"    {k}: {v}\n"
        result["report"] += extra
        result["artifact"] = {
            **_artifact_base(self.main_window, data, "XGBoost (Optimized)", "xgboost"),
            "model": search.best_estimator_,
            "prob_threshold": None,
        }
        return result


class RandomForestTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("Random Forest", main_window)

    def setup_params(self):
        self.add_int(0, 0, "n_estimators", "n_estimators", 200, 50, 2000, 50)
        self.add_int(0, 1, "max_depth", "max_depth", 10, 1, 100)
        self.add_int(1, 0, "min_samples_split", "min_samples_split", 10, 2, 50)
        self.add_int(1, 1, "min_samples_leaf", "min_samples_leaf", 1, 1, 50)
        self.add_combo(2, 0, "criterion", "criterion",
                       ["gini", "entropy", "log_loss"], "gini")
        self.add_int(2, 1, "max_features_pct", "max_features (%)",
                     100, 10, 100, 5)
        self.add_bool(3, 0, "auto_optimize", "Otomatik optimizasyon", False)
        self.add_bool(3, 1, "use_prob_threshold", "Val. olasılık eşiği", False)

    def train_model(self, data, params):
        from sklearn.ensemble import RandomForestClassifier

        if params.get("auto_optimize", False):
            return self._run_optimization(data)

        max_feat = params["max_features_pct"] / 100.0
        if max_feat >= 1.0:
            max_feat = None

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            criterion=params["criterion"],
            max_features=max_feat,
            random_state=42,
            n_jobs=-1,
        )

        if params.get("use_prob_threshold", False):
            y_pred, thr_meta, fitted = probability_threshold_predict(
                model,
                data["X_train"],
                data["y_train"],
                data["X_test"],
            )
            result = calc_metrics(
                data["y_test"], y_pred, "RF (olasılık eşiği)")
            result["report"] += (
                f"\n\nDoğrulama tabanlı olasılık eşiği:\n"
                f"  Seçilen eşik (P≥t → yukarı): {thr_meta['threshold']:.4f}\n"
                f"  Doğrulama F1 (bu eşikle): {thr_meta['val_f1_tuned']:.4f}\n"
                f"  Doğrulama örnek sayısı: {thr_meta['val_samples']}\n"
            )
            result["artifact"] = {
                **_artifact_base(self.main_window, data, "Random Forest", "random_forest"),
                "model": fitted,
                "prob_threshold": thr_meta["threshold"],
            }
            return result

        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])
        result = calc_metrics(data["y_test"], y_pred, "Random Forest")
        result["artifact"] = {
            **_artifact_base(self.main_window, data, "Random Forest", "random_forest"),
            "model": model,
            "prob_threshold": None,
        }
        return result

    def _run_optimization(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

        search_space = {
            "n_estimators": [100, 200, 300, 500, 700, 1000],
            "max_depth": [5, 7, 10, 15, 20, 30, 50, None],
            "min_samples_split": [2, 5, 10, 15, 20, 30],
            "min_samples_leaf": [1, 2, 4, 8, 12],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, None],
        }

        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            base, search_space, n_iter=40, scoring="f1_weighted",
            cv=cv, random_state=42, n_jobs=1, verbose=0)
        search.fit(data["X_train"], data["y_train"])

        y_pred = search.predict(data["X_test"])
        result = calc_metrics(data["y_test"], y_pred, "RF (Optimized)")

        bp = search.best_params_
        extra = (
            f"\n\nOtomatik optimizasyon sonuçları:\n"
            f"  CV F1 (en iyi): {search.best_score_:.4f}\n"
            f"  En iyi parametreler:\n"
        )
        for k, v in sorted(bp.items()):
            extra += f"    {k}: {v}\n"
        result["report"] += extra
        result["artifact"] = {
            **_artifact_base(self.main_window, data, "RF (Optimized)", "random_forest"),
            "model": search.best_estimator_,
            "prob_threshold": None,
        }
        return result


class SVMTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("SVM", main_window)

    def setup_params(self):
        self.add_combo(0, 0, "kernel", "kernel",
                       ["rbf", "linear", "poly", "sigmoid"], "rbf")
        self.add_float(0, 1, "C", "C (Regularization)", 1.0, 0.001, 100.0, 0.1, 3)
        self.add_combo(1, 0, "gamma", "gamma",
                       ["scale", "auto"], "scale")
        self.add_int(1, 1, "max_iter", "max_iter", 5000, 500, 50000, 500)

    def train_model(self, data, params):
        from sklearn.svm import SVC
        X_train_s, X_test_s = scale_data(data["X_train"], data["X_test"])
        model = SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            random_state=42,
            max_iter=params["max_iter"],
        )
        model.fit(X_train_s, data["y_train"])
        y_pred = model.predict(X_test_s)
        return calc_metrics(data["y_test"], y_pred, "SVM")


class MLPTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("MLP", main_window)

    def setup_params(self):
        self.add_text(0, 0, "hidden_layer_sizes", "hidden_layer_sizes",
                      "128, 64, 32", "Örnek: 128, 64, 32")
        self.add_combo(0, 1, "activation", "activation",
                       ["relu", "tanh", "logistic", "identity"], "relu")
        self.add_combo(1, 0, "solver", "solver",
                       ["adam", "sgd", "lbfgs"], "adam")
        self.add_int(1, 1, "max_iter", "max_iter", 300, 50, 2000, 50)
        self.add_float(2, 0, "learning_rate_init", "learning_rate_init",
                       0.001, 0.0001, 0.1, 0.0001, 4)
        self.add_float(2, 1, "validation_fraction", "validation_fraction",
                       0.1, 0.05, 0.3, 0.01, 2)
        self.add_bool(3, 0, "early_stopping", "early_stopping", True)
        self.add_bool(3, 1, "auto_optimize", "Otomatik optimizasyon", False)
        self.add_bool(4, 0, "use_prob_threshold", "Val. olasılık eşiği", False)

    def train_model(self, data, params):
        from sklearn.neural_network import MLPClassifier

        if params.get("auto_optimize", False):
            return self._run_optimization(data)


        try:
            hls = tuple(
                int(x.strip()) for x in params["hidden_layer_sizes"].split(",")
            )
        except ValueError:
            hls = (128, 64, 32)

        X_train_s, X_test_s, scaler = scale_data_fit(
            data["X_train"], data["X_test"])
        model = MLPClassifier(
            hidden_layer_sizes=hls,
            activation=params["activation"],
            solver=params["solver"],
            max_iter=params["max_iter"],
            learning_rate_init=params["learning_rate_init"],
            early_stopping=params["early_stopping"],
            validation_fraction=params["validation_fraction"],
            random_state=42,
            verbose=False,
        )
        base_art = _artifact_base(self.main_window, data, "MLP", "mlp")

        if params.get("use_prob_threshold", False):
            y_pred, thr_meta, fitted = probability_threshold_predict(
                model,
                X_train_s,
                data["y_train"],
                X_test_s,
            )
            result = calc_metrics(
                data["y_test"], y_pred, "MLP (olasılık eşiği)")
            result["report"] += (
                f"\n\nDoğrulama tabanlı olasılık eşiği:\n"
                f"  Seçilen eşik (P≥t → yukarı): {thr_meta['threshold']:.4f}\n"
                f"  Doğrulama F1 (bu eşikle): {thr_meta['val_f1_tuned']:.4f}\n"
                f"  Doğrulama örnek sayısı: {thr_meta['val_samples']}\n"
            )
            art = {**base_art, "model": fitted, "prob_threshold": thr_meta["threshold"]}
            art["scaler"] = scaler
            result["artifact"] = art
            return result

        model.fit(X_train_s, data["y_train"])
        y_pred = model.predict(X_test_s)
        result = calc_metrics(data["y_test"], y_pred, "MLP")
        result["artifact"] = {
            **base_art,
            "model": model,
            "scaler": scaler,
            "prob_threshold": None,
        }
        return result

    def _run_optimization(self, data):
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        search_space = {
            "mlp__hidden_layer_sizes": [
                (64, 32), (128, 64), (128, 64, 32),
                (256, 128), (256, 128, 64), (64, 32, 16),
            ],
            "mlp__activation": ["relu", "tanh"],
            "mlp__learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "mlp__alpha": [0.0001, 0.001, 0.01, 0.1],
        }

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                solver="adam", max_iter=400,
                early_stopping=True, validation_fraction=0.1,
                random_state=42, verbose=False)),
        ])
        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            pipe, search_space, n_iter=25, scoring="f1_weighted",
            cv=cv, random_state=42, n_jobs=-1, verbose=0)
        search.fit(data["X_train"], data["y_train"])

        y_pred = search.predict(data["X_test"])
        result = calc_metrics(data["y_test"], y_pred, "MLP (Optimized)")

        bp = search.best_params_
        extra = (
            f"\n\nOtomatik optimizasyon sonuçları:\n"
            f"  CV F1 (en iyi): {search.best_score_:.4f}\n"
            f"  En iyi parametreler:\n"
        )
        for k, v in sorted(bp.items()):
            extra += f"    {k.replace('mlp__', '')}: {v}\n"
        result["report"] += extra
        result["artifact"] = {
            **_artifact_base(self.main_window, data, "MLP (Optimized)", "mlp"),
            "model": search.best_estimator_,
            "scaler": None,
            "prob_threshold": None,
        }
        return result


class MovingAverageTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("Moving Average", main_window)

    def setup_params(self):
        self.add_int(0, 0, "sma_short", "SMA kısa periyot", 10, 3, 100)
        self.add_int(0, 1, "sma_long", "SMA uzun periyot", 50, 10, 500, 5)

    def train_model(self, data, params):
        close_series = pd.Series(data["close_prices"])
        sma_s = close_series.rolling(window=params["sma_short"]).mean()
        sma_l = close_series.rolling(window=params["sma_long"]).mean()

        signals = (sma_s > sma_l).astype(int)
        y_pred = signals.iloc[data["split_idx"]:].values
        y_test = data["y_test"]

        n = min(len(y_pred), len(y_test))
        return calc_metrics(y_test[:n], y_pred[:n], "Moving Average")


class ARIMATab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("ARIMA", main_window)

    def setup_params(self):
        self.add_int(0, 0, "p", "p  (AR Order)", 5, 0, 15)
        self.add_int(0, 1, "d", "d  (Differencing)", 1, 0, 2)
        self.add_int(1, 0, "q", "q  (MA Order)", 0, 0, 15)
        self.add_int(1, 1, "max_train", "Maks. eğitim verisi", 5000, 500, 20000, 500)

    def train_model(self, data, params):
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        warnings.filterwarnings("ignore")

        split_idx = data["split_idx"]
        close_prices = data["close_prices"]
        y_test = data["y_test"]

        train_close = close_prices[:split_idx]
        test_close = close_prices[split_idx:]
        n_test = len(y_test)

        if len(train_close) > params["max_train"]:
            train_close = train_close[-params["max_train"]:]

        model = ARIMA(train_close, order=(params["p"], params["d"], params["q"]))
        fitted = model.fit()
        forecast = fitted.forecast(steps=n_test)

        prev = np.concatenate([[train_close[-1]], test_close[:-1]])[:n_test]
        y_pred = (forecast > prev).astype(int)

        n = min(len(y_pred), n_test)
        warnings.filterwarnings("default")
        return calc_metrics(y_test[:n], y_pred[:n], "ARIMA")


# ──────────────────────────────────────────────────────────────
#  Hybrid Model Tabs
# ──────────────────────────────────────────────────────────────
class HybridFeatureAugTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("Hybrid: Feature Augmentation", main_window)

    def setup_params(self):
        self.add_combo(0, 0, "ml_model", "ML modeli",
                       ["XGBoost", "Random Forest", "MLP"], "XGBoost")
        self.add_int(0, 1, "n_estimators", "n_estimators", 200, 50, 2000, 50)
        self.add_int(1, 0, "sma_short", "SMA kısa periyot", 10, 3, 50)
        self.add_int(1, 1, "sma_long", "SMA uzun periyot", 50, 10, 200)
        self.add_int(2, 0, "max_depth", "max_depth", 6, 1, 20)
        self.add_float(2, 1, "learning_rate", "learning_rate",
                       0.1, 0.001, 1.0, 0.005, 3)
        self.add_int(3, 0, "max_iter", "max_iter (MLP)", 300, 50, 2000, 50)

    def train_model(self, data, params):
        from src.modeling.hybrid_models import run_feature_augmentation
        return run_feature_augmentation(data, params)


class HybridVotingTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("Hybrid: Weighted Voting", main_window)

    def setup_params(self):
        self.add_bool(0, 0, "use_xgboost", "XGBoost", True)
        self.add_bool(0, 1, "use_rf", "Random Forest", True)
        self.add_bool(1, 0, "use_mlp", "MLP", True)
        self.add_bool(1, 1, "use_svm", "SVM (yavaş)", False)
        self.add_bool(2, 0, "use_ma", "Moving Average", True)
        self.add_bool(2, 1, "use_arima", "ARIMA (yavaş)", False)
        self.add_float(3, 0, "val_ratio", "Validasyon oranı",
                       0.15, 0.05, 0.30, 0.01, 2)

    def train_model(self, data, params):
        from src.modeling.hybrid_models import run_weighted_voting
        return run_weighted_voting(data, params)


class HybridStackingTab(BaseModelTab):
    def __init__(self, main_window):
        super().__init__("Hybrid: Stacking", main_window)

    def setup_params(self):
        self.add_int(0, 0, "n_folds", "CV fold sayısı", 5, 3, 10)
        self.add_combo(0, 1, "meta_learner", "Üst model",
                       ["Logistic Regression", "Ridge Classifier"],
                       "Logistic Regression")
        self.add_bool(1, 0, "use_xgboost", "XGBoost", True)
        self.add_bool(1, 1, "use_rf", "Random Forest", True)
        self.add_bool(2, 0, "use_mlp", "MLP", True)
        self.add_bool(2, 1, "use_ma", "Moving Average", True)

    def train_model(self, data, params):
        from src.modeling.hybrid_models import run_stacking
        return run_stacking(data, params)


# ──────────────────────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC Trend Tahmin")
        self.setMinimumSize(1020, 720)
        self.resize(1140, 820)
        self.data = None
        self.last_bundle = None
        self.current_csv_path = None
        self.dataset_key = ""
        self.prepare_kwargs = {}
        self._build_ui()

    def set_last_bundle(self, bundle):
        """Son eğitilen XGB/RF/MLP paketi — üstteki veya Dosya menüsünden kaydedilir."""
        self.last_bundle = bundle
        if hasattr(self, "save_ml_btn"):
            self.save_ml_btn.setEnabled(True)
        self.statusBar().showMessage(
            "ML modeli hazır. «ML modelini kaydet» ile .joblib olarak saklayın; "
            "tahmin için tahmin_uygulamasi.py programını kullanın.",
            12000,
        )

    def _action_save_ml_bundle(self):
        if self.last_bundle is None:
            QMessageBox.warning(
                self, "Uyarı",
                "Önce XGBoost, Random Forest veya MLP sekmesinde model eğitin.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "ML modelini kaydet",
            os.path.join(PROJECT_DIR, "saved_models", "btc_ml_model.joblib"),
            "Joblib (*.joblib)",
        )
        if not path:
            return
        try:
            save_bundle(path, self.last_bundle)
            QMessageBox.information(self, "Tamam", f"Kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def _build_ui(self):
        self.setStyleSheet(DARK_STYLE)

        mb = self.menuBar()
        fm = mb.addMenu("Dosya")
        act_save_ml = QAction("ML modelini kaydet (.joblib)…", self)
        act_save_ml.triggered.connect(self._action_save_ml_bundle)
        fm.addAction(act_save_ml)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 18, 24, 14)
        root.setSpacing(12)

        # ── Ust kart: baslik + veri / hedef ──
        top_surface = QFrame()
        top_surface.setObjectName("topSurface")
        top_lay = QVBoxLayout(top_surface)
        top_lay.setContentsMargins(18, 16, 18, 14)
        top_lay.setSpacing(12)

        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        title = QLabel("BTC Trend Tahmin")
        title.setObjectName("appTitle")
        sub = QLabel("Bitcoin fiyat yönü — hibrit makine öğrenmesi yaklaşımı")
        sub.setObjectName("appSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        header.addLayout(title_col)
        header.addStretch()

        ds_row = QHBoxLayout()
        ds_row.setSpacing(10)
        ds_lbl = QLabel("Veri Seti:")
        ds_lbl.setObjectName("fieldLabel")
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(DATASETS.keys())
        self.ds_combo.setMinimumWidth(180)

        self.load_btn = QPushButton("Veri yükle")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.clicked.connect(self._load_dataset)

        self.save_ml_btn = QPushButton("ML modelini kaydet")
        self.save_ml_btn.setObjectName("saveBtn")
        self.save_ml_btn.setCursor(Qt.PointingHandCursor)
        self.save_ml_btn.setToolTip(
            "Son eğitilen XGBoost / Random Forest / MLP paketini .joblib olarak kaydeder.")
        self.save_ml_btn.clicked.connect(self._action_save_ml_bundle)
        self.save_ml_btn.setEnabled(False)

        ds_row.addWidget(ds_lbl)
        ds_row.addWidget(self.ds_combo)
        ds_row.addWidget(self.load_btn)
        ds_row.addWidget(self.save_ml_btn)
        header.addLayout(ds_row)
        top_lay.addLayout(header)

        # ── Target definition row ──
        tgt_row = QHBoxLayout()
        tgt_row.setSpacing(10)
        tgt_lbl = QLabel("Hedef tanımı:")
        tgt_lbl.setObjectName("fieldLabel")
        self.target_combo = QComboBox()
        self._target_order = [
            ("next_bar", TARGET_MODES["next_bar"]),
            ("threshold_next", TARGET_MODES["threshold_next"]),
            ("horizon_k", TARGET_MODES["horizon_k"]),
        ]
        for key, text in self._target_order:
            self.target_combo.addItem(text, key)
        self.target_combo.setMinimumWidth(320)
        self.target_combo.currentIndexChanged.connect(self._on_target_mode_changed)

        thr_lbl = QLabel("Eşik (%):")
        thr_lbl.setObjectName("fieldLabel")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 3.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.20)
        self.threshold_spin.setToolTip(
            "Hedef B: Sonraki mumun mutlak getirisi bu yüzden küçükse örnek elenir "
            "(örneğin %0,20 = 0,002 oran).")

        h_lbl = QLabel("k (mum):")
        h_lbl.setObjectName("fieldLabel")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(2, 48)
        self.horizon_spin.setValue(3)
        self.horizon_spin.setToolTip(
            "Hedef C: k mum sonra kapanış şimdikinden yukarı mı?")

        tgt_row.addWidget(tgt_lbl)
        tgt_row.addWidget(self.target_combo)
        tgt_row.addWidget(thr_lbl)
        tgt_row.addWidget(self.threshold_spin)
        tgt_row.addWidget(h_lbl)
        tgt_row.addWidget(self.horizon_spin)
        tgt_row.addStretch()
        top_lay.addLayout(tgt_row)
        root.addWidget(top_surface)
        self._on_target_mode_changed()

        # ── Info Bar ──
        self.info_frame = QFrame()
        self.info_frame.setObjectName("infoBar")
        info_lay = QHBoxLayout(self.info_frame)
        info_lay.setContentsMargins(16, 10, 16, 10)
        self.info_label = QLabel(
            "Henüz veri seti yüklenmedi. "
            "Lütfen bir veri seti seçip «Veri yükle» düğmesine tıklayın.")
        self.info_label.setObjectName("infoLabel")
        info_lay.addWidget(self.info_label)
        root.addWidget(self.info_frame)

        # ── Tabs ──
        self.tabs = QTabWidget()
        self.tabs.setObjectName("modelTabs")
        self.tabs.setDocumentMode(True)

        self.tab_xgb = XGBoostTab(self)
        self.tab_rf = RandomForestTab(self)
        self.tab_svm = SVMTab(self)
        self.tab_mlp = MLPTab(self)
        self.tab_ma = MovingAverageTab(self)
        self.tab_arima = ARIMATab(self)

        self.tabs.addTab(self.tab_xgb, "  XGBoost  ")
        self.tabs.addTab(self.tab_rf, "  Random Forest  ")
        self.tabs.addTab(self.tab_svm, "  SVM  ")
        self.tabs.addTab(self.tab_mlp, "  MLP  ")
        self.tabs.addTab(self.tab_ma, "  Hareketli ortalama  ")
        self.tabs.addTab(self.tab_arima, "  ARIMA  ")

        self.tab_hfa = HybridFeatureAugTab(self)
        self.tab_hvote = HybridVotingTab(self)
        self.tab_hstack = HybridStackingTab(self)

        self.tabs.addTab(self.tab_hfa, "  Hibrit: özellik artırımı  ")
        self.tabs.addTab(self.tab_hvote, "  Hibrit: ağırlıklı oylama  ")
        self.tabs.addTab(self.tab_hstack, "  Hibrit: istifleme  ")

        root.addWidget(self.tabs)

        self.statusBar().showMessage("Hazır")

    def _on_target_mode_changed(self):
        mode = self.target_combo.currentData()
        self.threshold_spin.setEnabled(mode == "threshold_next")
        self.horizon_spin.setEnabled(mode == "horizon_k")

    # ── Data ──

    def _load_dataset(self):
        key = self.ds_combo.currentText()
        path = DATASETS[key]
        if not os.path.exists(path):
            QMessageBox.critical(
                self, "Hata", f"Veri dosyası bulunamadı:\n{path}")
            return

        self.statusBar().showMessage(f"{key} yükleniyor…")
        self.load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            df = load_data(path)
            mode = self.target_combo.currentData()
            thr = self.threshold_spin.value() / 100.0
            h = self.horizon_spin.value()
            X, y, feat_cols, close, tmeta = prepare_data(
                df,
                target_mode=mode,
                threshold_pct=thr,
                horizon=h,
            )
            if len(X) < 500:
                QMessageBox.warning(
                    self, "Uyarı",
                    f"Hedef tanımı sonrasında çok az örnek kaldı ({len(X)}). "
                    f"Eşik veya veri setini değiştirin.")
            Xtr, Xte, ytr, yte, sidx = split_data(X, y)

            self.data = {
                "X_train": Xtr, "X_test": Xte,
                "y_train": ytr, "y_test": yte,
                "close_prices": close,
                "split_idx": sidx,
                "feature_cols": feat_cols,
                "target_meta": tmeta,
            }

            up = np.mean(yte) * 100
            tdesc = tmeta.get("description", "")
            dropped = tmeta.get("rows_dropped", 0)
            self.info_label.setText(
                f"{key}   |   {tdesc}   |   "
                f"Toplam: {len(X):,} (elenen: {dropped:,})   |   "
                f"Eğitim: {len(Xtr):,}   |   Test: {len(Xte):,}   |   "
                f"Yukarı: %{up:.1f}   |   Aşağı: %{100-up:.1f}   |   "
                f"Öznitelik: {len(feat_cols)}")
            self.info_label.setStyleSheet("color: #b8bdd8; font-weight: 500;")

            self.current_csv_path = path
            self.dataset_key = key
            self.prepare_kwargs = {
                "target_mode": mode,
                "threshold_pct": thr,
                "horizon": h,
            }

            self.statusBar().showMessage(
                f"{key} başarıyla yüklendi!", 5000)
            self.save_ml_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Yükleme hatası", str(e))
            self.statusBar().showMessage("Veri yükleme başarısız!")
        finally:
            self.load_btn.setEnabled(True)

    def get_data(self):
        return self.data


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
