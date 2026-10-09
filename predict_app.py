# -*- coding: utf-8 -*-
"""
BTC Trend tahmin uygulaması (ayrı program).
Tek model (.joblib), oylama veya stacking paketi yükleyip CSV üzerinden tahmin.

Çalıştır: python predict_app.py
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIALOG_START = os.path.join(PROJECT_DIR, "data", "processed")

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSplitter,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.modeling.model_bundle import load_bundle
from src.modeling.ensemble_bundle import predict_any_bundle

STYLE = """
QMainWindow {
    background-color: #14151c;
}
QWidget#centralRoot {
    background-color: #14151c;
    color: #e8e9ef;
    font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
    font-size: 13px;
}
/* Üst doygun şerit */
QFrame#topStrip {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4c1d95, stop:0.35 #6366f1, stop:0.7 #7c3aed, stop:1 #5b21b6);
    border: none;
    min-height: 6px;
    max-height: 6px;
}
/* Ana kabuk */
QFrame#mainShell {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1b26, stop:0.45 #16171f, stop:1 #12131a);
    border: 1px solid #35364f;
    border-radius: 18px;
}
/* Kahraman başlık */
QFrame#heroPanel {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #252645, stop:0.5 #1e1f2e, stop:1 #1a1530);
    border: 1px solid #4c4d72;
    border-radius: 16px;
}
QLabel#brandBadge {
    background-color: rgba(99, 102, 241, 0.35);
    color: #e0e7ff;
    border: 1px solid #818cf8;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
}
QLabel#appTitle {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.03em;
}
QLabel#heroHint {
    color: #a5adc4;
    font-size: 12px;
    line-height: 1.5;
}
QFrame#decoGlow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #6366f1, stop:0.5 #7c3aed, stop:1 #4c1d95);
    border-radius: 14px;
    min-width: 100px;
    max-width: 140px;
}
QFrame#decoInner {
    background-color: #151621;
    border-radius: 11px;
    margin: 3px;
}
/* Adım rozeti */
QLabel#stepBadge {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #6366f1, stop:1 #4338ca);
    color: #ffffff;
    border: 1px solid #a5b4fc;
    border-radius: 14px;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    font-weight: 800;
    font-size: 12px;
}
/* Form kartı */
QFrame#formCard {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #232436, stop:1 #1b1c28);
    border: 1px solid #3f4066;
    border-radius: 14px;
}
QLabel#cardCaption {
    color: #c7c9ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
QLineEdit#pathField {
    background-color: #2a2b3d;
    color: #eef0f7;
    border: 1px solid #4b4c6e;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    selection-background-color: #6366f1;
}
QLineEdit#pathField:focus {
    border: 1px solid #8b9cff;
    background-color: #2f3048;
}
QPushButton#sec {
    background-color: #373a5c;
    color: #f1f5f9;
    border: 1px solid #5c5e8a;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    min-width: 72px;
}
QPushButton#sec:hover {
    background-color: #4c4f7a;
    border-color: #8b9cff;
}
QPushButton#go {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #5b21b6, stop:0.45 #6366f1, stop:1 #7c3aed);
    color: #ffffff;
    border: 2px solid #a78bfa;
    border-radius: 14px;
    padding: 14px 22px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.04em;
}
QPushButton#go:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #6d28d9, stop:0.5 #818cf8, stop:1 #8b5cf6);
}
QPushButton#go:pressed {
    background-color: #4c1d95;
}
/* Sağ panel */
QFrame#rightPanel {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1c1d2a, stop:1 #151621);
    border: 1px solid #393a55;
    border-radius: 16px;
}
QLabel#panelTitle {
    color: #c4b5fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
}
QFrame#chipRow {
    background-color: #1e1b3a;
    border: 1px solid #4c1d95;
    border-radius: 14px;
    padding: 4px 0;
}
QLabel#chipMuted {
    background-color: transparent;
    border: none;
    color: #9ca3c9;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 4px;
}
QLabel#predChipWait {
    background-color: #2d2f48;
    color: #e8e9ef;
    border: 1px solid #4b4d6e;
    border-radius: 12px;
    padding: 12px 22px;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.06em;
}
QLabel#predChipUp {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #14532d, stop:1 #166534);
    color: #ecfdf5;
    border: 1px solid #4ade80;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.05em;
}
QLabel#predChipDown {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7f1d1d, stop:1 #991b1b);
    color: #fef2f2;
    border: 1px solid #f87171;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.05em;
}
QLabel#chip {
    background-color: #312e81;
    color: #e0e7ff;
    border: 1px solid #6366f1;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
}
QLabel#chipAccent {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4c1d95, stop:1 #6366f1);
    color: #ffffff;
    border: 1px solid #c4b5fd;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#chipUp {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #14532d, stop:1 #166534);
    color: #dcfce7;
    border: 1px solid #4ade80;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#chipDown {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7f1d1d, stop:1 #991b1b);
    color: #fecaca;
    border: 1px solid #f87171;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
}
QTextEdit#resultConsole {
    background-color: #12131c;
    color: #dce3f0;
    border: 1px solid #2e2f45;
    border-radius: 12px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    padding: 12px;
    selection-background-color: #6366f1;
}
QLabel#footNote {
    color: #6b7190;
    font-size: 11px;
    padding-top: 4px;
}
QSplitter::handle {
    background-color: #2a2b3d;
    width: 4px;
    margin: 2px 6px;
    border-radius: 2px;
}
QSplitter::handle:hover {
    background-color: #6366f1;
}
"""


def _pretty_details(raw: str) -> str:
    """Bileşen satırlarını (oyu=/sinif=) sade Türkçe özetlere çevirir."""
    if not raw or not str(raw).strip():
        return ""
    out = []
    for raw_line in str(raw).strip().split("\n"):
        s = raw_line.strip()
        if not s:
            continue
        if ": oyu=" in s:
            left, right = s.split(": oyu=", 1)
            name = left.strip()
            parts = right.split()
            try:
                oy = int(parts[0])
            except (IndexError, ValueError):
                out.append(f"  • {name}: {right.strip()}")
                continue
            yon = "Yukarı" if oy == 1 else "Aşağı"
            out.append(f"  • {name} → {yon}")
        elif ": sinif=" in s:
            left, right = s.split(": sinif=", 1)
            name = left.strip()
            try:
                cl = int(right.strip()[0])
            except (IndexError, ValueError):
                out.append(f"  • {name}: {right.strip()}")
                continue
            yon = "Yukarı" if cl == 1 else "Aşağı"
            out.append(f"  • {name} → {yon}")
        else:
            out.append(f"  • {s.lstrip('•').strip()}")
    return "\n".join(out)


def _form_card(step_num: str, caption: str, field: QLineEdit, pick_btn: QPushButton) -> QFrame:
    card = QFrame()
    card.setObjectName("formCard")
    outer = QHBoxLayout(card)
    outer.setContentsMargins(14, 12, 14, 12)
    outer.setSpacing(12)

    badge = QLabel(step_num)
    badge.setObjectName("stepBadge")
    badge.setAlignment(Qt.AlignCenter)

    col = QVBoxLayout()
    col.setSpacing(8)
    cap = QLabel(caption)
    cap.setObjectName("cardCaption")
    row = QHBoxLayout()
    row.setSpacing(10)
    row.addWidget(field, stretch=1)
    row.addWidget(pick_btn, stretch=0)
    col.addWidget(cap)
    col.addLayout(row)

    outer.addWidget(badge, alignment=Qt.AlignTop)
    outer.addLayout(col, stretch=1)
    return card


class TahminPenceresi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC Trend — Tahmin")
        self.resize(960, 680)
        self.bundle_path = None
        self.csv_path = None

        root = QWidget()
        root.setObjectName("centralRoot")
        self.setCentralWidget(root)
        main_lay = QVBoxLayout(root)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        strip = QFrame()
        strip.setObjectName("topStrip")
        main_lay.addWidget(strip)

        shell = QFrame()
        shell.setObjectName("mainShell")
        shell_lay = QVBoxLayout(shell)
        shell_lay.setContentsMargins(18, 16, 18, 16)
        shell_lay.setSpacing(14)

        hero = QFrame()
        hero.setObjectName("heroPanel")
        hero_lay = QHBoxLayout(hero)
        hero_lay.setContentsMargins(18, 16, 18, 16)
        hero_lay.setSpacing(16)

        left_hero = QVBoxLayout()
        left_hero.setSpacing(8)
        badge = QLabel("4H · SON MUM")
        badge.setObjectName("brandBadge")
        title = QLabel("Tahmin konsolu")
        title.setObjectName("appTitle")
        hint = QLabel(
            "Eğitim uygulamasından dışa aktardığınız <b>.joblib</b> paketini ve "
            "eğitimle <b>aynı öznitelik sütunlarına</b> sahip <b>CSV</b> dosyasını "
            "«Seç…» ile seçin. Tahmin, dosyanın <b>son satırı</b> (son kapanan mum) üzerindendir."
        )
        hint.setObjectName("heroHint")
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        left_hero.addWidget(badge)
        left_hero.addWidget(title)
        left_hero.addWidget(hint)

        deco = QFrame()
        deco.setObjectName("decoGlow")
        deco_l = QVBoxLayout(deco)
        deco_l.setContentsMargins(0, 0, 0, 0)
        inner = QFrame()
        inner.setObjectName("decoInner")
        inner.setMinimumHeight(72)
        deco_l.addWidget(inner)

        hero_lay.addLayout(left_hero, stretch=1)
        hero_lay.addWidget(deco, alignment=Qt.AlignVCenter)
        shell_lay.addWidget(hero)

        split = QSplitter(Qt.Horizontal)

        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.setContentsMargins(0, 0, 8, 0)
        left_v.setSpacing(12)

        self.ed_bundle = QLineEdit()
        self.ed_bundle.setObjectName("pathField")
        self.ed_bundle.setReadOnly(True)
        self.ed_bundle.setPlaceholderText("Model yolu…")
        b1 = QPushButton("Seç…")
        b1.setObjectName("sec")
        b1.setCursor(Qt.PointingHandCursor)
        b1.clicked.connect(self._pick_bundle)
        card1 = _form_card("1", "MODEL PAKETİ (.JOBLIB)", self.ed_bundle, b1)

        self.ed_csv = QLineEdit()
        self.ed_csv.setObjectName("pathField")
        self.ed_csv.setReadOnly(True)
        self.ed_csv.setPlaceholderText("CSV dosyası seçilmedi…")
        b2 = QPushButton("Seç…")
        b2.setObjectName("sec")
        b2.setCursor(Qt.PointingHandCursor)
        b2.clicked.connect(self._pick_csv)
        card2 = _form_card("2", "VERİ (CSV)", self.ed_csv, b2)

        self.btn = QPushButton("TAHMİN YAP")
        self.btn.setObjectName("go")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn.clicked.connect(self._run)

        left_v.addWidget(card1)
        left_v.addWidget(card2)
        left_v.addWidget(self.btn)
        left_v.addStretch()

        right_w = QFrame()
        right_w.setObjectName("rightPanel")
        right_v = QVBoxLayout(right_w)
        right_v.setContentsMargins(14, 14, 14, 14)
        right_v.setSpacing(10)

        pt = QLabel("Özet sonuç")
        pt.setObjectName("panelTitle")

        chip_wrap = QFrame()
        chip_wrap.setObjectName("chipRow")
        self._chip_lay = QHBoxLayout(chip_wrap)
        self._chip_lay.setSpacing(12)
        self._chip_lay.setContentsMargins(14, 10, 14, 10)

        self.chip_kind = QLabel("Paket: —")
        self.chip_kind.setObjectName("chipMuted")
        self.chip_pred = QLabel("—")
        self.chip_pred.setObjectName("predChipWait")
        self.chip_pred.setAlignment(Qt.AlignCenter)
        self.chip_time = QLabel("Zaman: —")
        self.chip_time.setObjectName("chipMuted")
        self._chip_lay.addWidget(self.chip_kind, stretch=0)
        self._chip_lay.addStretch(1)
        self._chip_lay.addWidget(self.chip_pred, stretch=0)
        self._chip_lay.addStretch(1)
        self._chip_lay.addWidget(self.chip_time, stretch=0)

        self.out = QTextEdit()
        self.out.setObjectName("resultConsole")
        self.out.setReadOnly(True)
        self.out.setPlaceholderText("Ayrıntılı çıktı burada listelenir…")

        right_v.addWidget(pt)
        right_v.addWidget(chip_wrap)
        right_v.addWidget(self.out, stretch=1)

        split.addWidget(left_w)
        split.addWidget(right_w)
        split.setSizes([400, 520])

        shell_lay.addWidget(split, stretch=1)

        foot = QLabel(
            "CSV sütunları, paketin beklediği özelliklerle uyumlu olmalıdır. "
            "Tahmin dosyanın son kapanan mum satırına göredir."
        )
        foot.setObjectName("footNote")
        foot.setWordWrap(True)
        shell_lay.addWidget(foot)

        main_lay.addWidget(shell)

        self.setStyleSheet(STYLE)

    def _reset_chips(self):
        self.chip_kind.setText("Paket: —")
        self.chip_kind.setObjectName("chipMuted")
        self.chip_pred.setText("—")
        self.chip_pred.setObjectName("predChipWait")
        self.chip_time.setText("Zaman: —")
        self.chip_time.setObjectName("chipMuted")
        self._restyle_chips()

    def _restyle_chips(self):
        for w in (self.chip_kind, self.chip_pred, self.chip_time):
            w.style().unpolish(w)
            w.style().polish(w)

    def _pick_bundle(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Model paketi", PROJECT_DIR, "Joblib (*.joblib)")
        if path:
            self.bundle_path = path
            self.ed_bundle.setText(path)

    def _pick_csv(self):
        start = CSV_DIALOG_START if os.path.isdir(CSV_DIALOG_START) else PROJECT_DIR
        path, _ = QFileDialog.getOpenFileName(
            self, "Veri CSV", start, "CSV (*.csv)")
        if path:
            self.csv_path = path
            self.ed_csv.setText(path)

    def _run(self):
        self._reset_chips()
        if not self.bundle_path or not os.path.isfile(self.bundle_path):
            QMessageBox.warning(self, "Uyarı", "Model dosyası seçin.")
            return
        if not self.csv_path or not os.path.isfile(self.csv_path):
            QMessageBox.warning(self, "Uyarı", "CSV dosyası seçin.")
            return
        try:
            bundle = load_bundle(self.bundle_path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))
            return
        try:
            r = predict_any_bundle(bundle, self.csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))
            return
        if not r.get("ok"):
            self.out.setPlainText(r.get("error", "Hata"))
            return

        kind = bundle.get("kind", "single")
        lbl = r.get("label") or ""
        pred = r.get("prediction")
        self.chip_kind.setText(f"Paket: {kind}")
        if pred == 1:
            self.chip_pred.setObjectName("predChipUp")
            self.chip_pred.setText("Yukarı")
        elif pred == 0:
            self.chip_pred.setObjectName("predChipDown")
            self.chip_pred.setText("Aşağı")
        else:
            self.chip_pred.setObjectName("predChipWait")
            self.chip_pred.setText(lbl or "—")
        self.chip_time.setText(f"Zaman: {r.get('open_time')}")
        self._restyle_chips()

        lines = [
            f"Paket            : {kind}",
            f"Son mum          : {r.get('open_time')}",
            f"Kapanış          : {r.get('close')}",
            f"Yön tahmini      : {lbl}",
        ]
        if r.get("meta_type"):
            lines.append(f"Birleştirme      : {r['meta_type']}")
        if r.get("detail_lines"):
            lines.append("")
            lines.append("Modeller:")
            lines.append(_pretty_details(r["detail_lines"]))
        self.out.setPlainText("\n".join(lines))


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
    w = TahminPenceresi()
    w.show()
    sys.exit(app.exec_())
