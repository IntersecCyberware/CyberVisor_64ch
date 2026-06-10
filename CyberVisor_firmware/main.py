import sys
import numpy as np

import pyqtgraph as pg
from scipy.ndimage import uniform_filter1d
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont

from test_code import CyberKinesis_v1p2_64_receiver, BUNDLE_SIZE

# ── Configuration ─────────────────────────────────────────────────────────────
ESP_IP       = "192.168.4.1"
PORT         = 2323
SAMPLE_RATE  = 1000         # Hz
N_CHANNELS   = 16           # how many channels to display (1–32)
WINDOW_S     = 5            # seconds of history to show
REFRESH_MS   = 30           # plot refresh rate in milliseconds (~33 FPS)
COLS         = 4            # max graphs per row before wrapping
SMOOTH_MS    = 200          # display smoothing in ms — 0 to disable

DISPLAY_MODE = "rms"        # "rms"  → RMS envelope, one value per packet
                            # "data" → filtered signal, 10 samples per packet

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COLOR     = "#0f1117"
PANEL_COLOR  = "#1a1d27"
RMS_COLOR    = "#ff4d6d"
DATA_COLOR   = "#00d4ff"
GRID_COLOR   = "#2a2d3a"
TEXT_COLOR   = "#e0e0e0"
BTN_START    = "#00c896"
BTN_STOP     = "#ff4d6d"
BTN_DISABLED = "#2a2d3a"


class EMGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CyberKinesis_v1p2_64")
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        # ── Receiver ──────────────────────────────────────────────────────────
        self.rx = CyberKinesis_v1p2_64_receiver(
            port=PORT,
            sample_rate=SAMPLE_RATE,
        )
        #self.rx.add_vref()
        self.rx.add_spike_removal(window_ms=100.0, threshold=1.0)
        self.rx.add_HPF(cutoff=1.0)
        self.rx.add_LPF(cutoff=200.0)
        #self.rx.add_notch(freq=60.0)

        # Only add RMS to the pipeline in RMS mode.
        # In data mode, process() returns filtered signal samples.
        if DISPLAY_MODE == "rms":
            self.rx.add_rms(window_ms=1000)

        self.streaming = False

        # ── Rolling display buffer ─────────────────────────────────────────────
        # RMS mode  : 1 value per packet  → SAMPLE_RATE / BUNDLE_SIZE updates/s
        # Data mode : BUNDLE_SIZE values per packet → SAMPLE_RATE samples/s
        self.rms_per_sec = SAMPLE_RATE / BUNDLE_SIZE            # used for smoothing calc

        if DISPLAY_MODE == "rms":
            self.buf_size = int(self.rms_per_sec * WINDOW_S)   # e.g. 500
        else:
            self.buf_size = int(SAMPLE_RATE * WINDOW_S)        # e.g. 5000

        self.sig_buf = np.zeros((self.buf_size, N_CHANNELS), dtype=np.float32)
        self.buf_ptr = 0

        # ── UI ────────────────────────────────────────────────────────────────
        self._build_ui()

        # ── Refresh timer ─────────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self._update)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root   = QWidget()
        v_root = QVBoxLayout(root)
        v_root.setContentsMargins(16, 16, 16, 16)
        v_root.setSpacing(12)
        self.setCentralWidget(root)

        mode_str   = "RMS mode" if DISPLAY_MODE == "rms" else "Raw data mode"
        title_color = RMS_COLOR if DISPLAY_MODE == "rms" else DATA_COLOR

        title = QLabel(f"CyberKinesis_v1p2_64 plotter — {mode_str}")
        title.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {title_color}; letter-spacing: 2px;")
        v_root.addWidget(title)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self.btn_start = QPushButton("▶  START")
        self.btn_stop  = QPushButton("■  STOP")

        for btn in (self.btn_start, self.btn_stop):
            btn.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
            btn.setFixedHeight(36)
            btn.setFixedWidth(120)

        self.btn_start.setStyleSheet(self._btn_style(BTN_START))
        self.btn_stop.setStyleSheet(self._btn_style(BTN_STOP, disabled=True))
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        self.lbl_status = QLabel("● IDLE")
        self.lbl_status.setFont(QFont("Courier New", 10))
        self.lbl_status.setStyleSheet(f"color: {TEXT_COLOR};")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.lbl_status)
        ctrl.addStretch()
        v_root.addLayout(ctrl)

        grid_widget = QWidget()
        self.grid   = QGridLayout(grid_widget)
        self.grid.setSpacing(8)
        v_root.addWidget(grid_widget)

        curve_color = RMS_COLOR if DISPLAY_MODE == "rms" else DATA_COLOR
        y_label     = "RMS (µV)" if DISPLAY_MODE == "rms" else "µV"

        self.plots  = []
        self.curves = []

        for ch in range(N_CHANNELS):
            row = ch // COLS
            col = ch  % COLS

            pw = pg.PlotWidget()
            pw.setBackground(PANEL_COLOR)
            pw.setMinimumSize(280, 180)

            for axis in ("left", "bottom"):
                ax = pw.getAxis(axis)
                ax.setTextPen(pg.mkPen(color=TEXT_COLOR))
                ax.setPen(pg.mkPen(color=GRID_COLOR))

            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel("left",   y_label)
            pw.setLabel("bottom", "s")
            pw.setTitle(
                f"<span style='color:{TEXT_COLOR}; font-family:Courier New;'>CH {ch}</span>"
            )

            # Lock x-axis to the display window — prevents horizontal autozoom.
            # Y-axis stays auto-ranging so each channel scales to its own amplitude
            # without jarring jumps from pyqtgraph trying to fit every new sample.
            pw.setXRange(-WINDOW_S, 0, padding=0)
            pw.setMouseEnabled(x=False, y=False)
            pw.enableAutoRange(axis="y", enable=True)

            curve = pw.plot(pen=pg.mkPen(color=curve_color, width=2))

            self.plots.append(pw)
            self.curves.append(curve)
            self.grid.addWidget(pw, row, col)

        n_rows = ((N_CHANNELS - 1) // COLS) + 1
        n_cols = min(COLS, N_CHANNELS)
        self.resize(n_cols * 300 + 40, n_rows * 210 + 120)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_start(self):
        if self.streaming:
            return
        self.rx.start()
        self.streaming = True
        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start.setStyleSheet(self._btn_style(BTN_START, disabled=True))
        self.btn_stop.setStyleSheet(self._btn_style(BTN_STOP))
        self.lbl_status.setText("● STREAMING")
        self.lbl_status.setStyleSheet(f"color: {BTN_START};")

    def _on_stop(self):
        if not self.streaming:
            return
        self.timer.stop()
        self.rx.stop()
        self.streaming = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_start.setStyleSheet(self._btn_style(BTN_START))
        self.btn_stop.setStyleSheet(self._btn_style(BTN_STOP, disabled=True))
        self.lbl_status.setText("● STOPPED")
        self.lbl_status.setStyleSheet(f"color: {BTN_STOP};")

    # ── Plot update ───────────────────────────────────────────────────────────

    def _update(self):
        new_samples = []
        while True:
            try:
                ts, packet = self.rx.get(n_channels=N_CHANNELS, timeout=0.0)
            except Exception:
                break

            result = self.rx.process(packet)

                # === TEMPORARY RAW LOGGER - CHANNEL 7 ONLY ===
            with open("raw_debug.log", "a") as f:
                if DISPLAY_MODE == "data" and len(result) > 0:
                    ch7_value = result[0][7]          # first sample, channel 7
                    f.write(f"TS: {ts} | CH7: {ch7_value:.2f}\n")

            if DISPLAY_MODE == "rms":
                new_samples.append(result.reshape(1, -1))
            else:
                new_samples.append(result)


        if not new_samples:
            return

        new_arr = np.vstack(new_samples)  # shape (n_new, N_CHANNELS)

        # Circular buffer insert
        n_new = len(new_arr)
        start = self.buf_ptr % self.buf_size
        end = start + n_new

        if end <= self.buf_size:
            self.sig_buf[start:end] = new_arr
        else:
            self.sig_buf[start:] = new_arr[:self.buf_size - start]
            self.sig_buf[:end % self.buf_size] = new_arr[self.buf_size - start:]

        self.buf_ptr += n_new

        # Extract ordered view for plotting
        if self.buf_ptr < self.buf_size:
            r = self.sig_buf[:self.buf_ptr]
        else:
            r = np.roll(self.sig_buf, - (self.buf_ptr % self.buf_size), axis=0)

        dt = (BUNDLE_SIZE / SAMPLE_RATE) if DISPLAY_MODE == "rms" else (1.0 / SAMPLE_RATE)
        t = np.linspace(-len(r) * dt, 0, len(r))

        if DISPLAY_MODE == "rms" and SMOOTH_MS > 0:
            n_smooth = max(1, int(SMOOTH_MS * self.rms_per_sec / 1000))
            r = uniform_filter1d(r, size=n_smooth, axis=0, mode='nearest')

        for ch in range(N_CHANNELS):
            self.curves[ch].setData(t, r[:, ch])



    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.streaming:
            self._on_stop()
        event.accept()

    # ── Style helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _btn_style(color: str, disabled: bool = False) -> str:
        bg = BTN_DISABLED if disabled else color
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {"#555" if disabled else "#000"};
                border: none;
                border-radius: 4px;
                font-family: 'Courier New';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color if not disabled else BTN_DISABLED};
            }}
        """


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = EMGPlotter()
    win.show()
    sys.exit(app.exec())