import logging
import queue
import socket
import struct
import threading

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


CHIP_NUM:          int   = 4
CHANNELS_PER_CHIP: int   = 8
BYTES_PER_CHANNEL: int   = 3
BUNDLE_SIZE:       int   = 10
TS_SIZE:           int   = 4
UV_SCALE:          float = 0.0223517

TOTAL_CHANNELS: int = CHIP_NUM * CHANNELS_PER_CHIP        # 32
DATA_SIZE:      int = TOTAL_CHANNELS * BYTES_PER_CHANNEL  # 96
FRAME_SIZE:     int = DATA_SIZE + TS_SIZE                 # 100
PACKET_SIZE:    int = FRAME_SIZE * BUNDLE_SIZE            # 1000

UDP_IP:   str = "0.0.0.0"
UDP_PORT: int = 2323
ESP_IP:   str = "192.168.4.1"


class CyberKinesis_v1p2_64_receiver:

    def __init__(self, esp_ip:str = ESP_IP, port:int = UDP_PORT, sample_rate:float = 1000.0, uv:bool = True, queue_size:int = 100, recv_buf:int = 4 * 1024 * 1024) -> None:
        self.esp_ip = esp_ip
        self.port = port
        self.sample_rate = sample_rate
        self.uv = uv
        self.base_ts: int | None = None
        self._filters: list[list] = []

        # Virtual reference flag — set by add_vref()
        # Subtracts mean of all channels from each channel before filtering
        self._use_vref: bool = False

        # RMS mode flag — set by add_rms()
        # When True, process() returns RMS values instead of filtered data
        self._use_rms:    bool = False
        self._rms_window: int  = int(sample_rate * 0.2)
        self._rms_buffer: np.ndarray = np.zeros((self._rms_window, TOTAL_CHANNELS), dtype=np.float32)
        self._rms_ptr: int = 0

        self._last_ts:   np.ndarray | None = None
        self._last_data: np.ndarray | None = None
        self._last_rms:  np.ndarray | None = None

        self._sock = self._create_socket(recv_buf)
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="ESP32-Recv")


    def _create_socket(self, recv_buf: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf)
        sock.bind((UDP_IP, self.port))
        return sock

    def start(self) -> None:
        self._sock.sendto(b"STR", (self.esp_ip, self.port))
        self._thread.start()
        log.info("Streaming started from %s:%d", self.esp_ip, self.port)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._sock.sendto(b"STOP", (self.esp_ip, self.port))
        finally:
            self._sock.close()
        self._thread.join(timeout=2.0)
        log.info("Streaming stopped.")

    def __enter__(self) -> "CyberKinesis_v1p2_64_receiver":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def get(self, n_channels: int = TOTAL_CHANNELS, timeout: float = 2.0,) -> tuple[np.ndarray, np.ndarray]:
        n_channels = max(1, min(n_channels, TOTAL_CHANNELS))
        ch_ts, ch_data = self._queue.get(timeout=timeout)
        ch_data = ch_data[:, :n_channels]
        self._last_ts   = ch_ts
        self._last_data = ch_data
        return ch_ts, ch_data

    def add_vref(self) -> "CyberKinesis_v1p2_64_receiver":
        """
        Enable virtual reference subtraction.

        Virtual reference = mean across all channels at each sample.
        Each channel becomes: CH - mean(all channels).
        Removes common-mode noise shared across all electrodes.
        Call this before add_HPF / add_LPF / add_notch.
        """
        self._use_vref = True
        log.info("Virtual reference enabled.")
        return self

    def add_HPF(self, cutoff: float, order: int = 4) -> "CyberKinesis_v1p2_64_receiver":
        sos = butter(order, cutoff, btype="high", fs=self.sample_rate, output="sos")
        self._filters.append(self._init_filter(sos))
        log.info("Added HPF: cutoff=%.1f Hz, order=%d", cutoff, order)
        return self

    def add_LPF(self, cutoff: float, order: int = 4) -> "CyberKinesis_v1p2_64_receiver":
        sos = butter(order, cutoff, btype="low", fs=self.sample_rate, output="sos")
        self._filters.append(self._init_filter(sos))
        log.info("Added LPF: cutoff=%.1f Hz, order=%d", cutoff, order)
        return self

    def add_notch(self, freq: float, q: float = 30.0) -> "CyberKinesis_v1p2_64_receiver":
        b, a = iirnotch(freq, q, fs=self.sample_rate)
        sos  = tf2sos(b, a)
        self._filters.append(self._init_filter(sos))
        log.info("Added notch: freq=%.1f Hz, Q=%.1f", freq, q)
        return self

    def add_rms(self, window_ms: float = 200.0) -> "CyberKinesis_v1p2_64_receiver":
        """
        Enable RMS mode. Must be called last in the pipeline.

        When enabled, process() returns RMS values rounded to the nearest
        whole µV instead of filtered signal data.
        Rounding fixes sub-1 µV floating point noise cycling (0.003, 400, 300...).

        Parameters
        ----------
        window_ms : RMS window in ms. Default 200 ms is standard for EMG.
        """
        self._use_rms    = True
        self._rms_window = max(1, int(self.sample_rate * window_ms / 1000.0))
        self._rms_buffer = np.zeros((self._rms_window, TOTAL_CHANNELS), dtype=np.float32)
        self._rms_ptr    = 0
        log.info("RMS mode enabled: window=%.0f ms (%d samples)", window_ms, self._rms_window)
        return self

    def clear_filters(self) -> None:
        self._filters.clear()
        self._use_vref = False
        self._use_rms  = False
        log.info("Pipeline cleared.")

    def process(self, ch_data: np.ndarray) -> np.ndarray:
        """
        Run the full pipeline on one packet of data.

        Pipeline order:
            1. Virtual reference  (if add_vref() was called)
            2. Filters            (HPF, LPF, notch in order added)
            3. RMS                (if add_rms() was called)

        Returns
        -------
        (BUNDLE_SIZE, n_channels) filtered data  — if add_rms() was NOT called
        (n_channels,) RMS values in whole µV     — if add_rms() WAS called
        """
        # ── Step 1: Virtual reference ─────────────────────────────────────────
        if self._use_vref:
            # mean across channels per sample, shape (BUNDLE_SIZE, 1)
            vref    = ch_data.mean(axis=1, keepdims=True)
            ch_data = ch_data - vref

        # ── Step 2: Filters ───────────────────────────────────────────────────
        if self._filters:
            n_ch = ch_data.shape[1]
            out  = ch_data.T.astype(np.float64)

            for f in self._filters:
                sos, zi           = f
                out, zi_new       = sosfilt(sos, out, zi=zi[:, :n_ch, :])
                f[1][:, :n_ch, :] = zi_new

            ch_data = out.T.astype(np.float32)

        self._last_data = ch_data

        # ── Step 3: RMS ───────────────────────────────────────────────────────
        if self._use_rms:
            n_samples, n_ch = ch_data.shape

            for i in range(n_samples):
                idx = self._rms_ptr % self._rms_window
                self._rms_buffer[idx, :n_ch] = ch_data[i]
                self._rms_ptr += 1

            rms = np.sqrt(np.mean(self._rms_buffer[:, :n_ch] ** 2, axis=0))

            # Round to nearest whole µV — prevents sub-1 values from producing
            # floating point noise that cycles through large numbers on the graph
            rms = np.round(rms).astype(np.float32)
            self._last_rms = rms
            return rms

        return ch_data

    def RMS(self, ch_data: np.ndarray, window_ms: float | None = None,) -> np.ndarray:
        """Standalone RMS — use this if you did NOT call add_rms()."""
        n_samples, n_ch = ch_data.shape

        if window_ms is not None:
            new_size = max(1, int(self.sample_rate * window_ms / 1000.0))
            if new_size != self._rms_window:
                self._rms_window  = new_size
                self._rms_buffer  = np.zeros((self._rms_window, TOTAL_CHANNELS), dtype=np.float32)
                self._rms_ptr = 0

        for i in range(n_samples):
            idx = self._rms_ptr % self._rms_window
            self._rms_buffer[idx, :n_ch] = ch_data[i]
            self._rms_ptr += 1

        rms = np.sqrt(np.mean(self._rms_buffer[:, :n_ch] ** 2, axis=0))
        rms = np.round(rms).astype(np.float32)
        self._last_rms = rms
        return rms

    def print_data(self, data: bool = True, rms: bool = False, decimals: int = 1,) -> None:
        unit = "uV" if self.uv else "raw"

        if data and self._last_data is not None:
            for i, row in enumerate(self._last_data):
                ch_str = " ".join(f"{v:8.{decimals}f}" for v in row)
                print(f"Data({unit}): {self._last_ts[i]:06d}ms | {ch_str}")

        if rms and self._last_rms is not None:
            rms_str = " ".join(f"{v:8.{decimals}f}" for v in self._last_rms)
            print(f"RMS({unit}):  {self._last_ts[-1]:06d}ms | {rms_str}")

        if data or rms:
            print()

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(PACKET_SIZE + 64)
            except OSError:
                break

            if len(data) != PACKET_SIZE:
                log.warning("Dropped packet: received %d bytes, expected %d", len(data), PACKET_SIZE)
                continue

            if self.base_ts is None:
                self.base_ts = struct.unpack_from("<I", data, 0)[0]
                log.info("Base timestamp latched: %d ms", self.base_ts)

            ch_ts, ch_data = self._parse_packet(data)

            try:
                self._queue.put_nowait((ch_ts, ch_data))
            except queue.Full:
                log.warning("Queue full — frame dropped. Consumer is too slow.")

    def _parse_frame(self, raw: bytes) -> np.ndarray:
        arr    = np.frombuffer(raw, dtype=np.uint8).reshape(TOTAL_CHANNELS, BYTES_PER_CHANNEL)
        values = (arr[:, 0].astype(np.int32) << 16 | arr[:, 1].astype(np.int32) << 8 | arr[:, 2].astype(np.int32))
        values[values >= 0x800000] -= 0x1000000
        return values

    def _parse_packet(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        ch_ts   = np.empty(BUNDLE_SIZE, dtype=np.uint32)
        ch_data = np.empty((BUNDLE_SIZE, TOTAL_CHANNELS), dtype=np.float32)

        for i in range(BUNDLE_SIZE):
            frame_start = i * FRAME_SIZE
            ts          = struct.unpack_from("<I", data, frame_start)[0]
            ch_ts[i]    = ts - self.base_ts
            raw_frame   = data[frame_start + TS_SIZE : frame_start + FRAME_SIZE]
            values      = self._parse_frame(raw_frame)
            ch_data[i]  = values * UV_SCALE if self.uv else values

        return ch_ts, ch_data

    def _init_filter(self, sos: np.ndarray) -> list:
        zi = sosfilt_zi(sos)
        zi = np.stack([zi] * TOTAL_CHANNELS, axis=1)
        return [sos, zi]