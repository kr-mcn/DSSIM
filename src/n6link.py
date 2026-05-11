from param import ParameterClass, TimeManager
from collections import deque
import numpy as np
from typing import Deque, Tuple

import math
import numpy as np
from collections import deque
from typing import Deque, Tuple

class BottleneckLink(ParameterClass, TimeManager):
    """
    Drop-in replacement that preserves all original attribute names and interface.
    Adds a single authoritative get_inflight_bits method and stores pkt_bits per entry
    so bookkeeping is robust while keeping functional behavior unchanged.
    """

    def __init__(self, logger,
                 bandwidth_bps: float = 200e6,
                 delay_ti: int = 40,
                 loss_prob: float = 0,
                 max_length: int = 1e4,
                 max_volume: int = 1e9,
                 log: bool = False):
        self.logger = logger
        # Preserve original attribute name. Interpret as bits-per-time-index.
        # TIME_SLOT_WINDOW must be the seconds represented by one time index.
        self.bandwidth_Bps_per_ms = float(bandwidth_bps) * float(ParameterClass.TIME_SLOT_WINDOW)
        if self.bandwidth_Bps_per_ms <= 0:
            raise ValueError("bandwidth and TIME_SLOT_WINDOW must be positive")
        self.delay_ti = int(delay_ti)
        self.loss_prob = float(loss_prob)
        self.max_length = int(max_length)
        self.max_volume = int(max_volume)
        self.log = log

        # FIFO that stores tuples of
        # (departure_time_ti_float, pkt_row, departure_time_ti_int, pkt_bits)
        self.content: Deque[Tuple[float, np.ndarray, int, int]] = deque()
        self.bits_in_q = 0

        # Internal tracker for last serialization finish time in time-index units (float).
        self._last_serial_finish_ti = float(-1e18)

    # ---------- Ingress ----------
    def enqueue(self, pkts: np.ndarray) -> None:
        if pkts is None:
            return

        if self.log:
            print("Time index: ", TimeManager.time_index, " / Links bandwidth (bits per ti):", self.bandwidth_Bps_per_ms)

        for pkt in pkts:
            # Random loss decision
            if np.random.random() < self.loss_prob:
                if not ParameterClass.UDP_MODE:
                    self.logger.store(
                        "N6LINK", f"UE{pkt[ParameterClass.INDEX_ue_id]}", "packet_loss_in_N6", f"time={TimeManager.time_index}: {pkt} (Random loss)")
                continue

            # Determine packet size in bits (use MTU constant or packet field if available)
            pkt_bits = int(ParameterClass.MTU_BIT_SIZE)

            # Buffer overflow check (both by packet count and by bit volume)
            if (len(self.content) >= self.max_length or
                    self.bits_in_q + pkt_bits > self.max_volume):
                # tail-drop
                if not ParameterClass.UDP_MODE:
                    self.logger.store(
                        "N6LINK", f"UE{pkt[ParameterClass.INDEX_ue_id]}", "packet_loss_in_N6", f"time={TimeManager.time_index}: {pkt} (Buffer overflow)")
                if self.log:
                    print("DROP due to volume/length cap", TimeManager.time_index, "bits_in_q", self.bits_in_q, "pkt_bits", pkt_bits, "max_volume", self.max_volume)
                continue

            # Compute serialization start time
            ser_start_ti = max(float(TimeManager.time_index), self._last_serial_finish_ti)

            # serialization duration in time-index units (float)
            ser_duration_ti = pkt_bits / self.bandwidth_Bps_per_ms

            # serialization finish time (float)
            ser_finish_ti = ser_start_ti + ser_duration_ti

            # departure time: include fixed delay as a float
            departure_ti_float = ser_finish_ti + float(self.delay_ti)

            # integer release time: floor to avoid adding an extra time-index when fraction exists
            # Using floor() with 0.98 calibration factor in bbrv1.py to compensate for systematic bias
            departure_ti_int = int(math.floor(departure_ti_float))

            # Enqueue and update bookkeeping (store pkt_bits with the entry)
            self.content.append((departure_ti_float, pkt, departure_ti_int, pkt_bits))
            self.bits_in_q += pkt_bits
            self._last_serial_finish_ti = ser_finish_ti


    # ---------- Egress ----------
    def dequeue(self) -> np.ndarray:
        ready = []
        cur_ti = TimeManager.time_index
        # compare integer departure times to current integer time index
        while self.content and self.content[0][2] <= cur_ti:
            _, pkt, _, pkt_bits = self.content.popleft()
            self.bits_in_q -= int(pkt_bits)
            ready.append(pkt)

        # Reset last-serial finish when queue is empty so new arrivals aren't delayed by stale future time
        if not self.content:
            self._last_serial_finish_ti = float(TimeManager.time_index)

        # Repair possible tiny bookkeeping drift (keeps behavior unchanged)
        real_sum = sum(int(entry[3]) for entry in self.content)
        if real_sum != self.bits_in_q:
            if self.log:
                print("bits_in_q drift corrected", TimeManager.time_index, "was", self.bits_in_q, "now", real_sum)
            self.bits_in_q = real_sum

        if ready:
            return np.stack(ready, axis=0)
        return np.empty((0, ParameterClass.NUM_INFO_PACKET), dtype=int)

    # ---------- Single authoritative inflight accessor ----------
    def get_inflight_bits(self) -> int:
        return int(self.bits_in_q)