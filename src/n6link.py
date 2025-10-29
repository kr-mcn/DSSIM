from param import ParameterClass, TimeManager
from collections import deque
import numpy as np
from typing import Deque, Tuple


class BottleneckLink(ParameterClass, TimeManager):
    """
    - Holds an internal tail-drop FIFO queue
    - Drops packets either when capacity is exceeded or randomly by a given probability
    - Dequeues after a fixed DELAY
    - Intended to be instantiated per UE (each UE has its own link)
    """

    def __init__(self, logger,
                 # [bps]. Rate limiting is applied with this parameter.
                 bandwidth_bps: float = 10*1e9,
                 delay_ti: int = 50,  # [time_index]
                 loss_prob: float = 1e-4,
                 max_length: int = 1e4,  # [packets]
                 max_volume: int = 1e9,  # [bit]
                 ):
        # super().__init__()
        self.logger = logger
        self.bandwidth_Bps_per_ms = bandwidth_bps * \
            ParameterClass.TIME_SLOT_WINDOW  # bits per ms (uses simulation time slot)
        self.delay_ti = int(delay_ti)
        self.loss_prob = float(loss_prob)
        self.max_length = int(max_length)
        self.max_volume = int(max_volume)

        # FIFO that stores tuples of (departure_time_ti, pkt_row)
        self.content: Deque[Tuple[int, np.ndarray]] = deque()
        self.bits_in_q = 0

    # ---------- Ingress ----------
    def enqueue(self, pkts: np.ndarray) -> None:
        if pkts is None:
            return

        for pkt in pkts:
            # Random loss decision
            # np.random.random() returns a float in [0.0, 1.0)
            if np.random.random() < self.loss_prob:
                self.logger.store(
                    "main", f"UE{pkt[ParameterClass.INDEX_ue_id]}", "packet_loss_in_N6", f"time={TimeManager.time_index}: {pkt}")
                continue

            # Buffer overflow check (both by bit volume and by packet count)
            if (len(self.content) >= self.max_length or
                    self.bits_in_q + ParameterClass.MTU_BIT_SIZE > self.max_volume):
                # tail-drop
                continue

            # Admission complete: compute serialization delay and assign a departure time
            tx_delay_ti = ParameterClass.MTU_BIT_SIZE / self.bandwidth_Bps_per_ms
            depature_ti = TimeManager.time_index + \
                self.delay_ti + int(tx_delay_ti)  # int() truncates toward zero
            self.content.append((depature_ti, pkt))
            self.bits_in_q += ParameterClass.MTU_BIT_SIZE

    # ---------- Egress ----------
    def dequeue(self) -> np.ndarray:
        ready = []
        while self.content and self.content[0][0] <= TimeManager.time_index:
            _, pkt = self.content.popleft()
            self.bits_in_q -= ParameterClass.MTU_BIT_SIZE
            ready.append(pkt)

        if ready:
            return np.stack(ready, axis=0)
        # Return an empty array with shape (0, NUM_INFO_PACKET) as l4_outer expects
        return np.empty((0, ParameterClass.NUM_INFO_PACKET), dtype=int)
