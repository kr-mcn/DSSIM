"""
================================================================================
CAMFController  —  Capacity-Aware MASQUE Feed back Controller
================================================================================

Units:
    Rproxy, Rfb : pkt/ms  (= BBR pacing_rate units)
    q_min       : packets
    θ · Rproxy  : ms * pkt/ms = packets  (dimensionally consistent)

Requirements:
    MPQUIC_CC == "BBR"  (pacing_rate is obtained from the BBR subflow implementation)
================================================================================
"""

from param import ParameterClass


class CAMFController:
    """
    Per-UE CAMF feedback rate controller.

    Lifecycle per simulation period (CAMF_PERIOD_TI time_indices):
        tick_upf_queue()      — called each time_index with current UPF send buffer length
        set_capacity_sample() — called once per period with Rproxy (sum of pacing_rate)
        compute_rfb()         — called once per period; returns rfb_pps
    """

    def __init__(self, num_ue: int):
        self._num_ue = num_ue

        # EWMA state for utility Ū (initialized to 1.0 = no penalty)
        self._u_ewma = {i: 1.0 for i in range(num_ue)}

        # Per-period accumulators (reset inside compute_rfb)
        self._q_latest_pkts  = {i: 0            for i in range(num_ue)}
        self._cap_pps_sample = {i: 0.0          for i in range(num_ue)}

        # Public: current Rfb [pkt/ms] (last computed value, for logging)
        self.rfb_pps = {i: 0.0 for i in range(num_ue)}

    # ------------------------------------------------------------------
    # Per-time_index interface
    # ------------------------------------------------------------------

    def tick_upf_queue(self, ue_id: int, queue_len: int) -> None:
        """
        Record the latest UPF send buffer length at each time_index.
        queue_len = send_buffer.length + retransmission_buffer.length [packets].
        Call once per time_index per UE, after the subflow send_data() calls.
        """
        self._q_latest_pkts[ue_id] = queue_len

    def set_capacity_sample(self, ue_id: int, cap_pps: float) -> None:
        """
        Set Rproxy [pkt/ms] for the current period.
        Should be the sum of both MPQUIC subflow pacing_rate values (inflation-adjusted).
        Call once per period per UE, just before compute_rfb().
        """
        self._cap_pps_sample[ue_id] = cap_pps

    # ------------------------------------------------------------------
    # Per-period interface
    # ------------------------------------------------------------------

    def compute_rfb(self, ue_id: int) -> float | None:
        """
        Compute Rfb [pkt/ms] using TECC Algorithm 1 with Tr replaced by Rproxy,
        and q_min replaced by q_latest (queue length at the end of the period).

        Returns rfb_pps [pkt/ms], or None if no capacity estimate available.
        """
        p = ParameterClass

        rproxy   = self._cap_pps_sample[ue_id]    # [pkt/ms]
        q_latest = self._q_latest_pkts[ue_id]     # [pkts] — latest queue length

        # Always reset accumulators before any early return
        self._reset_period(ue_id)

        # ── Guard: no capacity estimate yet ────────────────────────
        if rproxy <= 0:
            return None

        Ts    = p.CAMF_SERVER_RTT                 # [ms]
        theta = p.CAMF_THETA_RATIO * Ts           # [ms]

        # ── CAMF ────────────────────────────────
        e      = (q_latest - p.CAMF_Q_TARGET) / (theta * rproxy)
        u_inst = max(1.0 - e, 1.0 - p.CAMF_MAX_PF)
        u_inst = min(1.0, max(0.0, u_inst))

        w = p.CAMF_EWMA_WEIGHT
        self._u_ewma[ue_id] = (1.0 - w) * self._u_ewma[ue_id] + w * u_inst

        rfb_new = self._u_ewma[ue_id] * rproxy

        self.rfb_pps[ue_id] = rfb_new

        return rfb_new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_period(self, ue_id: int) -> None:
        self._cap_pps_sample[ue_id] = 0.0
