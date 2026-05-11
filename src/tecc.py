from param import ParameterClass, TimeManager


class TECC:
    """
    Paper formula:
        e(t) = (q(t) + δ·r(t)·Tr(t)·τ) / (θ·Tr(t))
        U(t) = max{1 - e(t), 1 - max_pf}
        U̅  = (1 - w)·U̅ + w·U(t)
        Sr(t) = U̅·Tr(t) + MSS/Ts

    Inputs (via on_feedback):
        - tunnel_rate_pps: Tr [pps]  Tunnel server transmission rate
        - q_len_min_pkts:  q [packets]  Minimum queue length during FB period
        - retransmit_ratio: r [0..1]  Retransmission ratio (fixed at 0 for now)
    Output:
        - Sr [pps]  Server pacing rate

    Pacing (via get_send_count):
        - Called every slot → accumulates fractional part and returns integer packet count

    TECC switching:
        - main.py detects end of server-side BBR STARTUP and calls activate()
        - Before activate() is called, get_send_count() returns None (send with BBR)
    """

    def __init__(
        self,
        init_server_rtt: float = (ParameterClass.N3_DELAY - 1 + ParameterClass.N6_DELAY_TI) * 2 * ParameterClass.TIME_SLOT_WINDOW,
        init_feedback_interval: float = (ParameterClass.N3_DELAY - 1) * 2 * ParameterClass.TIME_SLOT_WINDOW,

        ewma_weight: float = 0.2,
        delta: float = 0.95,
        max_pf: float = 0.5,
        theta_ratio: float = 2.0 / 3.0,
        mss_packets: float = 1.0,

        min_rtt: float = 1e-6,

        # Tunnel RTT (used to adjust the loss detection threshold on the QUIC side)
        tunnel_rtt_cap: float = ParameterClass.TUNNEL_RTT_CAP,  # Tunnel RTT upper limit [seconds] (configured in param.py)
    ):
        self.ewma_weight = float(ewma_weight)
        self.delta = float(delta)
        self.max_pf = float(max_pf)
        self.theta_ratio = float(theta_ratio)
        self.mss_packets = float(mss_packets)
        self.min_rtt = float(min_rtt)

        # RTT estimate [seconds]
        self.server_rtt = float(init_server_rtt)
        self.feedback_interval = float(init_feedback_interval)

        # EWMA state
        self._u_ewma = 1.0

        # Pacing state
        self._sr_pps = 0.0
        self._accumulator = 0.0
        self._time_slot_sec = ParameterClass.TIME_SLOT_WINDOW

        # Tunnel RTT (observed value, with cap)
        self._tunnel_rtt_cap = float(tunnel_rtt_cap)
        self.tunnel_rtt = 0.0

        # TECC activation state (toggled by activate())
        self._active = False

    def activate(self):
        """
        Activate TECC from outside.
        Called when main.py detects that the server-side BBR STARTUP phase has ended.
        """
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def on_feedback(
        self,
        tunnel_rate_pps: float,
        q_len_min_pkts: float,
        retransmit_ratio: float,
        server_rtt: float | None = None,
        feedback_interval: float | None = None,
        tunnel_rtt: float | None = None,
    ) -> float:
        """
        Called on FB arrival. Updates the internal pacing rate and returns Sr (pps).
        If server_rtt / feedback_interval / tunnel_rtt are provided, the internal values are updated; if None, the previous values are used.
        tunnel_rtt is the observed MPQUIC subflow value and is used to adjust the QUIC-side loss detection threshold.
        """
        if server_rtt is not None and server_rtt > 0:
            self.server_rtt = float(server_rtt)
        if feedback_interval is not None and feedback_interval > 0:
            self.feedback_interval = float(feedback_interval)
        if tunnel_rtt is not None and tunnel_rtt > 0:
            self.tunnel_rtt = min(float(tunnel_rtt), self._tunnel_rtt_cap)

        # Do not update EWMA state before TECC is activated (prevents _u_ewma contamination by early FB)
        if self._active:
            self._sr_pps = self._calc_server_rate(tunnel_rate_pps, q_len_min_pkts, retransmit_ratio)

        return self._sr_pps

    def get_send_count(self) -> int | None:
        """
        Called every slot. Returns the number of packets to send.
        None: TECC not yet activated (caller should send with BBR)
        0:    Do not send this slot
        >=1:  Send this many packets
        """
        if not self._active:
            return None
        self._accumulator += self._sr_pps * self._time_slot_sec
        send_count = int(self._accumulator)
        self._accumulator -= send_count
        return send_count

    def _calc_server_rate(
        self,
        tunnel_rate_pps: float,
        q_len_min_pkts: float,
        retransmit_ratio: float,
    ) -> float:
        """
        Rate calculation.
        Implementation formula (dimensionally consistent):
            retrans_pkts = δ · r · Tr · τ   [packets]
            e = (q + retrans_pkts) / (θ · Tr) [dimensionless]
            U(t) = max{1 - e, 1 - max_pf}
            U̅ = (1-w)·U̅ + w·U(t)
            Sr = U̅ · Tr + MSS/Ts             [pps]
        """
        Ts = max(self.server_rtt, self.min_rtt)
        tau = max(self.feedback_interval, self.min_rtt)

        Tr = max(0.0, float(tunnel_rate_pps))
        q = max(0.0, float(q_len_min_pkts))
        r = min(1.0, max(0.0, float(retransmit_ratio)))

        # rai [pps] — minimum additive increase even when Tr=0
        rai = self.mss_packets / Ts

        if Tr <= 0.0:
            return rai

        theta = max(self.theta_ratio * Ts, self.min_rtt)

        # Retransmission packet volume [packets]
        retrans_pkts = self.delta * r * Tr * tau

        # Penalty ratio e [dimensionless]
        e = (q + retrans_pkts) / (theta * Tr)

        # U(t)
        u_inst = max(1.0 - e, 1.0 - self.max_pf)
        u_inst = min(1.0, max(0.0, u_inst))

        # EWMA
        w = self.ewma_weight
        self._u_ewma = (1.0 - w) * self._u_ewma + w * u_inst

        # Sr [pps]
        return max(0.0, self._u_ewma * Tr + rai)

