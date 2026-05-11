class Min_rtt_scheduler:
    """
    Minimum RTT (Round-Trip Time) scheduler.

    This is the default scheduler used in MPTCP/MPQUIC implementations.
    It selects the path with the lowest RTT to minimize latency and
    preferentially fill the fastest path's congestion window.

    The MinRTT scheduler aims to maximize throughput by opportunistically
    utilizing the lowest-latency path first, then using other paths when
    the primary path's congestion window is full.

    Reference: This is the default MPTCP scheduler as described in RFC 6824
    and used in Linux kernel MPTCP implementation.
    """

    def schedule(self, rtt1, rtt2):
        """
        Select the path with minimum round-trip time.

        Args:
            rtt1: Round-trip time of path 1 (time units)
            rtt2: Round-trip time of path 2 (time units)

        Returns:
            bool: True if path 1 has lower or equal RTT (prioritize path 1),
                  False if path 2 has lower RTT (prioritize path 2)
        """
        return rtt1 <= rtt2
