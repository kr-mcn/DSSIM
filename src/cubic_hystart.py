from param import ParameterClass, TimeManager
import math


class CUBIC_HyStart:
    def __init__(self, log_dir1, log_dir2, logger, C=0.4, beta_cubic=0.7):
        # Parameters for CUBIC
        self.C = C  # Constant C
        self.beta_cubic = beta_cubic  # Multiplicative decrease factor
        self.cwnd = ParameterClass.INIT_CWND  # Initial cwnd (in segments)
        self.ssthresh = float('inf')  # Initial slow start threshold
        self.W_max = self.cwnd  # Initial W_max (max cwnd before last loss)
        self.cwnd_prior = 1.0  # cwnd at last congestion event
        self.t_epoch = self._current_time()  # Timestamp of last congestion event
        self.cwnd_epoch = self.cwnd  # cwnd at the start of this epoch
        self.W_est = self.cwnd_epoch  # Reno-equivalent cwnd estimate
        self.alpha_cubic = (3 * (1 - self.beta_cubic)) / \
            (1 + self.beta_cubic)  # Alpha for Reno-friendly region
        # Current mode: SLOW_START / CSS / CONGESTION_AVOIDANCE
        self.current_mode = "SLOW_START"
        # To start directly in congestion avoidance (skip slow start), uncomment below:
        # self.current_mode = "CONGESTION_AVOIDANCE"
        self.current_region = "CUBIC"  # Region: CUBIC or RENO
        self.cubic_log = []
        self.cubic_rtt_log = []
        self.cubic_debug_log = []

        # Parameters for HyStart++
        self.MIN_RTT_THRESH = 0.004  # 4 ms minimum RTT threshold
        self.MAX_RTT_THRESH = 0.016  # 16 ms maximum RTT threshold
        self.MIN_RTT_DIVISOR = 8
        self.N_RTT_SAMPLE = 8  # Number of RTT samples per round
        # Growth divisor for Conservative Slow Start (CSS)
        self.CSS_GROWTH_DIVISOR = 4
        self.CSS_ROUNDS = 5  # Number of CSS rounds before switching to congestion avoidance
        # Limit for cwnd growth in non-paced mode (paced = evenly spaced packet transmission)
        self.L = 8

        # State variables for HyStart++
        self.lastRoundMinRTT = float('inf')
        self.currentRoundMinRTT = float('inf')
        self.currRTT = float('inf')

        self.windowEnd = self.cwnd  # Used to determine the end of a round
        self.round_active = False

        self.rttSampleCount = 0
        self.cssBaselineMinRtt = float('inf')
        self.cssRoundsCompleted = 0

        self.max_seq_num_acked = 0

        self.app_limited = False  # Whether the sender is currently app-limited

        self.log_dir1 = log_dir1
        self.log_dir2 = log_dir2
        self.logger = logger

        self.WITHOUT_RENO_MODE = True

    def _current_time(self):
        """Convert TimeManager.time_index to real time in seconds"""
        return round(TimeManager.time_index * ParameterClass.TIME_SLOT_WINDOW, 6)

    def on_ack(self, segments_num_acked, rtt, smoothed_rtt, seq_num_acked, next_seq_num, app_limited):
        now = self._current_time()
        self.currRTT = rtt

        # ---------- APP-LIMITED detection & epoch handling ---------------------
        # NEW: Handle app-limited behavior
        # ----------------------------------------------------------------------
        if app_limited:
            self.app_limited = True  # Stop cwnd growth while app-limited
        else:
            # Reset epoch when transitioning from app-limited to sending data again
            if self.app_limited:
                self.t_epoch = now
                self.cwnd_epoch = self.cwnd
            self.app_limited = False

        # Check if a new round has started
        if self._is_new_round(seq_num_acked):
            self._start_new_round(next_seq_num)

        # Update maximum ACKed sequence number
        if seq_num_acked > self.max_seq_num_acked:
            self.max_seq_num_acked = seq_num_acked

        # Update the minimum RTT observed in the current round
        self.currentRoundMinRTT = min(self.currentRoundMinRTT, self.currRTT)
        self.rttSampleCount += 1

        # Skip cwnd growth while app-limited ------------------------------------
        if self.app_limited:
            return  # Do not update cwnd

        # ---------- cwnd update -----------------------------------------------
        # Update cwnd depending on current mode
        if self.current_mode == "SLOW_START":
            self._slow_start_update(segments_num_acked)
        elif self.current_mode == "CSS":
            self._conservative_slow_start_update(segments_num_acked)
        elif self.current_mode == "CONGESTION_AVOIDANCE":
            self._cubic_update(segments_num_acked, smoothed_rtt)

        # Check for state transitions after cwnd update
        if self.current_mode == "SLOW_START":
            self._check_exit_slow_start()
        elif self.current_mode == "CSS":
            self._check_exit_css()

        # Ensure cwnd does not fall below minimum
        self.cwnd = max(self.cwnd, 2.0)

    def _is_new_round(self, seq_num_acked):
        # A new round starts when an ACK acknowledges data beyond windowEnd
        return seq_num_acked >= self.windowEnd

    def _start_new_round(self, next_seq_num):
        # Start a new round
        self.lastRoundMinRTT = self.currentRoundMinRTT
        self.currentRoundMinRTT = float('inf')
        self.rttSampleCount = 0
        self.windowEnd = next_seq_num
        if self.current_mode == "CSS":
            self.cssRoundsCompleted += 1

    def _slow_start_update(self, segments_acked):
        increment = min(segments_acked, self.L)
        self.cwnd += increment

    def _conservative_slow_start_update(self, segments_acked):
        increment = min(segments_acked, self.L) / self.CSS_GROWTH_DIVISOR
        self.cwnd += increment

    def _cubic_update(self, segments_acked, RTT):
        # Elapsed time since last congestion event
        t = self._current_time() - self.t_epoch

        # Calculate K
        K = math.copysign(abs((self.W_max - self.cwnd_epoch) / self.C)
                          ** (1/3), (self.W_max - self.cwnd_epoch) / self.C)

        # Calculate W_cubic(t)
        W_cubic = self.C * ((t - K) ** 3) + self.W_max

        # Update Reno-equivalent W_est
        self.W_est += self.alpha_cubic * (segments_acked / self.cwnd)

        # Calculate target cwnd
        W_cubic_future = self.C * math.pow((t + RTT) - K, 3) + self.W_max
        target = W_cubic_future
        if W_cubic_future < self.cwnd:
            target = self.cwnd
        elif W_cubic_future > 1.5 * self.cwnd:
            target = 1.5 * self.cwnd

        if self.WITHOUT_RENO_MODE is True:
            delta = (target - self.cwnd) / self.cwnd
            self.cwnd += delta
        else:
            # Determine if we are in Reno-friendly region
            if W_cubic < self.W_est:
                self.cwnd = self.W_est
                if self.current_region == "CUBIC":
                    self.current_region = "Reno"
                    self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                                      f"time_index={TimeManager.time_index}: mode is changed to \"Reno\"")
            else:
                delta = (target - self.cwnd) / self.cwnd
                self.cwnd += delta
                if self.current_region == "Reno":
                    self.current_region = "CUBIC"
                    self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                                      f"time_index={TimeManager.time_index}: mode is changed to \"CUBIC\"")

        # Ensure cwnd is not less than 2
        self.cwnd = max(self.cwnd, 2.0)

    def _check_exit_slow_start(self):
        # Determine whether to exit SLOW_START and enter CSS based on RTT growth
        if (self.rttSampleCount >= self.N_RTT_SAMPLE and
            self.currentRoundMinRTT != float('inf') and
                self.lastRoundMinRTT != float('inf')):

            RttThresh = max(
                self.MIN_RTT_THRESH,
                min(self.lastRoundMinRTT /
                    self.MIN_RTT_DIVISOR, self.MAX_RTT_THRESH)
            )

            if self.currentRoundMinRTT >= (self.lastRoundMinRTT + RttThresh):
                self.cssBaselineMinRtt = self.currentRoundMinRTT
                self.current_mode = "CSS"
                self.cssRoundsCompleted = 0
                self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                                  f"time_index={TimeManager.time_index}: RTT increased. Shift to CSS from SLOW_START.")

    def _check_exit_css(self):
        # Determine whether to exit CSS and enter CUBIC or return to SLOW_START
        if self.cssRoundsCompleted >= self.CSS_ROUNDS:
            # Transition to congestion avoidance state
            self.current_mode = "CONGESTION_AVOIDANCE"
            self._initialize_cubic()
            self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                              f"time_index={TimeManager.time_index}: All CSS rounds completed. Shift to CONGESTION_AVOIDANCE.")
            return

        if (self.rttSampleCount >= self.N_RTT_SAMPLE and
            self.currentRoundMinRTT != float('inf') and
                self.lastRoundMinRTT != float('inf')):

            # If RTT becomes smaller than CSS entry baseline by 3ms, revert to SLOW_START
            if self.currentRoundMinRTT < self.cssBaselineMinRtt - 0.003:  # Apply 3ms offset
                # Consider RTT growth as false positive, return to SLOW_START
                self.cssBaselineMinRtt = float('inf')
                self.current_mode = "SLOW_START"
                self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                                  f"time_index={TimeManager.time_index}: RTT decreased. Return to SLOW_START.")

    def on_congestion_event(self, fast_convergence_enabled=True):
        if self.current_mode == "CONGESTION_AVOIDANCE":
            # Update W_max
            if fast_convergence_enabled and self.cwnd < self.W_max:  # Apply fast convergence
                self.W_max = (self.cwnd * (1 + self.beta_cubic)) / 2
            else:
                self.W_max = self.cwnd

            # Set ssthresh
            self.ssthresh = max(self.cwnd * self.beta_cubic, 2.0)

            # Reduce cwnd
            self.cwnd = max(self.ssthresh, 2.0)
            self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                              f"time_index={TimeManager.time_index}: cwnd is decreased to {self.cwnd}")

            # Reinitialize congestion avoidance stage
            self.t_epoch = self._current_time()
            self.cwnd_epoch = self.cwnd
            self.W_est = self.cwnd_epoch
        else:  # current_mode = "CSS" or "SLOW_START"
            self.current_mode = "CONGESTION_AVOIDANCE"
            self.W_max = self.cwnd
            self.ssthresh = max(self.cwnd * self.beta_cubic, 2.0)
            self.cwnd = max(self.ssthresh, 2.0)
            self._initialize_cubic()
            self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                              f"time_index={TimeManager.time_index}: Congestion event occured. Shift to CONGESTION_AVOIDANCE.")

    def _initialize_cubic(self):
        """
        Initialization procedure when entering CUBIC mode
        """
        self.ssthresh = self.cwnd
        self.W_max = self.cwnd
        self.cwnd_prior = self.cwnd
        self.t_epoch = self._current_time()
        self.cwnd_epoch = self.cwnd
        self.W_est = self.cwnd

    def _on_timeout(self):
        # Timeout handling (not currently used)
        self.ssthresh = max(self.cwnd * self.beta_cubic, 2.0)
        self.cwnd = 1.0
        self.W_max = 1.0
        self.t_epoch = self._current_time()
        self.cwnd_epoch = self.cwnd
        self.W_est = self.cwnd_epoch

        # Reset HyStart++ state and return to SLOW_START
        self.current_mode = "SLOW_START"
        self.cssBaselineMinRtt = float('inf')
        self.cssRoundsCompleted = 0
        self.lastRoundMinRTT = float('inf')
        self.currentRoundMinRTT = float('inf')
        self.rttSampleCount = 0
        self.windowEnd = self.cwnd
        self.logger.store(self.log_dir1, self.log_dir2, "cubic_event_log",
                          f"time_index={TimeManager.time_index}: Timeout occured. Reset to SLOW_START.")
