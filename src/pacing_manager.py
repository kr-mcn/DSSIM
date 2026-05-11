import math
from collections import deque

# Pacing manager class adds sending rate functionality to the QUIC / MPQUIC protocol
class PacingManager:
    """
    Token bucket-based pacing manager for BBR congestion control.

    Implements a token bucket algorithm to pace packet transmission with fractional
    packet accumulation, which is essential for handling coarse simulation timesteps (1ms).

    Units:
        - sim_step: milliseconds (ms)
        - pacing_rate: packets per millisecond (packets/ms)
        - tokens: packets (fractional values allowed)

    Algorithm:
        1. Each timestep, accumulate fractional tokens based on pacing_rate
        2. Send floor(tokens) whole packets
        3. Retain fractional remainder for next timestep
        4. Clamp accumulated tokens to prevent unbounded bursts

    Public Interface:
        __init__(sim_step, num_sim_steps, max_time_step, initial_pacing_rate)
        advance_pacing_manager(pacing_rate) -> int (packets to send this timestep)
    """

    def __init__(self, sim_step, num_sim_steps, max_time_step, initial_pacing_rate):
        """
        Initialize the pacing manager.

        Args:
            sim_step: Duration of one simulation timestep in ms (typically 1.0)
            num_sim_steps: Total number of simulation timesteps
            max_time_step: Maximum time between packets in ms (used for burst limiting)
            initial_pacing_rate: Initial pacing rate in packets/ms
        """
        # Simulation attributes
        self.sim_index = 0
        self.sim_step = float(sim_step)           # Duration of one timestep [ms]
        self.num_sim_steps = int(num_sim_steps)  # Total number of timesteps

        # Pacing rate in packets/ms
        self.pacing_rate = float(initial_pacing_rate)

        # Token bucket state: accumulates fractional packets over time
        self._token_packets = 0.0  # Start with zero tokens (no initial burst)

        # Burst limiting parameters
        self.max_time_step = float(max_time_step)  # Maximum inter-packet time [ms]
        self._max_burst_ms = min(self.max_time_step, 100.0)  # Cap burst window at 100ms
        self._max_burst_packets = max(1.0, self.pacing_rate * self._max_burst_ms)

        # Per-step emission cap: allow up to 3x the normal rate per timestep
        # This handles rate changes and natural burstiness while preventing huge spikes
        # Factor of 3 allows smooth operation during PROBEBW's 5/4 pacing gain phase
        self._max_packets_per_step = max(1, int(math.ceil(self.pacing_rate * self.sim_step * 3)))

        # Legacy buffer for backward compatibility (maintained but not used in token bucket logic)
        self.buffer_size = int(math.floor(self.max_time_step / self.sim_step)) + 1
        self.buffer = deque([{'num packets': 0, 'wait': False} for _ in range(self.buffer_size)])

    def _update_burst_cap(self):
        """
        Update burst cap based on current pacing rate and clamp accumulated tokens.
        Called whenever pacing_rate changes to adapt the burst limit.
        """
        # Recalculate maximum burst based on current rate
        self._max_burst_packets = max(1.0, self.pacing_rate * self._max_burst_ms)

        # Clamp tokens to prevent excessive accumulation
        if self._token_packets > self._max_burst_packets:
            self._token_packets = self._max_burst_packets

    def _update_per_step_cap(self):
        """
        Update the per-timestep packet cap based on current pacing rate.
        Allows up to 3x the normal rate to handle BBR's PROBEBW gain cycling.
        """
        self._max_packets_per_step = max(1, int(math.ceil(self.pacing_rate * self.sim_step * 3)))

    def _advance_buffer_window(self):
        """Maintain legacy buffer window for backward compatibility."""
        self.buffer.popleft()
        self.buffer.append({'num packets': 0, 'wait': False})

    def advance_pacing_manager(self, pacing_rate):
        """
        Advance the pacing manager by one timestep and return packets to send.

        This is the main public method called each simulation timestep.

        Args:
            pacing_rate: Current pacing rate in packets/ms

        Returns:
            int: Number of whole packets to send this timestep

        Algorithm:
            1. Update pacing rate and related caps
            2. Accumulate fractional tokens: tokens += rate × timestep
            3. Determine whole packets to send: floor(tokens)
            4. Clamp to per-step limit (prevents huge bursts)
            5. Consume tokens for packets being sent
            6. Return number of packets
        """
        # Update pacing rate
        self.pacing_rate = float(pacing_rate)

        # Update caps based on new rate
        self._update_burst_cap()
        self._update_per_step_cap()

        # Advance legacy buffer window
        self._advance_buffer_window()

        # Accumulate tokens based on pacing rate
        # Units: packets += (packets/ms) × (ms) = packets
        self._token_packets += self.pacing_rate * self.sim_step

        # Clamp tokens to burst cap (already done in _update_burst_cap, but be safe)
        if self._token_packets > self._max_burst_packets:
            self._token_packets = self._max_burst_packets

        # Determine whole packets to send this timestep
        num_to_send = int(math.floor(self._token_packets))

        # Apply per-step cap to prevent unrealistic bursts
        # This limits how many packets can be sent in a single timestep
        if num_to_send > self._max_packets_per_step:
            num_to_send = self._max_packets_per_step

        # Consume tokens for packets being sent
        # The fractional remainder stays in the bucket for next timestep
        if num_to_send > 0:
            self._token_packets -= num_to_send

        # Update legacy buffer for backward compatibility
        self.buffer[0]['num packets'] = num_to_send
        self.buffer[0]['wait'] = False

        # Advance simulation index
        self.sim_index += 1

        return num_to_send