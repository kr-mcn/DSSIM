import numpy as np
from param import ParameterClass, TimeManager


class UDP(ParameterClass, TimeManager):
    def __init__(self, logger=None):
        self.logger = logger
        self.ue_states = {}

    def initialize_ue(self, ue_id):
        self.ue_states[ue_id] = {
            "next_seq_num": 0,
            "udp_carry_bits": 0.0,   # per-UE carry-over bits
        }

    def sender_send(self, ue_id):
        """
        Transmit at UDP_RATE [bps] every TIME_SLOT_WINDOW seconds in MTU_BIT_SIZE units.
        Any fractional bits are carried over per UE via udp_carry_bits.
        """
        # Convenience local variables
        mtu_bits = int(ParameterClass.MTU_BIT_SIZE)
        bits_per_slot = float(ParameterClass.UDP_RATE) * \
            float(ParameterClass.TIME_SLOT_WINDOW)

        state = self.ue_states[ue_id]
        carry = float(state.get("udp_carry_bits", 0.0))

        # Transmittable bits this slot = previous carry-over + this slot's allocation
        budget_bits = carry + bits_per_slot

        # Number of packets to send (floor in MTU units)
        num_packets = int(budget_bits // mtu_bits)

        # Update carry-over (to next slot)
        state["udp_carry_bits"] = budget_bits - (num_packets * mtu_bits)

        if num_packets <= 0:
            return np.zeros((0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        # Packet generation
        seq_base = state["next_seq_num"]
        state["next_seq_num"] = seq_base + num_packets

        packets = np.zeros(
            (num_packets, ParameterClass.NUM_INFO_PACKET), dtype=int)
        ti = TimeManager.time_index
        for i in range(num_packets):
            packets[i, ParameterClass.INDEX_PACKET_ID] = seq_base + i
            packets[i, ParameterClass.INDEX_ue_id] = ue_id
            packets[i, ParameterClass.INDEX_PAYLOAD_SIZE] = mtu_bits
            packets[i, ParameterClass.INDEX_SERVER_TIMESTAMP_ID] = ti

        # Log actual transmitted bits (MTU * count)
        self.logger.store(
            "UDP", f"UE{ue_id}", "server_send_throughput",
            [ti, num_packets * mtu_bits / ParameterClass.TIME_SLOT_WINDOW]
        )

        return packets

    # The following is a description for compatibility.
    def sender_recv(self, ue_id, temp):
        return False

    def receiver(self, ue_id, packets):
        # Do not send ACK packets
        return np.empty((0, ParameterClass.NUM_INFO_PACKET), dtype=int)

    def check_timer(self, ue_id):
        return False

    def show_results(self):
        pass

    def onetime_logger(self):
        pass
