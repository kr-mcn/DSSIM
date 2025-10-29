import numpy as np
from param import ParameterClass, TimeManager
from buffer import Buffer, TBBuffer
from collections import deque
import sys


class MAC_INFO_MANAGER(ParameterClass, TimeManager):
    def __init__(self):
        self.packet_TB_number_relation_matrix = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC], dtype=int)  # [UE id, packet id] — how many TBs a given MAC packet is split into
        self.number_ACK_received_TBs_MAC_packet = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC], dtype=int)  # [UE id, packet id] — how many TBs of a MAC packet have been ACKed

        self.packet_TB_ids_relation_matrix = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC, self.MAX_NUM_TB_per_MAC_PACKET], dtype=int)  # [UE id, packet id, ...] — TB IDs associated with a MAC packet
        self.whether_packet_converted_TB = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC], dtype=int
        )  # [UE id, packet id] — whether the payload of the MAC packet has been fully split into TBs: 0 = not started, 1 = in progress, 2 = complete

        # Handling this structure is tricky...
        self.ideal_MAC_floating_packets_infos = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC, self.NUM_INFO_PACKET],
            dtype=int,
        )  # Metadata for packets currently residing in the MAC layer (per-UE, per-MAC packet)

        self.number_received_TBs_MAC_packet = np.zeros(
            [self.NUM_UE, self.NUM_CONTROL_PACKET_ON_MAC], dtype=int)  # Per UE & MAC packet, count of TBs correctly received at the UE


class MAC(ParameterClass, TimeManager):
    def __init__(
        self,
        mac_info_manager,
        ue_id_list,
        ran_name,
        bandwidth=10e6,
    ):
        # Instance variables (per-instance state)

        self.ue_id_list = ue_id_list
        self.bandwidth = bandwidth  # 10 MHz
        self.ran_name = ran_name

        self.mac_info_manager = mac_info_manager

        self.num_UE_in_MAC = len(self.ue_id_list)

        self.dl_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)
        self.ul_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)
        self.rt_dl_buffer_list = np.zeros(self.NUM_UE, dtype=TBBuffer)

        for ue_id in range(self.NUM_UE):
            self.dl_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_MAC_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_MAC_DL)
            self.ul_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_MAC_UL, max_length=ParameterClass.BUFF_MAX_LENGTH_MAC_UL)
            self.rt_dl_buffer_list[ue_id] = TBBuffer()

        self.next_mac_packet_id = np.zeros(self.NUM_UE, dtype=int)

        self.dl_temporal_buffer = np.zeros(
            [self.NUM_UE, self.NUM_INFO_PACKET], dtype=int)

        self.MAC_packets_ids_per_TBS = [[[] for _ in range(
            self.NUM_SIMULATION_TIME_SLOTS + 1)] for _ in range(self.NUM_UE)]

        self.half_index = int(self.NUM_CONTROL_PACKET_ON_MAC / 2)
        self.three_fourth_index = int(self.NUM_CONTROL_PACKET_ON_MAC * 0.75)
        self.one_fourth_index = int(self.NUM_CONTROL_PACKET_ON_MAC * 0.25)

        # Per-UE TB index; increment as TBs are created
        self.TB_id_list = np.full(self.NUM_UE, 0, dtype=int)
        self.TB_size = np.zeros(
            [self.NUM_UE, self.NUM_SIMULATION_TIME_SLOTS], dtype=int)  # TB size (bits) per UE per time slot

        self.total_buffer_size = np.zeros([self.NUM_UE])

        self.assigned_TBS = np.zeros([self.NUM_UE])

        self.estimated_channel_condition = np.zeros([self.NUM_UE])
        self.estimated_spectral_efficiency = np.zeros([self.NUM_UE])
        self.experienced_throughput = np.zeros(
            [self.NUM_UE]) + 1e-9  # Historical average throughput experienced so far
        self.PF_metic = np.zeros([self.NUM_UE])
        self.assigned_bandwidth = np.zeros([self.NUM_UE])
        self.measured_spectrum_efficiency = np.zeros([self.NUM_UE])

        self.PF_metric_window = 10  # seconds

        self.priority_mask = np.zeros(self.NUM_UE)
        self.priority_order = np.zeros(self.NUM_UE)

        # --- For recent_experienced_throughput -----------------------
        self.recent_window_slots = max(
            1, int(ParameterClass.EXP_THPT_RANGE_SEC / ParameterClass.TIME_SLOT_WINDOW))
        self.recent_TB_history = [
            # For each UE, a FIFO of length recent_window_slots to compute a moving average of TB sizes
            deque(maxlen=self.recent_window_slots) for _ in range(self.NUM_UE)
        ]
        self.recent_experienced_throughput = np.zeros(
            [self.NUM_UE]
        )

        # --- For responsiveness evaluation -----------------------
        self.mac_tx_capacity = [0]*ParameterClass.NUM_UE

    def flesh_MAC_info_manager(self, flesh_type, ue_id):
        if flesh_type == "three_forth":
            # Approaching the tail region; initialize the first half
            print("ue id", ue_id, "flesh first half")
            self.mac_info_manager.packet_TB_number_relation_matrix[ue_id, : self.half_index] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.number_ACK_received_TBs_MAC_packet[ue_id, : self.half_index] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.packet_TB_ids_relation_matrix[ue_id, : self.half_index] = np.zeros(
                [self.half_index, self.MAX_NUM_TB_per_MAC_PACKET], dtype=int)
            self.mac_info_manager.whether_packet_converted_TB[ue_id, : self.half_index] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.ideal_MAC_floating_packets_infos[ue_id, : self.half_index] = np.zeros(
                [self.half_index, self.NUM_INFO_PACKET], dtype=int)
            self.mac_info_manager.number_received_TBs_MAC_packet[ue_id, : self.half_index] = np.zeros(
                self.half_index, dtype=int)

        if flesh_type == "one_forth":
            # Approaching the middle; initialize the latter half
            print("ue id", ue_id, "flesh last half")
            self.mac_info_manager.packet_TB_number_relation_matrix[ue_id, self.half_index:] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.number_ACK_received_TBs_MAC_packet[ue_id, self.half_index:] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.packet_TB_ids_relation_matrix[ue_id, self.half_index:] = np.zeros(
                [self.half_index, self.MAX_NUM_TB_per_MAC_PACKET], dtype=int)
            self.mac_info_manager.whether_packet_converted_TB[ue_id, self.half_index:] = np.zeros(
                self.half_index, dtype=int)
            self.mac_info_manager.ideal_MAC_floating_packets_infos[ue_id, self.half_index:] = np.zeros(
                [self.half_index, self.NUM_INFO_PACKET], dtype=int)
            self.mac_info_manager.number_received_TBs_MAC_packet[ue_id, self.half_index:] = np.zeros(
                self.half_index, dtype=int)

    def receive_ACK_NACK(self, ACK_NACK, TB_id_for_ACK_NACK, ue_id):
        """Load ACK/NACK from UE"""
        if TB_id_for_ACK_NACK == -1:
            if self.time_index < self.PF_metric_window / self.TIME_SLOT_WINDOW:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.time_index) / (self.time_index + 1)
            else:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.PF_metric_window / self.TIME_SLOT_WINDOW) / (self.PF_metric_window / self.TIME_SLOT_WINDOW + 1)
            self._update_recent_exp_thpt(ue_id, 0)
            return 0
        if ACK_NACK == 0:
            if self.time_index < self.PF_metric_window / self.TIME_SLOT_WINDOW:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.time_index) / (self.time_index + 1)
            else:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.PF_metric_window / self.TIME_SLOT_WINDOW) / (self.PF_metric_window / self.TIME_SLOT_WINDOW + 1)

            self._update_recent_exp_thpt(ue_id, 0)
            return 0
        if ACK_NACK == 1:  # On ACK reception
            corresponding_packet_ids_in_float = [
                x % self.NUM_CONTROL_PACKET_ON_MAC for x in self.MAC_packets_ids_per_TBS[ue_id][TB_id_for_ACK_NACK]]
            self.mac_info_manager.number_ACK_received_TBs_MAC_packet[
                ue_id][corresponding_packet_ids_in_float] += 1

            success_received_transmitted_MAC_packet_in_this_TS = (
                self.mac_info_manager.number_ACK_received_TBs_MAC_packet[
                    ue_id][corresponding_packet_ids_in_float]
                == self.mac_info_manager.packet_TB_number_relation_matrix[ue_id][corresponding_packet_ids_in_float]
            )  # Among transmitted MAC packet IDs in this TS, those for which all TBs were successfully received

            complete_received_MAC_packet_in_this_TS = success_received_transmitted_MAC_packet_in_this_TS * (
                self.mac_info_manager.whether_packet_converted_TB[
                    ue_id][corresponding_packet_ids_in_float] == 2
            )  # MAC packets that have been fully transmitted (all TBs sent) and also fully received
            complete_received_MAC_packet_id = like_value_binary_in_list(
                corresponding_packet_ids_in_float,
                complete_received_MAC_packet_in_this_TS,
            )
            packet_mac2rlc = self.mac_info_manager.ideal_MAC_floating_packets_infos[
                ue_id][complete_received_MAC_packet_id]

            # update PF metric
            transmitted_TBS = self.TB_size[ue_id][TB_id_for_ACK_NACK]
            if self.time_index < self.PF_metric_window / self.TIME_SLOT_WINDOW:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.time_index + transmitted_TBS) / (self.time_index + 1)
            else:
                self.experienced_throughput[ue_id] = (self.experienced_throughput[ue_id] * self.PF_metric_window / self.TIME_SLOT_WINDOW + transmitted_TBS) / (
                    self.PF_metric_window / self.TIME_SLOT_WINDOW + 1
                )

            self._update_recent_exp_thpt(ue_id, transmitted_TBS)
            return packet_mac2rlc
        if ACK_NACK == 2:
            self.rt_dl_buffer_list[ue_id].enqueue(
                np.array([[TB_id_for_ACK_NACK, self.TB_size[ue_id][TB_id_for_ACK_NACK]]]))

            # update PF metric
            if self.time_index < self.PF_metric_window / self.TIME_SLOT_WINDOW:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.time_index) / (self.time_index + 1)
            else:
                self.experienced_throughput[ue_id] = (
                    self.experienced_throughput[ue_id] * self.PF_metric_window / self.TIME_SLOT_WINDOW) / (self.PF_metric_window / self.TIME_SLOT_WINDOW + 1)
            self._update_recent_exp_thpt(ue_id, 0)

            # self.retransmission_flag[ue_id] = 1

    def _update_recent_exp_thpt(self, ue_id, tb_size):
        hist = self.recent_TB_history[ue_id]
        hist.append(tb_size)

        # Weighted moving average (more weight to recent TBs)
        weights = list(range(1, len(hist) + 1))
        weighted_sum = sum(w * x for w, x in zip(weights, hist))
        total_weight = sum(weights)
        self.recent_experienced_throughput[ue_id] = weighted_sum / total_weight

    def dl_enqueue(self, packet_from_rlc, ue_id):
        self.dl_buffer_list[ue_id].enqueue(packet_from_rlc)

    def measure_channel_condition(self, channel_condition):
        self.estimated_channel_condition = channel_condition  # SINR in dB

    def estimate_spectral_efficiency(self):
        self.estimated_spectral_efficiency = self.estimated_channel_condition

    def calculate_PF_metic(self):
        self.PF_metic = self.estimated_spectral_efficiency / \
            (self.experienced_throughput + 1e-9)

    def assign_bandwidth(self):
        self.estimate_spectral_efficiency()
        self.calculate_PF_metic()

        self.priority_order = np.argsort(self.PF_metic)[::-1]

        self.available_bandwidth = self.bandwidth
        self.assigned_bandwidth = np.zeros([self.NUM_UE])

        # Determine whether retransmission TBs exist and prioritize those UEs
        self.priority_mask = np.full([self.NUM_UE], False, dtype=bool)
        for ue_id in self.priority_order:
            if self.rt_dl_buffer_list[ue_id].length > 0:
                self.priority_mask[ue_id] = True

        sorted_priority_mask = self.priority_mask[self.priority_order]
        self.priority_order = np.concatenate(
            [self.priority_order[sorted_priority_mask],
                self.priority_order[~sorted_priority_mask]]
        )  # First: UEs with retransmission; Then: UEs without

        self.mac_tx_capacity = [0] * ParameterClass.NUM_UE

        for ue_id in self.priority_order:
            if (ue_id in self.ue_id_list) is False:
                # If the UE is not served by this cell
                continue
            self.total_buffer_size[ue_id] = self.rt_dl_buffer_list[ue_id].volume + \
                self.dl_buffer_list[ue_id].volume + \
                self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
            if self.total_buffer_size[ue_id] <= 0:
                continue
            if self.available_bandwidth <= 0:
                break

            maximum_TRB_size = self.available_bandwidth * \
                self.estimated_spectral_efficiency[ue_id] * \
                self.TIME_SLOT_WINDOW

            self.mac_tx_capacity[ue_id] = maximum_TRB_size  # bits per slot

            if self.total_buffer_size[ue_id] > maximum_TRB_size:
                # This UE will consume the entire remaining bandwidth
                self.assigned_bandwidth[ue_id] = self.available_bandwidth
                self.available_bandwidth = 0.0
                break

            if self.total_buffer_size[ue_id] <= maximum_TRB_size:
                # This UE will not consume all available bandwidth
                self.assigned_bandwidth[ue_id] = self.total_buffer_size[ue_id] / (
                    self.estimated_spectral_efficiency[ue_id] * self.TIME_SLOT_WINDOW)
                self.available_bandwidth -= self.assigned_bandwidth[ue_id]

    def assign_packet_to_TB(self):
        self.assigned_TBS = (self.assigned_bandwidth *
                             self.estimated_spectral_efficiency * self.TIME_SLOT_WINDOW).astype(int)

        # Remaining TB size (bits) available in this slot
        self.available_TBS = np.copy(self.assigned_TBS)

        # For this TS: TB ID if a TB is transmitted; -1 if nothing is transmitted
        self.TBS_id = np.full(self.NUM_UE, -1, dtype=int)

        for ue_id in self.ue_id_list:
            if self.assigned_TBS[ue_id] <= 0:
                continue

            whether_transmitted = False  # Did we transmit a TB in this TS?

            # If retransmission exists
            if self.rt_dl_buffer_list[ue_id].length > 0:
                TBS = self.rt_dl_buffer_list[ue_id].dequeue(
                    dequeue_type="length", length=1)
                self.TBS_id[ue_id] = TBS[0][0]
                continue

            # If there is a leftover fragment from the previous round
            if self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE] > 0:
                packet_id_in_MAC_float = int(
                    self.dl_temporal_buffer[ue_id][self.INDEX_MAC_PACKET_ID]) % self.NUM_CONTROL_PACKET_ON_MAC  # MAC-stack-local ID of the leftover packet
                packet_TB_number_relation = self.mac_info_manager.packet_TB_number_relation_matrix[
                    ue_id][packet_id_in_MAC_float] + 1
                self.mac_info_manager.packet_TB_number_relation_matrix[
                    ue_id][packet_id_in_MAC_float] = packet_TB_number_relation

                # Update the mapping from MAC packet to TB IDs
                self.mac_info_manager.packet_TB_ids_relation_matrix[ue_id][
                    packet_id_in_MAC_float][packet_TB_number_relation - 1] = self.TB_id_list[ue_id]

                # Check whether the leftover packet still remains
                # Leftover decision
                remaining_packet_size = self.dl_temporal_buffer[ue_id][
                    self.INDEX_PAYLOAD_SIZE] - self.available_TBS[ue_id]
                if remaining_packet_size > 0:
                    # Still leftover
                    self.TB_size[ue_id][self.TB_id_list[ue_id]
                                        ] += remaining_packet_size
                    self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE] = remaining_packet_size
                    self.available_TBS[ue_id] = 0
                    self.mac_info_manager.whether_packet_converted_TB[ue_id][packet_id_in_MAC_float] = 1
                else:
                    # No leftover
                    self.available_TBS[ue_id] -= self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
                    self.TB_size[ue_id][self.TB_id_list[ue_id]
                                        ] += self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
                    self.dl_temporal_buffer[ue_id] = np.zeros(
                        [self.NUM_INFO_PACKET])
                    self.mac_info_manager.whether_packet_converted_TB[ue_id][packet_id_in_MAC_float] = 2
                self.MAC_packets_ids_per_TBS[ue_id][self.TB_id_list[ue_id]].append(
                    packet_id_in_MAC_float)

                # self.MAC_packets_ids_per_TBS[self.TB_id_list[ue_id]]
                whether_transmitted = True

            if (self.available_TBS[ue_id] > 0) and (self.dl_buffer_list[ue_id].length > 0):
                # Pull packets from the MAC buffer
                packets = self.dl_buffer_list[ue_id].dequeue(
                    dequeue_type="volume", volume=self.available_TBS[ue_id])
                packets[:, self.INDEX_MAC_PACKET_ID] = np.arange(
                    self.next_mac_packet_id[ue_id], self.next_mac_packet_id[ue_id] + len(packets)) % self.NUM_CONTROL_PACKET_ON_MAC

                if self.next_mac_packet_id[ue_id] <= self.one_fourth_index < self.next_mac_packet_id[ue_id] + len(packets):
                    self.flesh_MAC_info_manager("one_forth", ue_id)

                if self.next_mac_packet_id[ue_id] <= self.three_fourth_index < self.next_mac_packet_id[ue_id] + len(packets):
                    self.flesh_MAC_info_manager("three_forth", ue_id)

                self.next_mac_packet_id[ue_id] = (
                    packets[-1, self.INDEX_MAC_PACKET_ID] + 1) % self.NUM_CONTROL_PACKET_ON_MAC  # Wrap around as well

                first_packet_id_in_MAC_float = packets[0,
                                                       self.INDEX_MAC_PACKET_ID] % self.NUM_CONTROL_PACKET_ON_MAC
                last_packet_id_in_MAC_float = packets[-1,
                                                      self.INDEX_MAC_PACKET_ID] % self.NUM_CONTROL_PACKET_ON_MAC

                if first_packet_id_in_MAC_float > last_packet_id_in_MAC_float:
                    # Wrap-around case
                    part1 = np.arange(first_packet_id_in_MAC_float,
                                      self.NUM_CONTROL_PACKET_ON_MAC)
                    part2 = np.arange(0, last_packet_id_in_MAC_float + 1)
                    mac_packet_id_range = np.concatenate([part1, part2])
                else:
                    # No wrap-around
                    mac_packet_id_range = np.arange(
                        first_packet_id_in_MAC_float, last_packet_id_in_MAC_float + 1)

                # If first and last are reversed, a full cycle occurred

                self.mac_info_manager.ideal_MAC_floating_packets_infos[
                    ue_id][mac_packet_id_range] = packets
                if len(packets) == 1:
                    # Case: only one new packet fetched from the MAC queue
                    packet_id_in_MAC_float = first_packet_id_in_MAC_float
                    self.mac_info_manager.packet_TB_number_relation_matrix[
                        ue_id][packet_id_in_MAC_float] = 1
                    self.mac_info_manager.packet_TB_ids_relation_matrix[ue_id][
                        packet_id_in_MAC_float][0] = self.TB_id_list[ue_id]
                    # Leftover decision
                    remaining_packet_size = packets[0,
                                                    self.INDEX_PAYLOAD_SIZE] - self.available_TBS[ue_id]
                    self.TB_size[ue_id][self.TB_id_list[ue_id]] = min(
                        packets[0, self.INDEX_PAYLOAD_SIZE], self.available_TBS[ue_id])
                    if remaining_packet_size > 0:
                        # There is leftover
                        self.dl_temporal_buffer[ue_id] = np.copy(packets[0])
                        self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE] = remaining_packet_size
                        self.mac_info_manager.whether_packet_converted_TB[
                            ue_id][packet_id_in_MAC_float] = 1
                    if remaining_packet_size <= 0:
                        # No leftover
                        self.available_TBS[ue_id] -= self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
                        self.dl_temporal_buffer[ue_id] = np.zeros(
                            [self.NUM_INFO_PACKET])
                        self.mac_info_manager.whether_packet_converted_TB[
                            ue_id][packet_id_in_MAC_float] = 2
                    self.MAC_packets_ids_per_TBS[ue_id][self.TB_id_list[ue_id]].append(
                        packet_id_in_MAC_float)
                    whether_transmitted = True

                if len(packets) > 1:
                    # Case: two or more new packets fetched from the MAC queue

                    self.mac_info_manager.packet_TB_number_relation_matrix[
                        ue_id][mac_packet_id_range] = 1
                    self.mac_info_manager.packet_TB_ids_relation_matrix[ue_id,
                                                                        mac_packet_id_range, 0] = self.TB_id_list[ue_id]
                    self.TB_size[ue_id][self.TB_id_list[ue_id]] = min(
                        np.sum(packets[:, self.INDEX_PAYLOAD_SIZE]),
                        self.available_TBS[ue_id],
                    )

                    remaining_packet_size = np.sum(
                        packets[:, self.INDEX_PAYLOAD_SIZE]) - self.assigned_TBS[ue_id]  # Leftover decision
                    if remaining_packet_size > 0:
                        # There is leftover
                        self.dl_temporal_buffer[ue_id] = np.copy(packets[-1])
                        self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE] = remaining_packet_size
                        self.mac_info_manager.whether_packet_converted_TB[
                            ue_id][mac_packet_id_range[:-1]] = 2
                        self.mac_info_manager.whether_packet_converted_TB[
                            ue_id][last_packet_id_in_MAC_float] = 1
                    if remaining_packet_size <= 0:
                        # No leftover
                        self.available_TBS[ue_id] -= self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
                        self.dl_temporal_buffer[ue_id] = np.zeros(
                            [self.NUM_INFO_PACKET])
                        self.mac_info_manager.whether_packet_converted_TB[
                            ue_id,
                            mac_packet_id_range,
                        ] = 2
                    self.MAC_packets_ids_per_TBS[ue_id][self.TB_id_list[ue_id]].extend(
                        mac_packet_id_range)

                    whether_transmitted = True

            if whether_transmitted:
                self.TBS_id[ue_id] = self.TB_id_list[ue_id]
                self.TB_id_list[ue_id] += 1


    def dl_transmit_TBS_to_air(self, ue_id):
        if self.TBS_id[ue_id] != -1:
            return self.TBS_id[ue_id]
        return -1

    def return_buffer_status(self):
        for ue_id in self.ue_id_list:
            self.total_buffer_size[ue_id] = self.rt_dl_buffer_list[ue_id].volume + \
                self.dl_buffer_list[ue_id].volume + \
                self.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
        return self.total_buffer_size


class Air(ParameterClass, TimeManager):
    def __init__(self, target_BLER=0.1):
        # Instance variables (per-instance state)
        self.itarget_bler = 0.1
        self.count_id = 0
        self.random_pass_or_drop_list = np.random.choice(
            [False, True], size=int(1e6), replace=True, p=[target_BLER, 1 - target_BLER])

    def PDSCH_transmit(self, TBS):
        # True => received successfully, False => lost
        ans = self.random_pass_or_drop_list[self.count_id % int(1e6)]
        self.count_id += 1
        return TBS, ans



class MAC_UE(ParameterClass, TimeManager):

    def __init__(self, mac_info_manager, load_path, ue_id, ue_id_for_load_channel_condition, ran_name):
        # Instance variables (per-instance state)
        self.load_path = load_path
        self.ue_id = ue_id
        self.ue_id_for_load_channel_condition = ue_id_for_load_channel_condition
        self.ran_name = ran_name
        self.mac_info_manager = mac_info_manager

        # Equivalent to K2? (ACK/NACK feedback timing offset)
        self.ACKNACK_offset = 4
        # Transmit ACK/NACK. ACK = 1, NACK = 2, no feedback = 0. Assumes simulation window size of 1 ms.
        self.ACK_NACK_list = np.zeros(
            self.NUM_SIMULATION_TIME_SLOTS + self.ACKNACK_offset, dtype=int)
        self.TB_id_for_ACK_NACK = np.zeros(
            self.NUM_SIMULATION_TIME_SLOTS + self.ACKNACK_offset, dtype=int)  # TB ID corresponding to the ACK/NACK

    def load(self):
        self.spectral_eff = np.loadtxt(self.load_path + "spectral_eff" + str(
            self.ue_id_for_load_channel_condition) + ".csv", dtype="float")
        self.snr = np.loadtxt(
            self.load_path + "snr" + str(self.ue_id_for_load_channel_condition) + ".csv", dtype="float")

    def report_channel_condition(self):
        if self.time_index == 0:
            return 0
        else:
            return self.spectral_eff[int((self.time_index - 1) / self.NUM_CONTINUE_AIR)]

    def report_measured_spectrum_efficiency(self):
        return self.spectral_eff[self.time_index]

    def receive_TBS(self, TB_ids, pass_or_drop, MAC_packets_ids_per_TBS):
        if TB_ids == -1:  # No transmission in this time slot
            return []
        else:
            if pass_or_drop:  # Successful reception
                self.ACK_NACK_list[self.time_index + self.ACKNACK_offset] = 1
                self.TB_id_for_ACK_NACK[self.time_index +
                                        self.ACKNACK_offset] = TB_ids
                packet_mac2rlc = self.convert_TBS_2_MAC_packet(
                    MAC_packets_ids_per_TBS[TB_ids])
                return packet_mac2rlc
            else:
                self.ACK_NACK_list[self.time_index + self.ACKNACK_offset] = 2
                self.TB_id_for_ACK_NACK[self.time_index +
                                        self.ACKNACK_offset] = TB_ids
                return []

    def transmit_ACKNACK(self):
        return (self.ACK_NACK_list[self.time_index], self.TB_id_for_ACK_NACK[self.time_index])

    def convert_TBS_2_MAC_packet(self, MAC_packets_ids_per_TBS):

        self.mac_info_manager.number_received_TBs_MAC_packet[self.ue_id][MAC_packets_ids_per_TBS] += 1

        success_received_transmitted_MAC_packet_in_this_TS = (
            self.mac_info_manager.number_received_TBs_MAC_packet[self.ue_id][
                MAC_packets_ids_per_TBS] == self.mac_info_manager.packet_TB_number_relation_matrix[self.ue_id][MAC_packets_ids_per_TBS]
        )  # Among MAC packet IDs transmitted this TS, those for which all TBs were successfully received
        complete_received_MAC_packet_in_this_TS = success_received_transmitted_MAC_packet_in_this_TS * (
            self.mac_info_manager.whether_packet_converted_TB[self.ue_id][MAC_packets_ids_per_TBS] == 2
        )  # MAC packets that are both fully transmitted (no remaining fragments) and fully received
        complete_received_MAC_packet_id = like_value_binary_in_list(
            MAC_packets_ids_per_TBS, complete_received_MAC_packet_in_this_TS)
        packet_mac2rlc = self.mac_info_manager.ideal_MAC_floating_packets_infos[
            self.ue_id][complete_received_MAC_packet_id]
        return packet_mac2rlc


def like_value_binary_in_list(values, binary):
    """Equivalent to values[binary] for a 1D list/array of indices and a boolean mask"""
    answer = []
    if binary.size == 0:
        return []
    for i in range(len(binary)):
        if binary[i]:
            answer.append(values[i])
    return answer
