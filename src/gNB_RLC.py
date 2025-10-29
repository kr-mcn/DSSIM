import numpy as np
from param import ParameterClass, TimeManager
from buffer import Buffer


class RLC(ParameterClass, TimeManager):

    def __init__(self, ue_id_list, ran_name):
        # Instance variables (which have different values for each instance)

        self.ue_id_list = ue_id_list
        self.ran_name = ran_name
        self.num_UE_in_rlc = len(self.ue_id_list)

        self.dl_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)
        self.ul_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)

        self.num_incoming_packets = np.zeros(self.NUM_UE)
        self.num_outgoing_packets = np.zeros(self.NUM_UE)

        self.next_rlc_packet_id = np.zeros(self.NUM_UE, dtype=int)

        for ue_id in range(self.NUM_UE):
            self.dl_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_RLC_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_RLC_DL)
            self.ul_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_RLC_UL, max_length=ParameterClass.BUFF_MAX_LENGTH_RLC_UL)

    def dl_enqueue(self, packet_from_PDCP, ue_id):
        if len(packet_from_PDCP) > 0:
            packet_from_PDCP[:,
                             self.INDEX_RLC_INCOMING_TIMESTAMP_ID] = self.time_index
        self.num_incoming_packets[ue_id] += len(packet_from_PDCP)
        self.dl_buffer_list[ue_id].enqueue(packet_from_PDCP)

    def dl_dequeue(self, ue_id):
        dequeued_packets = self.dl_buffer_list[ue_id].dequeue(
            dequeue_type="all")

        if len(dequeued_packets) > 0:
            self.num_outgoing_packets[ue_id] += len(dequeued_packets)
            dequeued_packets[:, self.INDEX_RLC_PACKET_ID] = np.arange(
                self.next_rlc_packet_id[ue_id],
                self.next_rlc_packet_id[ue_id] + len(dequeued_packets),
            )
            self.next_rlc_packet_id[ue_id] = dequeued_packets[-1,
                                                              self.INDEX_RLC_PACKET_ID] + 1

        return dequeued_packets

    def request_buffer_to_PDCP(self, mac_total_buffer_size, max_mac_buffer_length):
        ans = max_mac_buffer_length - \
            np.array(mac_total_buffer_size / (1500 * 8)) - 1
        ans = np.where(ans < 0, 0, ans).astype(int)

        return ans
