import numpy as np
from param import ParameterClass, TimeManager
from buffer import Buffer


class PDCP_woDC(ParameterClass, TimeManager):
    def __init__(self,
                 ue_id_list,
                 ran_name,
                 max_volume_dl,
                 max_length_dl,
                 max_volume_ul,
                 max_length_ul,
                 ):
        self.ue_id_list = ue_id_list
        self.ran_name = ran_name
        self.num_UE_in_PDCP = len(self.ue_id_list)

        self.dl_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)
        self.ul_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)

        for ue_id in range(self.NUM_UE):
            self.dl_buffer_list[ue_id] = Buffer(
                max_volume=max_volume_dl, max_length=max_length_dl)
            self.ul_buffer_list[ue_id] = Buffer(
                max_volume=max_volume_ul, max_length=max_length_ul)

        self.next_pdcp_packet_id = np.zeros(self.NUM_UE, dtype=int)
        self.DC = False
        self.whether_request_buffer_to_PDCP = np.zeros(
            len(self.ue_id_list), dtype=bool)
        self.num_incoming_packets = np.zeros(self.NUM_UE)
        self.num_outgoing_packets = np.zeros(self.NUM_UE)

    # hogehoge
    def update_ue_ids(self, ue_id_list):
        self.ue_id_list = ue_id_list
        self.num_UE_in_PDCP = len(self.ue_id_list)

    def dl_enqueue(self, N3_packets):
        UE_index_per_packet = N3_packets[:, self.INDEX_ue_id]

        for ue_id in range(self.NUM_UE):
            dequeued_for_UE = N3_packets[UE_index_per_packet == ue_id]
            if len(dequeued_for_UE) == 0:
                continue
            self.num_incoming_packets[ue_id] += len(dequeued_for_UE)
            self.dl_buffer_list[ue_id].enqueue(dequeued_for_UE)

    def dl_dequeue_woDC(self, ue_id):
        if self.request_buffer_to_PDCP[ue_id] <= 0:
            return []
        dequeued_for_UE = self.dl_buffer_list[ue_id].dequeue(
            dequeue_type="length", length=self.request_buffer_to_PDCP[ue_id])
        if len(dequeued_for_UE) == 0:
            return []
        dequeued_for_UE[:, self.INDEX_PDCP_PACKET_ID] = np.arange(
            self.next_pdcp_packet_id[ue_id],
            self.next_pdcp_packet_id[ue_id] + len(dequeued_for_UE),
        )
        self.next_pdcp_packet_id[ue_id] = dequeued_for_UE[-1,
                                                          self.INDEX_PDCP_PACKET_ID] + 1

        self.num_outgoing_packets[ue_id] += len(dequeued_for_UE)
        return dequeued_for_UE

    def exposure(self):
        pass

    def load_rlc_info(self, rlc_info):
        self.request_buffer_to_PDCP = rlc_info


class PDCP_wDC(ParameterClass, TimeManager):
    def __init__(self, ue_id_list, ran_name):
        self.ran_name = ran_name
        self.ue_id_list = ue_id_list
        self.num_UE_in_PDCP = len(self.ue_id_list)

        self.num_incoming_packets = np.zeros(self.NUM_UE)
        self.num_outgoing_packets = np.zeros(self.NUM_UE)

        self.dl_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)
        self.dl_buffer_list_mn = np.zeros(self.NUM_UE, dtype=Buffer)
        self.dl_buffer_list_sn = np.zeros(self.NUM_UE, dtype=Buffer)
        self.ul_buffer_list = np.zeros(self.NUM_UE, dtype=Buffer)

        for ue_id in range(self.NUM_UE):
            self.dl_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_PDCP_WITHDC_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_PDCP_WITHDC_DL)
            self.dl_buffer_list_mn[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_PDCP_WITHDC_MN_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_PDCP_WITHDC_MN_DL)
            self.dl_buffer_list_sn[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_PDCP_WITHDC_SN_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_PDCP_WITHDC_SN_DL)
            self.ul_buffer_list[ue_id] = Buffer(
                max_volume=ParameterClass.BUFF_MAX_VOLUME_PDCP_WITHDC_UL, max_length=ParameterClass.BUFF_MAX_LENGTH_PDCP_WITHDC_UL)

        self.next_pdcp_packet_id = np.zeros(self.NUM_UE, dtype=int)

        self.sn_buffer_size = np.zeros(self.NUM_UE, dtype=int)
        self.mn_buffer_size = np.zeros(self.NUM_UE, dtype=int)
        self.last_serving_node = np.zeros(
            self.NUM_UE, dtype=int)  # 0:mn, 1: sn

        self.request_buffer_to_PDCP_sn = None
        self.request_buffer_to_PDCP_mn = None

        self.theta_S_for_own_RLC = 2 * self.TIME_SLOT_WINDOW
        self.measured_TBS = np.zeros([self.NUM_UE])
        self.max_buffer = 1e9
        self.experienced_throughput = np.zeros([self.NUM_UE]) + 1e-9
        self.RLC_buffer_size = np.zeros([self.NUM_UE]) + 1e-9

        self.desired_payload = np.zeros([self.NUM_UE])
        self.transmitted_time = np.zeros([self.NUM_UE])
        self.void_flag = np.zeros([self.NUM_UE])

        self.threshold_to_distribute_SgNB = np.zeros([self.NUM_UE])

        self.xn_delay = 0.05  # ms
        self.theta_S_for_SgNB = 0.02  # ms

    def update_ue_ids(self, ue_id_list):
        self.ue_id_list = ue_id_list
        self.num_UE_in_PDCP = len(self.ue_id_list)

    def distribute_buffer_to_sn_mn(self):
        splitting_method = "ideal buffer"
        if splitting_method == "random":
            for ue_id in range(self.NUM_UE):
                if self.dl_buffer_list[ue_id].length == 0:
                    continue
                packets = self.dl_buffer_list[ue_id].dequeue(
                    dequeue_type="all")
                if self.last_serving_node[ue_id] == 0:
                    self.dl_buffer_list_sn[ue_id].enqueue(packets)
                    self.last_serving_node[ue_id] = 1
                    continue
                if self.last_serving_node[ue_id] == 1:
                    self.dl_buffer_list_mn[ue_id].enqueue(packets)
                    self.last_serving_node[ue_id] = 0
        if splitting_method == "ideal buffer":
            for ue_id in range(self.NUM_UE):
                if self.request_buffer_to_PDCP_mn[ue_id] > self.request_buffer_to_PDCP_sn[ue_id]:
                    packet_pdcp2rlc_mn = self.dl_buffer_list[ue_id].dequeue(
                        dequeue_type="length",
                        length=self.request_buffer_to_PDCP_mn[ue_id],
                    )
                    if len(packet_pdcp2rlc_mn) > 0:
                        self.dl_buffer_list_mn[ue_id].enqueue(
                            packet_pdcp2rlc_mn)
                    packet_pdcp2rlc_sn = self.dl_buffer_list[ue_id].dequeue(
                        dequeue_type="length",
                        length=self.request_buffer_to_PDCP_sn[ue_id],
                    )
                    if len(packet_pdcp2rlc_sn) > 0:
                        self.dl_buffer_list_sn[ue_id].enqueue(
                            packet_pdcp2rlc_sn)

                else:
                    packet_pdcp2rlc_sn = self.dl_buffer_list[ue_id].dequeue(
                        dequeue_type="length",
                        length=self.request_buffer_to_PDCP_sn[ue_id],
                    )
                    if len(packet_pdcp2rlc_sn) > 0:
                        self.dl_buffer_list_sn[ue_id].enqueue(
                            packet_pdcp2rlc_sn)
                    packet_pdcp2rlc_mn = self.dl_buffer_list[ue_id].dequeue(
                        dequeue_type="length",
                        length=self.request_buffer_to_PDCP_mn[ue_id],
                    )
                    if len(packet_pdcp2rlc_mn) > 0:
                        self.dl_buffer_list_mn[ue_id].enqueue(
                            packet_pdcp2rlc_mn)

    def dl_dequeue(self, ue_id):
        packet_pdcp2rlc_mn = self.dl_buffer_list_mn[ue_id].dequeue()
        if len(packet_pdcp2rlc_mn) != 0:
            # assign pdcp id
            packet_pdcp2rlc_mn[:, self.INDEX_PDCP_PACKET_ID] = np.arange(
                self.next_pdcp_packet_id[ue_id],
                self.next_pdcp_packet_id[ue_id] + len(packet_pdcp2rlc_mn),
            )
            self.next_pdcp_packet_id[ue_id] = packet_pdcp2rlc_mn[-1,
                                                                 self.INDEX_PDCP_PACKET_ID] + 1

        packet_pdcp2rlc_sn = self.dl_buffer_list_sn[ue_id].dequeue()
        if len(packet_pdcp2rlc_sn) != 0:
            # assign pdcp id
            packet_pdcp2rlc_sn[:, self.INDEX_PDCP_PACKET_ID] = np.arange(
                self.next_pdcp_packet_id[ue_id],
                self.next_pdcp_packet_id[ue_id] + len(packet_pdcp2rlc_sn),
            )
            self.next_pdcp_packet_id[ue_id] = packet_pdcp2rlc_sn[-1,
                                                                 self.INDEX_PDCP_PACKET_ID] + 1
        return packet_pdcp2rlc_mn, packet_pdcp2rlc_sn

    def dl_enqueue(self, N3_packets):
        UE_index_per_packet = N3_packets[:, self.INDEX_ue_id]

        for ue_id in range(self.NUM_UE):
            dequeued_for_UE = N3_packets[UE_index_per_packet == ue_id]
            if len(dequeued_for_UE) == 0:
                continue
            self.num_incoming_packets[ue_id] += len(dequeued_for_UE)
            self.dl_buffer_list[ue_id].enqueue(dequeued_for_UE)

    def dl_dequeue_woDC(self, ue_id):
        if self.request_buffer_to_PDCP[ue_id] <= 0:
            return []
        dequeued_for_UE = self.dl_buffer_list[ue_id].dequeue(
            dequeue_type="length", length=self.request_buffer_to_PDCP[ue_id])
        if len(dequeued_for_UE) == 0:
            return []
        dequeued_for_UE[:, self.INDEX_PDCP_PACKET_ID] = np.arange(
            self.next_pdcp_packet_id[ue_id],
            self.next_pdcp_packet_id[ue_id] + len(dequeued_for_UE),
        )
        self.next_pdcp_packet_id[ue_id] = dequeued_for_UE[-1,
                                                          self.INDEX_PDCP_PACKET_ID] + 1
        return dequeued_for_UE

    def load_rlc_info(self, request_buffer_to_PDCP_mn, request_buffer_to_PDCP_sn):
        self.request_buffer_to_PDCP_mn = request_buffer_to_PDCP_mn
        self.request_buffer_to_PDCP_sn = request_buffer_to_PDCP_sn

    def distribute_PDCP_buffer(self):
        TRB_for_own_RLC = np.zeros([self.NUM_UE])
        TRB_for_SgNB = np.zeros([self.NUM_UE])
        if self.time_index == 0:
            pass
        else:
            self.threshold_to_distribute_SgNB = self.experienced_throughput * \
                (self.theta_S_for_SgNB + self.xn_delay)
            for i in range(self.NUM_UE):
                if self.threshold_to_distribute_SgNB[i] > self.buffer_size[i]:
                    TRB_for_SgNB[i] = 0
                    TRB_for_own_RLC[i] = np.min(
                        self.experienced_throughput[i] *
                        self.theta_S_for_own_RLC - self.RLC_buffer_size[i],
                        self.buffer_size[i],
                    )
                    self.buffer_size[i] -= TRB_for_own_RLC[i]
                if self.threshold_to_distribute_SgNB[i] <= self.buffer_size[i]:
                    TRB_for_SgNB[i] = np.min(
                        self.desired_payload[i], self.buffer_size[i])
                    self.buffer_size[i] -= TRB_for_SgNB[i]
                    TRB_for_own_RLC[i] = np.min(
                        self.experienced_throughput[i] *
                        self.theta_S_for_own_RLC - self.RLC_buffer_size[i],
                        self.buffer_size[i],
                    )
                    self.buffer_size[i] -= TRB_for_own_RLC[i]

        return TRB_for_own_RLC, TRB_for_SgNB
