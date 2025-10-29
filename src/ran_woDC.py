import numpy as np
from pathlib import Path
from param import ParameterClass, TimeManager
from buffer import Buffer, SaveBuffer
from logger import Logger
from wiredlink import WiredLink
from upf import UPF
from gNB_PDCP import PDCP_wDC, PDCP_woDC
from gNB_RLC import RLC
from mac import MAC, MAC_UE, Air, MAC_INFO_MANAGER
from ue_pdcprlc import PDCP_RLC_UE
from quic import QUIC
from mpquic import MPQUIC
import pdb
import os
import datetime
import csv
import time
import matplotlib.pyplot as plt


class RAN_wo_DC(ParameterClass, TimeManager):
    def __init__(
        self,
        propagation_load_path=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_LB,
        log_save_path=ParameterClass.HEAVY_DATA_PATH + ParameterClass.LOG_SAVE_PATH,
        ran_name="mn",
        max_volume_dl=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_DL_DEFAULT,
        max_length_dl=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_DL_DEFAULT,
        max_volume_ul=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_UL_DEFAULT,
        max_length_ul=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_UL_DEFAULT,
    ):
        ue_id_list = np.arange(ParameterClass.NUM_UE)
        ue_id_list_for_load_channel_condition = np.array([0, 1, 6, 7, 9])
        ue_id_list_for_load_channel_condition = ue_id_list

        mac_info_manager = MAC_INFO_MANAGER()

        self.logger = Logger()

        self.pdcp = PDCP_woDC(ue_id_list=ue_id_list, ran_name=ran_name,
                              max_volume_dl=max_volume_dl,
                              max_length_dl=max_length_dl,
                              max_volume_ul=max_volume_ul,
                              max_length_ul=max_length_ul,)
        self.rlc = RLC(ue_id_list=ue_id_list, ran_name=ran_name)
        self.mac = MAC(mac_info_manager, ue_id_list=ue_id_list,
                       ran_name=ran_name)
        self.air = Air(target_BLER=0.1)

        self.load_path = propagation_load_path
        self.mac_ue_list = np.zeros([ParameterClass.NUM_UE], dtype=MAC_UE)
        self.rlc_ue_list = np.zeros([ParameterClass.NUM_UE], dtype=PDCP_RLC_UE)
        self.pdcp_ue_list = np.zeros(
            [ParameterClass.NUM_UE], dtype=PDCP_RLC_UE)

        # SaveBuffer for storing packets

        self.save_path = log_save_path
        self.max_mac_buffer_length = int(
            self.mac.dl_buffer_list[0].max_length * 0.9)

        for ue_id in range(ParameterClass.NUM_UE):
            self.mac_ue_list[ue_id] = MAC_UE(
                mac_info_manager=mac_info_manager,
                load_path=self.load_path,
                ue_id=ue_id,
                ran_name=ran_name,
                ue_id_for_load_channel_condition=ue_id_list_for_load_channel_condition[ue_id],
            )
            self.mac_ue_list[ue_id].load()
            self.rlc_ue_list[ue_id] = PDCP_RLC_UE(
                ue_id=ue_id, class_type="rlc", ran_name=ran_name)
            self.pdcp_ue_list[ue_id] = PDCP_RLC_UE(
                ue_id=ue_id, class_type="pdcp", ran_name=ran_name)

        self.channel_condition = np.zeros([ParameterClass.NUM_UE])
        self.measured_spectral_efficiency = np.zeros([ParameterClass.NUM_UE])

        self.debug_log = []
        self.logger.init_savebuffer(self.save_path)

        # For responsiveness evaluation
        self.mac_buff_before = [0]*ParameterClass.NUM_UE
        self.mac_buff_after = [0]*ParameterClass.NUM_UE
        self.received_mac = [0]*ParameterClass.NUM_UE

    def perform_one_time_slot(self, packet_N32pdcp, dl_ip_ue_buffer_list):
        num_packet_per_ts = 0
        num_mac_packet_per_ts = 0
        num_rlc_packet_per_ts = 0

        num_gnb_pdcp_per_ts = 0
        num_gnb_rlc_per_ts = 0
        communicate_ue = []

        # Add packets from UPF into the gNB PDCP buffer
        self.pdcp.dl_enqueue(packet_N32pdcp)
        num_gnb_pdcp_per_ts += len(packet_N32pdcp)
        UE_index_per_packet = packet_N32pdcp[:, self.INDEX_ue_id]
        packet_count_per_ue = np.bincount(
            UE_index_per_packet, minlength=self.NUM_UE)

        self.logger.load_packet(packet_N32pdcp, name="N32pdcp")

        # Load RLC information into PDCP
        mac_total_buffer_size = self.mac.return_buffer_status()
        request_buffer_to_PDCP = self.rlc.request_buffer_to_PDCP(
            mac_total_buffer_size, self.max_mac_buffer_length)
        self.pdcp.load_rlc_info(request_buffer_to_PDCP)

        for ue_id in self.pdcp.ue_id_list:
            # Enqueue from PDCP to each RLC buffer
            packet_pdcp2rlc = self.pdcp.dl_dequeue_woDC(ue_id)
            self.rlc.dl_enqueue(packet_pdcp2rlc, ue_id)

            # Logging
            num_gnb_rlc_per_ts += len(packet_pdcp2rlc)
            self.logger.load_packet(
                packet_pdcp2rlc, ue_id=ue_id, name="pdcp2rlc")
            received_pdcp = packet_count_per_ue[ue_id]
            capacity_pdcp = request_buffer_to_PDCP[ue_id]
            transmitted_pdcp = len(packet_pdcp2rlc)
            self.logger.load_tx_opportunity_loss(
                "pdcp", ue_id, capacity_pdcp, transmitted_pdcp)
            self.logger.load_over_reception(
                "pdcp", ue_id, received_pdcp, transmitted_pdcp)

            # Enqueue from RLC to the MAC buffer
            packet_rlc2mac = self.rlc.dl_dequeue(ue_id)
            self.mac.dl_enqueue(packet_rlc2mac, ue_id)

            # Logging
            self.logger.load_packet(
                packet_rlc2mac, ue_id=ue_id, name="rlc2mac")
            self.received_mac[ue_id] = len(packet_rlc2mac)

        self.logger.load_per_step_gNB_pdcp_log(self.pdcp)
        self.logger.load_per_step_gNB_rlc_log(self.rlc)

        for ue_id in self.pdcp.ue_id_list:
            self.channel_condition[ue_id] = self.mac_ue_list[ue_id].report_channel_condition(
            )
            self.measured_spectral_efficiency[ue_id] = self.mac_ue_list[ue_id].report_measured_spectrum_efficiency(
            )

        # gNB MAC processing
        self.mac.measure_channel_condition(self.channel_condition)
        self.mac.assign_bandwidth()
        self.mac_buff_before = [self.mac.total_buffer_size[ue_id]
                                for ue_id in range(ParameterClass.NUM_UE)]
        self.mac.assign_packet_to_TB()
        self.mac_buff_after = [
            self.mac.rt_dl_buffer_list[ue_id].volume +
            self.mac.dl_buffer_list[ue_id].volume +
            self.mac.dl_temporal_buffer[ue_id][self.INDEX_PAYLOAD_SIZE]
            for ue_id in range(ParameterClass.NUM_UE)
        ]

        for ue_id in range(ParameterClass.NUM_UE):
            capacity_mac = self.mac.mac_tx_capacity[ue_id]
            transmitted_mac = max(
                0, self.mac_buff_before[ue_id] - self.mac_buff_after[ue_id])  # bit

            self.logger.load_tx_opportunity_loss(
                "mac", ue_id, capacity_mac, transmitted_mac)  # bit
            self.logger.load_over_reception(
                "mac", ue_id, self.received_mac[ue_id]*ParameterClass.MTU_BIT_SIZE, transmitted_mac)  # bit
        # Logging
        self.logger.load_per_step_gNB_mac_log(self.mac)

        for ue_id in self.rlc.ue_id_list:
            # Per-UE MAC transmission processing
            TBS_PDSCH = self.mac.dl_transmit_TBS_to_air(ue_id)
            TBS, pass_or_drop = self.air.PDSCH_transmit(TBS_PDSCH)

            # Logging
            self.logger.load_per_step_ue_mac_log(ue_id, pass_or_drop, TBS)

            # UE-side processing: MAC -> RLC
            packet_mac2rlc = self.mac_ue_list[ue_id].receive_TBS(
                TBS, pass_or_drop, self.mac.MAC_packets_ids_per_TBS[ue_id])
            self.rlc_ue_list[ue_id].load_data(packet_mac2rlc)

            # Logging
            self.logger.load_packet(
                packet_mac2rlc, name="mac2rlc", ue_id=ue_id)
            num_mac_packet_per_ts += len(packet_mac2rlc)

            if TBS != -1:
                communicate_ue.append(ue_id)

            # UE ACK/NACK processing
            ACK_NACK, TB_id_for_ACK_NACK = self.mac_ue_list[ue_id].transmit_ACKNACK(
            )
            self.mac.receive_ACK_NACK(ACK_NACK, TB_id_for_ACK_NACK, ue_id)

            # RLC reordering
            packet_RLC2PDCP = self.rlc_ue_list[ue_id].reorder()

            # Logging
            num_rlc_packet_per_ts += len(packet_RLC2PDCP)
            self.logger.load_packet(
                packet_RLC2PDCP, name="rlc2pdcp", ue_id=ue_id)

            # PDCP processing
            self.pdcp_ue_list[ue_id].load_data(packet_RLC2PDCP)
            packet_PDCP2IP = self.pdcp_ue_list[ue_id].reorder()

            # Logging
            num_packet_per_ts += len(packet_PDCP2IP)
            self.logger.load_per_step_ue_rlc_log(
                ue_id=ue_id, rlc_ue=self.rlc_ue_list[ue_id])
            self.logger.load_per_step_ue_pdcp_log(
                ue_id=ue_id, pdcp_ue=self.pdcp_ue_list[ue_id])
            self.logger.load_packet(
                packet_PDCP2IP, name="pdcp2ip", ue_id=ue_id)

            dl_ip_ue_buffer_list[ue_id].enqueue(packet_PDCP2IP)

        if TimeManager.time_index % 1000 == 0 and TimeManager.time_index != 0:
            self.save_all_info()
        return dl_ip_ue_buffer_list

    def save_all_info(self, csv_conversion=False, plot=False):
        self.logger.save(self.save_path)

        file_name = os.path.join(self.save_path, "_debug_log.txt")
        with open(file_name, mode="w", newline="") as file:
            for value in self.debug_log:
                # If the value is a list or tuple, join elements with commas
                if isinstance(value, (list, tuple)):
                    file.write(", ".join(map(str, value)) + "\n")
                else:
                    file.write(str(value) + "\n")
        if csv_conversion:
            self.logger.make_csv_files(self.save_path)
        if plot:
            self.logger.plot(self.save_path)
