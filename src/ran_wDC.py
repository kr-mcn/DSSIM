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


class RAN_w_DC(ParameterClass, TimeManager):
    def __init__(
        self,
        ran_name,
        propagation_load_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_LB,
        log_save_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "mn/",
        propagation_load_path_sn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_HB,
        log_save_path_sn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "sn/",
    ):
        super().__init__()
        ue_id_list = np.arange(ParameterClass.NUM_UE)
        self.ue_id_for_load_channel_condition = ue_id_list

        self.pdcp = PDCP_wDC(ue_id_list=ue_id_list, ran_name=ran_name)

        self.mac_info_manager_mn = MAC_INFO_MANAGER()
        mac_info_manager_sn = MAC_INFO_MANAGER()

        self.rlc_mn = RLC(ue_id_list=ue_id_list, ran_name="mn")
        self.mac_mn = MAC(mac_info_manager=self.mac_info_manager_mn,
                          ue_id_list=ue_id_list, ran_name="mn")
        self.air_mn = Air(target_BLER=0.1)

        self.rlc_sn = RLC(ue_id_list=ue_id_list, ran_name="sn")
        self.mac_sn = MAC(mac_info_manager=mac_info_manager_sn,
                          ue_id_list=ue_id_list, ran_name="sn")
        self.air_sn = Air(target_BLER=0.1)

        self.logger_mn = Logger()
        self.logger_sn = Logger()

        self.load_path_sn = propagation_load_path_sn
        self.load_path_mn = propagation_load_path_mn

        self.save_path_mn = log_save_path_mn
        self.save_path_sn = log_save_path_sn

        self.logger_mn.init_savebuffer(self.save_path_mn)
        self.logger_sn.init_savebuffer(self.save_path_sn)

        self.pdcp_ue_list = np.zeros([ParameterClass.NUM_UE], dtype=Buffer)

        self.mac_ue_list_mn = np.zeros([ParameterClass.NUM_UE], dtype=MAC_UE)
        self.rlc_ue_list_mn = np.zeros(
            [ParameterClass.NUM_UE], dtype=PDCP_RLC_UE)
        self.mac_ue_list_sn = np.zeros([ParameterClass.NUM_UE], dtype=MAC_UE)
        self.rlc_ue_list_sn = np.zeros(
            [ParameterClass.NUM_UE], dtype=PDCP_RLC_UE)

        for ue_id in ue_id_list:
            self.mac_ue_list_mn[ue_id] = MAC_UE(
                mac_info_manager=self.mac_info_manager_mn,
                load_path=self.load_path_mn,
                ue_id=ue_id_list[ue_id],
                ue_id_for_load_channel_condition=self.ue_id_for_load_channel_condition[ue_id],
                ran_name="mn",
            )
            self.mac_ue_list_mn[ue_id].load()
            self.rlc_ue_list_mn[ue_id] = PDCP_RLC_UE(
                ue_id=ue_id, class_type="rlc", ran_name="mn")

            self.mac_ue_list_sn[ue_id] = MAC_UE(
                mac_info_manager=mac_info_manager_sn, load_path=self.load_path_sn, ue_id=ue_id_list[
                    ue_id], ue_id_for_load_channel_condition=self.ue_id_for_load_channel_condition[ue_id], ran_name="sn"
            )
            self.mac_ue_list_sn[ue_id].load()
            self.rlc_ue_list_sn[ue_id] = PDCP_RLC_UE(
                ue_id=ue_id, class_type="rlc", ran_name="sn")

            self.pdcp_ue_list[ue_id] = PDCP_RLC_UE(
                ue_id, class_type="pdcp", ran_name="mn")

        self.channel_condition_mn = np.zeros([ParameterClass.NUM_UE])
        self.measured_spectral_efficiency_mn = np.zeros(
            [ParameterClass.NUM_UE])
        self.channel_condition_sn = np.zeros([ParameterClass.NUM_UE])
        self.measured_spectral_efficiency_sn = np.zeros(
            [ParameterClass.NUM_UE])

        self.max_mac_buffer_length = int(
            self.mac_mn.dl_buffer_list[0].max_length * 0.9)

        self.debug_log = []

    def perform_one_time_slot(self, packet_N32pdcp, dl_ip_ue_buffer_list):

        num_packet_per_ts = 0
        num_mac_packet_per_ts = 0
        num_rlc_packet_per_ts = 0
        communicate_ue = []

        self.logger_mn.load_packet(packet_N32pdcp, name="N32pdcp")

        self.pdcp.dl_enqueue(packet_N32pdcp)
        mac_total_buffer_size_mn = self.mac_mn.return_buffer_status()
        request_buffer_to_PDCP_mn = self.rlc_mn.request_buffer_to_PDCP(
            mac_total_buffer_size_mn, self.max_mac_buffer_length)
        mac_total_buffer_size_sn = self.mac_sn.return_buffer_status()
        request_buffer_to_PDCP_sn = self.rlc_sn.request_buffer_to_PDCP(
            mac_total_buffer_size_sn, self.max_mac_buffer_length)

        self.pdcp.load_rlc_info(request_buffer_to_PDCP_mn,
                                request_buffer_to_PDCP_sn)
        self.pdcp.distribute_buffer_to_sn_mn()

        for ue_id in self.pdcp.ue_id_list:
            # Enqueue from PDCP to each RLC buffer
            packet_pdcp2rlc_mn, packet_pdcp2rlc_sn = self.pdcp.dl_dequeue(
                ue_id)

            # Logging
            self.logger_mn.load_packet(
                packet_pdcp2rlc_mn, ue_id=ue_id, name="pdcp2rlc")
            self.logger_sn.load_packet(
                packet_pdcp2rlc_sn, ue_id=ue_id, name="pdcp2rlc")

            self.rlc_mn.dl_enqueue(packet_pdcp2rlc_mn, ue_id)
            packet_rlc2mac_mn = self.rlc_mn.dl_dequeue(ue_id)
            self.mac_mn.dl_enqueue(packet_rlc2mac_mn, ue_id)

            self.rlc_sn.dl_enqueue(packet_pdcp2rlc_sn, ue_id)
            packet_rlc2mac_sn = self.rlc_sn.dl_dequeue(ue_id)
            self.mac_sn.dl_enqueue(packet_rlc2mac_sn, ue_id)

            self.logger_mn.load_packet(
                packet_rlc2mac_mn, ue_id=ue_id, name="rlc2mac")
            self.logger_sn.load_packet(
                packet_rlc2mac_sn, ue_id=ue_id, name="rlc2mac")

        self.logger_mn.load_per_step_gNB_pdcp_log(self.pdcp)

        # Per-UE measurement step
        for ue_id in range(ParameterClass.NUM_UE):
            self.channel_condition_mn[ue_id] = self.mac_ue_list_mn[ue_id].report_channel_condition(
            )
            self.measured_spectral_efficiency_mn[ue_id] = self.mac_ue_list_mn[ue_id].report_measured_spectrum_efficiency(
            )
            self.channel_condition_sn[ue_id] = self.mac_ue_list_sn[ue_id].report_channel_condition(
            )
            self.measured_spectral_efficiency_sn[ue_id] = self.mac_ue_list_sn[ue_id].report_measured_spectrum_efficiency(
            )

        self.mac_mn.measure_channel_condition(self.channel_condition_mn)
        self.mac_mn.assign_bandwidth()
        self.mac_mn.assign_packet_to_TB()
        self.logger_mn.load_per_step_gNB_mac_log(self.mac_mn)

        self.mac_sn.measure_channel_condition(self.channel_condition_sn)
        self.mac_sn.assign_bandwidth()
        self.mac_sn.assign_packet_to_TB()
        self.logger_sn.load_per_step_gNB_mac_log(self.mac_sn)

        for ue_id in self.pdcp.ue_id_list:
            # MAC transmission on the master node
            TBS_PDSCH_mn = self.mac_mn.dl_transmit_TBS_to_air(ue_id)
            TBS, pass_or_drop = self.air_mn.PDSCH_transmit(TBS_PDSCH_mn)
            self.logger_mn.load_per_step_ue_mac_log(ue_id, pass_or_drop, TBS)

            packet_mac2rlc_mn = self.mac_ue_list_mn[ue_id].receive_TBS(
                TBS,
                pass_or_drop,
                self.mac_mn.MAC_packets_ids_per_TBS[ue_id],
            )
            self.logger_mn.load_packet(
                packet_mac2rlc_mn, name="mac2rlc", ue_id=ue_id)

            if TBS != -1:
                communicate_ue.append(ue_id)
            ACK_NACK, TB_id_for_ACK_NACK = self.mac_ue_list_mn[ue_id].transmit_ACKNACK(
            )
            self.mac_mn.receive_ACK_NACK(ACK_NACK, TB_id_for_ACK_NACK, ue_id)

            # MAC transmission on the secondary node
            TBS_PDSCH_sn = self.mac_sn.dl_transmit_TBS_to_air(ue_id)
            TBS, pass_or_drop = self.air_sn.PDSCH_transmit(TBS_PDSCH_sn)
            self.logger_sn.load_per_step_ue_mac_log(ue_id, pass_or_drop, TBS)

            packet_mac2rlc_sn = self.mac_ue_list_sn[ue_id].receive_TBS(
                TBS,
                pass_or_drop,
                self.mac_sn.MAC_packets_ids_per_TBS[ue_id],
            )

            self.logger_sn.load_packet(
                packet_mac2rlc_sn, name="mac2rlc", ue_id=ue_id)

            if TBS != -1:
                communicate_ue.append(ue_id)
            num_mac_packet_per_ts += len(packet_mac2rlc_mn) + \
                len(packet_mac2rlc_sn)

            ACK_NACK, TB_id_for_ACK_NACK = self.mac_ue_list_sn[ue_id].transmit_ACKNACK(
            )
            self.mac_sn.receive_ACK_NACK(ACK_NACK, TB_id_for_ACK_NACK, ue_id)

            # RLC processing on the master node
            self.rlc_ue_list_mn[ue_id].load_data(packet_mac2rlc_mn)
            packet_RLC2PDCP_mn = self.rlc_ue_list_mn[ue_id].reorder()
            self.logger_mn.load_per_step_ue_rlc_log(
                ue_id=ue_id, rlc_ue=self.rlc_ue_list_mn[ue_id])
            self.logger_mn.load_packet(
                packet_RLC2PDCP_mn, name="rlc2pdcp", ue_id=ue_id)

            # PDCP processing on the master node
            self.pdcp_ue_list[ue_id].load_data(packet_RLC2PDCP_mn)
            packet_PDCP2IP_mn = self.pdcp_ue_list[ue_id].reorder()

            # Logging
            self.logger_mn.load_per_step_ue_rlc_log(
                ue_id=ue_id, rlc_ue=self.rlc_ue_list_mn[ue_id])
            self.logger_mn.load_per_step_ue_pdcp_log(
                ue_id=ue_id, pdcp_ue=self.pdcp_ue_list[ue_id])
            self.logger_mn.load_packet(
                packet_PDCP2IP_mn, name="pdcp2ip", ue_id=ue_id)

            # RLC processing on the secondary node
            self.rlc_ue_list_sn[ue_id].load_data(packet_mac2rlc_sn)
            packet_RLC2PDCP_sn = self.rlc_ue_list_sn[ue_id].reorder()
            num_rlc_packet_per_ts += len(packet_RLC2PDCP_sn)
            self.logger_sn.load_per_step_ue_rlc_log(
                ue_id=ue_id, rlc_ue=self.rlc_ue_list_sn[ue_id])

            # Logging
            self.logger_sn.load_packet(
                packet_RLC2PDCP_sn, name="rlc2pdcp", ue_id=ue_id)
            num_rlc_packet_per_ts += len(packet_RLC2PDCP_mn) + \
                len(packet_RLC2PDCP_sn)

            # PDCP processing on the secondary node
            self.pdcp_ue_list[ue_id].load_data(packet_RLC2PDCP_sn)
            packet_PDCP2IP_sn = self.pdcp_ue_list[ue_id].reorder()

            # Logging
            self.logger_sn.load_per_step_ue_rlc_log(
                ue_id=ue_id, rlc_ue=self.rlc_ue_list_sn[ue_id])
            self.logger_sn.load_per_step_ue_pdcp_log(
                ue_id=ue_id, pdcp_ue=self.pdcp_ue_list[ue_id])
            self.logger_mn.load_packet(
                packet_PDCP2IP_sn, name="pdcp2ip", ue_id=ue_id)

            num_packet_per_ts += len(packet_PDCP2IP_mn) + \
                len(packet_PDCP2IP_sn)

            dl_ip_ue_buffer_list[ue_id].enqueue(packet_PDCP2IP_mn)
            dl_ip_ue_buffer_list[ue_id].enqueue(packet_PDCP2IP_sn)

        print(
            "Time index is ",
            TimeManager.time_index,
        )
        if TimeManager.time_index % 100 == 0 and TimeManager.time_index != 0:
            self.save_all_info()
        return dl_ip_ue_buffer_list

    def save_all_info(self, csv_conversion=False, plot=False):
        self.logger_mn.save(self.save_path_mn)
        self.logger_sn.save(self.save_path_sn)

        file_name = os.path.join(self.save_path_mn, "_debug_log.txt")
        with open(file_name, mode="w", newline="") as file:
            for value in self.debug_log:
                # If the value is a list or tuple, join elements with commas
                if isinstance(value, (list, tuple)):
                    file.write(", ".join(map(str, value)) + "\n")
                else:
                    file.write(str(value) + "\n")
        if csv_conversion:
            self.logger_mn.make_csv_files(self.save_path_mn)
            self.logger_sn.make_csv_files(self.save_path_sn)
        if plot:
            self.logger_mn.plot(self.save_path_mn)
            self.logger_sn.plot(self.save_path_sn)
