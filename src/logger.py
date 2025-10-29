import numpy as np
from pathlib import Path
from param import ParameterClass, TimeManager
import os
from buffer import SaveBuffer
import shutil
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numbers
from collections import defaultdict


class Logger(ParameterClass, TimeManager):
    def __init__(self):
        # gnb mac
        self.total_buffer_size = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])  # MAC buffer size per UE per time index
        self.assigned_TBS = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.estimated_channel_condition = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.estimated_spectral_efficiency = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.experienced_throughput = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.PF_metic = np.zeros([self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.assigned_bandwidth = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.measured_spectrum_efficiency = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.priority_order = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.rt_dl_buffer_length = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        self.priority_mask = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # gnb pdcp
        self.gnb_pdcp_buffer_length = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])  # PDCP buffer size per UE per time index
        # Cumulative number of packets delivered to gNB PDCP per UE per time index (cumulative!)
        self.gnb_pdcp_num_incoming_packets = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        # Cumulative number of packets that gNB PDCP forwarded to RLC per UE per time index (cumulative!)
        self.gnb_pdcp_num_outgoing_packets = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # gnb rlc
        self.gnb_rlc_buffer_length = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])  # RLC buffer size per UE per time index
        # Cumulative number of packets delivered from gNB PDCP to RLC per UE per time index (cumulative!)
        self.gnb_rlc_num_incoming_packets = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        # Cumulative number of packets that gNB RLC forwarded to MAC per UE per time index (cumulative!)
        self.gnb_rlc_num_outgoing_packets = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # gnb pdcp rlc log
        # Number of packets delivered from PDCP to RLC per UE per time index
        self.packet_pdcp2rlc_len = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # UE MAC log
        # Whether reception succeeded (True) or failed (False) per UE per time index
        self.pass_or_drop = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        # Transport Block Size (bit) per UE per time index. -1 if not transmitted
        self.TBS = np.zeros([self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # UE RLC log
        # Cumulative number of packets delivered from UE MAC to RLC per UE per time index (accumulated from time index = 0)
        self.ue_rlc_incoming = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        # Cumulative number of packets delivered from UE RLC to PDCP per UE per time index (accumulated from time index = 0)
        self.ue_rlc_outgoing = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        # UE PDCP log
        # Cumulative number of packets delivered from UE RLC to PDCP per UE per time index (accumulated from time index = 0)
        self.ue_pdcp_incoming = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])
        # Cumulative number of packets delivered from UE PDCP to IP per UE per time index (accumulated from time index = 0)
        self.ue_pdcp_outgoing = np.zeros(
            [self.NUM_SIMULATION_TIME_SLOTS, self.NUM_UE])

        self.dl_rlc_gnb_buffer_for_save_list = np.zeros(
            [ParameterClass.NUM_UE], dtype=SaveBuffer)
        self.dl_mac_gnb_buffer_for_save_list = np.zeros(
            [ParameterClass.NUM_UE], dtype=SaveBuffer)
        self.dl_ip_packet_ue_list_for_save = np.zeros(
            [ParameterClass.NUM_UE], dtype=SaveBuffer)
        self.dl_pdcp_packet_ue_list_for_save = np.zeros(
            [ParameterClass.NUM_UE], dtype=SaveBuffer)
        self.dl_rlc_packet_ue_list_for_save = np.zeros(
            [ParameterClass.NUM_UE], dtype=SaveBuffer)

        # log for L4
        # e.g. Dict[QUIC, Dict[UE3, Dict[send_thpt, List[value]]]] / dict_key becomes the directory name.
        self.L4_logs: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}

        # For Responsiveness Evaluation
        # tx_opportunity_loss_occurrences: Number of transmission-opportunity loss occurrences
        # tx_opportunity_loss_amount: Cumulative amount of transmission-opportunity loss
        # over_reception_occurrences: Number of over-reception occurrences
        # over_reception_amount: Cumulative amount when over-reception occurred
        self.tx_opportunity_loss_occurrences_pdcp = [0] * ParameterClass.NUM_UE
        self.tx_opportunity_loss_amount_pdcp = [0] * ParameterClass.NUM_UE
        self.tx_opportunity_loss_occurrences_mac = [0] * ParameterClass.NUM_UE
        self.tx_opportunity_loss_amount_mac = [0] * ParameterClass.NUM_UE
        self.tx_opportunity_pdcp = [0] * ParameterClass.NUM_UE
        self.tx_opportunity_mac = [0] * ParameterClass.NUM_UE
        self.over_reception_occurrences_pdcp = [0] * ParameterClass.NUM_UE
        self.over_reception_amount_pdcp = [0] * ParameterClass.NUM_UE
        self.over_reception_occurrences_mac = [0] * ParameterClass.NUM_UE
        self.over_reception_amount_mac = [0] * ParameterClass.NUM_UE

    def init_savebuffer(self, log_save_path):
        save_path = log_save_path
        npy_save_path = log_save_path + "npy/"
        directory = Path(save_path)
        if not directory.exists():
            directory.mkdir(parents=True)
        directory = Path(npy_save_path)
        if not directory.exists():
            directory.mkdir(parents=True)

        self.dl_pdcp_gnb_buffer_for_save = SaveBuffer(
            savefilename=npy_save_path + "dl_pdcp_gnb_buffer_for_save_")

        for ue_id in range(ParameterClass.NUM_UE):
            self.dl_ip_packet_ue_list_for_save[ue_id] = SaveBuffer(
                savefilename=npy_save_path + "dl_ip_packet_ue_" + str(ue_id) + "_")
            self.dl_pdcp_packet_ue_list_for_save[ue_id] = SaveBuffer(
                savefilename=npy_save_path + "dl_pdcp_packet_ue_" + str(ue_id) + "_")
            self.dl_rlc_packet_ue_list_for_save[ue_id] = SaveBuffer(
                savefilename=npy_save_path + "dl_rlc_packet_ue_" + str(ue_id) + "_")

            self.dl_rlc_gnb_buffer_for_save_list[ue_id] = SaveBuffer(
                savefilename=npy_save_path + "dl_rlc_gnb_buffer_for_save" + str(ue_id) + "_")
            self.dl_mac_gnb_buffer_for_save_list[ue_id] = SaveBuffer(
                savefilename=npy_save_path + "dl_mac_gnb_buffer_for_save" + str(ue_id) + "_")

    def load_packet(self, packet, name="N32pdcp", ue_id=None):
        if name == "N32pdcp":
            self.dl_pdcp_gnb_buffer_for_save.enqueue(packet)

        if name == "pdcp2rlc":
            self.dl_rlc_gnb_buffer_for_save_list[ue_id].enqueue(packet)

        if name == "rlc2mac":
            self.dl_mac_gnb_buffer_for_save_list[ue_id].enqueue(packet)

        if name == "mac2rlc":
            self.dl_rlc_packet_ue_list_for_save[ue_id].enqueue(packet)

        if name == "rlc2pdcp":
            self.dl_pdcp_packet_ue_list_for_save[ue_id].enqueue(packet)

        if name == "pdcp2ip":
            self.dl_ip_packet_ue_list_for_save[ue_id].enqueue(packet)

    def load_per_step_gNB_mac_log(self, mac):
        # kaizennkaizenn  testtest
        self.total_buffer_size[self.time_index] = mac.total_buffer_size
        self.assigned_TBS[self.time_index] = mac.assigned_TBS
        self.estimated_channel_condition[self.time_index] = mac.estimated_channel_condition
        self.estimated_spectral_efficiency[self.time_index] = mac.estimated_spectral_efficiency
        self.experienced_throughput[self.time_index] = mac.experienced_throughput
        self.PF_metic[self.time_index] = mac.PF_metic
        self.assigned_bandwidth[self.time_index] = mac.assigned_bandwidth
        self.measured_spectrum_efficiency[self.time_index] = mac.measured_spectrum_efficiency
        self.priority_order[self.time_index] = mac.priority_order

        for ue_id in mac.ue_id_list:
            self.rt_dl_buffer_length[self.time_index][ue_id] = mac.rt_dl_buffer_list[ue_id].length
        self.priority_mask[self.time_index] = mac.priority_mask

    def load_per_step_gNB_pdcp_log(self, pdcp):
        for ue_id in pdcp.ue_id_list:
            self.gnb_pdcp_buffer_length[self.time_index][ue_id] = pdcp.dl_buffer_list[ue_id].length
        self.gnb_pdcp_num_incoming_packets[self.time_index] = pdcp.num_incoming_packets
        self.gnb_pdcp_num_outgoing_packets[self.time_index] = pdcp.num_outgoing_packets

    def load_per_step_gNB_rlc_log(self, rlc):
        for ue_id in rlc.ue_id_list:
            self.gnb_rlc_buffer_length[self.time_index][ue_id] = rlc.dl_buffer_list[ue_id].length
        self.gnb_rlc_num_incoming_packets[self.time_index] = rlc.num_incoming_packets
        self.gnb_rlc_num_outgoing_packets[self.time_index] = rlc.num_outgoing_packets

    def load_per_step_ue_mac_log(self, ue_id, pass_or_drop, TBS):
        self.pass_or_drop[self.time_index][ue_id] = pass_or_drop
        self.TBS[self.time_index][ue_id] = TBS

    def load_per_step_ue_rlc_log(self, ue_id, rlc_ue):
        self.ue_rlc_incoming[self.time_index,
                             ue_id] = rlc_ue.num_incoming_packets
        self.ue_rlc_outgoing[self.time_index,
                             ue_id] = rlc_ue.num_outgoing_packets

    def load_per_step_ue_pdcp_log(self, ue_id, pdcp_ue):
        self.ue_pdcp_incoming[self.time_index,
                              ue_id] = pdcp_ue.num_incoming_packets
        self.ue_pdcp_outgoing[self.time_index,
                              ue_id] = pdcp_ue.num_outgoing_packets

    def load_tx_opportunity_loss(self, layer, ue_id, capacity, transmitted):
        loss_amount = capacity - transmitted
        if capacity > 0:
            if layer == "pdcp":
                self.tx_opportunity_pdcp[ue_id] += 1
            if layer == "mac":
                self.tx_opportunity_mac[ue_id] += 1
        if loss_amount > 0:
            if layer == "pdcp":
                self.tx_opportunity_loss_occurrences_pdcp[ue_id] += 1
                self.tx_opportunity_loss_amount_pdcp[ue_id] += loss_amount
            if layer == "mac":
                self.tx_opportunity_loss_occurrences_mac[ue_id] += 1
                self.tx_opportunity_loss_amount_mac[ue_id] += loss_amount

    def load_over_reception(self, layer, ue_id, received, transmitted):
        over_amount = received - transmitted
        if over_amount > 0:
            if layer == "pdcp":
                self.over_reception_occurrences_pdcp[ue_id] += 1
                self.over_reception_amount_pdcp[ue_id] += over_amount
            if layer == "mac":
                self.over_reception_occurrences_mac[ue_id] += 1
                self.over_reception_amount_mac[ue_id] += over_amount

    def save(self, save_path):
        directory = Path(save_path)
        # Create the folder if it doesn't exist
        if not directory.exists():
            directory.mkdir(parents=True)
        npy_path = save_path + "npy/"
        directory = Path(npy_path)
        if not directory.exists():
            directory.mkdir(parents=True)

        for i in range(self.NUM_UE):
            self.dl_ip_packet_ue_list_for_save[i].save()
            self.dl_pdcp_packet_ue_list_for_save[i].save()
            self.dl_rlc_packet_ue_list_for_save[i].save()

            self.dl_rlc_gnb_buffer_for_save_list[i].save()
            self.dl_mac_gnb_buffer_for_save_list[i].save()
        self.dl_pdcp_gnb_buffer_for_save.save()

        # gnb pdcp
        np.save(npy_path + "gnb_pdcp_buffer_length.npy",
                self.gnb_pdcp_buffer_length)
        np.save(
            npy_path + "gnb_pdcp_num_incoming_packets.npy",
            self.gnb_pdcp_num_incoming_packets,
        )
        np.save(
            npy_path + "gnb_pdcp_num_outgoing_packets.npy",
            self.gnb_pdcp_num_outgoing_packets,
        )

        # gnb rlc
        np.save(npy_path + "gnb_rlc_buffer_length.npy",
                self.gnb_rlc_buffer_length)
        np.save(
            npy_path + "gnb_rlc_num_incoming_packets.npy",
            self.gnb_rlc_num_incoming_packets,
        )
        np.save(
            npy_path + "gnb_rlc_num_outgoing_packets.npy",
            self.gnb_rlc_num_outgoing_packets,
        )

        # Save around gNB MAC
        np.save(npy_path + "total_buffer_size.npy", self.total_buffer_size)
        np.save(npy_path + "measured_TBS.npy", self.assigned_TBS)
        np.save(
            npy_path + "estimated_channel_condition.npy",
            self.estimated_channel_condition,
        )
        np.save(
            npy_path + "estimated_spectral_efficiency.npy",
            self.estimated_spectral_efficiency,
        )
        np.save(npy_path + "experienced_throughput.npy",
                self.experienced_throughput)
        np.save(npy_path + "PF_metic.npy", self.PF_metic)
        np.save(npy_path + "assigned_bandwidth.npy", self.assigned_bandwidth)
        np.save(
            npy_path + "measured_spectrum_efficiency.npy",
            self.measured_spectrum_efficiency,
        )
        np.save(
            npy_path + "priority_order.npy",
            self.priority_order,
        )
        np.save(npy_path + "priority_mask.npy", self.priority_mask)
        np.save(npy_path + "rt_dl_buffer_length.npy", self.rt_dl_buffer_length)

        np.save(npy_path + "packet_pdcp2rlc.npy", self.packet_pdcp2rlc_len)

        # Save around UE MAC
        np.save(npy_path + "UE_pass_or_drop_size.npy", self.pass_or_drop)
        np.save(npy_path + "UE_TBS.npy", self.TBS)

        # Save around UE RLC
        np.save(npy_path + "ue_rlc_incoming.npy", self.ue_rlc_incoming)
        np.save(npy_path + "ue_rlc_outgoing.npy", self.ue_rlc_outgoing)

        # Save around UE PDCP
        np.save(npy_path + "ue_pdcp_incoming.npy", self.ue_pdcp_incoming)
        np.save(npy_path + "ue_pdcp_outgoing.npy", self.ue_pdcp_outgoing)

        with open(save_path + "text_log.txt", mode="w", newline="") as file:
            for value in self.TXTLOG:
                # If it's a list or tuple, join with commas
                if isinstance(value, (list, tuple)):
                    file.write(", ".join(map(str, value)) + "\n")
                else:
                    file.write(str(value) + "\n")

    def safe_div(self, numerator, denominator):
        return '{:.3g}'.format(numerator / denominator) if denominator != 0 else 'NaN'

    def make_csv_files(self, load_path):

        csv_save_path = load_path + "csv/"
        load_path = load_path + "npy/"
        # fig_save_path = "../heavy_data/figure/20250327_0/sn/"
        directory = Path(csv_save_path)
        # Create the folder if it doesn't exist
        if not directory.exists():
            directory.mkdir(parents=True)

        num_ue = ParameterClass.NUM_UE
        packet_data_list_ue = ["dl_ip_packet_ue_",
                               "dl_pdcp_packet_ue_", "dl_rlc_packet_ue_"]
        packet_data_list_gnb_per_ue = [
            "dl_rlc_gnb_buffer_for_save", "dl_mac_gnb_buffer_for_save"]
        packet_data_list_gnb = ("dl_pdcp_gnb_buffer_for_save",)

        index = [
            "Packet ID",
            "Payload Size",
            "UE ID",
            "MAC Packet ID",
            "PDCP Packet ID",
            "Server Timestamp",
            "UE Timestamp(save time)",
            "RLC Incoming Timestamp",
            "Outer Packet ID",
            "Outer ACK Flag",
            "RLC Packet ID",
        ]

        for i in range(num_ue):
            for data_type in packet_data_list_ue + packet_data_list_gnb_per_ue:
                filename_start = data_type + str(i)
                files = [f for f in os.listdir(
                    load_path) if f.startswith(filename_start)][::-1]

                print(load_path, filename_start, files)

                for j, file in enumerate(files):
                    if j == 0:
                        array = np.load(load_path + file)
                    else:
                        array = np.vstack((array, np.load(load_path + file)))
                sorted_indices = np.argsort(
                    array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID])  # arr[1] = [5, 15, 25]
                array = array[sorted_indices]
                np.savetxt(
                    csv_save_path + data_type + str(i) + ".csv",
                    array,
                    delimiter=",",
                    fmt="%d",
                    header=",".join(index),
                )
                print(len(array))

        for data_type in packet_data_list_gnb:
            filename_start = data_type
            files = [f for f in os.listdir(
                load_path) if f.startswith(filename_start)][::-1]

            for j, file in enumerate(files):
                if j == 0:
                    array = np.load(load_path + file)
                else:
                    array = np.vstack((array, np.load(load_path + file)))
            sorted_indices = np.argsort(
                array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID])  # arr[1] = [5, 15, 25]
            array = array[sorted_indices]
            print(csv_save_path + data_type + ".csv")
            np.savetxt(
                csv_save_path + data_type + ".csv",
                array,
                delimiter=",",
                fmt="%d",
                header=",".join(index),
            )

        # for responsiveness data
        res_data_path = os.path.join(csv_save_path, "responsiveness_data.csv")
        with open(res_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(ParameterClass.NUM_UE):
                writer.writerow([f"UE{i}"])
                writer.writerow(
                    [f"tx_opportunity_pdcp =\t{self.tx_opportunity_pdcp[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_occurrences_pdcp =\t{self.tx_opportunity_loss_occurrences_pdcp[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_amount_pdcp =\t{self.tx_opportunity_loss_amount_pdcp[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_rate_pdcp =\t{self.safe_div(self.tx_opportunity_loss_occurrences_pdcp[i], self.tx_opportunity_pdcp[i])}"])
                writer.writerow(
                    [f"tx_opportunity_mac =\t{self.tx_opportunity_mac[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_occurrences_mac =\t{self.tx_opportunity_loss_occurrences_mac[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_amount_mac =\t{self.tx_opportunity_loss_amount_mac[i]}"])
                writer.writerow(
                    [f"tx_opportunity_loss_rate_mac =\t{self.safe_div(self.tx_opportunity_loss_occurrences_mac[i], self.tx_opportunity_mac[i])}"])
                writer.writerow(
                    [f"over_reception_occurrences_pdcp =\t{self.over_reception_occurrences_pdcp[i]}"])
                writer.writerow(
                    [f"over_reception_amount_pdcp =\t{self.over_reception_amount_pdcp[i]}"])
                writer.writerow(
                    [f"over_reception_occurrences_mac =\t{self.over_reception_occurrences_mac[i]}"])
                writer.writerow(
                    [f"over_reception_amount_mac =\t{self.over_reception_amount_mac[i]}"])
                writer.writerow([])  # Separate with a blank line

    def plot(self, path):
        csv_load_path = path + "csv/"
        figure_sav_path = path + "figure/"
        np_path = path + "npy/"
        directory = Path(figure_sav_path)
        # Create the folder if it doesn't exist
        if not directory.exists():
            directory.mkdir(parents=True)

        window_size = 100
        for i in range(ParameterClass.NUM_UE):
            dl_rlc_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_rlc_packet_ue_" + str(i) + ".csv", delimiter=",", skiprows=1, dtype=int)
            dl_pdcp_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_pdcp_packet_ue_" + str(i) + ".csv", delimiter=",", skiprows=1, dtype=int)
            dl_IP_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_ip_packet_ue_" + str(i) + ".csv", delimiter=",", skiprows=1, dtype=int)

            x = (np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS - window_size +
                 1) + int(window_size / 2)) * ParameterClass.TIME_SLOT_WINDOW

            plt.plot(x, moving_average(dl_rlc_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="UE MAC")  # 1e6 = 1M
            plt.plot(x, moving_average(dl_pdcp_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="UE RLC")  # 1e6 = 1M
            plt.plot(x, moving_average(dl_IP_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="UE PDCP")  # 1e6 = 1M

            plt.legend()
            plt.ylim(bottom=0)
            plt.ylabel("THP(M bps)")
            plt.xlabel("Time(s)")
            plt.savefig(figure_sav_path + "ue_thp_" + str(i) + ".png")
            plt.cla()

        # Plot sizes of packets received at the gNB
        for i in range(ParameterClass.NUM_UE):
            dl_rlc_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_mac_gnb_buffer_for_save" + str(i) + ".csv", delimiter=",", skiprows=1, dtype=int)
            dl_pdcp_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_rlc_gnb_buffer_for_save" + str(i) + ".csv", delimiter=",", skiprows=1, dtype=int)
            dl_IP_packet_ue_0 = np.loadtxt(
                csv_load_path + "dl_pdcp_gnb_buffer_for_save.csv", delimiter=",", skiprows=1, dtype=int)

            x = (np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS - window_size +
                 1) + int(window_size / 2)) * ParameterClass.TIME_SLOT_WINDOW

            plt.plot(x, moving_average(dl_rlc_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="gNB MAC")  # 1e6 = 1M
            plt.plot(x, moving_average(dl_pdcp_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="gNB RLC")  # 1e6 = 1M
            plt.plot(x, moving_average(dl_IP_packet_ue_0, window_size) /
                     (1e6 * ParameterClass.TIME_SLOT_WINDOW), label="gNB PDCP")  # 1e6 = 1M

            plt.legend()
            plt.ylim(bottom=0)
            plt.ylabel("THP(M bps)")
            plt.xlabel("Time(s)")
            plt.savefig(figure_sav_path + "gNB_thp_" + str(i) + ".png")
            plt.cla()

        # gNB buffer sizes
        gnb_mac_buffer_size = np.load(np_path + "total_buffer_size.npy")
        gnb_rlc_buffer_size = np.load(np_path + "gnb_rlc_buffer_length.npy")
        gnb_pdcp_buffer_size = np.load(np_path + "gnb_pdcp_buffer_length.npy")
        x = np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS) * \
            ParameterClass.TIME_SLOT_WINDOW
        for i in range(ParameterClass.NUM_UE):
            plt.plot(x, gnb_pdcp_buffer_size[:, i], label="gNB PDCP")
            plt.plot(x, gnb_rlc_buffer_size[:, i], label="gNB RLC")
            plt.plot(x, gnb_mac_buffer_size[:, i] / 12000, label="gNB MAC")
            plt.legend()
            plt.ylim(bottom=0)
            plt.ylabel("buffer length")
            plt.xlabel("Time(s)")
            plt.savefig(figure_sav_path +
                        "gNB_buffer_length_" + str(i) + ".png")
            plt.cla()

        # UE receive buffer sizes
        pdcp_outgoing = np.load(np_path + "ue_pdcp_outgoing.npy")
        pdcp_incoming = np.load(np_path + "ue_pdcp_incoming.npy")
        rlc_outgoing = np.load(np_path + "ue_rlc_outgoing.npy")
        rlc_incoming = np.load(np_path + "ue_rlc_incoming.npy")
        x = np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS) * \
            ParameterClass.TIME_SLOT_WINDOW

        for i in range(ParameterClass.NUM_UE):
            plt.plot(x, pdcp_incoming[:, i] -
                     pdcp_outgoing[:, i], label="pdcp")
            plt.plot(x, rlc_incoming[:, i] - rlc_outgoing[:, i], label="rlc")
            plt.ylabel("buffer length")
            plt.xlabel("Time(s)")
            plt.legend()
            plt.savefig(figure_sav_path + "ue_buffer_length" + str(i) + ".png")
            plt.clf()

        estimated_channel_condition = np.load(
            np_path + "estimated_channel_condition.npy")
        x = np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS) * \
            ParameterClass.TIME_SLOT_WINDOW
        for i in range(ParameterClass.NUM_UE):
            plt.plot(x, estimated_channel_condition[:, i], label="UE" + str(i))
        plt.legend()
        plt.ylabel("spectrum efficiency bps/Hz")
        plt.xlabel("Time(s)")
        plt.savefig(figure_sav_path + "estimated_channel_condition.png")
        plt.clf()
        for i in range(ParameterClass.NUM_UE):
            plt.plot(x, estimated_channel_condition[:, i], label="UE" + str(i))
            plt.legend()
            plt.ylabel("spectrum efficiency bps/Hz")
            plt.xlabel("Time(s)")
            plt.savefig(figure_sav_path +
                        f"estimated_channel_condition{i}.png")
            plt.clf()

    def tmp_plot(self, path):
        """
        Temporary plotting for debugging
        """
        csv_load_path = path + "csv/"
        figure_sav_path = path + "figure/"
        np_path = path + "npy/"
        PF_metric = np.load(np_path + "PF_metic.npy")
        x = np.arange(ParameterClass.NUM_SIMULATION_TIME_SLOTS) * \
            ParameterClass.TIME_SLOT_WINDOW
        print(PF_metric.shape)
        for i in range(ParameterClass.NUM_UE):
            plt.plot(x, PF_metric[:, i], label="UE" + str(i))
        plt.legend()
        plt.ylabel("PF metric")
        plt.xlabel("Time(s)")
        plt.xlim(left=1)
        plt.ylim([0, 0.02])
        plt.savefig(figure_sav_path + "PF_metric.png")
        plt.clf()

        priority_order = np.load(np_path + "priority_order.npy")
        np.savetxt(
            csv_load_path + "priority_order.csv",
            priority_order,
            delimiter=",",
            fmt="%d",
        )
        priority_order = np.abs(priority_order - 6)
        priority_order = np.argmin(priority_order, axis=1)

        plt.plot(priority_order)
        plt.savefig(figure_sav_path + "priority_order.png")
        plt.clf()

        assigned_bandwidth = np.load(np_path + "assigned_bandwidth.npy")
        np.savetxt(
            csv_load_path + "assigned_bandwidth.csv",
            assigned_bandwidth,
            delimiter=",",
            fmt="%d",
        )
        print(assigned_bandwidth.shape)

        priority_mask = np.load(np_path + "priority_mask.npy")
        np.savetxt(
            csv_load_path + "priority_mask.csv",
            priority_mask,
            delimiter=",",
            fmt="%d",
        )

        rt_dl_buffer_length = np.load(np_path + "rt_dl_buffer_length.npy")
        np.savetxt(
            csv_load_path + "rt_dl_buffer_length.csv",
            rt_dl_buffer_length,
            delimiter=",",
            fmt="%d",
        )
        PF_metic = np.load(np_path + "PF_metic.npy")
        np.savetxt(csv_load_path + "PF_metic.csv",
                   PF_metic, delimiter=",", fmt="%.3g")
        print(assigned_bandwidth.shape)


def moving_average(array, window_size):
    """"""
    if len(array) == 0:
        x = np.zeros(ParameterClass.NUM_SIMULATION_TIME_SLOTS -
                     window_size + 1) + int(window_size / 2)
        return x
    times = array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID]
    packets_sizes = array[:, ParameterClass.INDEX_PAYLOAD_SIZE]
    max_time = ParameterClass.NUM_SIMULATION_TIME_SLOTS
    result = np.bincount(times, weights=packets_sizes, minlength=max_time)
    kernel = np.ones(window_size) / window_size  # Kernel for averaging
    # Compute moving average using convolution ('valid' excludes edges)
    moving_avg = np.convolve(result, kernel, mode="valid")
    return moving_avg


if __name__ == "__main__":
    logger = Logger()
    logger.plot(ParameterClass.HEAVY_DATA_PATH +
                ParameterClass.LOG_SAVE_PATH + "sn/")



##########################################
##                for L4                ##
##########################################


class L4Logger(ParameterClass, TimeManager):
    def __init__(self):
        self.logs: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
        # Record file_format
        self.formats: Dict[str, Dict[str, Dict[str, str]]] = {}
        # Names of logs whose averages should be computed ⇒ the average values will be written to files named logname_avg.txt
        self.culc_avg_value_targets: List[str] = [
            "latest_rtt_change_log", "SF1_latest_rtt_change_log", "SF2_latest_rtt_change_log", "MPQUIC-level_one_way_delay"]
        self.plot_targets = {
            "send_throughput", "recv_throughput", "goodput",
            "smoothed_RTT", "latest_RTT", "cwnd_size_log", "SF1_cwnd_size_log", "SF2_cwnd_size_log"
        }

        self.plot_meta = {
            ("MPQUIC", "send_throughput"): {
                "title": "UE{ue} Send Throughput",
                "xlabel": "Time [s]",
                "ylabel": "Throughput [Mbps]",
                "unit": 1e-6,
            },
            ("MPQUIC", "recv_throughput"): {
                "title": "UE{ue} Receive Throughput",
                "xlabel": "Time [s]",
                "ylabel": "Throughput [Mbps]",
                "unit": 1e-6,
            },
            ("MPQUIC", "goodput"): {
                "title": "UE{ue} Goodput",
                "xlabel": "Time [s]",
                "ylabel": "Throughput [Mbps]",
                "unit": 1e-6,
            },
            ("MPQUIC", "SF1_smoothed_RTT"): {
                "title": "UE{ue} SF1 Smoothed RTT",
                "xlabel": "Time [s]",
                "ylabel": "RTT [s]",
                "unit": 1,
            },
            ("MPQUIC", "SF2_smoothed_RTT"): {
                "title": "UE{ue} SF2 Smoothed RTT",
                "xlabel": "Time [s]",
                "ylabel": "RTT [s]",
                "unit": 1,
            },
            ("MPQUIC", "SF1_latest_RTT"): {
                "title": "UE{ue} SF1 Latest RTT",
                "xlabel": "Time [s]",
                "ylabel": "RTT [s]",
                "unit": 1,
            },
            ("MPQUIC", "SF2_latest_RTT"): {
                "title": "UE{ue} SF2 Latest RTT",
                "xlabel": "Time [s]",
                "ylabel": "RTT [s]",
                "unit": 1,
            },
            ("MPQUIC", "cwnd_size_log"): {
                "title": "UE{ue} CWND Size",
                "xlabel": "Time [s]",
                "ylabel": "CWND [packets]",
                "unit": 1.0,
            },
            ("MPQUIC", "SF1_cwnd_size_log"): {
                "title": "UE{ue} SF1 CWND Size",
                "xlabel": "Time [s]",
                "ylabel": "CWND [packets]",
                "unit": 1.0,
            },
            ("MPQUIC", "SF2_cwnd_size_log"): {
                "title": "UE{ue} SF2 CWND Size",
                "xlabel": "Time [s]",
                "ylabel": "CWND [packets]",
                "unit": 1.0,
            },
        }

        # Targets to draw
        self.plot_targets = set(name for _, name in self.plot_meta)

        self._thpt_params = {
            "send_throughput":  "Send Throughput",
            "SF1_server_send_throughput":  "SF1 Send Throughput",
            "SF2_server_send_throughput":  "SF2 Send Throughput",
            "SF1_UE_recv_throughput":  "SF1 Receive Throughput",
            "SF2_UE_recv_throughput":  "SF2 Receive Throughput",
            "recv_throughput":  "Receive Throughput",
            "goodput":          "Goodput",
            "MPQUIC-level_send_throughput":     "MPQUIC-level Send Throughput",
            "MPQUIC-level_recv_throughput":     "MPQUIC-level Receive Throughput",
            "MPQUIC-level_goodput":     "MPQUIC-level Goodput",
            "server_send_throughput": "Send Throughput",
        }
        # Average over 100 ms and 1 s
        self._avg_windows = [
            (0.1, "100ms"),
            (1.0, "1000ms"),
        ]

        self._rtt_params = {"smoothed_RTT", "latest_RTT", "SF1_smoothed_RTT",
                            "SF1_latest_RTT", "SF2_smoothed_RTT", "SF2_latest_RTT"}

    # ---------- Internal --------------------------------------
    def _set_L4log_dir_name(self):
        base_dir = ParameterClass.HEAVY_DATA_PATH
        date_str = ParameterClass.LOG_SAVE_PATH
        parameters_str = "L4_results"
        dir_name = os.path.join(base_dir, date_str, parameters_str)
        print(dir_name)
        return dir_name

    def _dump_txt(self, path, data, footer: list[str] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for v in data:
                f.write(f"{v}\n")
            if footer:
                for line in footer:
                    f.write(line)

    def _dump_csv(self, path, data, footer: list[str] = None):
        """
        Save a CSV file.
        - data: data to write (list)
        - footer: lines appended at the end (optional)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            for v in data:
                w.writerow([v])
            if footer:
                for line in footer:
                    w.writerow([line])

    def _is_numeric_list(self, lst) -> bool:
        """Return True if all elements are numeric."""
        return all(isinstance(v, numbers.Number) for v in lst)

    def _is_slot_value_list(self, lst) -> bool:
        """
        Check if it's a list of [[int_slot, num_value], ...]
        """
        return (
            isinstance(lst, list)
            and len(lst) > 0
            and all(isinstance(v, list) and len(v) == 2 for v in lst)
            and all(isinstance(v[0], int) and isinstance(v[1], (int, float)) for v in lst)
        )

    def _plot_series(self, xs, ys, meta, path, label=None):
        ys = [v * meta["unit"] for v in ys]
        plt.plot(xs, ys, marker='o', linestyle='-', color='b', linewidth=1.6)
        plt.title(meta["title"])
        plt.xlabel(meta["xlabel"])
        plt.ylabel(meta["ylabel"])
        plt.grid(True)
        if label is not None:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _get_meta(self, cls, ue_id, param):
        meta = self.plot_meta.get((cls, param), {})
        # Replace placeholders (e.g., {ue})
        meta = {k: (v.format(ue=ue_id) if isinstance(v, str) else v)
                for k, v in meta.items()}
        # Fill defaults
        meta.setdefault("title", f"{param}")
        meta.setdefault("xlabel", "time [s]")
        meta.setdefault("ylabel", param)
        meta.setdefault("unit", 1.0)
        return meta

    def _avg_sparse_series(self, series: list[list[int, float]], window_sec: float) -> list[float]:
        """
        series : [[slot_idx, value_bps], ...] (slots may be sparse)
        window_sec : e.g., 0.1 (100 ms), 1.0 (1 s)
        return     : list of averages per window
        """
        ts = self.TIME_SLOT_WINDOW                # seconds per slot
        # number of slots per window (1 s or 100 ms)
        win = max(1, int(round(window_sec / ts)))
        # convert to dict for O(1) access
        d = {slot: val for slot, val in series}
        max_slot = ParameterClass.NUM_SIMULATION_TIME_SLOTS
        num_win = max_slot // win

        avg_list = []
        for i in range(num_win):
            start = i * win
            # end is exclusive in Python range
            end = (i + 1) * win
            # sum throughput within the window
            s = sum(d.get(sl, 0.0) for sl in range(start, end))
            avg_list.append(s / win)              # keep in bps
        avg = sum(avg_list)/len(avg_list)
        avg_list.insert(0, 0)  # insert 0 at the beginning
        return avg, avg_list

    def _avg_dense_series(self, dense: list[float], window_sec: float) -> list[float]:
        ts = self.TIME_SLOT_WINDOW
        win = max(1, int(window_sec / ts))
        return [sum(dense[i:i+win]) / len(dense[i:i+win])
                for i in range(0, len(dense), win)]

    def _copy_all_files_to_source_code(self, src_dir, dest_root):
        dest_dir = os.path.join(dest_root, "source_code")
        os.makedirs(dest_dir, exist_ok=True)

        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(dest_dir, filename))

    # ---------- External --------------------------------------

    def store(self, class_name: str, ue_id: str, param_name: str, value: Any, file_format: Optional[str] = None):
        # Initialize log structure
        self.logs.setdefault(class_name, {}).setdefault(
            ue_id, {}).setdefault(param_name, []).append(value)

        # Save output format (use the first specified one)
        if file_format:
            self.formats.setdefault(class_name, {}).setdefault(
                ue_id, {})[param_name] = file_format
        else:
            # If not specified, tentatively assume "txt" (auto-detection later is fine)
            self.formats.setdefault(class_name, {}).setdefault(
                ue_id, {}).setdefault(param_name, "txt")

    def export_all(self):
        root = self._set_L4log_dir_name()          # Reuse original function
        os.makedirs(root, exist_ok=True)
        # shutil.copyfile("param.py", os.path.join(root, "param.txt"))
        src_dir = "."
        self._copy_all_files_to_source_code(src_dir, root)

        # --- 1. Save all logs to files ---------------------------
        for cls, ue_dict in self.logs.items():
            for ue, param_dict in ue_dict.items():
                for param, data in param_dict.items():
                    fmt = self.formats[cls][ue][param]
                    subdir = os.path.join(root, cls, ue)
                    fname = f"{param}.{fmt}"
                    fpath = os.path.join(subdir, fname)

                    if fmt == "csv":
                        self._dump_csv(fpath, data)
                    else:
                        self._dump_txt(fpath, data)

                    # --- for rtt only (calculate average value) ----------------------
                    if param in self.culc_avg_value_targets and fmt == "txt":
                        values_for_culc_avgvalue = []
                        for v in data:
                            try:
                                if "\t" in v:
                                    rtt = float(v.strip().split("\t")[1])
                                else:
                                    # For MPQUIC-level_one_way_delay. Convert units from [time_index] to [s].
                                    rtt = float(v.strip()) * \
                                        ParameterClass.TIME_SLOT_WINDOW
                                values_for_culc_avgvalue.append(rtt)
                            except:
                                pass
                        if len(values_for_culc_avgvalue) > 0:
                            values_for_culc_avgvalue_np = np.array(
                                values_for_culc_avgvalue, dtype=float)
                            avg_rtt = np.mean(values_for_culc_avgvalue_np)
                            p95 = np.percentile(
                                values_for_culc_avgvalue_np, 95)
                            p99 = np.percentile(
                                values_for_culc_avgvalue_np, 99)
                            avg_file_path = os.path.join(
                                subdir, f"{param}_avg.txt")
                            print(f"{avg_file_path}:  {avg_rtt}")
                            with open(avg_file_path, "w") as f:
                                f.write(
                                    f"{avg_rtt:.6f}\t[s]  (Average value)\n")
                                f.write(f"{p95:.6f}\t[s]  (p95 value)\n")
                                f.write(f"{p99:.6f}\t[s]  (p99 value)\n")

                    # --- 2. Generate graphs if conditions match ------------
                    if param in self._rtt_params and self._is_numeric_list(data):
                        # Fetch title etc. from plot_meta
                        meta = self._get_meta(cls, ue, param)
                        xs = [
                            i * self.TIME_SLOT_WINDOW for i in range(len(data))]
                        gpath = os.path.join(subdir, f"{param}.png")

                        # Average RTT
                        avg_rtt = sum(data) / len(data) if data else 0.0

                        # Graph
                        self._plot_series(xs, [v * meta["unit"]
                                          for v in data], meta, gpath)

                        # CSV (append average value at the end)
                        footer = [
                            f"average {param} = {avg_rtt:.6f}\t[s]",
                            f"{avg_rtt*1000:.2f} [ms]"
                        ]
                        self._dump_csv(os.path.join(
                            subdir, f"{param}.csv"), data, footer)

                        continue

                    if param in self.plot_targets and self._is_numeric_list(data):
                        meta = self._get_meta(cls, ue, param)
                        xs = [
                            i*self.TIME_SLOT_WINDOW for i in range(len(data))]
                        gpath = os.path.join(subdir, f"{param}.png")
                        self._plot_series(
                            xs, [v*meta["unit"] for v in data], meta, gpath)
                    if param in self._thpt_params:
                        # Determine whether sparse or dense
                        is_sparse = self._is_slot_value_list(data)
                        is_dense = self._is_numeric_list(data)

                        if not (is_sparse or is_dense):
                            # Skip plotting if not numeric
                            continue

                        meta_base = self._get_meta(cls, ue, param)

                        for w_sec, tag in self._avg_windows:
                            # Average
                            if is_sparse:
                                avg_footer, avg_series = self._avg_sparse_series(
                                    data, w_sec)
                            else:
                                avg_series = self._avg_dense_series(
                                    data, w_sec)

                            xs = [i * w_sec for i in range(len(avg_series))]

                            meta = dict(meta_base)
                            meta["title"] = f"{meta_base['title']} (avg {tag})"
                            gname = f"{param}_avg_{tag}.png"
                            csvname = f"{param}_avg_{tag}.csv"

                            self._plot_series(
                                xs, [v * meta["unit"] for v in avg_series],
                                meta, os.path.join(subdir, gname)
                            )
                            footer = [
                                f"{avg_footer/1e6}\t[Mbps]",
                            ]
                            self._dump_csv(os.path.join(
                                subdir, csvname), avg_series, footer)

    def export_all_UE_results(self):
        dir_name = os.path.join(self._set_L4log_dir_name(), "all_UE_results")
        os.makedirs(dir_name, exist_ok=True)

        colors = ["red", "orange", "yellow", "green", "blue",
                  "purple", "cyan", "magenta", "brown", "gray"]
        class_names = ["MPQUIC", "QUIC", "UDP"]

        for metric in self._thpt_params:
            for class_name in class_names:
                # Only process class_name that actually has the metric
                if class_name not in self.logs or not any(metric in ue_logs for ue_logs in self.logs[class_name].values()):
                    continue

                for avg_window_sec, tag in self._avg_windows:
                    plt.figure(figsize=(12, 6))
                    for ue_id in range(self.NUM_UE):
                        try:
                            series = self.logs[class_name][f"UE{ue_id}"][metric]
                        except KeyError:
                            continue

                        if self._is_slot_value_list(series):
                            avg_value, dense = self._avg_sparse_series(
                                series, avg_window_sec)
                        elif self._is_numeric_list(series):
                            dense = self._avg_dense_series(
                                series, avg_window_sec)
                        else:
                            continue

                        xs = [i * avg_window_sec for i in range(len(dense))]
                        ys = [v * 1e-6 for v in dense]

                        plt.plot(
                            xs, ys, label=f"UE{ue_id}", color=colors[ue_id % len(colors)])

                    plt.xlabel("Time [s]")
                    plt.ylabel(f"{metric} [Mbps]")
                    plt.title(f"{class_name}: All UE {metric} (avg {tag})")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        dir_name, f"{class_name}_all_ue_{metric}_avg_{tag}.png"))
                    plt.close()

        # Throughput summary bar chart (for MPQUIC only)
        bar_width = 0.25  # Bar width
        index = np.arange(self.NUM_UE)  # X-axis positions
        param_sets = [
            ("MPQUIC", "SF1_server_send_throughput",
             "SF1_UE_recv_throughput", "SF1_all_UEs_thpt", "SF1"),
            ("MPQUIC", "SF2_server_send_throughput",
             "SF2_UE_recv_throughput", "SF2_all_UEs_thpt", "SF2"),
        ]
        for layer, param1, param2, fig_tag, txt_tag in param_sets:
            list_for_txt_file = []
            sector_send_throughput = 0
            sector_recv_throughput = 0
            plt.figure(figsize=(12, 6))
            for ue_id in range(self.NUM_UE):
                send_series = self.logs[layer][f"UE{ue_id}"].get(param1)
                recv_series = self.logs[layer][f"UE{ue_id}"].get(param2)
                if send_series is None or recv_series is None:
                    continue  # Skip if throughput data is missing
                avg_send_thpt, _ = self._avg_sparse_series(
                    self.logs[layer][f"UE{ue_id}"][param1], 1)
                avg_recv_thpt, _ = self._avg_sparse_series(
                    self.logs[layer][f"UE{ue_id}"][param2], 1)
                list_for_txt_file.extend([
                    f"UE{ue_id} {txt_tag} AVG SEND THPT: \t{avg_send_thpt/1000/1000:.2f}\t[Mbps]",
                    f"UE{ue_id} {txt_tag} AVG RECV THPT: \t{avg_recv_thpt/1000/1000:.2f}\t[Mbps]",
                ])
                print(
                    f"UE{ue_id} {txt_tag} AVG SEND THPT: \t{avg_send_thpt/1000/1000:.2f}\t[Mbps]")
                print(
                    f"UE{ue_id} {txt_tag} AVG RECV THPT: \t{avg_recv_thpt/1000/1000:.2f}\t[Mbps]")
                sector_send_throughput += avg_send_thpt
                sector_recv_throughput += avg_recv_thpt
                plt.bar(index[ue_id] - bar_width/2, avg_send_thpt,
                        width=bar_width, label="Send" if ue_id == 0 else None, color="skyblue")
                plt.bar(index[ue_id] + bar_width/2, avg_recv_thpt,
                        width=bar_width, label="Recv" if ue_id == 0 else None, color="lightgreen")

            # plt.bar(index + bar_width, good_avg, bar_width, label="Goodput", color="orange")
            plt.xlabel("UE")
            plt.ylabel("Throughput [bps]")
            plt.title(f"{fig_tag} Bitrate Comparison per UE")
            plt.xticks(index, [f"UE{i}" for i in range(self.NUM_UE)])
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_name, f"{fig_tag}_all_UEs_thpt"))
            plt.close()
            list_for_txt_file.extend([
                f"\nSECTOR SEND THPT: \t{sector_send_throughput/1000/1000:.2f}\t[Mbps]",
                f"SECTOR RECV THPT: \t{sector_recv_throughput/1000/1000:.2f}\t[Mbps]",
            ])
            self._dump_txt(
                os.path.join(
                    dir_name, f"{fig_tag}_all_UEs_thpt.txt"),
                list_for_txt_file,
                footer=None
            )

        if ParameterClass.UDP_MODE is not True:
            # Throughput summary bar chart (three bar series)
            bar_width = 0.25  # Bar width
            index = np.arange(self.NUM_UE)  # X-axis positions
            param_sets = [
                ("QUIC", "send_throughput",
                 "recv_throughput", "goodput", "all_UEs_thpt"),
            ]
            for layer, param1, param2, param3, fig_tag in param_sets:
                list_for_txt_file = []
                sector_send_throughput = 0
                sector_recv_throughput = 0
                sector_goodput = 0
                plt.figure(figsize=(12, 6))
                for ue_id in range(self.NUM_UE):
                    avg_send_thpt, _ = self._avg_sparse_series(
                        self.logs[layer][f"UE{ue_id}"][param1], 1)
                    avg_recv_thpt, _ = self._avg_sparse_series(
                        self.logs[layer][f"UE{ue_id}"][param2], 1)
                    avg_goodput, _ = self._avg_sparse_series(
                        self.logs[layer][f"UE{ue_id}"][param3], 1)
                    list_for_txt_file.extend([
                        f"UE{ue_id} {layer} AVG SEND THPT: \t{avg_send_thpt/1000/1000:.2f}\t[Mbps]",
                        f"UE{ue_id} {layer} AVG RECV THPT: \t{avg_recv_thpt/1000/1000:.2f}\t[Mbps]",
                        f"UE{ue_id} {layer} AVG GOODPUT: \t{avg_goodput/1000/1000:.2f}\t[Mbps]",
                    ])
                    print(
                        f"UE{ue_id} {layer} AVG SEND THPT: \t{avg_send_thpt/1000/1000:.2f}\t[Mbps]")
                    print(
                        f"UE{ue_id} {layer} AVG RECV THPT: \t{avg_recv_thpt/1000/1000:.2f}\t[Mbps]")
                    print(
                        f"UE{ue_id} {layer} AVG GOODPUT: \t{avg_recv_thpt/1000/1000:.2f}\t[Mbps]")
                    sector_send_throughput += avg_send_thpt
                    sector_recv_throughput += avg_recv_thpt
                    sector_goodput += avg_goodput
                    plt.bar(index[ue_id] - bar_width, avg_send_thpt,
                            width=bar_width, label="Send" if ue_id == 0 else None, color="skyblue")
                    plt.bar(index[ue_id], avg_recv_thpt,
                            width=bar_width, label="Recv" if ue_id == 0 else None, color="lightgreen")
                    plt.bar(index[ue_id] + bar_width, avg_goodput,
                            width=bar_width, label="Recv" if ue_id == 0 else None, color="orange")

                plt.xlabel("UE")
                plt.ylabel("Throughput [bps]")
                plt.title(f"{fig_tag} Bitrate Comparison per UE")
                plt.xticks(index, [f"UE{i}" for i in range(self.NUM_UE)])
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(dir_name, f"{fig_tag}_all_UEs_thpt"))
                plt.close()
                list_for_txt_file.extend([
                    f"\nSECTOR SEND THPT: \t{sector_send_throughput/1000/1000:.2f}\t[Mbps]",
                    f"SECTOR RECV THPT: \t{sector_recv_throughput/1000/1000:.2f}\t[Mbps]",
                    f"SECTOR GOODPUT: \t{sector_goodput/1000/1000:.2f}\t[Mbps]",
                ])
                self._dump_txt(
                    os.path.join(
                        dir_name, f"{fig_tag}_all_UEs_thpt.txt"),
                    list_for_txt_file,
                    footer=None
                )

    def export_from_the_beggining(self):
        root = self._set_L4log_dir_name()
        os.makedirs(root, exist_ok=True)
        src_dir = "."
        self._copy_all_files_to_source_code(src_dir, root)


global_l4_logger = Logger()
