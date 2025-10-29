import numpy as np
from pathlib import Path
from param import ParameterClass, TimeManager
from buffer import Buffer, SaveBuffer
from logger import Logger, L4Logger
from wiredlink import WiredLink
from upf import UPF
from gNB_PDCP import PDCP_wDC, PDCP_woDC
from gNB_RLC import RLC
from mac import MAC, MAC_UE, Air, MAC_INFO_MANAGER
from ue_pdcprlc import PDCP_RLC_UE
from quic import QUIC
from mpquic import MPQUIC
from ran_woDC import RAN_wo_DC
from ran_wDC import RAN_w_DC
from n6link import BottleneckLink
from udp import UDP
import pdb
import os
import datetime
import csv
import time
import cProfile
import math

np.random.seed(0)


def dc_sim():
    # --- Initialize Downlink (DL) components ---
    ue_id_list = np.arange(ParameterClass.NUM_UE)
    upf_routing_table = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    num_N3 = 2

    upf = UPF(routing_table=upf_routing_table, num_N3=num_N3)
    dl_N3_0 = WiredLink()  # Wired link between RAN and UPF (downlink path)

    server_list = np.zeros([ParameterClass.NUM_UE])
    dl_ip_ue_buffer_list = np.zeros([ParameterClass.NUM_UE], dtype=Buffer)
    ul_ip_server_buffer_list = np.zeros([ParameterClass.NUM_UE], dtype=Buffer)

    l4_logger = L4Logger()
    l4 = QUIC(l4_logger)

    for ue_id in ue_id_list:
        server_list[ue_id] = l4.initialize_ue(ue_id)
        dl_ip_ue_buffer_list[ue_id] = Buffer()
        ul_ip_server_buffer_list[ue_id] = Buffer()

    simulation_type = "DC"
    if simulation_type == "DC":
        # Dual Connectivity (DC) RAN configuration
        ran = RAN_w_DC(
            propagation_load_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_LB,
            log_save_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "mn/",
            propagation_load_path_sn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_HB,
            log_save_path_sn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "sn/",
            ran_name="hoge"
        )
    if simulation_type == "wo DC":
        # Non-DC RAN configuration (single connectivity)
        ran = RAN_wo_DC(
            propagation_load_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_LB,
            log_save_path_mn=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "mn/",
        )

    # --- Initialize Uplink (UL) components ---
    ul_wiredlink = np.zeros([ParameterClass.NUM_UE], dtype=WiredLink)
    for i in ue_id_list:
        # Wired link between RAN and UPF (uplink path)
        ul_wiredlink[i] = WiredLink()

    # --- Main Simulation Loop ---
    for _ in range(ParameterClass.NUM_SIMULATION_TIME_SLOTS):
        for ue_id in ue_id_list:
            if l4.check_timer(ue_id):
                incoming_N6_packets = l4.on_loss_detection_timeout(ue_id)
            else:
                incoming_N6_packets = l4.sender_send(ue_id)
            upf.dl_enqueue(incoming_N6_packets)
        upf.route()
        packet_upf2N3 = upf.dl_dequeue(N3id=0)

        dl_N3_0.enqueue(packet_upf2N3)
        dl_N3_0.do_timeslot()
        packet_N32pdcp = dl_N3_0.dequeue()
        dl_ip_ue_buffer_list = ran.perform_one_time_slot(
            packet_N32pdcp=packet_N32pdcp, dl_ip_ue_buffer_list=dl_ip_ue_buffer_list)
        for ue_id in ue_id_list:
            packets_in_l4_buff = dl_ip_ue_buffer_list[ue_id].dequeue()
            packet_ue2ullink = l4.receiver(ue_id, packets_in_l4_buff)
            ul_wiredlink[ue_id].enqueue(packet_ue2ullink)
            ul_wiredlink[ue_id].do_timeslot()
            ul_ip_server_buffer_list[ue_id].enqueue(
                ul_wiredlink[ue_id].dequeue())

        for ue_id in ue_id_list:
            ack_packet = ul_ip_server_buffer_list[ue_id].dequeue()
            l4.sender_recv(ue_id, ack_packet)
        TimeManager.time_index += 1

    ran.save_all_info()
    l4.show_results()


def ds_sim():
    # --- Initialize components for DualSteer / MPQUIC simulation ---
    ue_id_list = np.arange(ParameterClass.NUM_UE)
    dl_N3_0 = WiredLink()  # Downlink RAN–UPF wired path for subflow 1 (sn/6G)
    dl_N3_1 = WiredLink()  # Downlink RAN–UPF wired path for subflow 2 (mn/5G)
    dl_ip_ue_buffer_list_0 = np.zeros([ParameterClass.NUM_UE], dtype=Buffer)
    dl_ip_ue_buffer_list_1 = np.zeros([ParameterClass.NUM_UE], dtype=Buffer)
    ul_ip_server_buffer_list_0 = np.zeros(
        [ParameterClass.NUM_UE], dtype=Buffer)
    ul_ip_server_buffer_list_1 = np.zeros(
        [ParameterClass.NUM_UE], dtype=Buffer)
    dl_N6 = np.zeros([ParameterClass.NUM_UE], dtype=BottleneckLink)
    ul_N6 = np.zeros([ParameterClass.NUM_UE], dtype=BottleneckLink)

    l4_logger = L4Logger()
    l4_logger.export_from_the_beggining()
    if ParameterClass.UDP_MODE is True:
        l4_inner = UDP(l4_logger)
    else:
        l4_inner = QUIC(l4_logger)
    l4_outer = MPQUIC(l4_logger)

    # For RAN feedback: keep track of requested-but-undelivered data and queued send sizes
    total_requested_data = {}  # {ue_id: total_data_requested_but_not_arrived}
    txsize_queue = {}
    last_mac_capacity = [1] * ParameterClass.NUM_UE

    for ue_id in ue_id_list:
        l4_inner.initialize_ue(ue_id)
        l4_outer.initialize_ue(ue_id)
        dl_N6[ue_id] = BottleneckLink(logger=l4_logger, delay_ti=ParameterClass.N6_DELAY_TI, loss_prob=ParameterClass.N6_LOSS_RATE_DL,
                                      max_length=ParameterClass.BUFF_MAX_LENGTH_N6_DL, max_volume=ParameterClass.BUFF_MAX_VOLUME_N6_DL)
        ul_N6[ue_id] = BottleneckLink(logger=l4_logger, delay_ti=ParameterClass.N6_DELAY_TI, loss_prob=ParameterClass.N6_LOSS_RATE_UL,
                                      max_length=ParameterClass.BUFF_MAX_LENGTH_N6_UL, max_volume=ParameterClass.BUFF_MAX_VOLUME_N6_UL)
        dl_ip_ue_buffer_list_0[ue_id] = Buffer()
        dl_ip_ue_buffer_list_1[ue_id] = Buffer()
        ul_ip_server_buffer_list_0[ue_id] = Buffer()
        ul_ip_server_buffer_list_1[ue_id] = Buffer()
        total_requested_data[ue_id] = 0
        txsize_queue[ue_id] = []

    simulation_type = "MPQUIC"
    if simulation_type == "MPQUIC":
        # Configure two independent RANs for DualSteer (no DC aggregation inside RAN)
        ran_sn = RAN_wo_DC(
            propagation_load_path=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_HB,
            log_save_path=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "sn/",
            ran_name="sn",
            max_volume_dl=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_SN_DL,
            max_length_dl=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_SN_DL,
            max_volume_ul=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_SN_UL,
            max_length_ul=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_SN_UL,
        )  # 5 GHz, treated as 6G RAN (sn)
        ran_mn = RAN_wo_DC(
            propagation_load_path=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.PROPAGATION_LOAD_PATH_LB,
            log_save_path=ParameterClass.HEAVY_DATA_PATH +
            ParameterClass.LOG_SAVE_PATH + "mn/",
            ran_name="mn",
            max_volume_dl=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_MN_DL,
            max_length_dl=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_MN_DL,
            max_volume_ul=ParameterClass.BUFF_MAX_VOLUME_PDCP_WODC_MN_UL,
            max_length_ul=ParameterClass.BUFF_MAX_LENGTH_PDCP_WODC_MN_UL,
        )  # 2 GHz, treated as 5G RAN (mn)

    # --- Initialize Uplink (UL) wired paths ---
    ul_wiredlink_sn = np.zeros([ParameterClass.NUM_UE], dtype=WiredLink)
    ul_wiredlink_mn = np.zeros([ParameterClass.NUM_UE], dtype=WiredLink)
    for i in ue_id_list:
        ul_wiredlink_sn[i] = WiredLink()  # Uplink wired path for sn/6G subflow
        ul_wiredlink_mn[i] = WiredLink()  # Uplink wired path for mn/5G subflow

    # --- Main Simulation Loop ---
    for _ in range(ParameterClass.NUM_SIMULATION_TIME_SLOTS):
        print(
            "\nTime index is ",
            TimeManager.time_index,
        )

        packet_upf2N3_0 = np.empty(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)
        packet_upf2N3_1 = np.empty(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        for ue_id in ue_id_list:
            # --- SERVER PROCESSING (DL) ---
            if l4_inner.check_timer(ue_id):
                quic_packets_to_n6 = l4_inner.on_loss_detection_timeout(ue_id)
            else:
                quic_packets_to_n6 = l4_inner.sender_send(ue_id)

            # --- N6 PROCESSING (DL) ---
            dl_N6[ue_id].enqueue(quic_packets_to_n6)
            quic_packets_to_upf = dl_N6[ue_id].dequeue()

            # --- UPF PROCESSING (DL) ---
            # Receive QUIC packets from server and push into MPQUIC outer layer
            l4_outer.receive_data(ue_id, quic_packets_to_upf)
            l4_logger.store("main", f"UE{ue_id}", "mpquic_recv_buff_length_at_UPF_before_send",
                            f"time={TimeManager.time_index}: {l4_outer.ue_states[ue_id]['send_buffer_to_lower_layer'].length}")
            l4_outer.ue_states[ue_id]["subflow_1"].check_timer_subflow(ue_id)
            l4_outer.ue_states[ue_id]["subflow_2"].check_timer_subflow(ue_id)
            # Mode: use feedback from one RAN (sn/6G)
            if ParameterClass.RAN_FB_OPTION == "SINGLE":
                # Check whether to trigger feedback  (periodic)
                # Use fixed constants for the first three periods
                if TimeManager.time_index in [0, ParameterClass.RAN_FB_CYCLE*1, ParameterClass.RAN_FB_CYCLE*2]:
                    txsize_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index,
                        "data_size": ParameterClass.INIT_CWND*10,
                    })
                    total_requested_data[ue_id] += ParameterClass.INIT_CWND*10
                elif TimeManager.time_index > ParameterClass.RAN_FB_CYCLE*2 and (TimeManager.time_index - ParameterClass.N3_DELAY+1) % ParameterClass.RAN_FB_CYCLE == 0:
                    # Convert experienced throughput (bit/s) to packets/s
                    exp_thpt_pps = ran_sn.mac.recent_experienced_throughput[ue_id] / \
                        ParameterClass.TIME_SLOT_WINDOW / ParameterClass.MTU_BIT_SIZE
                    mac_tx_capacity = math.ceil(
                        ran_sn.mac.mac_tx_capacity[ue_id] * 0.9 / ParameterClass.MTU_BIT_SIZE)
                    mac_total_buff_size = ran_sn.mac.return_buffer_status()
                    mac_buff_size = mac_total_buff_size[ue_id]  # bit
                    mac_buff_size_packet = mac_buff_size / \
                        ParameterClass.MTU_BIT_SIZE  # packets
                    pdcp_buff_size = ran_sn.pdcp.dl_buffer_list[ue_id].length
                    pdcp_buff_filling_ratio = max(
                        0.8 - pdcp_buff_size / ran_sn.pdcp.dl_buffer_list[ue_id].max_length, 0)
                    l4_logger.store(
                        "main", f"UE{ue_id}", "ran_fb_pdcp_filling_ratio", f"time = {TimeManager.time_index}, pdcp_buff_filling_ratio = {pdcp_buff_filling_ratio}")
                    l4_logger.store("main", f"UE{ue_id}", "mac_capacity",
                                    f"time = {TimeManager.time_index},\tmac_tx_capacity={mac_tx_capacity},\tlast_mac_capacity[ue_id]={last_mac_capacity[ue_id]}")
                    # Guard buffer: at least 40 packets or N3 delay (minus 1) worth of time
                    guard_buff = max(40, ParameterClass.N3_DELAY-1)
                    # Required packets over the next period minus current in-flight/queued amounts
                    tx_data_size = max(0, int(exp_thpt_pps * (ParameterClass.RAN_FB_CYCLE + guard_buff)
                                              * ParameterClass.TIME_SLOT_WINDOW - mac_buff_size_packet - pdcp_buff_size
                                              # - total_requested_data[ue_id]
                                              ))

                    l4_logger.store(
                        "main", f"UE{ue_id}", "ran_fb_log", f"time = {TimeManager.time_index},\t{exp_thpt_pps} [pps],\tmac_tx_capacity={mac_tx_capacity},\tlast_mac_capacity={last_mac_capacity[ue_id]}, \t{mac_buff_size_packet:.2f} [packets in mac_buff],\t{pdcp_buff_size} [packets in pdcp_buff],\t{total_requested_data[ue_id]}[unarrived packets],\t{tx_data_size}[requested packets]")

                    # Update "requested but not yet delivered" coueter
                    total_requested_data[ue_id] += tx_data_size
                    last_mac_capacity[ue_id] = max(mac_tx_capacity, 1)

                    # Reflected in UPF at t = reflection_time
                    txsize_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index + ParameterClass.N3_DELAY - 1,
                        "data_size": tx_data_size,
                    })

                if not txsize_queue[ue_id]:
                    if ParameterClass.PACING_OPTION is True:
                        packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb_with_pacing(
                            ue_id)
                    else:
                        packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb(
                            ue_id)
                else:
                    # Only look at the head of the FB queue; once consumed, delete and then the next item becomes the head
                    if TimeManager.time_index >= txsize_queue[ue_id][0]["reflection_time"]:
                        l4_logger.store("main", f"UE{ue_id}", "ran_fb_queue_log",
                                        f"time = {TimeManager.time_index}, queue = {txsize_queue}")
                        if ParameterClass.PACING_OPTION is True:
                            packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb_with_pacing(
                                ue_id, txsize_queue[ue_id][0]["data_size"])
                        else:
                            packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb(
                                ue_id, txsize_queue[ue_id][0]["data_size"])
                        total_requested_data[ue_id] -= txsize_queue[ue_id][0]["data_size"]
                        del txsize_queue[ue_id][0]
                    else:
                        if ParameterClass.PACING_OPTION is True:
                            packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb_with_pacing(
                                ue_id)
                        else:
                            packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_ran_fb(
                                ue_id)
            # Mode: use feedback from both RANs (sn & mn)
            if ParameterClass.RAN_FB_OPTION == "BOTH":
                # Trigger feedback periodically; use constants for the first three cycles
                if TimeManager.time_index in [0, ParameterClass.RAN_FB_CYCLE*1, ParameterClass.RAN_FB_CYCLE*2]:
                    txsize_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index,
                        "data_size_for_sf1": ParameterClass.INIT_CWND*10,
                        "data_size_for_sf2": ParameterClass.INIT_CWND*10,
                    })
                elif TimeManager.time_index > ParameterClass.RAN_FB_CYCLE*2 and (TimeManager.time_index - ParameterClass.N3_DELAY+1) % ParameterClass.RAN_FB_CYCLE == 0:
                    # Compute for sf1 (6G/sn)
                    exp_thpt_pps_sf1 = ran_sn.mac.recent_experienced_throughput[ue_id] / \
                        ParameterClass.TIME_SLOT_WINDOW / ParameterClass.MTU_BIT_SIZE
                    mac_total_buff_size_sf1 = ran_sn.mac.return_buffer_status()
                    mac_buff_size_sf1 = mac_total_buff_size_sf1[ue_id] / \
                        ParameterClass.MTU_BIT_SIZE
                    pdcp_buff_size_sf1 = ran_sn.pdcp.dl_buffer_list[ue_id].length
                    guard_buff_sf1 = max(40, ParameterClass.N3_DELAY-1)
                    data_size_for_sf1 = max(0, int(exp_thpt_pps_sf1 * (ParameterClass.RAN_FB_CYCLE + guard_buff_sf1)
                                            * ParameterClass.TIME_SLOT_WINDOW - mac_buff_size_sf1 - pdcp_buff_size_sf1))

                    # Compute for sf2 (5G/mn)
                    exp_thpt_pps_sf2 = ran_mn.mac.recent_experienced_throughput[ue_id] / \
                        ParameterClass.TIME_SLOT_WINDOW / ParameterClass.MTU_BIT_SIZE
                    mac_total_buff_size_sf2 = ran_mn.mac.return_buffer_status()
                    mac_buff_size_sf2 = mac_total_buff_size_sf2[ue_id] / \
                        ParameterClass.MTU_BIT_SIZE
                    pdcp_buff_size_sf2 = ran_mn.pdcp.dl_buffer_list[ue_id].length
                    guard_buff_sf2 = max(40, ParameterClass.N3_DELAY-1)
                    data_size_for_sf2 = max(0, int(exp_thpt_pps_sf2 * (ParameterClass.RAN_FB_CYCLE + guard_buff_sf2)
                                            * ParameterClass.TIME_SLOT_WINDOW - mac_buff_size_sf2 - pdcp_buff_size_sf2))

                    txsize_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index + ParameterClass.N3_DELAY - 1,
                        "data_size_for_sf1": data_size_for_sf1,
                        "data_size_for_sf2": data_size_for_sf2,
                    })

                if not txsize_queue[ue_id]:
                    packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_both_ran_fb(
                        ue_id)
                else:
                    # Only look at the head of the FB queue; once consumed, delete it
                    if TimeManager.time_index >= txsize_queue[ue_id][0]["reflection_time"]:
                        l4_logger.store("main", f"UE{ue_id}", "ran_fb_queue_log",
                                        f"time = {TimeManager.time_index}, queue = {txsize_queue}")
                        packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_both_ran_fb(
                            ue_id, txsize_queue[ue_id][0]["data_size_for_sf1"], txsize_queue[ue_id][0]["data_size_for_sf2"])
                        del txsize_queue[ue_id][0]
                    else:
                        packets_for_sf1, packets_for_sf2 = l4_outer.send_data_considering_both_ran_fb(
                            ue_id)

            if ParameterClass.RAN_FB_OPTION == "NONE":  # Normal mode without RAN feedback
                packets_for_sf1, packets_for_sf2 = l4_outer.send_data(
                    ue_id)  # Schedule to each subflow

            l4_logger.store("main", f"UE{ue_id}", "mpquic_recv_buff_length_at_UPF_after_send",
                            f"time={TimeManager.time_index}: {l4_outer.ue_states[ue_id]['send_buffer_to_lower_layer'].length}")

            # Subflow-specific send
            mpquic_packets_by_sf1 = l4_outer.ue_states[ue_id]["subflow_1"].send_data(
                ue_id, packets_for_sf1)
            mpquic_packets_by_sf2 = l4_outer.ue_states[ue_id]["subflow_2"].send_data(
                ue_id, packets_for_sf2)
            sf1_len = len(
                mpquic_packets_by_sf1) if mpquic_packets_by_sf1 is not None else 0
            sf2_len = len(
                mpquic_packets_by_sf2) if mpquic_packets_by_sf2 is not None else 0
            send_throughput = (
                sf1_len + sf2_len) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
            l4_logger.store("MPQUIC", f"UE{ue_id}", "MPQUIC-level_send_throughput", [
                            TimeManager.time_index, send_throughput])
            # if needed: measure real DL processing time here
            if mpquic_packets_by_sf1 is not None:
                packet_upf2N3_0 = np.vstack(
                    (packet_upf2N3_0, mpquic_packets_by_sf1))
            if mpquic_packets_by_sf2 is not None:
                packet_upf2N3_1 = np.vstack(
                    (packet_upf2N3_1, mpquic_packets_by_sf2))

        # --- Wired RAN–UPF transmission (DL) ---
        dl_N3_0.enqueue(packet_upf2N3_0)
        dl_N3_0.do_timeslot()
        packet_N32pdcp_0 = dl_N3_0.dequeue()

        dl_N3_1.enqueue(packet_upf2N3_1)
        dl_N3_1.do_timeslot()
        packet_N32pdcp_1 = dl_N3_1.dequeue()

        # --- RAN processing for one timeslot (DL to UE) ---
        dl_ip_ue_buffer_list_0 = ran_sn.perform_one_time_slot(
            packet_N32pdcp=packet_N32pdcp_0, dl_ip_ue_buffer_list=dl_ip_ue_buffer_list_0)
        dl_ip_ue_buffer_list_1 = ran_mn.perform_one_time_slot(
            packet_N32pdcp=packet_N32pdcp_1, dl_ip_ue_buffer_list=dl_ip_ue_buffer_list_1)

        for ue_id in ue_id_list:
            # --- UE PROCESSING (DL & UL) ---
            # Receive subflow packets
            mpquic_packets_to_sf1_buff = dl_ip_ue_buffer_list_0[ue_id].dequeue(
            )
            mpquic_packets_to_sf2_buff = dl_ip_ue_buffer_list_1[ue_id].dequeue(
            )

            # Subflow ACK generation (each subflow sends ACKs)
            packet_ue2ullink_sf1 = l4_outer.ue_states[ue_id]["subflow_1"].receive_data_send_ack(
                ue_id, mpquic_packets_to_sf1_buff)
            packet_ue2ullink_sf2 = l4_outer.ue_states[ue_id]["subflow_2"].receive_data_send_ack(
                ue_id, mpquic_packets_to_sf2_buff)

            # Merge received packets from both subflows
            if mpquic_packets_to_sf1_buff.size == 0:
                packets_to_quic_buff = mpquic_packets_to_sf2_buff
            elif mpquic_packets_to_sf2_buff.size == 0:
                packets_to_quic_buff = mpquic_packets_to_sf1_buff
            else:
                # Note: order is always SF1 -> SF2;
                packets_to_quic_buff = np.vstack(
                    (mpquic_packets_to_sf1_buff, mpquic_packets_to_sf2_buff))

            # When enabling stream-level reordering across subflows
            if ParameterClass.STREAM_LEVEL_REORDERING_OPTION == True:
                l4_outer.receive_data_from_subflow_layer(
                    ue_id, packets_to_quic_buff)  # Enqueue into stream-level reorder buffer
                l4_logger.store("main", f"UE{ue_id}", "mpquic_reordering_buff_length_at_UE",
                                f"time={TimeManager.time_index}: {l4_outer.ue_states[ue_id]['send_buffer_to_upper_layer'].length}")
                packets_to_quic_buff = l4_outer.send_data_to_upper_layer(
                    ue_id)  # Dequeue from stream-level reorder buffer

            # QUIC receive & ACK generation
            packet_ue2ullink_quic = l4_inner.receiver(
                ue_id, packets_to_quic_buff)

            # --- RAN PROCESSING (UL) ---
            ul_wiredlink_sn[ue_id].enqueue(packet_ue2ullink_sf1)
            # QUIC ACK packets should always be enqueued into sn path here.
            ul_wiredlink_sn[ue_id].enqueue(packet_ue2ullink_quic)
            ul_wiredlink_mn[ue_id].enqueue(packet_ue2ullink_sf2)

            ul_wiredlink_sn[ue_id].do_timeslot()
            ul_wiredlink_mn[ue_id].do_timeslot()

        for ue_id in ue_id_list:
            # --- UPF PROCESSING (UL) ---
            packets_from_sn = ul_wiredlink_sn[ue_id].dequeue()
            # Separate MPQUIC subflow ACKs and inner QUIC ACKs; diff_ack dispatches ACK packet types
            mpquic_ack_to_sf1_buff, quic_ack = l4_outer.diff_ack(
                packets_from_sn)
            # Subflow ACKs from mn path
            mpquic_ack_to_sf2_buff = ul_wiredlink_mn[ue_id].dequeue()

            # Subflow ACK handling
            l4_outer.ue_states[ue_id]["subflow_1"].receive_ack(
                ue_id, mpquic_ack_to_sf1_buff)
            l4_outer.ue_states[ue_id]["subflow_2"].receive_ack(
                ue_id, mpquic_ack_to_sf2_buff)

            # --- N6 PROCESSING (UL) ---
            ul_N6[ue_id].enqueue(quic_ack)
            quic_ack_from_n6 = ul_N6[ue_id].dequeue()

            # --- SERVER PROCESSING ---
            l4_inner.sender_recv(ue_id, quic_ack_from_n6)

            # --- Log current RAN buffer sizes (monitoring) ---
            l4_logger.store(
                "main", f"UE{ue_id}", "mn_5G_pdcp_buff_size", ran_mn.pdcp.dl_buffer_list[ue_id].length)
            l4_logger.store(
                "main", f"UE{ue_id}", "sn_6G_pdcp_buff_size", ran_sn.pdcp.dl_buffer_list[ue_id].length)
            l4_logger.store(
                "main", f"UE{ue_id}", "mn_5G_mac_buff_size", ran_mn.mac.return_buffer_status()[ue_id])
            l4_logger.store(
                "main", f"UE{ue_id}", "sn_6G_mac_buff_size", ran_sn.mac.return_buffer_status()[ue_id])

        TimeManager.time_index += 1

    # --- Persist results and export logs ---
    ran_sn.save_all_info(csv_conversion=True, plot=True)
    ran_mn.save_all_info(csv_conversion=True, plot=True)
    l4_inner.onetime_logger()
    l4_outer.onetime_logger()
    l4_logger.export_all()
    l4_logger.export_all_UE_results()


if __name__ == "__main__":
    if ParameterClass.SIM_MODE == "DC":
        print("DC simulation")
        dc_sim()
    if ParameterClass.SIM_MODE == "MPQUIC":
        print("MPQUIC simulation")
        start_time = time.time()
        print(start_time)
        ds_sim()
        print(time.time() - start_time)
