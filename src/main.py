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
from tecc import TECC
from bbrv1_definitions import BBRState
from camf import CAMFController
import pdb
import os
import datetime
import csv
import time
import cProfile
import math
from collections import deque
import itertools
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'validation'))
from test_bbr import test_bbr
import pstats

np.random.seed(0)

def ds_sim():
    # --- Initialize components for DualSteer / MPQUIC simulation ---
    ue_id_list = np.arange(ParameterClass.NUM_UE)
    dl_N3_0 = WiredLink(bandwidth_pps=ParameterClass.N3_BANDWIDTH_PPS, bandwidth_bps=ParameterClass.N3_BANDWIDTH_BPS, loss_prob=ParameterClass.N3_LOSS_RATE_DL)  # Downlink RAN–UPF wired path for subflow 1 (sn/6G)
    dl_N3_1 = WiredLink(bandwidth_pps=ParameterClass.N3_BANDWIDTH_PPS, bandwidth_bps=ParameterClass.N3_BANDWIDTH_BPS, loss_prob=ParameterClass.N3_LOSS_RATE_DL)  # Downlink RAN–UPF wired path for subflow 2 (mn/5G)
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
    upf_fb_queue = {}
    last_mac_capacity = [1] * ParameterClass.NUM_UE

    # For TECC
    tecc_fb_queue = {}
    tecc_tx_amount_per_fb = {}
    tecc_q_min = {}
    tecc_by_ue = {}

    # For RA-TECC
    ran_cap_fb_queue = {}

    # For CAMF
    camf_fb_queue = {}   # {ue_id: [{"reflection_time": int, "rfb_pps": float, "fb_id": int}, ...]}
    _camf_fb_id_counter = itertools.count(1)  # Global FB ID counter (starts from 1)
    camf_active = {}     # {ue_id: bool} Becomes True after slow start exits
    if ParameterClass.CAMF_OPTION:
        camf_ctrl = CAMFController(ParameterClass.NUM_UE)

    for ue_id in ue_id_list:
        l4_inner.initialize_ue(ue_id)
        l4_outer.initialize_ue(ue_id)
        dl_N6[ue_id] = BottleneckLink(logger=l4_logger, bandwidth_bps=ParameterClass.N6_BANDWIDTH_BPS, delay_ti=ParameterClass.N6_DELAY_TI, loss_prob=ParameterClass.N6_LOSS_RATE_DL,
                                      max_length=ParameterClass.BUFF_MAX_LENGTH_N6_DL, max_volume=ParameterClass.BUFF_MAX_VOLUME_N6_DL)
        ul_N6[ue_id] = BottleneckLink(logger=l4_logger, bandwidth_bps=ParameterClass.N6_BANDWIDTH_BPS, delay_ti=ParameterClass.N6_DELAY_TI, loss_prob=ParameterClass.N6_LOSS_RATE_UL,
                                      max_length=ParameterClass.BUFF_MAX_LENGTH_N6_UL, max_volume=ParameterClass.BUFF_MAX_VOLUME_N6_UL)
        dl_ip_ue_buffer_list_0[ue_id] = Buffer()
        dl_ip_ue_buffer_list_1[ue_id] = Buffer()
        ul_ip_server_buffer_list_0[ue_id] = Buffer()
        ul_ip_server_buffer_list_1[ue_id] = Buffer()
        total_requested_data[ue_id] = 0
        txsize_queue[ue_id] = []
        upf_fb_queue[ue_id] = []
        camf_fb_queue[ue_id] = []
        camf_active[ue_id] = False

        # TECC initialization
        tecc_fb_queue[ue_id] = []
        tecc_by_ue[ue_id] = TECC()
        tecc_tx_amount_per_fb[ue_id] = 0
        tecc_q_min[ue_id] = ParameterClass.BUFF_MAX_LENGTH_MPQUIC_RECV_UL
        ran_cap_fb_queue[ue_id] = []

        # Set the transmission start time for each UE
        l4_inner.ue_states[ue_id]['start_time'] = ParameterClass.UE_START_TIMES.get(ue_id, 0)

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
        ul_wiredlink_sn[i] = WiredLink(bandwidth_pps=ParameterClass.N3_BANDWIDTH_PPS, bandwidth_bps=ParameterClass.N3_BANDWIDTH_BPS, loss_prob=ParameterClass.N3_LOSS_RATE_UL)  # Uplink wired path for sn/6G subflow
        ul_wiredlink_mn[i] = WiredLink(bandwidth_pps=ParameterClass.N3_BANDWIDTH_PPS, bandwidth_bps=ParameterClass.N3_BANDWIDTH_BPS, loss_prob=ParameterClass.N3_LOSS_RATE_UL)  # Uplink wired path for mn/5G subflow

    # --- Main Simulation Loop ---
    for _ in range(ParameterClass.NUM_SIMULATION_TIME_SLOTS):
        print(
            "\nTime index is ",
            TimeManager.time_index,
        )

        # --- Apply N6 bandwidth time-change schedule ---
        _ti = TimeManager.time_index
        for _t, _bps in ParameterClass.N6_BANDWIDTH_SCHEDULE:
            if _t == _ti:
                _bw = float(_bps) * float(ParameterClass.TIME_SLOT_WINDOW)
                for ue_id in ue_id_list:
                    dl_N6[ue_id].bandwidth_Bps_per_ms = _bw
                    ul_N6[ue_id].bandwidth_Bps_per_ms = _bw

        packet_upf2N3_0 = np.empty(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)
        packet_upf2N3_1 = np.empty(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        for ue_id in ue_id_list:
            # --- Skip server transmission for UEs before their start time ---
            if TimeManager.time_index < l4_inner.ue_states[ue_id]['start_time']:
                quic_packets_to_n6 = None
                dl_N6[ue_id].enqueue(quic_packets_to_n6)
                quic_packets_to_upf = dl_N6[ue_id].dequeue()
                l4_outer.receive_data(ue_id, quic_packets_to_upf)
                continue

            # --- CAMF-A: Apply FBs whose delivery time has arrived to the server ---
            # FBs are computed at UPF, queued in a FIFO queue, and applied here after CAMF_FB_DELAY_TI.
            if ParameterClass.CAMF_OPTION:
                if camf_fb_queue[ue_id] and \
                        TimeManager.time_index >= camf_fb_queue[ue_id][0]["reflection_time"]:
                    fb = camf_fb_queue[ue_id].pop(0)
                    l4_inner.ue_states[ue_id]['camf_target_pps'] = fb['rfb_pps']
                    l4_inner.ue_states[ue_id]['camf_fb_id']      = fb['fb_id']
                    # tunnel_rtt is also retrieved from the FB packet (CAMF_FB_DELAY_TI delay already accounted for)
                    l4_inner.ue_states[ue_id]['tunnel_rtt'] = fb['tunnel_rtt']
                    l4_logger.store("CAMF", f"UE{ue_id}", "camf_fb_apply",
                        f"time={TimeManager.time_index}: rfb_pps={fb['rfb_pps']:.4f} [pkt/ms] fb_id={fb['fb_id']}")

            # --- SERVER PROCESSING (DL) ---
            if l4_inner.check_timer(ue_id):
                quic_packets_to_n6 = l4_inner.on_loss_detection_timeout(ue_id)
            else:
                if ParameterClass.TECC_OPTION is True:
                    # --- TECC: Tunnel-endpoint FB-based rate control ---
                    # Slow start end detection -> activate TECC
                    if not tecc_by_ue[ue_id].is_active:
                        cc = l4_inner.ue_states[ue_id]['cc_algo']
                        startup_ended = False
                        if ParameterClass.QUIC_CC == "BBR":
                            startup_ended = cc.state != BBRState.STARTUP
                        elif ParameterClass.QUIC_CC == "CUBIC":
                            startup_ended = cc.current_mode not in ("SLOW_START", "CSS")
                        if startup_ended:
                            tecc_by_ue[ue_id].activate()
                            l4_logger.store("TECC", f"UE{ue_id}", "tecc_sr_log",
                                f"time={TimeManager.time_index}: TECC activated (CC exited slow start)")
                    # TECC FB arrival -> update rate
                    while tecc_fb_queue[ue_id] and TimeManager.time_index >= tecc_fb_queue[ue_id][0]["reflection_time"]:
                        fb = tecc_fb_queue[ue_id][0]
                        sr_pps = tecc_by_ue[ue_id].on_feedback(
                            tunnel_rate_pps=fb["tunnel_rate_pps"],
                            q_len_min_pkts=fb["q_len_min_pkts"],
                            retransmit_ratio=fb["retransmit_ratio"],
                            tunnel_rtt=fb.get("tunnel_rtt"))
                        l4_inner.ue_states[ue_id]['tunnel_rtt'] = tecc_by_ue[ue_id].tunnel_rtt
                        l4_logger.store("TECC", f"UE{ue_id}", "tecc_sr_log",
                            f"time={TimeManager.time_index}: sr_pps={sr_pps}, tunnel_rtt={tecc_by_ue[ue_id].tunnel_rtt}")
                        del tecc_fb_queue[ue_id][0]
                    # Pacing
                    send_count = tecc_by_ue[ue_id].get_send_count()
                    if send_count is None:
                        quic_packets_to_n6 = l4_inner.sender_send(ue_id)
                    elif send_count > 0:
                        quic_packets_to_n6 = l4_inner.sender_send_by_tecc(ue_id, send_count)
                    else:
                        quic_packets_to_n6 = l4_inner.sender_send_by_tecc(ue_id)
                else:
                    quic_packets_to_n6 = l4_inner.sender_send(ue_id)

            # --- N6 PROCESSING (DL) ---
            dl_N6[ue_id].enqueue(quic_packets_to_n6)
            quic_packets_to_upf = dl_N6[ue_id].dequeue()

            # --- UPF PROCESSING (DL) ---
            l4_outer.receive_data(ue_id, quic_packets_to_upf)
            l4_logger.store("MPQUIC", f"UE{ue_id}", "mpquic-level_recv_buff_length_at_UPF_before_send",
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
                        "RANFB", f"UE{ue_id}", "ran_fb_pdcp_filling_ratio", f"time = {TimeManager.time_index}, pdcp_buff_filling_ratio = {pdcp_buff_filling_ratio}")
                    l4_logger.store("RANFB", f"UE{ue_id}", "mac_capacity",
                                    f"time = {TimeManager.time_index},\tmac_tx_capacity={mac_tx_capacity},\tlast_mac_capacity[ue_id]={last_mac_capacity[ue_id]}")
                    # Guard buffer: at least 40 packets or N3 delay (minus 1) worth of time
                    guard_buff = max(40, ParameterClass.N3_DELAY-1)
                    # Required packets over the next period minus current in-flight/queued amounts
                    tx_data_size = max(0, int(exp_thpt_pps * (ParameterClass.RAN_FB_CYCLE + guard_buff)
                                              * ParameterClass.TIME_SLOT_WINDOW - mac_buff_size_packet - pdcp_buff_size
                                              # - total_requested_data[ue_id]
                                              ))

                    l4_logger.store(
                        "RANFB", f"UE{ue_id}", "ran_fb_log", f"time = {TimeManager.time_index},\t{exp_thpt_pps} [pps],\tmac_tx_capacity={mac_tx_capacity},\tlast_mac_capacity={last_mac_capacity[ue_id]}, \t{mac_buff_size_packet:.2f} [packets in mac_buff],\t{pdcp_buff_size} [packets in pdcp_buff],\t{total_requested_data[ue_id]}[unarrived packets],\t{tx_data_size}[requested packets]")

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
                        l4_logger.store("RANFB", f"UE{ue_id}", "ran_fb_queue_log",
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
                        l4_logger.store("RANFB", f"UE{ue_id}", "ran_fb_queue_log",
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

            l4_logger.store("MPQUIC", f"UE{ue_id}", "mpquic-level_recv_buff_length_at_UPF_after_send",
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

            # --- CAMF-C: Measure q_min of UPF send buffer (same measurement point as TECC q_min) ---
            if ParameterClass.CAMF_OPTION:
                _q_len_upf = (l4_outer.ue_states[ue_id]['send_buffer_to_lower_layer'].length
                          + l4_outer.ue_states[ue_id]['retranmission_buffer_to_lower_layer'].length)
                camf_ctrl.tick_upf_queue(ue_id, _q_len_upf)
            # if needed: measure real DL processing time here
            if mpquic_packets_by_sf1 is not None:
                packet_upf2N3_0 = np.vstack(
                    (packet_upf2N3_0, mpquic_packets_by_sf1))
            if mpquic_packets_by_sf2 is not None:
                packet_upf2N3_1 = np.vstack(
                    (packet_upf2N3_1, mpquic_packets_by_sf2))
            
            # --- TECC FB generation ---
            if ParameterClass.TECC_OPTION is True:
                tecc_tx_amount_per_fb[ue_id] += (sf1_len + sf2_len)
                # retransmit_enqueue_count is automatically accumulated at the MPQUIC layer (read and reset when generating FB)
                tecc_q_min[ue_id] = min(tecc_q_min[ue_id], l4_outer.ue_states[ue_id]['send_buffer_to_lower_layer'].length + l4_outer.ue_states[ue_id]['retranmission_buffer_to_lower_layer'].length)
                if TimeManager.time_index > ParameterClass.TECC_FB_CYCLE and TimeManager.time_index % ParameterClass.TECC_FB_CYCLE == 0:
                    tunnel_rate_pps = tecc_tx_amount_per_fb[ue_id] / (ParameterClass.TECC_FB_CYCLE * ParameterClass.TIME_SLOT_WINDOW)
                    retransmit_count = l4_outer.ue_states[ue_id]['retransmit_enqueue_count']
                    retransmit_ratio = retransmit_count / tecc_tx_amount_per_fb[ue_id] if tecc_tx_amount_per_fb[ue_id] != 0 else 0
                    sf1_rtt = l4_outer.ue_states[ue_id]["subflow_1"].ue_states[ue_id]['smoothed_rtt']
                    sf2_rtt = l4_outer.ue_states[ue_id]["subflow_2"].ue_states[ue_id]['smoothed_rtt']
                    observed_tunnel_rtt = max(sf1_rtt, sf2_rtt)
                    tecc_fb_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index + (ParameterClass.N3_DELAY-1)*2 + ParameterClass.N6_DELAY_TI,
                        "tunnel_rate_pps": tunnel_rate_pps,
                        "q_len_min_pkts": tecc_q_min[ue_id],
                        "retransmit_ratio": retransmit_ratio,
                        "tunnel_rtt": observed_tunnel_rtt,
                    })
                    l4_logger.store("TECC", f"UE{ue_id}", "tecc_fb_log", f"time={TimeManager.time_index}: tunnel_rate_pps={tunnel_rate_pps}, q_min={tecc_q_min[ue_id]}, retransmit_count={retransmit_count}, retransmit_ratio={retransmit_ratio}, tunnel_rtt={observed_tunnel_rtt}")
                    tecc_tx_amount_per_fb[ue_id] = 0
                    l4_outer.ue_states[ue_id]['retransmit_enqueue_count'] = 0
                    tecc_q_min[ue_id] = ParameterClass.BUFF_MAX_LENGTH_MPQUIC_RECV_UL

            # --- CAMF-D: End-of-period processing (compute Rfb & enqueue FB / BN detection) ---
            # Slow start end detection -> activate CAMF
            if (ParameterClass.CAMF_OPTION) \
                    and not camf_active[ue_id]:
                cc = l4_inner.ue_states[ue_id]['cc_algo']
                startup_ended = False
                if ParameterClass.QUIC_CC == "BBR":
                    startup_ended = cc.state != BBRState.STARTUP
                elif ParameterClass.QUIC_CC == "CUBIC":
                    startup_ended = cc.current_mode not in ("SLOW_START", "CSS")
                if startup_ended:
                    camf_active[ue_id] = True
                    l4_logger.store("CAMF", f"UE{ue_id}", "camf_activate_log",
                        f"time={TimeManager.time_index}: CAMF activated (CC exited slow start)")

            if (ParameterClass.CAMF_OPTION) \
                    and camf_active[ue_id] \
                    and TimeManager.time_index > 0 \
                    and TimeManager.time_index % ParameterClass.CAMF_PERIOD_TI == 0:

                # Rproxy = sum of per-path RTT-adaptive bandwidth estimates [pkt/ms]
                # Congestion on each path is determined by RTT ratio = latest_rtt / rt_prop.
                #   No congestion (rtt_ratio < RTT_CONG_THRESH):
                #       btlbw is a nearly accurate available bandwidth estimate -> use as-is
                #   Congestion (rtt_ratio >= RTT_CONG_THRESH):
                #       btlbw may stagnate for up to 10 RTTs and is unreliable.
                #       Use delivery_rate (measured value responsive within 1-2 RTTs), corrected for RTT inflation.
                #       delivery_rate / rtt_ratio: removes the slowdown due to queue buildup,
                #       restoring the "estimated rate that would have flowed without a queue".
                # Computed independently per subflow and summed (congestion on one path does not affect the other).
                _RTT_CONG_THRESH = ParameterClass.CAMF_RTT_CONG_THRESH
                sf1_cc     = l4_outer.ue_states[ue_id]['subflow_1'].ue_states[ue_id]['cc_algo']
                sf2_cc     = l4_outer.ue_states[ue_id]['subflow_2'].ue_states[ue_id]['cc_algo']
                sf1_rtt_ms = l4_outer.ue_states[ue_id]['subflow_1'].ue_states[ue_id]['latest_rtt'] * 1000
                sf2_rtt_ms = l4_outer.ue_states[ue_id]['subflow_2'].ue_states[ue_id]['latest_rtt'] * 1000

                sf1_rtt_ratio = sf1_rtt_ms / sf1_cc.rt_prop \
                    if sf1_cc.rt_prop > 0 and sf1_cc.rt_prop != float('inf') else 1.0
                sf2_rtt_ratio = sf2_rtt_ms / sf2_cc.rt_prop \
                    if sf2_cc.rt_prop > 0 and sf2_cc.rt_prop != float('inf') else 1.0

                if sf1_rtt_ratio < _RTT_CONG_THRESH:
                    sf1_cap = sf1_cc.btlbw                              # No congestion: adopt btlbw
                    sf1_mode = "btlbw"
                else:
                    sf1_cap = sf1_cc.delivery_rate / sf1_rtt_ratio      # Congestion: apply delivery_rate correction
                    sf1_mode = "dlvr"

                if sf2_rtt_ratio < _RTT_CONG_THRESH:
                    sf2_cap = sf2_cc.btlbw
                    sf2_mode = "btlbw"
                else:
                    sf2_cap = sf2_cc.delivery_rate / sf2_rtt_ratio
                    sf2_mode = "dlvr"

                cap_pps = sf1_cap + sf2_cap  # [pkt/ms]
                l4_logger.store("CAMF", f"UE{ue_id}", "cap_pps_log",
                    f"time={TimeManager.time_index}: cap_pps={cap_pps:.4f}, "
                    f"sf1=({sf1_mode} rtt_ratio={sf1_rtt_ratio:.3f} cap={sf1_cap:.4f}), "
                    f"sf2=({sf2_mode} rtt_ratio={sf2_rtt_ratio:.3f} cap={sf2_cap:.4f})")

                if ParameterClass.CAMF_OPTION:
                    camf_ctrl.set_capacity_sample(ue_id, cap_pps)
                    rfb = camf_ctrl.compute_rfb(ue_id)
                    _sf1_srtt = l4_outer.ue_states[ue_id]['subflow_1'].ue_states[ue_id]['smoothed_rtt']
                    _sf2_srtt = l4_outer.ue_states[ue_id]['subflow_2'].ue_states[ue_id]['smoothed_rtt']
                    # FB is always enqueued. When rfb=None, rfb_pps=0 (rate FB suppressed).
                    _fb_id = next(_camf_fb_id_counter)
                    _rfb_pps = rfb if rfb is not None else 0.0
                    camf_fb_queue[ue_id].append({
                        "reflection_time": TimeManager.time_index
                                           + ParameterClass.CAMF_FB_DELAY_TI,
                        "rfb_pps":        _rfb_pps,
                        "fb_id":          _fb_id,
                        "tunnel_rtt":     min(max(_sf1_srtt, _sf2_srtt), ParameterClass.TUNNEL_RTT_CAP),
                    })
                    if rfb is not None:
                        l4_logger.store("CAMF", f"UE{ue_id}", "camf_rfb_log",
                            f"time={TimeManager.time_index}: rfb_pps={rfb:.4f} [pkt/ms], "
                            f"rproxy={cap_pps:.4f}, u_ewma={camf_ctrl._u_ewma[ue_id]:.4f}, "
                            f"deliver_at={TimeManager.time_index + ParameterClass.CAMF_FB_DELAY_TI}")
                    else:
                        l4_logger.store("CAMF", f"UE{ue_id}", "camf_rfb_log",
                            f"time={TimeManager.time_index}: rfb_pps=None (no capacity), "
                            f"rproxy={cap_pps:.4f}")

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
                l4_logger.store("MPQUIC", f"UE{ue_id}", "mpquic-level_reordering_buff_length_at_UE",
                                f"time={TimeManager.time_index}: {l4_outer.ue_states[ue_id]['send_buffer_to_upper_layer'].length}")
                packets_to_quic_buff = l4_outer.send_data_to_upper_layer(
                    ue_id)  # Dequeue from stream-level reorder buffer

            # Subflow ACK generation (each subflow sends ACKs)
            packet_ue2ullink_sf1 = l4_outer.ue_states[ue_id]["subflow_1"].receive_data_send_ack(
                ue_id, mpquic_packets_to_sf1_buff)
            packet_ue2ullink_sf2 = l4_outer.ue_states[ue_id]["subflow_2"].receive_data_send_ack(
                ue_id, mpquic_packets_to_sf2_buff)

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
                "RAN", f"UE{ue_id}", "mn_5G_pdcp_buff_size", ran_mn.pdcp.dl_buffer_list[ue_id].length)
            l4_logger.store(
                "RAN", f"UE{ue_id}", "sn_6G_pdcp_buff_size", ran_sn.pdcp.dl_buffer_list[ue_id].length)
            l4_logger.store(
                "RAN", f"UE{ue_id}", "mn_5G_mac_buff_size", ran_mn.mac.return_buffer_status()[ue_id])
            l4_logger.store(
                "RAN", f"UE{ue_id}", "sn_6G_mac_buff_size", ran_sn.mac.return_buffer_status()[ue_id])
            # --- Log UPF / N6 buffer sizes (monitoring, per-UE) ---
            l4_logger.store(
                "MPQUIC", f"UE{ue_id}", "upf_send_buff_pkts", l4_outer.ue_states[ue_id]['send_buffer_to_lower_layer'].length)
            l4_logger.store(
                "N6LINK", f"UE{ue_id}", "n6_inflight_pkts", len(dl_N6[ue_id].content))

        # --- Log N3 buffer sizes (monitoring, shared link) ---
        l4_logger.store("N3LINK", "ALL", "n3_sf1_buff_pkts", dl_N3_0.length)
        l4_logger.store("N3LINK", "ALL", "n3_sf2_buff_pkts", dl_N3_1.length)

        # FCT mode: check completion for all UEs
        if ParameterClass.FCT_MODE:
            all_complete = all(
                l4_inner.ue_states[ue_id].get('fct_recv_complete', False)
                for ue_id in ue_id_list
            )
            if all_complete:
                print(f"\nFCT mode: All UEs completed at time_index={TimeManager.time_index}")
                break

        TimeManager.time_index += 1

    ### Plot the goodput ###
    # if ParameterClass.QUIC_CC == "BBR":
    #     plot_simple(BBRLog.log_goodput_quic, "quic_goodput_Mbps")
    # if ParameterClass.MPQUIC_CC == "BBR":
    #     plot_simple(BBRLog.log_goodput_mpquic, "mpquic_goodput_Mbps")
    ### end ###

    # --- FCT mode: output results ---
    if ParameterClass.FCT_MODE:
        fct_lines = ["=== FCT Results ==="]
        for ue_id in ue_id_list:
            state = l4_inner.ue_states[ue_id]
            start = state['start_time']
            send_end = state['fct_send_completion_time']
            end = state['fct_completion_time']
            if end is not None:
                fct_ms = (end - start) * ParameterClass.TIME_SLOT_WINDOW * 1000
                send_ms = (send_end - start) * ParameterClass.TIME_SLOT_WINDOW * 1000 if send_end is not None else None
                total_bytes = state['fct_flow_size'] * ParameterClass.MTU_SIZE
                avg_goodput_mbps = total_bytes * 8 / (fct_ms / 1000) / 1e6
                send_str = f", send_complete={send_ms:.1f}ms" if send_ms is not None else ""
                fct_lines.append(f"  UE{ue_id}: FCT={fct_ms:.1f}ms{send_str}, "
                      f"packets={state['fct_flow_size']}, "
                      f"avg_goodput={avg_goodput_mbps:.2f}Mbps")
            else:
                fct_lines.append(f"  UE{ue_id}: NOT COMPLETED (timeout)")
            # Also log to the logger
            fct_ti = (end - start) if end is not None else None
            send_ti = (send_end - start) if send_end is not None else None
            l4_logger.store("FCT", f"UE{ue_id}", "fct_result", {
                "start_time_index": start,
                "send_completion_time_index": send_end,
                "send_completion_seconds": send_ti * ParameterClass.TIME_SLOT_WINDOW if send_ti else None,
                "completion_time_index": end,
                "fct_time_index": fct_ti,
                "fct_seconds": fct_ti * ParameterClass.TIME_SLOT_WINDOW if fct_ti else None,
                "total_packets": state['fct_flow_size'],
                "total_bytes": state['fct_flow_size'] * ParameterClass.MTU_SIZE,
            })
        # Console output
        print("\n" + "\n".join(fct_lines))
        # File output
        fct_log_dir = os.path.join(ParameterClass.HEAVY_DATA_PATH, ParameterClass.LOG_SAVE_PATH, "L4_results", "all_UE_results")
        os.makedirs(fct_log_dir, exist_ok=True)
        with open(os.path.join(fct_log_dir, "fct_log.txt"), "w") as f:
            f.write("\n".join(fct_lines) + "\n")

    # --- Persist results and export logs ---
    ran_sn.save_all_info(csv_conversion=True, plot=True)
    ran_mn.save_all_info(csv_conversion=True, plot=True)
    l4_inner.onetime_logger()
    l4_outer.onetime_logger()
    l4_logger.export_all()
    l4_logger.export_all_UE_results()


if __name__ == "__main__":
    if ParameterClass.SIM_MODE == "MPQUIC":
        print("MPQUIC simulation")
        start_time = time.time()
        print(start_time)
        ds_sim()
        #cProfile.run("ds_sim()", "prof.out")
        print(time.time() - start_time)

        #stats = pstats.Stats("prof.out")
        #stats.sort_stats("cumtime").print_stats(30)
