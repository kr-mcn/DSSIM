import numpy as np
from pathlib import Path
from param import ParameterClass, TimeManager
from packet import Sent_packets, get_time
from cubic_hystart import CUBIC_HyStart
from buffer import Buffer, ReorderingBuffer
import os
import datetime
import matplotlib.pyplot as plt
import csv
from mpquic_subflow import MPQUIC_SUBFLOW
from collections import deque
import shutil
from logger import L4Logger
import math


class MPQUIC(ParameterClass, TimeManager):

    def __init__(self, logger):
        self.ue_states = {}  # Dictionary to manage per-UE states
        self.logger = logger

    def initialize_ue(self, ue_id):
        """
        Initialize per-UE state: subflows, buffers, and transmission control variables.
        """
        self.ue_states[ue_id] = {
            'subflow_1': MPQUIC_SUBFLOW(logger=self.logger, mpquic_instance=self),
            'subflow_2': MPQUIC_SUBFLOW(logger=self.logger, mpquic_instance=self),
            'send_buffer_to_lower_layer': Buffer(max_volume=ParameterClass.BUFF_MAX_VOLUME_MPQUIC_SEND_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_MPQUIC_SEND_DL),
            'retranmission_buffer_to_lower_layer': Buffer(max_volume=ParameterClass.BUFF_MAX_VOLUME_MPQUIC_RESEND_DL, max_length=ParameterClass.BUFF_MAX_LENGTH_MPQUIC_RESEND_DL),
            'send_buffer_to_upper_layer': ReorderingBuffer(max_volume=ParameterClass.BUFF_MAX_VOLUME_MPQUIC_RECV_UL, max_length=ParameterClass.BUFF_MAX_LENGTH_MPQUIC_RECV_UL, ue_id=ue_id, logger=self.logger),
            'throughput': [],  # Store throughput at each time index
            'goodput': [],  # Store goodput at each time index
            'event_log': [],  # Record occurred events for CSV output
            # Number of packets allowed to send on 6G path (RAN feedback mode)
            '6g_path_transmittable': 0,
            # Number of packets allowed to send on 5G path (both RAN feedback mode)
            '5g_path_transmittable': 0,
            'packet_num': 0,  # MPQUIC-layer packet number counter
            'feedback_amount': 0,  # Amount granted by RAN feedback
            'carry_over': 0,  # Amount carried over when pacing cannot send all packets in one slot
            'denominator': 1,  # Denominator used for pacing calculation
        }
        self.ue_states[ue_id]['subflow_1'].initialize_ue(
            ue_id=ue_id, subflow_id=1)
        self.ue_states[ue_id]['subflow_2'].initialize_ue(
            ue_id=ue_id, subflow_id=2)

    def receive_data(self, ue_id, received_packets):
        if received_packets is not None:
            state = self.ue_states[ue_id]
            # Determine how many packets can be enqueued to avoid buffer overflow
            enqueue_limit = state['send_buffer_to_lower_layer'].max_length - \
                state['send_buffer_to_lower_layer'].length
            start_num = state['packet_num']
            if enqueue_limit >= received_packets.shape[0]:
                end_num = state['packet_num'] + received_packets.shape[0]
                received_packets[:, ParameterClass.INDEX_STREAM_PACKET_ID] = np.arange(
                    start_num, end_num)
                state['packet_num'] += received_packets.shape[0]
            else:
                end_num = state['packet_num'] + enqueue_limit
                received_packets[:enqueue_limit, ParameterClass.INDEX_STREAM_PACKET_ID] = np.arange(
                    start_num, end_num)
                state['packet_num'] += enqueue_limit

            # Enqueue packets into the send buffer
            state['send_buffer_to_lower_layer'].enqueue(received_packets)

    def _take_from_buffers(self, state, cap: int):
        """
        Dequeue up to `cap` packets: prioritizing retransmission buffer first, then normal buffer.
        Return stacked array of packets or None if empty.
        """
        if cap <= 0:
            return None

        rbuf = state['retranmission_buffer_to_lower_layer']
        sbuf = state['send_buffer_to_lower_layer']

        r = rbuf.dequeue(dequeue_type="length", length=cap)
        rem = cap - len(r)
        n = None
        if rem > 0:
            n = sbuf.dequeue(dequeue_type="length", length=rem)

        len_r = len(r) if r is not None else 0
        len_n = len(n) if n is not None else 0
        if len_r + len_n == 0:
            return None
        if len_r == 0:
            return n
        if len_n == 0:
            return r
        return np.vstack((r, n))

    def _schedule_min_rtt(self, state, ue_id, cap1: int, cap2: int):
        """
        Min-RTT scheduler:
        Select packets from the subflow with the shorter RTT first, within given caps.
        If cap is 0, that subflow does not transmit.
        Return tuple: (sf1_packets, sf2_packets)
        """
        if cap1 <= 0 and cap2 <= 0:
            return None, None
        if cap1 > 0 and cap2 <= 0:
            return self._take_from_buffers(state, cap1), None
        if cap1 <= 0 and cap2 > 0:
            return None, self._take_from_buffers(state, cap2)

        sf1 = state['subflow_1'].ue_states[ue_id]
        sf2 = state['subflow_2'].ue_states[ue_id]
        rtt1 = sf1['smoothed_rtt']
        rtt2 = sf2['smoothed_rtt']

        if rtt1 <= rtt2:
            sf1_pkts = self._take_from_buffers(state, cap1)
            sf2_pkts = self._take_from_buffers(state, cap2)
        else:
            sf2_pkts = self._take_from_buffers(state, cap2)
            sf1_pkts = self._take_from_buffers(state, cap1)
        return sf1_pkts, sf2_pkts

    def send_data(self, ue_id):
        """
        Get number of transmittable packets for both subflows,
        decide which to send first using min-RTT scheduling,
        and dequeue retransmission packets first.
        """
        state = self.ue_states[ue_id]
        cap1 = state['subflow_1'].ue_states[ue_id]['transmittable_packets_num']
        cap2 = state['subflow_2'].ue_states[ue_id]['transmittable_packets_num']
        return self._schedule_min_rtt(state, ue_id, cap1, cap2)

    def send_data_considering_ran_fb(self, ue_id, ran_fb_info=None):
        """
        Consider RAN feedback: update 6G quota, send up to that quota on SF1 (6G path),
        then use remaining capacity on SF2.
        """
        state = self.ue_states[ue_id]
        if ran_fb_info is not None:
            if ran_fb_info == 0:
                state['6g_path_transmittable'] = max(
                    0, state['6g_path_transmittable'])
            else:
                state['6g_path_transmittable'] = ran_fb_info
        else:
            state['6g_path_transmittable'] = max(
                0, state['6g_path_transmittable'])

        quota_sf1 = state['6g_path_transmittable']
        cap_sf2 = state['subflow_2'].ue_states[ue_id]['transmittable_packets_num']
        sf1_packets = self._take_from_buffers(state, quota_sf1)
        sent_sf1 = 0 if sf1_packets is None else len(sf1_packets)
        state['6g_path_transmittable'] = max(0, quota_sf1 - sent_sf1)
        sf2_packets = self._take_from_buffers(state, cap_sf2)
        return sf1_packets, sf2_packets

    def send_data_considering_both_ran_fb(self, ue_id, ran_fb_sf1=None, ran_fb_sf2=None):
        """
        Consider feedback from both RANs: update quotas for SF1 (6G) and SF2 (5G),
        then schedule packets accordingly.
        """
        state = self.ue_states[ue_id]
        if ran_fb_sf1 is not None:
            if ran_fb_sf1 == 0:
                state['6g_path_transmittable'] = max(
                    0, state['6g_path_transmittable'])
            else:
                state['6g_path_transmittable'] = ran_fb_sf1
        else:
            state['6g_path_transmittable'] = max(
                0, state['6g_path_transmittable'])

        if ran_fb_sf2 is not None:
            if ran_fb_sf2 == 0:
                state['5g_path_transmittable'] = max(
                    0, state['5g_path_transmittable'])
            else:
                state['5g_path_transmittable'] = ran_fb_sf2
        else:
            state['5g_path_transmittable'] = max(
                0, state['5g_path_transmittable'])

        sf1_packets, sf2_packets = self._schedule_min_rtt(
            state, ue_id, state['6g_path_transmittable'], state['5g_path_transmittable'])
        sent_length_sf1 = 0 if sf1_packets is None else len(sf1_packets)
        sent_length_sf2 = 0 if sf2_packets is None else len(sf2_packets)
        state['6g_path_transmittable'] = max(
            0, state['6g_path_transmittable'] - sent_length_sf1)
        state['5g_path_transmittable'] = max(
            0, state['5g_path_transmittable'] - sent_length_sf2)

        return sf1_packets, sf2_packets

    def send_data_considering_ran_fb_with_pacing(self, ue_id, ran_fb_info=None):
        """
        When FB arrives:
        1) Overwrite feedback_amount
        2) Reset denominator to RAN_FB_CYCLE
        Then:
        Set ceil(feedback_amount/denominator) as the SF1 cap for the current timeslot
        After transmission, subtract len(actual number sent in SF1) from feedback_amount
        Set denominator to max(1, denominator-1)
        """
        state = self.ue_states[ue_id]

        # FB reflection
        if ran_fb_info is not None:
            state['feedback_amount'] = ran_fb_info
            state['denominator'] = ParameterClass.RAN_FB_CYCLE

        # Current slot SF1 upper limit (rounded up)
        quota_sf1 = math.ceil(state['feedback_amount'] / state['denominator'])
        state['6g_path_transmittable'] = quota_sf1

        # Reduce denominator by 1 for next slot (lower limit 1)
        state['denominator'] = max(1, state['denominator'] - 1)

        # SF2 capacity
        cap_sf2 = state['subflow_2'].ue_states[ue_id]['transmittable_packets_num']

        # transmit
        sf1_packets_to_send = self._take_from_buffers(
            state, quota_sf1)    # cap=quota_sf1
        sf2_packets_to_send = self._take_from_buffers(
            state, cap_sf2)      # cap=cap_sf2

        if sf1_packets_to_send is not None:
            print(f"len(sf1_packets_to_send)={len(sf1_packets_to_send)}")
        else:
            print("len(sf1_packets_to_send)=0")

        # Reduce FB_amount by the amount sent
        sent_sf1 = 0 if sf1_packets_to_send is None else len(
            sf1_packets_to_send)
        state['feedback_amount'] = max(0, state['feedback_amount'] - sent_sf1)

        return sf1_packets_to_send, sf2_packets_to_send

    def receive_data_from_subflow_layer(self, ue_id, packets):
        recv_thpt = len(packets) * ParameterClass.MTU_SIZE * \
            8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", "MPQUIC-level_recv_throughput", [TimeManager.time_index, recv_thpt])
        self.ue_states[ue_id]['send_buffer_to_upper_layer'].enqueue(packets)

    def send_data_to_upper_layer(self, ue_id):
        packets_to_send = self.ue_states[ue_id]['send_buffer_to_upper_layer'].dequeue(
        )
        if packets_to_send is not None and len(packets_to_send) > 0:
            pkts = np.atleast_2d(packets_to_send)
            send_times = pkts[:, ParameterClass.INDEX_UPF_TRANSMIT_TIMESTAMP].astype(
                np.int64)
            one_way_delays = TimeManager.time_index - send_times
            for delay in one_way_delays:
                self.logger.store(
                    "MPQUIC", f"UE{ue_id}", "MPQUIC-level_one_way_delay", str(int(delay)))
        self.logger.store("MPQUIC", f"UE{ue_id}", "MPQUIC-level_recv_packets_at_UE",
                          f"time={TimeManager.time_index}: received packets = {packets_to_send}")
        return packets_to_send

    def diff_ack(self, received_acks):
        acks_for_mpquic = []
        acks_for_quic = []
        for ack in received_acks:
            flag = ack[self.INDEX_OUTER_ACK_FLAG]
            if flag == 1:
                acks_for_mpquic.append(ack)
            else:
                acks_for_quic.append(ack)
        acks_for_mpquic = np.array(acks_for_mpquic)
        acks_for_quic = np.array(acks_for_quic)
        return acks_for_mpquic, acks_for_quic

    def onetime_logger(self):
        for ue_id in range(ParameterClass.NUM_UE):
            state = self.ue_states[ue_id]
            state['subflow_1'].onetime_logger(ue_id)
            state['subflow_2'].onetime_logger(ue_id)
