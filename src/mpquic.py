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
from scheduler import Min_rtt_scheduler
from bbrv1_definitions import BBRState

class MPQUIC(ParameterClass, TimeManager):

    def __init__(self, logger):
        self.ue_states = {}  # Dictionary to manage per-UE states
        self.logger = logger

        # Instantiate the schedular
        if ParameterClass.SCHEDULER_MODE == "MINRTT":
            self.schedular = Min_rtt_scheduler()

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
            'retransmit_enqueue_count': 0,  # For TECC r(t): number of packets enqueued in retransmission buffer during FB period

            # ── MPQUIC Flow Control (UPF->UE direction, RFC 9000 compliant) ──────────────────
            # UE side (receiver): manages cumulative amount passed from send_buffer_to_upper_layer to QUIC layer and MAX_DATA
            'fc_recv_window':      ParameterClass.MPQUIC_FC_RECV_WINDOW,
            'fc_recv_offset':      0,     # Cumulative packet count passed to QUIC layer (monotonically increasing)
            'fc_recv_max_offset':  ParameterClass.MPQUIC_FC_RECV_WINDOW,  # MAX_DATA to advertise
            # UPF side (sender): manages MAX_DATA received from UE and cumulative sent amount
            'fc_send_offset':      0,     # Cumulative packets sent as new data (retransmissions not counted)
            'fc_send_max_offset':  ParameterClass.MPQUIC_FC_RECV_WINDOW,  # MAX_DATA from UE (send limit)
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
                if not ParameterClass.UDP_MODE:
                    dropped_ids = received_packets[enqueue_limit:, ParameterClass.INDEX_PACKET_ID].tolist()
                    self.logger.store(
                        "MPQUIC", f"UE{ue_id}", "upf_buffer_overflow_log",
                        f"{TimeManager.time_index},{dropped_ids}")
                end_num = state['packet_num'] + enqueue_limit
                received_packets[:enqueue_limit, ParameterClass.INDEX_STREAM_PACKET_ID] = np.arange(
                    start_num, end_num)
                state['packet_num'] += enqueue_limit

            # Stamp UPF enqueue timestamp before buffering
            received_packets[:, ParameterClass.INDEX_UPF_ENQUEUE_TIMESTAMP] = TimeManager.time_index

            # Enqueue packets into the send buffer
            state['send_buffer_to_lower_layer'].enqueue(received_packets)

    def _take_from_buffers(self, state, cap: int, new_data_cap: int | None = None):
        """
        Dequeue up to `cap` packets: retransmission buffer first, then normal buffer.

        Parameters
        ----------
        cap          : total packet cap (from CCA / scheduler)
        new_data_cap : upper limit on new-data packets only [FC window remaining].
                       Re-transmissions are NOT subject to this limit.
                       None means no FC limit (= large window).

        Returns
        -------
        tuple (packets, n_new) where
          packets : stacked ndarray of all packets to send (retransmit + new), or None
          n_new   : number of new-data packets in `packets` (for fc_send_offset tracking)
        """
        if cap <= 0:
            return None, 0

        rbuf = state['retranmission_buffer_to_lower_layer']
        sbuf = state['send_buffer_to_lower_layer']

        # Retransmissions: not subject to FC constraints
        r = rbuf.dequeue(dequeue_type="length", length=cap)
        len_r = len(r) if r is not None else 0

        # New data: smaller of CCA remaining capacity and FC window
        rem = cap - len_r
        if new_data_cap is not None:
            rem = min(rem, new_data_cap)
        n = None
        if rem > 0:
            n = sbuf.dequeue(dequeue_type="length", length=rem)
        len_n = len(n) if n is not None else 0

        if len_r + len_n == 0:
            return None, 0
        if len_r == 0:
            return n, len_n
        if len_n == 0:
            return r, 0
        return np.vstack((r, n)), len_n

    def _schedule_packets(self, state, ue_id, cap1: int, cap2: int,
                          new_data_cap: int | None = None):
        """
        Dispatch packets to each subflow according to the configured scheduler.
        Applies BBR pacing limits first, then invokes the scheduler to decide
        which subflow gets first access to the shared send buffer.
        If cap is 0, that subflow does not transmit.

        Parameters
        ----------
        new_data_cap : FC remaining window for new data [pkts]. None = no limit.

        Return tuple: (sf1_packets, sf2_packets, n_new_total)
          n_new_total: number of new-data packets sent across both subflows.
        """

        # --- BBR pacing limit ---
        bbr_subflow_1 = None
        bbr_subflow_2 = None
        if ParameterClass.MPQUIC_CC == "BBR":
            bbr_subflow_1 = self.ue_states[ue_id]['subflow_1'].ue_states[ue_id]['cc_algo']
            pacing_window_1 = bbr_subflow_1.pacing_manager.advance_pacing_manager(
                bbr_subflow_1.pacing_rate)

            bbr_subflow_2 = self.ue_states[ue_id]['subflow_2'].ue_states[ue_id]['cc_algo']
            pacing_window_2 = bbr_subflow_2.pacing_manager.advance_pacing_manager(
                bbr_subflow_2.pacing_rate)

            cap1 = min(cap1, int(pacing_window_1))
            cap2 = min(cap2, int(pacing_window_2))

        # --- Early returns when one or both paths have no capacity ---
        if cap1 <= 0 and cap2 <= 0:
            return None, None, 0
        if cap1 > 0 and cap2 <= 0:
            pkts, n_new = self._take_from_buffers(state, cap1, new_data_cap)
            return pkts, None, n_new
        if cap1 <= 0 and cap2 > 0:
            pkts, n_new = self._take_from_buffers(state, cap2, new_data_cap)
            return None, pkts, n_new

        # --- Both paths have capacity: scheduler decides priority ---
        sf1 = state['subflow_1'].ue_states[ue_id]
        sf2 = state['subflow_2'].ue_states[ue_id]
        rtt1 = sf1['smoothed_rtt']
        rtt2 = sf2['smoothed_rtt']

        # Min-RTT scheduler
        if ParameterClass.SCHEDULER_MODE == "MINRTT":
            first_path_first_flag = self.schedular.schedule(rtt1, rtt2)

        # Consume FC window remaining capacity from the leading path first, then pass the remainder to the following path
        if first_path_first_flag:
            sf1_pkts, n_new1 = self._take_from_buffers(state, cap1, new_data_cap)
            rem_new = (new_data_cap - n_new1) if new_data_cap is not None else None
            sf2_pkts, n_new2 = self._take_from_buffers(state, cap2, rem_new)
        else:
            sf2_pkts, n_new2 = self._take_from_buffers(state, cap2, new_data_cap)
            rem_new = (new_data_cap - n_new2) if new_data_cap is not None else None
            sf1_pkts, n_new1 = self._take_from_buffers(state, cap1, rem_new)
        return sf1_pkts, sf2_pkts, n_new1 + n_new2

    def send_data(self, ue_id):
        """
        Get number of transmittable packets for both subflows,
        decide which to send first using min-RTT scheduling,
        and dequeue retransmission packets first.
        Applies MPQUIC flow control: new-data packets are capped by the remaining
        FC window (fc_send_max_offset - fc_send_offset). Retransmissions are free.
        """
        state = self.ue_states[ue_id]
        cap1 = state['subflow_1'].ue_states[ue_id]['transmittable_packets_num']
        cap2 = state['subflow_2'].ue_states[ue_id]['transmittable_packets_num']

        _cwnd1 = state['subflow_1'].ue_states[ue_id]['cc_algo'].cwnd
        _cwnd2 = state['subflow_2'].ue_states[ue_id]['cc_algo'].cwnd
        _inflight1 = int(_cwnd1) - cap1
        _inflight2 = int(_cwnd2) - cap2
        self.logger.store("MPQUIC", f"UE{ue_id}", "cwnd_inflight_log",
                          f"t={TimeManager.time_index}, cwnd={int(_cwnd1 + _cwnd2)}, inflight={_inflight1 + _inflight2}")

        # FC: New data send limit = MAX_DATA(UE) - already-sent offset
        new_data_cap = max(0, state['fc_send_max_offset'] - state['fc_send_offset'])

        sf1_pkts, sf2_pkts, n_new = self._schedule_packets(
            state, ue_id, cap1, cap2, new_data_cap=new_data_cap)

        # Advance offset only for new data (retransmissions are not counted)
        state['fc_send_offset'] += n_new

        return sf1_pkts, sf2_pkts

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
        sf1_packets, _ = self._take_from_buffers(state, quota_sf1)
        sent_sf1 = 0 if sf1_packets is None else len(sf1_packets)
        state['6g_path_transmittable'] = max(0, quota_sf1 - sent_sf1)
        sf2_packets, _ = self._take_from_buffers(state, cap_sf2)
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

        sf1_packets, sf2_packets, _ = self._schedule_packets(
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
        sf1_packets_to_send, _ = self._take_from_buffers(
            state, quota_sf1)    # cap=quota_sf1
        sf2_packets_to_send, _ = self._take_from_buffers(
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
        packets_to_send = self.ue_states[ue_id]['send_buffer_to_upper_layer'].dequeue()
        if packets_to_send is not None and len(packets_to_send) > 0:
            pkts = np.atleast_2d(packets_to_send)
            send_times = pkts[:, ParameterClass.INDEX_UPF_TRANSMIT_TIMESTAMP].astype(
                np.int64)
            one_way_delays = TimeManager.time_index - send_times
            for delay in one_way_delays:
                self.logger.store(
                    "MPQUIC", f"UE{ue_id}", "MPQUIC-level_one_way_delay", str(int(delay)))

            # FC (UE receiver side): advance receive offset by the amount passed to QUIC layer and update MAX_DATA
            state = self.ue_states[ue_id]
            state['fc_recv_offset'] += len(pkts)
            state['fc_recv_max_offset'] = (state['fc_recv_offset']
                                           + state['fc_recv_window'])

        #self.logger.store("MPQUIC", f"UE{ue_id}", "MPQUIC-level_recv_packets_at_UE", f"time={TimeManager.time_index}: received packets = {packets_to_send}")
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
