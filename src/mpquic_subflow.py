import numpy as np
from pathlib import Path
import math
from param import ParameterClass, TimeManager
from packet import Sent_packets, get_time
from cubic_hystart import CUBIC_HyStart

import os
import datetime
import matplotlib.pyplot as plt
import csv
import pdb


class MPQUIC_SUBFLOW(ParameterClass, TimeManager):
    def __init__(self, logger, mpquic_instance=None):
        self.ue_states = {}
        self.mpquic = mpquic_instance
        np.set_printoptions(threshold=np.inf, linewidth=200)
        self.logger = logger

    def initialize_ue(self, ue_id, subflow_id):
        """
        Initialize the state of each UE.
        """
        initial_seq_num = 0
        mode = ParameterClass.MPQUIC_CC
        if mode == "CUBIC":
            cc_instance = CUBIC_HyStart(
                logger=self.logger, log_dir1="MPQUIC", log_dir2=f"UE{ue_id}/SF{subflow_id}")
        else:
            raise ValueError(f"Unsupported congestion control mode: {mode}")
        self.ue_states[ue_id] = {
            'cc_algo': cc_instance,
            'seq_num': initial_seq_num,  # Sequence number for the next packet to be sent
            'l4_max_received_seq_num': initial_seq_num - 1,  # Maximum packet number received at L4 level
            'app_max_received_seq_num': initial_seq_num - 1,  # Maximum packet number received at the application level
            'server_sent_max_seq_num': 0,  # Maximum packet number sent from the server
            'send_throughput': [],
            'recv_throughput': [],  # Stores instantaneous throughput at each time_index
            'goodput': [],  # Stores instantaneous goodput at each time_index
            'missing_seq_nums': [],  # List of missing sequence numbers detected at receiver
            'retransmission_seq_nums': [],  # List of sequence numbers confirmed for retransmission
            'seq_num_for_ack': 0,  # Separate sequence space for ACK and data packets
            'ack_packet_dict': {},  # Master dictionary for ACK packets
            'ack_counter': 0,  # ACK counter
            'ack_thresh': 2,  # Send an ACK every N packets
            'pto_count': 0,  # Number of PTO events (reset when ACK is received)
            'latest_rtt': 0,  # Most recent RTT
            'smoothed_rtt': ParameterClass.K_INITIAL_RTT,
            'rtt_var': ParameterClass.K_INITIAL_RTT/2,
            'min_rtt': 0,
            'first_rtt_sample': 0,
            'largest_acked_packet': float('inf'),
            'time_of_last_ack_eliciting_packet': 0,
            'loss_time': 0,  # Time when loss is expected to occur upon reordering
            'sent_packets': Sent_packets(),
            'largest_acked_packet_on_sender': 0,  # Largest packet number ACKed at sender
            'loss_detection_timer': float('inf'),
            'ack_delay': 0,  # Stores ACK delay. Reset to 0 each time an ACK packet is sent.
            'subflow_id': subflow_id,  # Identifier used for CSV outputs etc.
            'transmittable_packets_num': self.INIT_CWND,
            'loss_epoch_flag': False,  # True while in loss_epoch. During this, no congestion event is triggered.
            'loss_epoch_end_num': 0,  # Packet number sent immediately after entering the epoch. Ends when this packet is ACKed.
            'counter_total_packet_loss': 0,
            'counter_total_sent_packets': 0,
        }
        for i in range(ParameterClass.NUM_UE):
            # Prevents missing logs for paths where no transmission/reception occurs (depends on logger behavior).
            self.logger.store(
                "MPQUIC", f"UE{ue_id}", f"SF1_server_send_throughput", [0, 0])
            self.logger.store(
                "MPQUIC", f"UE{ue_id}", f"SF2_server_send_throughput", [0, 0])
            self.logger.store(
                "MPQUIC", f"UE{ue_id}", f"SF1_UE_recv_throughput", [0, 0])
            self.logger.store(
                "MPQUIC", f"UE{ue_id}", f"SF2_UE_recv_throughput", [0, 0])

    def send_data(self, ue_id, received_data):
        """
        Executes the transmission process on the sending side (Server for downlink).
        """
        state = self.ue_states[ue_id]

        # Immediately return if no data is passed
        if received_data is None or len(received_data) == 0:
            return

        # Update time_of_last_ack_eliciting_packet (timestamp of last data packet transmission)
        state['time_of_last_ack_eliciting_packet'] = round(
            self.time_index * self.TIME_SLOT_WINDOW, 6)

        # Below: create outgoing packets
        packets_to_send = []

        # Do not modify received_packets; just add seq_num for this layer
        for packet in received_data:
            packet[self.INDEX_OUTER_PACKET_ID] = state['seq_num']
            packet[self.INDEX_UPF_TRANSMIT_TIMESTAMP] = TimeManager.time_index
            packets_to_send.append(packet)
            self.add_sent_packet(ue_id, packet_number=state['seq_num'], in_flight=True, sent_bits=self.MTU_BIT_SIZE,
                                 retransmission=False, inner_packet_number=packet[self.INDEX_PACKET_ID], stream_packet_number=packet[self.INDEX_STREAM_PACKET_ID])
            state['seq_num'] += 1

        # Start timer
        self.set_loss_detection_timer(ue_id)

        # Convert the list into a NumPy array
        packets_to_send = np.array(packets_to_send)

        # Logs transmission throughput.
        send_throughput = len(
            packets_to_send) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_server_send_throughput", [
                          TimeManager.time_index, send_throughput])
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_server_send_packets",
                          f"time_index={TimeManager.time_index}: SERVER sent packets = \n{packets_to_send}")
        state['counter_total_sent_packets'] += len(packets_to_send)

        return packets_to_send

    def receive_data_send_ack(self, ue_id, received_packets):
        """
        Executes the receiving (UE for downlink) and ACK sending process.
        """
        state = self.ue_states[ue_id]

        # When no packets are received
        if received_packets is None or len(received_packets) == 0:
            # If waiting to send an ACK, increase ack_delay
            if state['ack_counter'] != 0:
                state['ack_delay'] = round(
                    state['ack_delay'] + self.TIME_SLOT_WINDOW, 6)  # Increment ack_delay
            return

        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_UE_received_packets", f"time_index={TimeManager.time_index}: UE received packets = \n{received_packets}")

        # Initialize return array
        packets_to_send = np.zeros(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        # Update received sequence numbers
        for packet in received_packets:
            seq_num = packet[self.INDEX_OUTER_PACKET_ID]

            # Case: received packet is in missing list
            if seq_num in state['missing_seq_nums']:
                state['missing_seq_nums'].remove(seq_num)  # Remove from missing list
                if state['missing_seq_nums']:  # If list is not empty
                    state['app_max_received_seq_num'] = min(
                        state['missing_seq_nums']) - 1
                else:  # If missing list is empty, app = l4
                    state['app_max_received_seq_num'] = state['l4_max_received_seq_num']
            else:
                # Check packet continuity at L4 level
                if seq_num == state['l4_max_received_seq_num'] + 1:
                    state['l4_max_received_seq_num'] += 1
                elif seq_num > state['l4_max_received_seq_num'] + 1:
                    # If missing packets exist, add all in-between numbers to the missing list
                    for missing in range(state['l4_max_received_seq_num'] + 1, seq_num):
                        if missing not in state['missing_seq_nums']:
                            state['missing_seq_nums'].append(missing)
                    state['l4_max_received_seq_num'] = seq_num

                # Check packet continuity at the application level
                if seq_num == state['app_max_received_seq_num'] + 1:
                    state['app_max_received_seq_num'] += 1

            state['ack_counter'] += 1
            if state['ack_counter'] == state['ack_thresh']:  # Generate an ACK packet every N packets
                state['ack_counter'] = 0  # Reset ACK counter
                ack_packet = np.zeros(
                    (1, ParameterClass.NUM_INFO_PACKET), dtype=int)

                # Generate ACK packet
                ack_size = ParameterClass.IPV4_HEADER_SIZE + \
                    ParameterClass.UDP_HEADER_SIZE + ParameterClass.ACK_PAYLOAD
                ack_packet[:, self.INDEX_OUTER_PACKET_ID] = state['seq_num_for_ack']
                ack_packet[:, self.INDEX_PAYLOAD_SIZE] = ack_size
                ack_packet[:, self.INDEX_ue_id] = ue_id
                ack_packet[:, self.INDEX_OUTER_ACK_FLAG] = 1

                # Update the ACK master table, mapping packet ID to record
                state['ack_packet_dict'][state['seq_num_for_ack']] = [
                    state['l4_max_received_seq_num']+1, state['missing_seq_nums']]
                # Increment ACK sequence number for next ACK
                state['seq_num_for_ack'] += 1

                # Append ACK packet to return array
                packets_to_send = np.append(
                    packets_to_send, ack_packet, axis=0)

        # Calculate throughput
        recv_throughput = len(
            received_packets) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_UE_recv_throughput",  [TimeManager.time_index, recv_throughput])

        if packets_to_send.size == 0:  # Received packets but no ACK generated
            state['ack_delay'] = round(
                state['ack_delay'] + ParameterClass.TIME_SLOT_WINDOW, 6)
            return
        else:
            state['ack_delay'] = 0

        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_ue_send_ack", f"time_index={TimeManager.time_index}: UE sent ACK = \n{packets_to_send}")
        return packets_to_send

    def receive_ack(self, ue_id, ack_packets):
        """
        Executes ACK reception process on the server side.
        The argument ack_packets is assumed to contain only ACK packets.
        """
        state = self.ue_states[ue_id]
        if ack_packets.size > 0:
            # Process ACK packets one by one
            for ack_packet in ack_packets:
                if ack_packet[self.INDEX_OUTER_PACKET_ID] in state['ack_packet_dict']:
                    # Retrieve corresponding record from the ACK master table
                    ack_entry = state['ack_packet_dict'].pop(
                        ack_packet[self.INDEX_OUTER_PACKET_ID])
                    next_seq_num, missing_packets = ack_entry

                    # Update the largest ACKed packet number
                    state['largest_acked_packet_on_sender'] = max(
                        state['largest_acked_packet_on_sender'], next_seq_num - 1)

                    # List newly ACKed packet numbers
                    newly_acked_packets = [packet_number for packet_number in state['sent_packets']
                                           if packet_number < next_seq_num and packet_number not in missing_packets]

                    # Skip if no new ACKed packets
                    if not newly_acked_packets:
                        continue

                    # Check end of loss_epoch
                    if state['loss_epoch_flag'] and any(pn > state['loss_epoch_end_num'] for pn in newly_acked_packets):
                        self.end_loss_epoch(ue_id)

                    # Update RTT if the largest ACKed packet is newly acknowledged
                    if state['largest_acked_packet_on_sender'] in newly_acked_packets:
                        target_packet_number = state['largest_acked_packet_on_sender']
                        time_sent = state['sent_packets'][target_packet_number].time_sent
                        self.update_rtt(ue_id, time_sent)

                    # Remove newly ACKed packets from the sent packet list
                    for packet_number in newly_acked_packets:
                        del state['sent_packets'][packet_number]

                    lost_packets = self.detect_and_remove_lost_packets(ue_id)
                    if lost_packets:
                        self.on_packets_lost(ue_id, lost_packets)

                    # App-limited check (conforms to Linux TCP Cubic behavior)
                    if ParameterClass.APP_LIMITED_OPTION == True:
                        inflight_size = len(
                            [p for p in state['sent_packets'].values() if p.in_flight])
                        app_limited = (
                            # app_limited=True when no transmittable packets in MPQUIC layer
                            # and the estimated available bandwidth is not fully utilized
                            self.mpquic.ue_states[ue_id]['send_buffer_to_lower_layer'].length == 0 and
                            self.mpquic.ue_states[ue_id]['retranmission_buffer_to_lower_layer'].length == 0 and
                            inflight_size < state['cc_algo'].cwnd
                        )
                        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_app_limited_status_log",
                                          f"time={self.time_index}: {self.mpquic.ue_states[ue_id]['send_buffer_to_lower_layer'].length == 0}, {self.mpquic.ue_states[ue_id]['retranmission_buffer_to_lower_layer'].length == 0}, {inflight_size < state['cc_algo'].cwnd}, app_limited={app_limited}")
                    else:
                        app_limited = False

                    # Perform CUBIC control upon ACK reception
                    newly_acked_num = len(newly_acked_packets)
                    state['cc_algo'].on_ack(segments_num_acked=newly_acked_num, rtt=state['latest_rtt'], smoothed_rtt=state['smoothed_rtt'],
                                            seq_num_acked=state['largest_acked_packet_on_sender'], next_seq_num=state['seq_num']+1, app_limited=app_limited)

                    state['pto_count'] = 0
                    self.set_loss_detection_timer(ue_id)

        # Update the number of transmittable packets for this subflow
        self.update_transmittable_packets_num(ue_id)

        # Log cwnd size and RTT values at the end of loop
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_cwnd_size_log", state['cc_algo'].cwnd)
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_smoothed_RTT", state['smoothed_rtt'])
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_latest_RTT", state['latest_rtt'])

    def add_sent_packet(self, ue_id, packet_number, in_flight, sent_bits, retransmission, inner_packet_number, stream_packet_number):
        """
        packet_number refers to the packet number space at the subflow layer (outer_id).
        The inner_packet_number is kept for retransmission handling.
        """
        if ue_id not in self.ue_states:
            raise ValueError(f"UE ID '{ue_id}' has not been initialized.")
        sent_packets = self.ue_states[ue_id]['sent_packets']
        packet = sent_packets[packet_number]
        packet.in_flight = in_flight
        packet.sent_bits = sent_bits
        packet.retransmission = retransmission
        packet.inner_packet_number = inner_packet_number
        packet.stream_packet_number = stream_packet_number

    def get_pto_time(self, ue_id):
        state = self.ue_states[ue_id]
        duration = (state['smoothed_rtt'] + max(4 * state['rtt_var'],
                    ParameterClass.K_GRANULARITY)) * (2 ** state['pto_count'])

        has_in_flight_packets = any(
            packet.in_flight for packet in state['sent_packets'].values())
        if has_in_flight_packets == 0:  # This probably never happens...
            return TimeManager.time_index * ParameterClass.TIME_SLOT_WINDOW + duration

        pto_timeout = math.inf
        duration += self.MAX_ACK_DELAY * (2 ** state['pto_count'])
        t = state['time_of_last_ack_eliciting_packet'] + duration
        if t < pto_timeout:
            pto_timeout = t
        return pto_timeout

    def set_loss_detection_timer(self, ue_id):
        state = self.ue_states[ue_id]
        earliest_loss_time = state['loss_time']

        # If packet reordering has been observed, start the normal (loss) timer
        if earliest_loss_time != 0:
            state['loss_detection_timer'] = earliest_loss_time
            return

        # If inflight is 0, stop the timer
        has_in_flight_packets = any(
            packet.in_flight for packet in state['sent_packets'].values())
        if has_in_flight_packets == 0:
            state['loss_detection_timer'] = float('inf')
            return

        # If inflight exists and no reordering, start the PTO timer
        timeout = self.get_pto_time(ue_id)
        state['loss_detection_timer'] = timeout

    def update_rtt(self, ue_id, time_sent):
        state = self.ue_states[ue_id]
        # When newly_acked arrives, update latest_rtt first
        state['latest_rtt'] = round(
            self.time_index * self.TIME_SLOT_WINDOW - time_sent, 6)
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_latest_rtt_change_log",
                          f"time={self.time_index}\t{state['latest_rtt']}")

        # Then update various RTT values
        if state['first_rtt_sample'] == 0:
            state['min_rtt'] = state['latest_rtt']
            state['smoothed_rtt'] = state['latest_rtt']
            state['rtt_var'] = state['latest_rtt'] / 2
            state['first_rtt_sample'] = round(
                self.time_index * self.TIME_SLOT_WINDOW, 6)
            return

        state['min_rtt'] = min(state['min_rtt'], state['latest_rtt'])
        # Limit ack_delay by max_ack_delay
        state['ack_delay'] = min(state['ack_delay'], self.MAX_ACK_DELAY)

        adjusted_rtt = state['latest_rtt']
        if state['latest_rtt'] >= state['min_rtt'] + state['ack_delay']:
            adjusted_rtt = state['latest_rtt'] - state['ack_delay']

        state['rtt_var'] = 3/4 * state['rtt_var'] + 1 / \
            4 * abs(state['smoothed_rtt'] - adjusted_rtt)
        state['smoothed_rtt'] = 7/8 * \
            state['smoothed_rtt'] + 1/8 * adjusted_rtt

    def detect_and_remove_lost_packets(self, ue_id):
        """
        Return the list of packet numbers that were lost.
         - Only performs "loss detection" and "loss_time update".
        """
        state = self.ue_states[ue_id]

        state['loss_time'] = 0
        lost_packets = []
        loss_delay = self.K_TIME_THRESH * \
            max(state['latest_rtt'], state['smoothed_rtt'])

        loss_delay = max(loss_delay, self.K_GRANULARITY)

        lost_send_time = round(
            self.time_index * self.TIME_SLOT_WINDOW - loss_delay, 6)

        # for packet_number, packet in state['sent_packets'].items():
        for packet_number in list(state['sent_packets'].keys()):
            packet = state['sent_packets'][packet_number]
            if packet.in_flight:  # Process only those with in_flight == True
                if packet_number > state['largest_acked_packet_on_sender']:
                    continue
                if packet.time_sent <= lost_send_time:
                    lost_packets.append(packet_number)
                    self.logger.store(
                        "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_packet_loss_cause", f"time={self.time_index}:\tpacket loss (more than 9/8RTT delay)")
                elif state['largest_acked_packet_on_sender'] >= packet_number + self.K_PACKET_THRESH:
                    lost_packets.append(packet_number)
                    self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_packet_loss_cause",
                                      f"time={self.time_index}:\tpacket loss (reordering more than 3 packets)")
                else:
                    if state['loss_time'] == 0:
                        state['loss_time'] = state['sent_packets'][packet_number].time_sent + loss_delay
                    else:
                        state['loss_time'] = min(
                            state['loss_time'], state['sent_packets'][packet_number].time_sent + loss_delay)

        # Return the list of lost packet numbers
        return lost_packets

    def check_timer_subflow(self, ue_id):
        """
        Note: this behavior differs from standard QUIC.
        Check the timer, and if a timeout is detected, execute on_loss_detection_timeout.
        No return value.
        """
        state = self.ue_states[ue_id]
        if state['loss_detection_timer'] != float('inf'):
            current_time = round(self.time_index * self.TIME_SLOT_WINDOW, 6)
            if state['loss_detection_timer'] < current_time:
                self.on_loss_detection_timeout(ue_id)

    def on_loss_detection_timeout(self, ue_id):
        """
        When a PTO occurs, increase 'transmittable' by +2 so that probing packets can be sent.
        """
        state = self.ue_states[ue_id]

        earliest_loss_time = state['loss_time']
        if earliest_loss_time != 0:  # If the non-PTO timer expired
            lost_packets = self.detect_and_remove_lost_packets(ue_id)
            if lost_packets:  # Trigger a congestion event
                self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_packet_loss_cause",
                                  f"time={self.time_index}:\tpacket loss (Timer expiration)")
                self.on_packets_lost(ue_id, lost_packets)
            self.set_loss_detection_timer(ue_id)
            return

        # If the PTO timer expired ⇒ send 1 or 2 packets
        state['pto_count'] += 1
        self.set_loss_detection_timer(ue_id)

        # Update time_of_last_ack_eliciting_packet
        state['time_of_last_ack_eliciting_packet'] = round(
            self.time_index * self.TIME_SLOT_WINDOW, 6)

        for _ in range(self.PTO_TRANSMIT_NUM):
            state['transmittable_packets_num'] += 1
        self.logger.store(
            "MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_transmittable_packets_num", f"time_index={TimeManager.time_index}: transmittable_packets_num (PTO) = {state['transmittable_packets_num']}")

    def on_packets_lost(self, ue_id, lost_packets):
        state = self.ue_states[ue_id]
        sent_time_of_last_loss = 0
        lost_packets_for_del_from_inflight = []

        # Update retransmission list (do this before deleting from the dictionary)
        # - If at least 1 RTT has elapsed since the previous retransmission, add the packet to the retransmission list
        for lost_packet in lost_packets:
            time_thresh = round(
                self.time_index * self.TIME_SLOT_WINDOW - state['latest_rtt'], 6)
            if state['sent_packets'][lost_packet].retransmission == False:
                self.make_and_retransmit_packet(ue_id, lost_packet)
                lost_packets_for_del_from_inflight.append(lost_packet)
            elif state['sent_packets'][lost_packet].retransmission == True and state['sent_packets'][lost_packet].time_sent < time_thresh:
                self.make_and_retransmit_packet(ue_id, lost_packet)
                lost_packets_for_del_from_inflight.append(lost_packet)

        # Remove lost packets from in-flight
        for lost_seq_num in lost_packets_for_del_from_inflight:
            sent_time_of_last_loss = max(
                sent_time_of_last_loss, state['sent_packets'][lost_seq_num].time_sent)
            del state['sent_packets'][lost_seq_num]

        # Trigger a congestion event
        if sent_time_of_last_loss != 0 and state['loss_epoch_flag'] == False:
            state['cc_algo'].on_congestion_event()
            self.start_loss_epoch(ue_id)

        # If you do not block retransmission for 1 RTT, here is the update process for the retransmission list
        """
        state['retransmission_seq_nums'].extend(lost_packets) # Add packets to the retransmission list
        """

        # For log.
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_lost_packets",
                          f"time_index={TimeManager.time_index}: Lost packet={lost_packets}")
        state['counter_total_packet_loss'] += len(
            lost_packets_for_del_from_inflight)

    def start_loss_epoch(self, ue_id):
        state = self.ue_states[ue_id]
        state['loss_epoch_flag'] = True
        state['loss_epoch_end_num'] = state['seq_num']
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_packet_loss_cause",
                          f"time={self.time_index}:\tloss_epoch starts.")

    def end_loss_epoch(self, ue_id):
        state = self.ue_states[ue_id]
        state['loss_epoch_flag'] = False
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_packet_loss_cause",
                          f"time={self.time_index}:\tloss_epoch ends.")

    def log_for_debug(self, ue_id):
        state = self.ue_states[ue_id]
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_ack_dict",
                          f"time_index={TimeManager.time_index}: ack_dict = {state['ack_packet_dict']}")
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_sent_packets",
                          f"time_index={TimeManager.time_index}: sent_packets = {state['sent_packets']}")

    def make_and_retransmit_packet(self, ue_id, lost_packet_id):
        """
        Enqueue the lost packet data into the MPQUIC layer's retransmission queue.
            - lost_packet_id is the outer_id of the lost packet
            - Using outer_id, fetch inner_id from sent_packets, reconstruct the lost packet,
              and push it into the MPQUIC layer's retransmission queue.
        """
        state = self.ue_states[ue_id]

        # Create packet
        packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
        packet[:, self.INDEX_PACKET_ID] = state['sent_packets'][lost_packet_id].inner_packet_number
        packet[:, self.INDEX_PAYLOAD_SIZE] = self.MTU_BIT_SIZE
        packet[:, self.INDEX_ue_id] = ue_id
        packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = TimeManager.time_index
        packet[:, self.INDEX_OUTER_PACKET_ID] = state['seq_num']
        packet[:, self.INDEX_STREAM_PACKET_ID] = state['sent_packets'][lost_packet_id].stream_packet_number

        # Store into the sent packet data table
        self.add_sent_packet(ue_id, packet_number=state['seq_num'], in_flight=True, sent_bits=self.MTU_BIT_SIZE,
                             retransmission=True, inner_packet_number=packet[0][self.INDEX_PACKET_ID], stream_packet_number=packet[0][self.INDEX_STREAM_PACKET_ID])

        # Enqueue into the MPQUIC layer's retransmission queue
        self.mpquic.ue_states[ue_id]['retranmission_buffer_to_lower_layer'].enqueue(
            packet)

        # Update the number of transmittable packets referenced by the MPQUIC layer
        self.update_transmittable_packets_num(ue_id)

        # Log
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_retransmission_list",
                          f"time_index={TimeManager.time_index}: packets for retransmission = {packet}")

    def update_transmittable_packets_num(self, ue_id):
        state = self.ue_states[ue_id]
        cwnd = state['cc_algo'].cwnd
        inflight_size = sum(
            1 for p in state['sent_packets'].values() if p.in_flight)
        state['transmittable_packets_num'] = max(0, int(cwnd - inflight_size))
        self.logger.store("MPQUIC", f"UE{ue_id}", f"SF{state['subflow_id']}_transmittable_packets_num",
                          f"time_index={TimeManager.time_index}: transmittable_packets_num = {state['transmittable_packets_num']}")

    def onetime_logger(self, ue_id):
        state = self.ue_states[ue_id]
        self.logger.store("MPQUIC", f"UE{ue_id}/SF{state['subflow_id']}", f"packet_loss_rate",
                          f"Total Sent Packets = \t{state['counter_total_sent_packets']}")
        self.logger.store("MPQUIC", f"UE{ue_id}/SF{state['subflow_id']}", f"packet_loss_rate",
                          f"Total Packet Losses = \t{state['counter_total_packet_loss']}")
        self.logger.store("MPQUIC", f"UE{ue_id}/SF{state['subflow_id']}", "packet_loss_rate",
                          f"PLR = \t{state['counter_total_packet_loss'] / state['counter_total_sent_packets'] if state['counter_total_sent_packets'] else 'N/A (no packets sent)'}")
