import queue
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
import pdb  # pdb.set_trace()
import shutil

from bbrv1 import BBRv1

class QUIC(ParameterClass, TimeManager):
    def __init__(self, logger):
        self.ue_states = {}  # Dictionary to manage per-UE states
        np.set_printoptions(threshold=np.inf, linewidth=200)
        self.logger = logger

    def initialize_ue(self, ue_id):
        """
        Initialize per-UE state.
        """
        initial_seq_num = 0
        mode = ParameterClass.QUIC_CC
        if mode == "CUBIC":
            cc_instance = CUBIC_HyStart(
                logger=self.logger, log_dir1="QUIC", log_dir2=f"UE{ue_id}")
        elif mode == "BBR":
            cc_instance = BBRv1(logger=self.logger,
                              log_dir1="QUIC", log_dir2=f"UE{ue_id}")
        else:
            raise ValueError(f"Unsupported congestion control mode: {mode}")
        self.ue_states[ue_id] = {
            'cc_algo': cc_instance,
            'seq_num': initial_seq_num,  # Next sequence number to send
            # Max packet number received at L4
            'l4_max_received_seq_num': initial_seq_num - 1,
            # Max packet number received contiguously at the app layer
            'app_max_received_seq_num': initial_seq_num - 1,
            'server_sent_max_seq_num': 0,  # Max packet number sent on the server side
            'missing_seq_nums': [],  # List of missing sequence numbers observed at the receiver
            'retransmission_seq_nums': [],  # Sequence numbers confirmed for retransmission
            'seq_num_for_ack': 0,  # Separate packet number space for ACK vs data
            'ack_packet_dict': {},  # Master dict for ACK packets
            'ack_counter': 0,  # Counter for ACK generation
            'ack_thresh': 2,  # Send one ACK every N packets
            # Number of PTOs performed (reset when an ACK is received)
            'pto_count': 0,
            'latest_rtt': 0,  # Most recent RTT
            'smoothed_rtt': ParameterClass.K_INITIAL_RTT,
            'rtt_var': ParameterClass.K_INITIAL_RTT/2,
            'min_rtt': 0,
            'first_rtt_sample': 0,
            'largest_acked_packet': float('inf'),
            'time_of_last_ack_eliciting_packet': 0,
            'loss_time': 0,  # Time when loss is likely if reordering is observed
            'sent_packets': Sent_packets(),
            'largest_acked_packet_on_sender': 0,  # Sender-side largest ACKed packet number
            'loss_detection_timer': float('inf'),
            'ack_delay': 0,  # Stores ACK delay; reset to 0 whenever an ACK is sent
            # True while in a loss epoch; do not trigger congestion events during this
            'loss_epoch_flag': False,
            # Packet number sent right after epoch start; epoch ends when this gets ACKed
            'loss_epoch_end_num': 0,
            'counter_total_packet_loss': 0,
            'counter_total_sent_packets': 0,

            ### BBR attributes added ###
            'delivered_packets': 0,
            'packets_in_buffer': True,
            ### end ###

            # Tunnel RTT (dynamically updated via FB from TECC)
            'tunnel_rtt': 0.0,

            # Transmission start time per UE
            'start_time': 0,

            # FCT mode state
            'fct_flow_size': ParameterClass.FCT_TOTAL_PACKETS if ParameterClass.FCT_MODE else None,
            'fct_send_complete': False,
            'fct_send_completion_time': None,
            'fct_recv_complete': False,
            'fct_completion_time': None,

            # CAMF: Recommended transmission rate [pkt/ms]. 0.0 = CAMF disabled.
            # Set from main.py.
            'camf_target_pps': 0.0,
            # Fractional carry of leaky bucket.
            'camf_credit':     0.0,
            # Currently applied FB ID.
            'camf_fb_id':      0,
        }

    def sender_send(self, ue_id, mpquic_cwnd=None):
        """
        Execute sending on the data sender (server for DL).
        """
        state = self.ue_states[ue_id]

        # Calculate how many packets can be sent
        if mpquic_cwnd == None:
            cwnd = state['cc_algo'].cwnd                            ### Note: CCA sets cwnd here!
        else:
            cwnd = (mpquic_cwnd + state['cc_algo'].cwnd) / 2
        inflight_size = sum(
            1 for packet in state['sent_packets'].values() if packet.in_flight)

        self.logger.store("QUIC", f"UE{ue_id}", "cwnd_inflight_log",
                          f"t={TimeManager.time_index}, cwnd={int(cwnd)}, inflight={inflight_size}")

        ### BBR limits the pacing window ###
        if ParameterClass.QUIC_CC == "BBR":
            pacing_rate = state['cc_algo'].pacing_rate
            pacing_window = state['cc_algo'].pacing_manager.advance_pacing_manager(pacing_rate)
            available_packets = max(0, int(cwnd - inflight_size))
            available_packets = min(int(pacing_window), int(cwnd - inflight_size))
        ### end ###
        else:
            available_packets = max(0, int(cwnd - inflight_size))

        ### CAMF: direct rate synchronization (accumulator) ############################
        # When camf_target_pps [pkt/ms] is set, synchronize the server transmission rate directly.
        # Each time_slot adds target_pps to credit; the integer part is the send count for this slot.
        # The fractional part is carried over to the next slot (leaky bucket).
        if ParameterClass.CAMF_OPTION and state['camf_target_pps'] > 0.0:
            state['camf_credit'] += state['camf_target_pps']
            to_send = int(state['camf_credit'])
            state['camf_credit'] -= to_send
            cwnd_headroom = max(0, int(cwnd - inflight_size))
            if ParameterClass.CAMF_NOCAP_OPTION:
                available_packets = to_send
            else:
                available_packets = min(to_send, cwnd_headroom)
            self.logger.store("DEBUG", f"UE{ue_id}", "debug", f"time_index={TimeManager.time_index}: smaller={'equal' if to_send == cwnd_headroom else ('to_send' if to_send < cwnd_headroom else 'cwnd_headroom')}")
        else:
            # Reset credit when target becomes 0 (to prevent burst on re-activation)
            state['camf_credit'] = 0.0

        ### end CAMF ################################################################

        #self.logger.store("QUIC", f"UE{ue_id}", f"available_packets_num_log", f"time_index={TimeManager.time_index}: available_packets = {available_packets}")

        # If nothing can be sent, return immediately
        if available_packets <= 0:
            return
        
        ### For BBR: App limitation condition not possbile in QUIC file ###
        ### since the packets are generated here and not buffered ###
        if ParameterClass.APP_LIMITED_OPTION == True:
            state['packets_in_buffer'] = True
        ### end ### 

        # Update time_of_last_ack_eliciting_packet (last time a data packet was sent)
        state['time_of_last_ack_eliciting_packet'] = round(
            self.time_index * self.TIME_SLOT_WINDOW, 6)

        # Initialize return array
        packets_to_send = np.zeros(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        # Build packets to send
        # Priority: retransmissions
        for seq_num in sorted(state['retransmission_seq_nums']):
            if available_packets <= 0:
                break
            packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
            packet[:, self.INDEX_PACKET_ID] = seq_num
            packet[:, self.INDEX_PAYLOAD_SIZE] = self.MTU_BIT_SIZE
            packet[:, self.INDEX_ue_id] = ue_id
            packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = TimeManager.time_index
            packet[:, self.INDEX_CAMF_FB_ID] = state['camf_fb_id']
            packets_to_send = np.append(packets_to_send, packet, axis=0)
            available_packets -= 1
            self.add_sent_packet(ue_id, packet_number=seq_num, in_flight=True,
                                 sent_bits=self.MTU_BIT_SIZE, retransmission=True)
            state['retransmission_seq_nums'].remove(seq_num)

        # Send new packets if capacity remains
        while available_packets > 0:
            # FCT mode: stop generating new packets once the specified count is exceeded
            if ParameterClass.FCT_MODE and state['seq_num'] >= state['fct_flow_size']:
                state['fct_send_complete'] = True
                if state['fct_send_completion_time'] is None:
                    state['fct_send_completion_time'] = TimeManager.time_index
                break
            packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
            packet[:, self.INDEX_PACKET_ID] = state['seq_num']
            packet[:, self.INDEX_PAYLOAD_SIZE] = self.MTU_BIT_SIZE
            packet[:, self.INDEX_ue_id] = ue_id
            packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = TimeManager.time_index
            packet[:, self.INDEX_CAMF_FB_ID] = state['camf_fb_id']
            packets_to_send = np.append(packets_to_send, packet, axis=0)
            self.add_sent_packet(
                ue_id, packet_number=state['seq_num'], in_flight=True, sent_bits=self.MTU_BIT_SIZE, retransmission=False)
            state['seq_num'] += 1
            available_packets -= 1

        # Start timer
        self.set_loss_detection_timer(ue_id)
        # Logs transmission throughput and count
        state['counter_total_sent_packets'] += len(packets_to_send)
        send_throughput = len(
            packets_to_send) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store("QUIC", f"UE{ue_id}",
                          f"send_throughput", [TimeManager.time_index, send_throughput])
        self.logger.store("QUIC", f"UE{ue_id}", f"server_send_packet_log",
                          f"time_index={TimeManager.time_index}: SERVER sent packets = \n{packets_to_send}")

        return packets_to_send

    def sender_send_by_tecc(self, ue_id, mpquic_cwnd=None):
        state = self.ue_states[ue_id]

        # Calculate how many packets can be sent
        if mpquic_cwnd:
            available_packets = int(mpquic_cwnd)
        else:
            available_packets = 0
            
        # If nothing can be sent, return immediately
        if available_packets <= 0:
            return
        
        ### For BBR: App limitation condition not possbile in QUIC file ###
        ### since the packets are generated here and not buffered ###
        if ParameterClass.APP_LIMITED_OPTION == True:
            state['packets_in_buffer'] = True
        ### end ### 

        # Update time_of_last_ack_eliciting_packet (last time a data packet was sent)
        state['time_of_last_ack_eliciting_packet'] = round(
            self.time_index * self.TIME_SLOT_WINDOW, 6)

        # Initialize return array
        packets_to_send = np.zeros(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        # Build packets to send
        # Priority: retransmissions
        for seq_num in sorted(state['retransmission_seq_nums']):
            if available_packets <= 0:
                break
            packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
            packet[:, self.INDEX_PACKET_ID] = seq_num
            packet[:, self.INDEX_PAYLOAD_SIZE] = self.MTU_BIT_SIZE
            packet[:, self.INDEX_ue_id] = ue_id
            packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = TimeManager.time_index
            packet[:, self.INDEX_CAMF_FB_ID] = state['camf_fb_id']
            packets_to_send = np.append(packets_to_send, packet, axis=0)
            available_packets -= 1
            self.add_sent_packet(ue_id, packet_number=seq_num, in_flight=True,
                                 sent_bits=self.MTU_BIT_SIZE, retransmission=True)
            state['retransmission_seq_nums'].remove(seq_num)

        # Send new packets if capacity remains
        while available_packets > 0:
            # # FCT mode: stop generating new packets once the specified count is exceeded
            if ParameterClass.FCT_MODE and state['seq_num'] >= state['fct_flow_size']:
                state['fct_send_complete'] = True
                if state['fct_send_completion_time'] is None:
                    state['fct_send_completion_time'] = TimeManager.time_index
                break
            packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
            packet[:, self.INDEX_PACKET_ID] = state['seq_num']
            packet[:, self.INDEX_PAYLOAD_SIZE] = self.MTU_BIT_SIZE
            packet[:, self.INDEX_ue_id] = ue_id
            packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = TimeManager.time_index
            packet[:, self.INDEX_CAMF_FB_ID] = state['camf_fb_id']
            packets_to_send = np.append(packets_to_send, packet, axis=0)
            self.add_sent_packet(
                ue_id, packet_number=state['seq_num'], in_flight=True, sent_bits=self.MTU_BIT_SIZE, retransmission=False)
            state['seq_num'] += 1
            available_packets -= 1

        # Start timer
        self.set_loss_detection_timer(ue_id)
        # Logs transmission throughput and count
        state['counter_total_sent_packets'] += len(packets_to_send)
        send_throughput = len(
            packets_to_send) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store("QUIC", f"UE{ue_id}",
                          f"send_throughput", [TimeManager.time_index, send_throughput])
        self.logger.store("QUIC", f"UE{ue_id}", f"server_send_packet_log",
                          f"time_index={TimeManager.time_index}: SERVER sent packets = \n{packets_to_send}")
        return packets_to_send

    def receiver(self, ue_id, received_packets):
        """
        Execute receive/send processing on the data receiver (UE for DL).
        """
        state = self.ue_states[ue_id]

        # If there are no received packets
        if received_packets is None or len(received_packets) == 0:
            # If waiting to send an ACK, increase ack_delay
            if state['ack_counter'] != 0:
                state['ack_delay'] = round(
                    state['ack_delay'] + self.TIME_SLOT_WINDOW, 6)  # Increase ack_delay
                # max_ack_delay: force-send ACK if delay exceeds threshold
                if state['ack_delay'] >= ParameterClass.MAX_ACK_DELAY:
                    state['ack_counter'] = 0
                    ack_packet = np.zeros(
                        (1, ParameterClass.NUM_INFO_PACKET), dtype=int)
                    ack_size = ParameterClass.IPV4_HEADER_SIZE + \
                        ParameterClass.UDP_HEADER_SIZE + ParameterClass.ACK_PAYLOAD
                    ack_packet[:, self.INDEX_PACKET_ID] = state['seq_num_for_ack']
                    ack_packet[:, self.INDEX_PAYLOAD_SIZE] = ack_size
                    ack_packet[:, self.INDEX_ue_id] = ue_id
                    state['ack_packet_dict'][state['seq_num_for_ack']] = [
                        state['l4_max_received_seq_num']+1, state['missing_seq_nums']]
                    state['seq_num_for_ack'] += 1
                    state['ack_delay'] = 0
                    self.logger.store("QUIC", f"UE{ue_id}", f"ue_send_ack_log",
                                      f"time_index={TimeManager.time_index}:  UE sent ACK (max_ack_delay) = \n{ack_packet}")
                    return ack_packet
            return

        self.logger.store("QUIC", f"UE{ue_id}", f"ue_received_packet_log",
                          f"time_index={TimeManager.time_index}: UE received packets = \n{received_packets}")

        # Initialize return array
        packets_to_send = np.zeros(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        # Temporary holder for app_max_received_seq_num (for goodput calc)
        temp_app_max_received_seq_num = state['app_max_received_seq_num']

        # Update received sequence numbers
        for packet in received_packets:
            seq_num = packet[self.INDEX_PACKET_ID]

            # If the received packet is listed as missing
            if seq_num in state['missing_seq_nums']:
                state['missing_seq_nums'].remove(
                    seq_num)  # Remove from missing list
                if state['missing_seq_nums']:  # Still missing something
                    state['app_max_received_seq_num'] = min(
                        state['missing_seq_nums']) - 1
                else:  # No missing; app == l4
                    state['app_max_received_seq_num'] = state['l4_max_received_seq_num']
            else:
                # L4-level contiguity check
                if seq_num == state['l4_max_received_seq_num'] + 1:
                    state['l4_max_received_seq_num'] += 1
                elif seq_num > state['l4_max_received_seq_num'] + 1:
                    # Gap detected; add all in-between numbers to missing list
                    for missing in range(state['l4_max_received_seq_num'] + 1, seq_num):
                        if missing not in state['missing_seq_nums']:
                            state['missing_seq_nums'].append(missing)
                    state['l4_max_received_seq_num'] = seq_num

                # App-level contiguity check
                if seq_num == state['app_max_received_seq_num'] + 1:
                    state['app_max_received_seq_num'] += 1

            state['ack_counter'] += 1
            if state['ack_counter'] == state['ack_thresh']:  # Create an ACK every N packets
                state['ack_counter'] = 0  # Reset ACK counter
                ack_packet = np.zeros(
                    (1, ParameterClass.NUM_INFO_PACKET), dtype=int)

                # Build ACK packet
                ack_size = ParameterClass.IPV4_HEADER_SIZE + \
                    ParameterClass.UDP_HEADER_SIZE + ParameterClass.ACK_PAYLOAD
                ack_packet[:, self.INDEX_PACKET_ID] = state['seq_num_for_ack']
                ack_packet[:, self.INDEX_PAYLOAD_SIZE] = ack_size
                ack_packet[:, self.INDEX_ue_id] = ue_id

                # Update master ACK table (associate by packet ID)
                state['ack_packet_dict'][state['seq_num_for_ack']] = [
                    state['l4_max_received_seq_num']+1, state['missing_seq_nums']]
                # Increment for the next ACK
                state['seq_num_for_ack'] += 1

                # Append to return array
                packets_to_send = np.append(
                    packets_to_send, ack_packet, axis=0)

        # Throughput calculation
        recv_throughput = len(
            received_packets) * ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store("QUIC", f"UE{ue_id}",
                          f"recv_throughput", [TimeManager.time_index, recv_throughput])

        # Goodput calculation
        goodput = (state['app_max_received_seq_num'] - temp_app_max_received_seq_num) * \
            ParameterClass.MTU_SIZE * 8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store("QUIC", f"UE{ue_id}", f"goodput", [
                          TimeManager.time_index, goodput])

        # FCT mode: check that all packets have been received
        if ParameterClass.FCT_MODE and not state['fct_recv_complete']:
            if (state['app_max_received_seq_num'] >= state['fct_flow_size'] - 1
                    and len(state['missing_seq_nums']) == 0):
                state['fct_recv_complete'] = True
                state['fct_completion_time'] = TimeManager.time_index

        if packets_to_send.size == 0:  # Packets were received but no ACK was generated
            state['ack_delay'] = round(
                state['ack_delay'] + ParameterClass.TIME_SLOT_WINDOW, 6)
            # max_ack_delay: force-send ACK if delay exceeds threshold
            if state['ack_delay'] >= ParameterClass.MAX_ACK_DELAY:
                state['ack_counter'] = 0
                ack_packet = np.zeros(
                    (1, ParameterClass.NUM_INFO_PACKET), dtype=int)
                ack_size = ParameterClass.IPV4_HEADER_SIZE + \
                    ParameterClass.UDP_HEADER_SIZE + ParameterClass.ACK_PAYLOAD
                ack_packet[:, self.INDEX_PACKET_ID] = state['seq_num_for_ack']
                ack_packet[:, self.INDEX_PAYLOAD_SIZE] = ack_size
                ack_packet[:, self.INDEX_ue_id] = ue_id
                state['ack_packet_dict'][state['seq_num_for_ack']] = [
                    state['l4_max_received_seq_num']+1, state['missing_seq_nums']]
                state['seq_num_for_ack'] += 1
                state['ack_delay'] = 0
                packets_to_send = np.append(
                    packets_to_send, ack_packet, axis=0)
                self.logger.store("QUIC", f"UE{ue_id}", f"ue_send_ack_log",
                                  f"time_index={TimeManager.time_index}:  UE sent ACK (max_ack_delay) = \n{packets_to_send}")
                return packets_to_send
            return  # Without this, a single returned ACK could be overwritten by an empty packets_to_send
        else:
            state['ack_delay'] = 0

        self.logger.store("QUIC", f"UE{ue_id}", f"ue_send_ack_log",
                          f"time_index={TimeManager.time_index}:  UE sent ACK = \n{packets_to_send}")
        return packets_to_send

    def sender_recv(self, ue_id, ack_packets):
        """
        Execute ACK processing on the server side.
        The argument ack_packets is assumed to consist solely of ACK packets.
        """

        ### BBR cumulative ACK ###
        cumulative_newly_acked_packets = []
        cumulative_lost_packets = 0
        app_limited = False
        ### end ###

        state = self.ue_states[ue_id]

        ### BBR: Snapshot inflight before processing ACKs ###
        if ParameterClass.QUIC_CC == "BBR":
            state['cc_algo'].previous_inflight = state['cc_algo'].inflight_packet_data.inflight()
        ### end ###

        if ack_packets.size > 0:
            # Handle one ACK packet at a time
            for ack_packet in ack_packets:
                if ack_packet[self.INDEX_PACKET_ID] in state['ack_packet_dict']:
                    # Look up the record for this ACK in the master table and unpack (next_seq_num, missing_packets)
                    ack_entry = state['ack_packet_dict'].pop(
                        ack_packet[self.INDEX_PACKET_ID])
                    next_seq_num, missing_packets = ack_entry
                    self.logger.store("QUIC", f"UE{ue_id}", f"server_received_ack_log",
                                      f"time_index={TimeManager.time_index}: SERVER received ACK = {ack_packet[self.INDEX_PACKET_ID]}, {next_seq_num}, {missing_packets}")

                    # Update largest ACKed packet number
                    state['largest_acked_packet_on_sender'] = max(
                        state['largest_acked_packet_on_sender'], next_seq_num - 1)

                    # List newly ACKed packets
                    newly_acked_packets = [packet_number for packet_number in state['sent_packets']
                                           if packet_number < next_seq_num and packet_number not in missing_packets]

                    # If none newly ACKed, continue
                    if not newly_acked_packets:
                        continue

                    # Check loss epoch end
                    ### BUGFIX: Check if ANY packet sent AFTER epoch start is ACKed (matches mpquic_subflow.py) ###
                    if state['loss_epoch_flag'] and any(pn > state['loss_epoch_end_num'] for pn in newly_acked_packets):
                        self.end_loss_epoch(ue_id)
                    ### end ###

                    # Update RTT if the largest ACKed is newly acknowledged
                    if state['largest_acked_packet_on_sender'] in newly_acked_packets:
                        # Target packet_number
                        target_packet_number = state['largest_acked_packet_on_sender']
                        # Fetch time_sent from the sent packet table
                        time_sent = state['sent_packets'][target_packet_number].time_sent
                        self.update_rtt(ue_id, time_sent)

                    # Remove newly ACKed packets from the sent list
                    for packet_number in newly_acked_packets:
                        del state['sent_packets'][packet_number]

                    # [Alternative] Mark as "not inflight" instead of deleting
                    """
                    for packet_number, packet in state['sent_packets'].items():
                        if packet_number < next_seq_num and packet_number not in missing_packets:
                            packet.in_flight = False  # Mark as not in flight once received
                    """

                    lost_packets = self.detect_and_remove_lost_packets(ue_id)

                    ### BBR initial variable for number of lost packets ###
                    num_lost_packets = len(lost_packets)
                    ### end ###

                    if lost_packets:
                        con_event_present = self.on_packets_lost(ue_id, lost_packets)

                        ### BBR triggers congestion event and transmits number of newely lost packets this ACK ###
                        if ParameterClass.QUIC_CC == "BBR" and con_event_present:
                            state['cc_algo'].on_congestion_event()
                            self.start_loss_epoch(ue_id)
                        ### end ###

                    ### Code for app limitation was copied from implementation in mpquic_subflow.py and slightly adjusted ###
                    if ParameterClass.APP_LIMITED_OPTION == True:
                        inflight_size = len(
                            [p for p in state['sent_packets'].values() if p.in_flight])
                        app_limited = (
                            state['packets_in_buffer'] and inflight_size < state['cc_algo'].cwnd
                        )
                    else:
                        app_limited = False
                    ### end ###

                    # CUBIC control on ACK reception (this layer is always app_limited=False)
                    newly_acked_num = len(newly_acked_packets)

                    ### Note: Calling on_ack() function of CCA ###
                    ### BBR want all ACKed sequence numbers, not just length ###
                    if ParameterClass.QUIC_CC == "BBR":
                        cumulative_newly_acked_packets.extend(newly_acked_packets)
                        cumulative_lost_packets += num_lost_packets
                    else:
                        state['cc_algo'].on_ack(segments_num_acked=newly_acked_num, rtt=state['latest_rtt'], smoothed_rtt=state['smoothed_rtt'],
                                                seq_num_acked=state['largest_acked_packet_on_sender'], next_seq_num=state['seq_num']+1, app_limited=app_limited)
                    ### end ###

                    state['pto_count'] = 0
                    self.set_loss_detection_timer(ue_id)

        ### Cumulative ACK for BBR - bundeling all ack messages in one sim cycle together ###
        if ParameterClass.QUIC_CC == "BBR":
            state['cc_algo'].on_ack(segments_num_acked=cumulative_newly_acked_packets, rtt=state['latest_rtt'], smoothed_rtt=state['smoothed_rtt'],
                                    seq_num_acked=state['largest_acked_packet_on_sender'], next_seq_num=state['seq_num']+1, app_limited=app_limited, num_lost_packets=cumulative_lost_packets)

            ### BUGFIX: Update delivered counter AFTER on_ack() so delivery rate calculation is correct ###
            state['delivered_packets'] += len(cumulative_newly_acked_packets)
            state['cc_algo'].delivered = state['delivered_packets']
            ### end ###
        ### end ###

        ### BBR remove packet from internal packet manager after ACK ###
        if ParameterClass.QUIC_CC == "BBR":
            for packet_number in cumulative_newly_acked_packets:
                    state['cc_algo'].inflight_packet_data.delete_packet(packet_number)
        ### end ###

        # Log cwnd size and RTT at the end of the loop
        self.logger.store("QUIC", f"UE{ue_id}",
                          f"cwnd_size_log", state['cc_algo'].cwnd)
        self.logger.store(
            "QUIC", f"UE{ue_id}", f"smoothed_RTT", state['smoothed_rtt'])
        self.logger.store("QUIC", f"UE{ue_id}",
                          f"latest_RTT", state['latest_rtt'])

    def culc_throughput(self, ue_id, N, throughput_data):
        """
        Return "throughput averaged over N seconds" and "the average throughput over the entire simulation".
        """
        state = self.ue_states[ue_id]
        total_time = self.NUM_SIMULATION_TIME_SLOTS * \
            self.TIME_SLOT_WINDOW  # Total simulation time [s]
        throughput_dict = {time_index: value for time_index,
                           value in throughput_data}

        # Average throughput per N seconds
        num_intervals = int(np.ceil(total_time / N))
        avg_throughput_per_N = [0]

        for i in range(num_intervals):
            start_time = i * N
            end_time = (i + 1) * N
            start_index = int(start_time / self.TIME_SLOT_WINDOW)
            end_index = int(end_time / self.TIME_SLOT_WINDOW)

            sum_throughput = 0
            count = 0
            for time_index in range(start_index, end_index):
                if time_index in throughput_dict:
                    sum_throughput += throughput_dict[time_index]
                count += 1

            avg_throughput = sum_throughput / count if count > 0 else 0
            avg_throughput_per_N.append(avg_throughput)

        # Average throughput over the entire communication
        total_sum_throughput = sum(throughput_dict.values())
        avg_total_throughput = total_sum_throughput / self.NUM_SIMULATION_TIME_SLOTS

        return avg_throughput_per_N, avg_total_throughput

    def add_sent_packet(self, ue_id, packet_number, in_flight, sent_bits, retransmission):
        state = self.ue_states[ue_id]
        if ue_id not in self.ue_states:
            raise ValueError(f"UE ID '{ue_id}' has not been initialized.")
        sent_packets = self.ue_states[ue_id]['sent_packets']
        packet = sent_packets[packet_number]
        packet.in_flight = in_flight
        packet.sent_bits = sent_bits
        packet.retransmission = retransmission

        ### BBR add packet to internal packet manager ###
        state = self.ue_states[ue_id]
        if ParameterClass.QUIC_CC == "BBR":

            # Get all the important data before
            bbr_packet_number = packet_number
            bbr_packet_delivered = state['delivered_packets']
            bbr_packet_send_time = round(TimeManager.time_index * ParameterClass.TIME_SLOT_WINDOW * 1000, 6) # in ms
            bbr_packet_size = sent_bits
            bbr_packet_is_inflight = in_flight
            bbr_packet_is_app_limited = state['cc_algo'].app_limited
            bbr_packet_is_retransmitted = retransmission

            # Enter the information into the dict
            state['cc_algo'].inflight_packet_data.add_packet(bbr_packet_number, bbr_packet_send_time, bbr_packet_delivered, bbr_packet_size,
                                                            bbr_packet_is_inflight, bbr_packet_is_app_limited, bbr_packet_is_retransmitted, ue_id)
        ### end ### 

    def get_pto_time(self, ue_id):
        state = self.ue_states[ue_id]
        duration = (state['smoothed_rtt'] + max(4 * state['rtt_var'],
                    ParameterClass.K_GRANULARITY)) * (2 ** state['pto_count'])
        # TECC paper Section 4.2.5: relax PTO by the tunnel RTT amount (dynamic observed value)
        # CAMF similarly references tunnel_rtt (waiting for MPQUIC retransmission completion)
        if ParameterClass.TECC_OPTION is True or ParameterClass.CAMF_OPTION is True:
            duration += state['tunnel_rtt']

        has_in_flight_packets = any(
            packet.in_flight for packet in state['sent_packets'].values())
        if has_in_flight_packets == 0:  # Probably never happens...
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

        # If reordering has been observed, start the normal timer
        if earliest_loss_time != 0:
            state['loss_detection_timer'] = earliest_loss_time
            return

        # If no packets are in flight, stop the timer
        has_in_flight_packets = any(
            packet.in_flight for packet in state['sent_packets'].values())
        if has_in_flight_packets == 0:
            state['loss_detection_timer'] = float('inf')
            return

        # If in flight and no reordering, start PTO timer
        timeout = self.get_pto_time(ue_id)
        state['loss_detection_timer'] = timeout

    def update_rtt(self, ue_id, time_sent):
        state = self.ue_states[ue_id]
        # When a newly_acked arrives, first update latest_rtt
        state['latest_rtt'] = round(
            self.time_index * self.TIME_SLOT_WINDOW - time_sent, 6)
        self.logger.store("QUIC", f"UE{ue_id}", "latest_rtt_change_log",
                          f"time={self.time_index}\t{state['latest_rtt']}")

        # Then update various RTT metrics
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
        Return a list of lost packet numbers.
         - Only "loss detection" and "loss time update" are performed here.
        """
        state = self.ue_states[ue_id]

        state['loss_time'] = 0
        lost_packets = []

        ### BUGFIX: Use RFC 9002 formula (matches mpquic_subflow.py) ###
        loss_delay = state['smoothed_rtt'] + max(4.0 * state['rtt_var'], 0.005)
        # TECC paper Section 4.2.5: relax threshold by Tt to suppress spurious losses due to tunnel retransmissions (dynamic observed value)
        # CAMF similarly references tunnel_rtt (waiting for MPQUIC retransmission completion)
        if ParameterClass.TECC_OPTION is True or ParameterClass.CAMF_OPTION is True:
            loss_delay += state['tunnel_rtt']
        ### end ###

        loss_delay = max(loss_delay, self.K_GRANULARITY)

        lost_send_time = round(
            self.time_index * self.TIME_SLOT_WINDOW - loss_delay, 6)

        # for packet_number, packet in state['sent_packets'].items():
        for packet_number in list(state['sent_packets'].keys()):
            packet = state['sent_packets'][packet_number]
            if packet.in_flight:  # Process only those with in_flight=True
                if packet_number > state['largest_acked_packet_on_sender']:
                    continue
                if packet.time_sent <= lost_send_time:
                    lost_packets.append(packet_number)
                    self.logger.store(
                        "QUIC", f"UE{ue_id}", f"event_log", f"time_index={TimeManager.time_index}: packet loss (more than 9/8RTT delay)")
                elif (state['largest_acked_packet_on_sender'] >= packet_number + self.K_PACKET_THRESH
                      and not packet.retransmission):
                    lost_packets.append(packet_number)
                    self.logger.store("QUIC", f"UE{ue_id}", f"event_log", f"time_index={TimeManager.time_index}: packet loss (reordering more than 3 packets)")
                else:
                    if state['loss_time'] == 0:
                        state['loss_time'] = state['sent_packets'][packet_number].time_sent + loss_delay
                    else:
                        state['loss_time'] = min(
                            state['loss_time'], state['sent_packets'][packet_number].time_sent + loss_delay)

        # Return the list of lost packet numbers
        return lost_packets

    def check_timer(self, ue_id):
        """
        Check the timer and return whether on_loss_detection_timeout should be triggered (True = trigger).
        """
        state = self.ue_states[ue_id]
        if state['loss_detection_timer'] != float('inf'):
            current_time = round(self.time_index * self.TIME_SLOT_WINDOW, 6)
            if state['loss_detection_timer'] < current_time:
                return True
        return False

    def on_loss_detection_timeout(self, ue_id):
        state = self.ue_states[ue_id]

        ### BBR triggeres timeout here ###
        if ParameterClass.QUIC_CC == "BBR":
            state['cc_algo'].on_timeout()
        ### end ###

        ### BBR advances pacing manager because it is not executed in send function in case of on_loss_detection_timeout ###
        if ParameterClass.QUIC_CC == "BBR":
            pacing_rate = state['cc_algo'].pacing_rate
            pacing_window = state['cc_algo'].pacing_manager.advance_pacing_manager(pacing_rate)
        ### end ###

        earliest_loss_time = state['loss_time']
        if earliest_loss_time != 0:  # Non-PTO timer expired
            lost_packets = self.detect_and_remove_lost_packets(ue_id)
            if lost_packets:  # Trigger congestion event
                self.logger.store(
                    "QUIC", f"UE{ue_id}", f"event_log", f"time_index={TimeManager.time_index}: Packet loss (Timer expiration)")
                self.on_packets_lost(ue_id, lost_packets)
            self.set_loss_detection_timer(ue_id)
            return

        # PTO timer expired => send 1 or 2 probe packets
        state['pto_count'] += 1
        self.set_loss_detection_timer(ue_id)

        # Update time_of_last_ack_eliciting_packet
        state['time_of_last_ack_eliciting_packet'] = round(
            self.time_index * self.TIME_SLOT_WINDOW, 6)

        # Compute packet size
        header_size = ParameterClass.IPV4_HEADER_SIZE + \
            ParameterClass.UDP_HEADER_SIZE + ParameterClass.QUIC_HEADER_SIZE
        payload_size = ParameterClass.MTU_SIZE - header_size
        packet_size = (header_size + payload_size) * 8

        # Initialize return array
        packets_to_send = np.zeros(
            (0, ParameterClass.NUM_INFO_PACKET), dtype=int)

        for _ in range(self.PTO_TRANSMIT_NUM):
            # FCT mode: check new packet upper limit
            if ParameterClass.FCT_MODE and state['seq_num'] >= state['fct_flow_size']:
                state['fct_send_complete'] = True
                if state['fct_send_completion_time'] is None:
                    state['fct_send_completion_time'] = TimeManager.time_index
                break
            packet = np.zeros((1, ParameterClass.NUM_INFO_PACKET), dtype=int)
            packet[:, self.INDEX_PACKET_ID] = state['seq_num']
            packet[:, self.INDEX_PAYLOAD_SIZE] = packet_size
            packet[:, self.INDEX_ue_id] = ue_id
            packet[:, self.INDEX_SERVER_TIMESTAMP_ID] = self.time_index
            packet[:, self.INDEX_CAMF_FB_ID] = state['camf_fb_id']
            packets_to_send = np.append(packets_to_send, packet, axis=0)
            self.add_sent_packet(
                ue_id, packet_number=state['seq_num'], in_flight=True, sent_bits=packet_size, retransmission=False)
            self.logger.store("QUIC", f"UE{ue_id}", f"event_log",
                              f"time_index={TimeManager.time_index}: Transmission caused by PTO. Seq_num ={state['seq_num']}")
            state['seq_num'] += 1

        return packets_to_send

    def on_packets_lost(self, ue_id, lost_packets):
        state = self.ue_states[ue_id]
        sent_time_of_last_loss = 0
        lost_packets_for_del_from_inflight = []

        # Update retransmission list (before deleting from dict)
        # - If 1 RTT has passed since the previous retransmission, add to the list
        for lost_packet in lost_packets:
            time_thresh = round(
                self.time_index * self.TIME_SLOT_WINDOW - state['latest_rtt'], 6)
            if state['sent_packets'][lost_packet].retransmission == False:
                state['retransmission_seq_nums'].append(lost_packet)
                lost_packets_for_del_from_inflight.append(lost_packet)
            elif state['sent_packets'][lost_packet].retransmission == True and state['sent_packets'][lost_packet].time_sent < time_thresh:
                state['retransmission_seq_nums'].append(lost_packet)
                lost_packets_for_del_from_inflight.append(lost_packet)
        self.logger.store("QUIC", f"UE{ue_id}", f"retransmission_list_log",
                          f"time_index={TimeManager.time_index}: packets for retransmission = {state['retransmission_seq_nums']}")

        # Remove lost packets from in-flight
        for lost_seq_num in lost_packets_for_del_from_inflight:
            sent_time_of_last_loss = max(
                sent_time_of_last_loss, state['sent_packets'][lost_seq_num].time_sent)
            del state['sent_packets'][lost_seq_num]

            ### BBR removes lost packets from internal packet manager ###
            if ParameterClass.QUIC_CC == "BBR":
                state['cc_algo'].inflight_packet_data.delete_packet(lost_seq_num)
            ### end ###

        # Trigger a congestion event
        # Potential bug -> guard for BBR
        if ParameterClass.QUIC_CC != "BBR":
            if sent_time_of_last_loss != 0 and state['loss_epoch_flag'] == False:
                state['cc_algo'].on_congestion_event()
                self.start_loss_epoch(ue_id)

        # If not blocking retransmission for 1 RTT, we could simply extend the list:
        """
        state['retransmission_seq_nums'].extend(lost_packets) # Add to retransmission list
        """

        # For log.
        self.logger.store("QUIC", f"UE{ue_id}", f"packet_loss_log",
                          f"time_index={TimeManager.time_index}: Newly lost packet = {lost_packets_for_del_from_inflight}")
        state['counter_total_packet_loss'] += len(
            lost_packets_for_del_from_inflight)
        
        ### BBR: Return value for congestion event condition ###
        if sent_time_of_last_loss != 0 and state['loss_epoch_flag'] == False:
            return True
        else:
            return False
        ### end ###

    def start_loss_epoch(self, ue_id):
        state = self.ue_states[ue_id]
        state['loss_epoch_flag'] = True
        state['loss_epoch_end_num'] = state['seq_num']
        self.logger.store("QUIC", f"UE{ue_id}", f"event_log",
                          f"time_index={TimeManager.time_index}: loss_epoch starts.")

    def end_loss_epoch(self, ue_id):
        state = self.ue_states[ue_id]
        state['loss_epoch_flag'] = False
        self.logger.store("QUIC", f"UE{ue_id}", f"event_log",
                          f"time_index={TimeManager.time_index}: loss_epoch ends.")
        
        ### End loss recovery inside of BBR ###
        if ParameterClass.QUIC_CC == "BBR":
            state['cc_algo'].exit_congestion_event()
        ### end ###

    def log_for_debug(self, ue_id):
        state = self.ue_states[ue_id]
        self.logger.store("QUIC", f"UE{ue_id}", f"ack_dict_log",
                          f"time_index={TimeManager.time_index}: ack_dict_log = {state['ack_packet_dict']}")
        self.logger.store("QUIC", f"UE{ue_id}", f"sent_packets_log",
                          f"time_index={TimeManager.time_index}: sent_packets = {state['sent_packets']}")

    def onetime_logger(self):
        for i in range(ParameterClass.NUM_UE):
            state = self.ue_states[i]
            self.logger.store("QUIC", f"UE{i}", f"packet_loss_rate",
                              f"Total Sent Packets = \t{state['counter_total_sent_packets']}")
            self.logger.store("QUIC", f"UE{i}", f"packet_loss_rate",
                              f"Total Packet Losses = \t{state['counter_total_packet_loss']}")
            self.logger.store("QUIC", f"UE{i}", f"packet_loss_rate",
                              f"PLR = \t{state['counter_total_packet_loss']/state['counter_total_sent_packets']}")
