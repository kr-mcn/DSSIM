import random
import math

# Modules
from bbrv1_definitions import BBRState, WindowedFilter, PacketManager
from param import ParameterClass, TimeManager
from pacing_manager import PacingManager

#-----------------------------------------------------------#
# BBR Module                                                #
#-----------------------------------------------------------#

# Define BBRv1 class
class BBRv1:

    # Initializing the BBR algorithm
    def __init__(self, logger, log_dir1, log_dir2):

        # Bottleneck Bandwidth variables
        self.btlbw = 0
        self.delivery_rate = 0.0                  # latest per-ACK delivery rate [pkt/ms]
        self.btlbw_filter = WindowedFilter()
        self.btlbw_filter_length = 10
        self.next_round_delivered = 0
        self.round_start = False
        self.round_count = 0
        self.delivered = 0
        self.inflight_packet_data = PacketManager()
        self.app_limited = False

        # Round trip propagation time variables
        self.rt_prop = float('inf')
        self.rt_prop_expired = True
        self.rt_prop_stamp = self._now()
        self.rt_prop_filter_length = 10000 # 10s

        # Pacing rate
        self.initial_cwnd = ParameterClass.INIT_CWND
        self.initial_SRTT = 1
        self.pacing_gain = 2.89    # 2/ln(2)
        self.pacing_rate = self.pacing_gain * (self.initial_cwnd / self.initial_SRTT)
        self.pacing_manager = PacingManager(ParameterClass.TIME_SLOT_WINDOW * 1000, 
                                            ParameterClass.NUM_SIMULATION_TIME_SLOTS,
                                            max_time_step=100,                          
                                            initial_pacing_rate=self.pacing_rate)

        # Send quantum
        self.send_quantum = 0

        # Congestion window
        self.cwnd = self.initial_cwnd
        self.cwnd_gain = 2.89     # 2/ln(2)
        self.target_cwnd = self.initial_cwnd                 
        self.prior_cwnd = 0      
        self.min_in_pipe_cwnd = 4
        self.packet_conservation = False    
        self.in_loss_recovery = False           
        self.newely_delivered = 0
        self.newely_lost = 0
        # Deviation from Internet draft: estimation of loss recovery rounds
        self.loss_round_counter = 0
        self.loss_recovery_second_round_time = float('inf')

        # State variables
        self.state = BBRState.STARTUP

        # State machine - startup
        self.high_gain = 2.89
        self.filled_pipe = False                
        self.full_bw = 0
        self.full_bw_count = 0

        # State machine - probebw
        self.cycle_index = 0
        self.cycle_stamp = self._now()
    
        # State machine - probertt
        self.probe_rtt_done_stamp = 0 
        self.probe_rtt_round_done = False 
        self.probe_rtt_duration = 200 # 200ms
        self.previous_inflight = 0

        # State machine - idle restart
        self.idle_restart = False
        self.app_limited = False

        # Logger
        self.logger = logger
        self.log_dir1 = log_dir1
        self.log_dir2 = log_dir2

        # Call BBR init function
        self._bbr_init()

    # Init function
    def _bbr_init(self):
        self._init_round_counting()
        self._init_full_pipe()
        self._init_pacing_rate()
        self._enter_startup()

#-----------------------------------------------------------#
# General Methods                                           #
#-----------------------------------------------------------#

    # Current simulation time (ms for BBR estimates)
    def _now(self):
        return round(TimeManager.time_index * ParameterClass.TIME_SLOT_WINDOW * 1000, 6)  
    
    # Find the most recently sent packet for estimations
    def _filter_packets(self, segments_num_acked):

        # Filter out retransmitted packets for estimation
        first_transmitted_ack = []
        for seq_num in segments_num_acked:
            
            # Check if packet is in inflight data to avoid errors
            entry = self.inflight_packet_data.get_packet_info(seq_num)
            if entry is not None:
                
                # Add to first transmittion packets to list
                if not entry.is_retransmitted and entry.inflight:
                    first_transmitted_ack.append(seq_num)
        
        # Choose only the most recently ACKed packet for estimates
        if first_transmitted_ack:
            seq_num = max(first_transmitted_ack)
            #self.logger.store(self.log_dir1, self.log_dir2, "chosen_ack_by_filter", f"time_index={TimeManager.time_index}: chosen_ack={seq_num}")
            packet = self.inflight_packet_data.get_packet_info(seq_num)
            return packet
        else:
            return None
        
    # Do all necessary steps to update the model and state variables
    def _update_model_and_state(self, packet):

        # Update bottleneck bandwidth estimation
        self._update_btlbw(packet)

        # Handle cycle phase
        self._check_cycle_phase()

        # Check if the pipe is full -> Startup to Drain
        self._check_full_pipe(packet)

        # Chech if the pipe has drained -> Drain to ProbeRTT
        self._check_drain()

        # Update the RTprop estimation
        self._update_rtprop(packet)

        # Check if RTT probing is necessary
        self._check_probe_rtt()

    # Do all necessary steps to update the control parameters
    def _update_control_parameters(self):

        # Set the pacing rate
        self._set_pacing_rate(self.pacing_gain)

        # Set the send quantum
        self._set_send_quantum()

        # Set the congestion window
        self._set_cwnd()

    # The main ACK processing function
    def on_ack(self, segments_num_acked, rtt, smoothed_rtt, seq_num_acked, next_seq_num, app_limited, num_lost_packets):
        
        # Adjust the app limited status from QUIC / MPQUIC
        self.app_limited = app_limited
        if self.app_limited == True:
            self.logger.store(self.log_dir1, self.log_dir2, "bbr_app_limited", f"time_index={TimeManager.time_index}: app_limited={self.app_limited}")

        # Exit method is no packets have arrived
        if not segments_num_acked:
            return None

        # Find the most recently send packet that is not retransmitted
        packet = self._filter_packets(segments_num_acked)
        if packet is None:
            return None
        
        # Refresh the number newly delivered and lost packets this ACK 
        self.newely_lost = num_lost_packets
        self.newely_delivered = len(segments_num_acked)
        
        # Update model estimates and state variables
        self._update_model_and_state(packet)

        # Update the control parameters
        self._update_control_parameters()

    # Tasks performed when sending a packet
    def on_transmission(self):

        # Initiate restart after idle period
        self._handle_restart_from_idle()

#-----------------------------------------------------------#
# Bottleneck Bandwidth Estimation                           #
#-----------------------------------------------------------#

    # Initalization of round counting 
    def _init_round_counting(self):
        self.next_round_delivered = 0
        self.round_start = False
        self.round_count = 0

    # Updates round time used to get rid of expired RTprop samples
    def _update_round(self, packet):

        # Check if round time is reached
        if packet.delivered >= self.next_round_delivered:
            self.next_round_delivered = self.delivered + self.newely_delivered
            self.round_count += 1
            self.round_start = True
        else:
            self.round_start = False

    # Update the BtlBw filter and return max value
    def _update_windowed_max_filter(self, btlbw_filter, value, time, window_length):

        # Add value to filter
        btlbw_filter.add_value(value, time, window_length)

        # Remove old values
        btlbw_filter.remove_old_values(time)

        # Return max value
        return btlbw_filter.get_max()
    
    # Calculates the delivery rate for the BtlBw estimation
    def _get_delivery_rate(self, packet):
        
        # Check if packet is in packets dictionary
        entry = self.inflight_packet_data.get_packet_info(packet.id)

        # If packet is missing -> exit function
        if entry is None:
            return None
        
        # Calculate delivery rate
        #numerator = self.delivered - entry.delivered
        numerator = self.delivered + self.newely_delivered- entry.delivered
        denominator = self._now() - entry.send_time    # Calculate the delivery rate in [packets / ms]
        delivery_rate = numerator / denominator if denominator > 0 else 0
        self.logger.store(self.log_dir1, self.log_dir2, "bbr_delivery_rate", f"time_index={TimeManager.time_index}: delivery_rate={delivery_rate}")

        # Return values
        return {
            'delivery_rate': delivery_rate,
            'is_app_limited': entry.is_app_limited }

    # Update the BtlBw estimation
    def _update_btlbw(self, packet):

        # Perform the round time measurement
        self._update_round(packet)

        # Calculate the delivery rate
        result = self._get_delivery_rate(packet)
        if result is None:
            return None
        delivery_rate = result['delivery_rate']
        app_limited = result['is_app_limited']
        if delivery_rate > 0:
            self.delivery_rate = delivery_rate
        
        # Deviation from Internet Draft! Surpress updating BtlBw in ProbeRTT state
        if self.state != BBRState.PROBERTT:

            # Update the BtlBw estimation (app limit hook deviates from Internet draft)
            if not app_limited and delivery_rate > 0:
                self.btlbw = self._update_windowed_max_filter(self.btlbw_filter,
                                                            delivery_rate,
                                                            self.round_count,
                                                            self.btlbw_filter_length)
                self.logger.store(self.log_dir1, self.log_dir2, "bbr_btlbw",
                                  f"time_index={TimeManager.time_index}: btlbw={self.btlbw}")
            
#-----------------------------------------------------------#
# Round-Trip Propagation Delay                              #
#-----------------------------------------------------------#

    # Update the RTprop estimation
    def _update_rtprop(self, packet):

        # Calculate the RTT
        rtt = self._now() - packet.send_time

        # Check if the RTT measurement is expired
        self.rt_prop_expired = self._now() > self.rt_prop_stamp + self.rt_prop_filter_length

        # Update RTprop estimation
        if rtt >= 0 and (rtt <= self.rt_prop or self.rt_prop_expired):
            self.rt_prop = rtt
            self.rt_prop_stamp = self._now()

#-----------------------------------------------------------#
# Pacing Rate                                               #
#-----------------------------------------------------------#

    # Initialize the pacing rate at the connection start
    def _init_pacing_rate(self):

        # Default values for cwnd and SRTT
        nominal_bandwidth = self.initial_cwnd / self.initial_SRTT
        self.pacing_rate = self.pacing_gain * nominal_bandwidth

    # Set a new pacing rate
    def _set_pacing_rate(self, pacing_gain):

        # Derive minimum pacing rate from pacing manager (safe lower bound)
        min_rate = 1 / self.pacing_manager.max_time_step

        btlbw_calibration_factor = 0.98
        calibrated_btlbw = self.btlbw * btlbw_calibration_factor

        # Estimate the pacing rate using calibrated btlbw
        rate_raw = pacing_gain * calibrated_btlbw
        rate = max(rate_raw, min_rate)

        self.pacing_rate = rate
        self.logger.store(self.log_dir1, self.log_dir2, "pacing_rate", f"time_index={TimeManager.time_index}: pacing_rate={self.pacing_rate}")

#-----------------------------------------------------------#
# Send Quantum                                              #
#-----------------------------------------------------------#

### BBR uses number of packets instead of Bytes
### It also uses miliseconds instead of seconds
### Therefore:
### throughput = n packets/ms * (1500 Bytes * 8) * 1000 = 12e6 * n packets/ms = [bps]
### number of packets/ms = throughput / (1500 Bytes * 8 * 1000)

    # Update the send quantum -> rates in bits and quantum in bytes
    def _set_send_quantum(self):

        # Less than 1.2Mbps (0.1 packets/ms) -> 1 Packet
        if self.pacing_rate < 0.1:          
            self.send_quantum = 1

        # Less than 24Mbps (2 packets/ms) -> 2 Packets
        elif self.pacing_rate < 2:     

            self.send_quantum = 1

        # More than 24Mbps -> Min of pacing rate and 42 packets/ms (504 Mbps)
        else:
            self.send_quantum = min(self.pacing_rate , 42)

#-----------------------------------------------------------#
# Congestion Window                                         #
#-----------------------------------------------------------#

    # Calculate the data inflight based on estimate values
    def _inflight(self, cwnd_gain):

        # Use default cwnd in case no RTprop estimate exists
        if self.rt_prop == float('inf'):
            return self.initial_cwnd
        
        # Calculate target cwnd
        quanta = self.send_quantum if self.pacing_rate < 1.5 else 3 * self.send_quantum
        estimated_bdp = self.btlbw * self.rt_prop
        cwnd_window = int(math.ceil(cwnd_gain * estimated_bdp + quanta))
        return cwnd_window

    # Update the target cwnd based on the path model
    def _update_target_cwnd(self):
        self.target_cwnd = self._inflight(self.cwnd_gain)
    
    # Save current cwnd before entering loss recovery
    def _save_cwnd(self):
        if not self.in_loss_recovery and self.state != BBRState.PROBERTT:
            return self.cwnd
        else:
            return max(self.cwnd, self.prior_cwnd)
        
    # Restore old cwnd after exiting loss recovery
    def _restore_cwnd(self):
        self.cwnd = max(self.cwnd, self.prior_cwnd)

    # Modulate the cwnd in case of loss recovery
    def _modulate_cwnd_for_recovery(self):
        if self.newely_lost > 0:
            self.cwnd = max(self.cwnd - self.newely_lost, 1)
        if self.packet_conservation:
            # inflight_packet_data is cleaned up AFTER on_ack() in mpquic_subflow.py,
            # so inflight_packet_data.inflight() at this point still contains the newly
            # ACKed packets. This means:
            #   inflight_packet_data.inflight() == true_inflight + newely_delivered
            # Adding newely_delivered again would double-count ACKed packets and inflate cwnd.
            # The correct packet-conservation formula is:
            #   cwnd = max(cwnd, true_inflight + newely_delivered)
            #        = max(cwnd, inflight_packet_data.inflight())
            self.cwnd = max(self.cwnd, self.inflight_packet_data.inflight())

    # Reduce cwnd in case of RTprop estimation state
    def _modulate_cwnd_for_probe_rtt(self):
        if self.state == BBRState.PROBERTT:
            self.cwnd = min(self.cwnd, self.min_in_pipe_cwnd)

    # Enter second round of loss recovery
    def _check_loss_recovery_second_round(self):
        if self._now() >= self.loss_recovery_second_round_time:
            self.packet_conservation = False
            self.loss_recovery_second_round_time = float('inf')

    # Set the cwnd
    def _set_cwnd(self):

        ### Deviation from Internet draft: check for second round of loss recovery
        self._check_loss_recovery_second_round()

        # Update target congestion window
        self._update_target_cwnd()

        # Modulate cwnd for recovery
        self._modulate_cwnd_for_recovery()

        # Process the cwnd modulation in the normal case
        if self.packet_conservation == False:
            if self.filled_pipe:
                self.cwnd = min(self.cwnd + self.newely_delivered, self.target_cwnd)
            elif self.cwnd < self.target_cwnd or self.delivered < self.initial_cwnd:
                self.cwnd = self.cwnd + self.newely_delivered
            self.cwnd = max(self.cwnd, self.min_in_pipe_cwnd)

        # Modulate cwnd for RT probning state 
        self._modulate_cwnd_for_probe_rtt()

#-----------------------------------------------------------#
# Congestion Events                                         #
#-----------------------------------------------------------#    
    
    # Timeout event -> Saving old cwnd and lowering the current to 1
    def _cwnd_on_retransmission_timeout(self):
        self.in_loss_recovery = True
        self.prior_cwnd = self._save_cwnd()
        #self.cwnd = 1
        self.cwnd = self.inflight_packet_data.inflight() + 1

    # Fast recovery (packet loss but still packets inflight)
    def _cwnd_on_fast_recovery(self):
        self.in_loss_recovery = True
        self.prior_cwnd = self._save_cwnd()
        self.cwnd = self.inflight_packet_data.inflight() + max(self.newely_delivered, 1)
        self.packet_conservation = True

        ### Deviation from Internet draft: Estimate one loss round trip
        if self.loss_round_counter == 0 and self.rt_prop != float('inf'):    
            self.loss_recovery_second_round_time = self._now() + self.rt_prop
        self.loss_round_counter += 1

    # Exiting the loss recovery state and restoring the previous cwnd
    def _exit_loss_recovery(self):
        self.in_loss_recovery = False
        self.packet_conservation = False
        self._restore_cwnd()

        ### Deviation from Internet draft: Estimate one loss round trip
        self.loss_round_counter = 0
        self.loss_recovery_second_round_time = float('inf')

    # In case of timeout
    def on_timeout(self):
        self._cwnd_on_retransmission_timeout()

    # In case of other congestion event
    def on_congestion_event(self):
        self._cwnd_on_fast_recovery()

    # Exit congestion event (for both retransmission timeout and fast recovery)
    def exit_congestion_event(self):
        self._exit_loss_recovery()
    
#-----------------------------------------------------------#
# State Machine - Startup                                   #
#-----------------------------------------------------------#  

    # Setting variables for the startup state
    def _enter_startup(self):
        self.state = BBRState.STARTUP
        self.logger.store(self.log_dir1, self.log_dir2, "bbr_state_log", f"time_index={TimeManager.time_index}: state=STARTUP")
        self.pacing_gain = self.high_gain
        self.cwnd_gain = self.high_gain

    # Setting variables to check for full pipe
    def _init_full_pipe(self):
        self.filled_pipe = False
        self.full_bw = 0
        self.full_bw_count = 0
    
    # Check for full pipe each round trip
    def _check_full_pipe(self, packet):
        
        # See if checking the pipe is necessary 
        app_limit = packet.is_app_limited
        if self.filled_pipe or not self.round_start or app_limit:
            return None
        
        # Check if BtlBw is still growing
        if self.btlbw >= self.full_bw * 1.25:

            # Save new base level
            self.full_bw = self.btlbw
            self.full_bw_count = 0
            return None
        
        # Increment full pipe count if there is no significant growth of BtlBw
        self.full_bw_count += 1

        # Declare the pipe as full after three cycles with no growth
        if self.full_bw_count >= 3:
            self.filled_pipe = True

#-----------------------------------------------------------#
# State Machine - Drain                                     #
#-----------------------------------------------------------#  

    # Setting variables for the drain state
    def _enter_drain(self):
        self.state = BBRState.DRAIN
        self.logger.store(self.log_dir1, self.log_dir2, "bbr_state_log", f"time_index={TimeManager.time_index}: state=DRAIN")
        self.pacing_gain = 1/self.high_gain
        self.cwnd_gain = self.high_gain

    # Check for drained pipe
    def _check_drain(self):

        # Transition from starup to drain state
        if self.state == BBRState.STARTUP and self.filled_pipe:
            self._enter_drain()

        # Transition from drain to probebw state
        estimated_bdp = self._inflight(1.0)
        if self.state == BBRState.DRAIN and self.inflight_packet_data.inflight() <= estimated_bdp:
            self._enter_probebw()

#-----------------------------------------------------------#
# State Machine - ProbeBW                                   #
#-----------------------------------------------------------# 

    # Advancing to the next cycle phase
    def _advance_cycle_phase(self):

        # Refresh the cycle stamp
        self.cycle_stamp = self._now()

        # Interate to next cycle phase and set pacing gain
        self.cycle_index = (self.cycle_index + 1) % 8
        pacing_gain_cycle = [5/4, 3/4, 1, 1, 1, 1, 1, 1]
        self.pacing_gain = pacing_gain_cycle[self.cycle_index] 

    # Setting variables for the probebw state
    def _enter_probebw(self):

        # State variables
        self.state = BBRState.PROBEBW
        self.logger.store(self.log_dir1, self.log_dir2, "bbr_state_log", f"time_index={TimeManager.time_index}: state=PROBEBW")
        self.pacing_gain = 1
        self.cwnd_gain = 2

        # Choose random cycles gain index [5/4, 3/4, 1, 1, 1, 1, 1, 1] excluding value 3/4
        self.cycle_index = random.choice([i for i in range(8) if i != 0])
        
        # Advance to the next cycle
        self._advance_cycle_phase()

    # Check if transition to next cycle phase is due
    def _is_next_cycle_phase(self):

        # Check if one RTT has passed
        is_full_length = (self._now() - self.cycle_stamp) > self.rt_prop

        # In case pacing gain equal to 1
        if self.pacing_gain == 1:
            return is_full_length

        # In case pacing gain bigger than 1
        elif self.pacing_gain > 1:
            estimated_bdp = self._inflight(self.pacing_gain)
            conditional = is_full_length and (self.newely_lost > 0 or self.previous_inflight >= estimated_bdp)
            return conditional

        # In case pacing gain smaller than 1
        else:
            estimated_bdp = self._inflight(1)
            return is_full_length or self.previous_inflight <= estimated_bdp
    
    # Check if advancing the cycle phase is necessary and do it
    def _check_cycle_phase(self):

        # Check if state is ProbeBW and the next cycle phase is due
        if self.state == BBRState.PROBEBW and self._is_next_cycle_phase():

            # Advance Cycle phase
            self._advance_cycle_phase()
        
#-----------------------------------------------------------#
# State Machine - ProbeRTT                                  #
#-----------------------------------------------------------#

    # Setting variables for the probebw state
    def _enter_probertt(self):
        self.state = BBRState.PROBERTT
        self.logger.store(self.log_dir1, self.log_dir2, "bbr_state_log", f"time_index={TimeManager.time_index}: state=PROBERTT")
        self.pacing_gain = 1
        self.cwnd_gain = 1

    # Exit the ProbeRTT state
    def _exit_probertt(self):
        
        # If startup has already happend -> switch to ProbeBW state
        if self.filled_pipe:
            self._enter_probebw()

        # If startup has not happened yet -> switch to startup
        else:
            self._enter_startup()

    # Handle the RTT probing
    def _handle_probe_rtt(self):
        
        ### Deivation from Internet Draft
        # Mark as app limited to not include any samples in BtlBw estimation
        # This was done differently -> instead hook at BtlBw estimation when in ProbeBW

        # In case RTT probing has not started yet (first iteration)
        if self.probe_rtt_done_stamp == 0 and self.inflight_packet_data.inflight() <= self.min_in_pipe_cwnd:
            
            # ProbeRTT state will end at the defined timestamp (in 200ms)
            self.probe_rtt_done_stamp = self._now() + self.probe_rtt_duration
            
            # Set flag to False because ProbeRTT has only just started
            self.probe_rtt_round_done = False
            
            # Setup to detect the next round trip
            self.next_round_delivered = self.delivered
            
        # else if (BBR.probe_rtt_done_stamp != 0)
        elif self.probe_rtt_done_stamp != 0:
            
            # Did one round trip pass?
            if self.round_start:
                
                # Set flag to exit the ProbeRT state after one round trip
                self.probe_rtt_round_done = True
                
            # Ready to exit state
            if self.probe_rtt_round_done and self._now() > self.probe_rtt_done_stamp:
                
                # Refresh the timestamp of the last ProbeRTT event
                self.rt_prop_stamp = self._now()
                
                # Restore the cwnd
                self._restore_cwnd()
                
                # Exit ProbeRTT state
                self._exit_probertt()

    # Set variables and perform RTT probing
    def _check_probe_rtt(self):

        # Enter ProbeRTT state if the last RTT estimate has expired (after 10 seconds)
        if (self.state != BBRState.PROBERTT and self.rt_prop_expired and not self.idle_restart):

            # Enter ProbeRTT and set variables
            self._enter_probertt()
            
            # Save prior congestion window for latter restauration
            self.prior_cwnd = self._save_cwnd()
            
            # Set stamp to 0 to signal new start of the state
            self.probe_rtt_done_stamp = 0

        # Currenlty in ProbeRTT state
        if self.state == BBRState.PROBERTT:
            
            # Perform probing
            self._handle_probe_rtt()

        # Set idle restart flag to False
        self.idle_restart = False

#-----------------------------------------------------------#
# State Machine - Restart from idle                         #
#-----------------------------------------------------------#

    # Restart process in case of long idle period
    def _handle_restart_from_idle(self):

        # Idle conditions met?
        if self.inflight_packet_data.inflight() == 0 and self.app_limited:
            self.idle_restart = True

            # Is the state machine in ProbeBW?
            if self.state == BBRState.PROBEBW:
                self._set_pacing_rate(1)

