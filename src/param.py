import numpy as np
from datetime import datetime
import os as _os

np.random.seed(0)

class ParameterClass:
    # Class variables (shared across all instances)
    # --- Basic Simulation Parameters ----------------------
    NUM_UE = 1  # Number of User Equipments (UEs)
    NUM_SIMULATION_TIME_SLOTS = 30000  # Total number of simulation time slots 30k
    TIME_SLOT_WINDOW = 0.001  # Duration of a single time slot [seconds]
    SIM_MODE = "MPQUIC"
    TECC_OPTION = (
        #True  # Enable TECC rate control algorithm
        False
    )
    CAMF_OPTION = (# Enable CAMF control
        #True
        False
    )
    CAMF_NOCAP_OPTION    = (# Disable the cwnd_headroom capacity check in CAMF
        #True
        False
    )
    SCHEDULER_MODE = (
        "MINRTT"  # Min-RTT scheduler: Chooses the path with shortest RTT
    )
   
    UE_START_TIMES = {# Transmission start time for each UE (UE_ID: start time_index)
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
    }

    # --- FCT (Flow Completion Time) Mode ---
    FCT_MODE = (
        #True   # FCT mode: terminate after sending the specified number of packets (FCT_TOTAL_PACKETS)
        False   # Non-FCT mode: unlimited data, fixed duration
    )
    FCT_TOTAL_PACKETS = (# Total number of packets each UE transmits (valid only in FCT mode)
        10000 # 10000 = 15MB (120Mbit)
    ) 

    QUIC_CC = (
        #"CUBIC"  # Congestion control algorithm for QUIC layer
        "BBR"
    )
    MPQUIC_CC = (
        #"CUBIC"  # Congestion control algorithm for MPQUIC outer layer
        "BBR"
    )

    # --- Delay parameters -----------------------------------
    N3_DELAY = 11 # Delay applied on N3 link [unit: time_index]. Actual delay = (N3_DELAY - 1) * time_slot.
    N6_DELAY_TI = 10  # N6 link delay [time_index]

    # --- N3/N6 link bandwidth. 1e6=Mb/s ------------------------
    N3_BANDWIDTH_BPS = 10 * 1e9
    N3_BANDWIDTH_PPS = 1e7
    N6_BANDWIDTH_BPS = (
        #50 * 1e6
        #150 * 1e6
        200 * 1e6
    )
    #N6_BANDWIDTH_BPS = 150 * 1e6
    # N6 link bandwidth time-change schedule
    # Format: [(time_index, bps), ...]. Empty list means no change (N6_BANDWIDTH_BPS is used).
    N6_BANDWIDTH_SCHEDULE = [
        #(10000, 50 * 1e6)
        #(10000, 150 * 1e6)
    ]

    # --- Packet loss rates -----------------------------------
    N3_LOSS_RATE_DL = 0  # Downlink loss rate on N3 link (default: 0% - no loss)
    N3_LOSS_RATE_UL = 0  # Uplink loss rate on N3 link (default: 0% - no loss)
    N6_LOSS_RATE_DL = 0  # Downlink loss rate on N6 link
    N6_LOSS_RATE_UL = 0  # Uplink loss rate on N6 link

    # --- CAMF (Capacity Aware MASQUE Feedback) -----------------------------------
    CAMF_PERIOD_TI    = 20                              # Feedback period [time_index = ms]
    CAMF_THETA_RATIO  = 2.0 / 3.0                      # θ = CAMF_THETA_RATIO * SERVER_RTT (equivalent to TECC theta_ratio)
    CAMF_MAX_PF       = 0.5                             # Maximum penalty factor (equivalent to TECC max_pf)
    CAMF_EWMA_WEIGHT  = 0.2 # EWMA weight (equivalent to TECC ewma_weight)
    CAMF_SERVER_RTT   = (N3_DELAY - 1 + N6_DELAY_TI) * 2  # Server RTT estimate [ms] = (10+10)*2 = 40ms
    CAMF_FB_DELAY_TI  = 2 * (N3_DELAY - 1) + N6_DELAY_TI
    CAMF_RTT_CONG_THRESH = 1.1                             # RTT ratio threshold for switching from btlbw to delivery_rate of Rproxy estimation: 
    CAMF_Q_TARGET        = 200                              # Target queue length of UPF send buffer [pkts]

    # --- Rarely Changed Options ---------------------------
    RAN_FB_OPTION = (  # Feedback mode: receive periodic feedback from 6G RAN about desired transmission volume for next period
        "NONE"   # No feedback
        # "SINGLE"   # Feedback from a single RAN (e.g., 6G)
        #"BOTH"   # Feedback from both 5G/6G RANs
    )
    EXP_THPT_RANGE_SEC = 10 # Time window for measuring experienced throughput [seconds]
    RAN_FB_CYCLE = int(
        20  # Feedback cycle period from 6G RAN [unit: time_slot]
    )
    PACING_OPTION = (  # If True: pace out feedback-granted volume gradually instead of sending all at once
        # True
        False
    )
    UDP_MODE = (
        False  # Use QUIC (reliable transport)
        #True   # Use UDP (unreliable transport)
    )
    UDP_RATE = 400 * 1e6 # Transmission rate per UE in UDP mode [bps]. 1e6=Mb/s
    APP_LIMITED_OPTION = (
        #True  # Application-limited traffic mode
        False
    )
    TECC_FB_CYCLE = int(# [unit: time_slot]
        20
    )
    TUNNEL_RTT_CAP = 0.3  # Tunnel RTT upper limit [seconds]: upper limit of tunnel_rtt used for relaxing QUIC loss detection threshold

    # --- RAN Configurations ------------------------
    RAN_CONFIG = (
        "train0", "conf/large_2GHz_10UEs_60000step_25mps_train_0/", "conf/large_5GHz_10UEs_60000step_25mps_train_0/"
        #"train1", "conf/large_2GHz_10UEs_60000step_25mps_train_1/", "conf/large_5GHz_10UEs_60000step_25mps_train_1/"
        #"train2", "conf/large_2GHz_10UEs_60000step_25mps_train_2/", "conf/large_5GHz_10UEs_60000step_25mps_train_2/"
        #"train3", "conf/large_2GHz_10UEs_60000step_25mps_train_3/", "conf/large_5GHz_10UEs_60000step_25mps_train_3/"
        #"train4", "conf/large_2GHz_10UEs_60000step_25mps_train_4/", "conf/large_5GHz_10UEs_60000step_25mps_train_4/"
        #"train5", "conf/large_2GHz_10UEs_60000step_25mps_train_5/", "conf/large_5GHz_10UEs_60000step_25mps_train_5/"
        #"train6", "conf/large_2GHz_10UEs_60000step_25mps_train_6/", "conf/large_5GHz_10UEs_60000step_25mps_train_6/"
        #"train7", "conf/large_2GHz_10UEs_60000step_25mps_train_7/", "conf/large_5GHz_10UEs_60000step_25mps_train_7/"
        #"train8", "conf/large_2GHz_10UEs_60000step_25mps_train_8/", "conf/large_5GHz_10UEs_60000step_25mps_train_8/"
        #"train9", "conf/large_2GHz_10UEs_60000step_25mps_train_9/", "conf/large_5GHz_10UEs_60000step_25mps_train_9/"
        #"train10", "conf/large_2GHz_10UEs_60000step_25mps_train_10/", "conf/large_5GHz_10UEs_60000step_25mps_train_10/"
        #"train11", "conf/large_2GHz_10UEs_60000step_25mps_train_11/", "conf/large_5GHz_10UEs_60000step_25mps_train_11/"
        #"train12", "conf/large_2GHz_10UEs_60000step_25mps_train_12/", "conf/large_5GHz_10UEs_60000step_25mps_train_12/"
        #"train13", "conf/large_2GHz_10UEs_60000step_25mps_train_13/", "conf/large_5GHz_10UEs_60000step_25mps_train_13/"
        #"train14", "conf/large_2GHz_10UEs_60000step_25mps_train_14/", "conf/large_5GHz_10UEs_60000step_25mps_train_14/"
        #"train15", "conf/large_2GHz_10UEs_60000step_25mps_train_15/", "conf/large_5GHz_10UEs_60000step_25mps_train_15/"
        #"train16", "conf/large_2GHz_10UEs_60000step_25mps_train_16/", "conf/large_5GHz_10UEs_60000step_25mps_train_16/"
        #"train17", "conf/large_2GHz_10UEs_60000step_25mps_train_17/", "conf/large_5GHz_10UEs_60000step_25mps_train_17/"
        #"train18", "conf/large_2GHz_10UEs_60000step_25mps_train_18/", "conf/large_5GHz_10UEs_60000step_25mps_train_18/"
        #"train19", "conf/large_2GHz_10UEs_60000step_25mps_train_19/", "conf/large_5GHz_10UEs_60000step_25mps_train_19/"

        # Specific scenario
        # --- Fixed quality scenario ------------------------
        # Both RANs fixed (5bps/Hz)
        #"stable1", "conf/large_2GHz_10UEs_60000step_fix_5/", "conf/large_5GHz_10UEs_60000step_fix_5/"
        # --- Bottleneck adaptation evaluation scenario ------------------------
        # Total bandwidth lower bound: 5Mbps, upper bound: 36Mbps
        #"btleval", "conf/bottleneck_change_scenario_2GHz/", "conf/bottleneck_change_scenario_5GHz/"
    )

    # --- For Debugging ------------------------------------
    TXTLOG = []

    # --- Packet Structure ---------------------------------
    NUM_INFO_PACKET = 15  # Total number of fields per packet
    INDEX_PACKET_ID = 0
    INDEX_PAYLOAD_SIZE = 1
    INDEX_ue_id = 2
    INDEX_MAC_PACKET_ID = 3
    INDEX_PDCP_PACKET_ID = 4
    INDEX_SERVER_TIMESTAMP_ID = 5
    INDEX_UE_TIMESTAMP_ID = 6
    INDEX_RLC_INCOMING_TIMESTAMP_ID = 7
    INDEX_OUTER_PACKET_ID = 8  # Used for MPQUIC subflow identification
    INDEX_OUTER_ACK_FLAG = 9  # Flag to distinguish ACK packets for Inner or Outer layer
    INDEX_RLC_PACKET_ID = 10
    INDEX_STREAM_PACKET_ID = 11  # Stream-level packet ID (for MPQUIC)
    INDEX_UPF_TRANSMIT_TIMESTAMP = 12  # Timestamp when packet is sent from the UPF
    INDEX_CAMF_FB_ID = 13  # CAMF feedback ID stamped at server send (0 = not set)
    INDEX_UPF_ENQUEUE_TIMESTAMP = 14  # Timestamp when packet arrives at UPF N6_buffer

    # --- RAN Parameters -----------------------------------
    # Number of control packets tracked at each layer per UE; impacts table sizes in simulation
    NUM_CONTROL_PACKET_ON_MAC = 10000
    NUM_CONTROL_PACKET_ON_RLC = 10000
    NUM_CONTROL_PACKET_ON_PDCP = 10000
    NUM_CONTROL_TB_ON_MAC = 1000  # Number of TBs tracked per UE at MAC layer
    # Number of TBs per MAC packet (granularity of segmentation)
    MAX_NUM_TB_per_MAC_PACKET = 1000
    # Retrieve SINR every 5ms (continuous air interface sampling)
    NUM_CONTINUE_AIR = 1

    # --- L4 Parameters ------------------------------------
    MTU_SIZE = 1500  # Packet size [Bytes]
    MTU_BIT_SIZE = 12000  # Packet size [bits] (precomputed for efficiency)
    # IPv4 header size [Bytes] (usually 20 unless options are used)
    IPV4_HEADER_SIZE = 20
    IPV6_HEADER_SIZE = 40  # IPv6 header size [Bytes] (always 40)
    UDP_HEADER_SIZE = 8  # UDP header size [Bytes]
    QUIC_HEADER_SIZE = 12  # QUIC header size [Bytes]
    ACK_DELAY = 0.001  # Typical ACK delay [seconds] (e.g., 1ms)
    INIT_RET_TIMER = 1  # Initial retransmission timer [seconds]
    ACK_PAYLOAD = 16  # ACK payload size [Bytes] (simplified)

    # Max intentional ACK delay at receiver [seconds] (Chromium default: 25ms)
    MAX_ACK_DELAY = 0.025
    K_INITIAL_RTT = (
        MAX_ACK_DELAY  # Recommended value is 333ms, but reduced for plotting clarity
    )
    K_GRANULARITY = 0.001  # RTT granularity [seconds]
    K_TIME_THRESH = 9 / 8  # Loss detection time threshold multiplier
    K_PACKET_THRESH = (
        3  # Packet number threshold for considering loss (RFC recommended)
    )
    PTO_TRANSMIT_NUM = 2  # Number of packets sent on PTO timeout
    INIT_CWND = 10  # Initial congestion window (in packets)
    ROBUFF_TIMEOUT_TI = int(  # Timeout threshold for reordering buffer [time_index]
        #0.2 / TIME_SLOT_WINDOW  # Equivalent to 100 ms when TI = 1 ms
        NUM_SIMULATION_TIME_SLOTS  # Disable timeout-based loss detection in reordering buffer
    )
    STREAM_LEVEL_REORDERING_OPTION = (  # If True: enable stream-level reordering in MPQUIC
        True
        # False
    )

    # --- Default Buffer Sizes -----------------------------
    BUFF_MAX_VOLUME_DEFAULT = int(1e9)
    BUFF_MAX_LENGTH_DEFAULT = int(1e4)
    SVBUFF_MAX_LENGTH = int(1e4)
    TBBUFF_MAX_VOLUME = int(1e9)
    TBBUFF_MAX_LENGTH = int(1e2)
    ROBUFF_MAX_VOLUME = int(1e9)
    ROBUFF_MAX_LENGTH = int(1e4)
    # --- WiredLink Size -----------------------------------
    WIREDLINK_MAX_VOLUME = int(1e9)
    # WIREDLINK_MAX_LENGTH = int(1e6)
    WIREDLINK_MAX_LENGTH = int(1e4)
    # --- PDCP without DC Buffer Size ----------------------
    BUFF_MAX_VOLUME_PDCP_WODC_DL_DEFAULT = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_DL_DEFAULT = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WODC_UL_DEFAULT = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_UL_DEFAULT = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WODC_MN_DL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_MN_DL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WODC_MN_UL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_MN_UL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WODC_SN_DL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_SN_DL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WODC_SN_UL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WODC_SN_UL = int(1e3)
    # --- PDCP with DC Buffer Size -------------------------
    BUFF_MAX_VOLUME_PDCP_WITHDC_DL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WITHDC_DL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WITHDC_MN_DL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WITHDC_MN_DL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WITHDC_SN_DL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WITHDC_SN_DL = int(1e3)
    BUFF_MAX_VOLUME_PDCP_WITHDC_UL = int(1e9)
    BUFF_MAX_LENGTH_PDCP_WITHDC_UL = int(1e3)
    # --- RLC Buffer Size ----------------------------------
    BUFF_MAX_VOLUME_RLC_DL = int(1e9)
    BUFF_MAX_LENGTH_RLC_DL = int(1e3)
    BUFF_MAX_VOLUME_RLC_UL = int(1e9)
    BUFF_MAX_LENGTH_RLC_UL = int(1e3)
    # --- MAC Buffer Size ----------------------------------
    BUFF_MAX_VOLUME_MAC_DL = int(1e9)
    BUFF_MAX_LENGTH_MAC_DL = int(200)
    BUFF_MAX_VOLUME_MAC_UL = int(1e9)
    BUFF_MAX_LENGTH_MAC_UL = int(200)
    # --- MPQUIC Buffer Size -------------------------------
    BUFF_MAX_VOLUME_MPQUIC_SEND_DL = int(1e9)
    BUFF_MAX_LENGTH_MPQUIC_SEND_DL = int(1e3)
    BUFF_MAX_VOLUME_MPQUIC_RESEND_DL = int(1e9)
    BUFF_MAX_LENGTH_MPQUIC_RESEND_DL = int(1e3)
    BUFF_MAX_VOLUME_MPQUIC_RECV_UL = int(1e9)
    BUFF_MAX_LENGTH_MPQUIC_RECV_UL = int(1e3)
    # --- MPQUIC / QUIC Flow Control Window ----------------
    MPQUIC_FC_RECV_WINDOW = int(1e3)  # 1000 pkts
    QUIC_FC_RECV_WINDOW   = int(1e7)  # 1e7[pkts]=15GB. Effectively infinite
    # --- N6 Buffer Size -----------------------------------
    BUFF_MAX_VOLUME_N6_DL = int(1e9)
    #BUFF_MAX_LENGTH_N6_DL = int(1e2)
    MAX_QUEUEING_DELAY_TI = 200  # Maximum allowable queuing delay at N6 [time_index = ms]
    BUFF_MAX_LENGTH_N6_DL = int(
        #1e5
        #1e5 / 8
        N6_BANDWIDTH_BPS / 8 / MTU_SIZE * MAX_QUEUEING_DELAY_TI * TIME_SLOT_WINDOW
    )
    BUFF_MAX_VOLUME_N6_UL = int(1e9)
    BUFF_MAX_LENGTH_N6_UL = int(1e3)

    # --- Paths --------------------------------------------
    CONF_STR, PROPAGATION_LOAD_PATH_LB, PROPAGATION_LOAD_PATH_HB = RAN_CONFIG
    # Directory for storing large data (gitignored)
    HEAVY_DATA_PATH = "../heavy_data/"
    LOG_SAVE_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
    if RAN_FB_OPTION == "NONE":
        fb_option_str = ""
    elif RAN_FB_OPTION == "SINGLE":
        fb_option_str = "_PROPOSED"
    elif RAN_FB_OPTION == "BOTH":
        fb_option_str = "_IDEAL"

    if CAMF_OPTION:
        camf_option_str = "_CAMF=on"
    else:
        camf_option_str = ""

    if CAMF_NOCAP_OPTION:
        camf_nocap_str = "_NOCAP"
    else:
        camf_nocap_str = ""
    
    if TECC_OPTION == True:
        tecc_fb_option_str = "_TECC=on"
    else:
        tecc_fb_option_str = ""

    bwchange_str = f"_BWchange{len(N6_BANDWIDTH_SCHEDULE)}" if N6_BANDWIDTH_SCHEDULE else ""

    # LOG_SAVE_PATH = f"{LOG_SAVE_ID}_ue={NUM_UE}_slot={NUM_SIMULATION_TIME_SLOTS}_D={N3_DELAY-1}_{'UDP' if UDP_MODE else 'QUIC'}{f"{int(UDP_RATE/1000000)}M" if UDP_MODE else ''}_5G={STR_5G}_6G={STR_6G}_{f'UDPon{MPQUIC_CC[:3]}' if UDP_MODE else f'{QUIC_CC[:3]}on{MPQUIC_CC[:3]}'}{fb_option_str}/"
    
    def _fmt_bps(bps: float, decimals: int = 3) -> str:
        units = [
            (1_000_000_000, "G"),
            (1_000_000, "M"),
            (1_000, "K"),
        ]

        for base, suffix in units:
            if bps >= base:
                v = bps / base
                s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
                return f"{s}{suffix}"

        if float(bps).is_integer():
            return str(int(bps))
        return str(bps)
    
    LOG_SAVE_PATH = (
        f"{LOG_SAVE_ID}_{NUM_UE}UE_{NUM_SIMULATION_TIME_SLOTS*TIME_SLOT_WINDOW}[s]_"
        f"N3={N3_DELAY-1}ms_{N3_LOSS_RATE_DL}_{_fmt_bps(N3_BANDWIDTH_BPS)}_"
        f"N6={N6_DELAY_TI}ms_{N6_LOSS_RATE_DL}_{_fmt_bps(N6_BANDWIDTH_BPS)}_"
        f"{CONF_STR}_"
        f"{f'UDPon{MPQUIC_CC[:3]}' if UDP_MODE else f'{QUIC_CC[:3]}on{MPQUIC_CC[:3]}'}"
        f"{bwchange_str}"
        f"{'_' + _fmt_bps(UDP_RATE) if UDP_MODE else ''}"
        f"{'_FCT=' + str(FCT_TOTAL_PACKETS) if FCT_MODE else ''}"
        f"_{SCHEDULER_MODE}"
        f"{tecc_fb_option_str}"
        f"{fb_option_str}"
        f"{camf_option_str}"
        f"{camf_nocap_str}"
        f"/"
    )


class TimeManager:
    # Global simulation time index, shared across modules
    time_index = 0
