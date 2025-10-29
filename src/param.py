import numpy as np
from datetime import datetime

np.random.seed(0)


class ParameterClass:
    # Class variables (shared across all instances)
    # --- Basic Simulation Parameters ----------------------
    NUM_UE = 10  # Number of User Equipments (UEs)
    NUM_SIMULATION_TIME_SLOTS = 30000  # Total number of simulation time slots
    TIME_SLOT_WINDOW = 0.001  # Duration of a single time slot [seconds]
    SIM_MODE = (
        # "DC"  # Dual Connectivity mode
        "MPQUIC"  # Multipath QUIC mode
    )
    RAN_FB_OPTION = (  # Feedback mode: receive periodic feedback from 6G RAN about desired transmission volume for next period
        # "NONE"   # No feedback
        # "SINGLE"   # Feedback from a single RAN (e.g., 6G)
        "BOTH"   # Feedback from both 5G/6G RANs
    )
    PACING_OPTION = (  # If True: pace out feedback-granted volume gradually instead of sending all at once
        # True
        False
    )
    UDP_MODE = (
        # False  # Use QUIC (reliable transport)
        True   # Use UDP (unreliable transport)
    )
    N3_DELAY = int(  # Delay applied on N3 link [unit: time_index]. Actual delay = (N3_DELAY - 1) * time_slot.
        0.01 / TIME_SLOT_WINDOW + 1  # Ensures fixed delay of 10ms regardless of slot duration
    )
    QUIC_CC = (
        "CUBIC"  # Congestion control algorithm for QUIC layer
    )
    MPQUIC_CC = (
        "CUBIC"  # Congestion control algorithm for MPQUIC outer layer
    )
    # Transmission rate per UE in UDP mode [bps]. 1e6=Mb/s
    UDP_RATE = 400 * 1e6
    N6_DELAY_TI = 0  # N6 link delay [time_index]
    N6_LOSS_RATE_DL = 0  # Downlink loss rate on N6 link
    N6_LOSS_RATE_UL = 0  # Uplink loss rate on N6 link
    # Time window for measuring experienced throughput [seconds]
    EXP_THPT_RANGE_SEC = 10
    RAN_FB_CYCLE = int(
        20  # Feedback cycle period from 6G RAN [unit: time_slot]
    )
    APP_LIMITED_OPTION = (
        # True  # Application-limited traffic mode
        False
    )
    # 5G Radio Wave Propagation Configuration
    RAN_CONFIG_5G = (
        # "walk0", "conf/large_2GHz_10UEs_60000step_1-5mps_walk_0/"
        # "car0", "conf/large_2GHz_10UEs_60000step_15mps_car_0/"
        "train0", "conf/large_2GHz_10UEs_60000step_25mps_train_0/"
    )
    # 5G Radio Wave Propagation Configuration
    RAN_CONFIG_6G = (
        # "walk0", "conf/large_5GHz_10UEs_60000step_1-5mps_walk_0/"
        # "car0", "conf/large_5GHz_10UEs_60000step_15mps_car_0/"
        "train0", "conf/large_5GHz_10UEs_60000step_25mps_train_0/"
    )

    # --- For Debugging ------------------------------------
    TXTLOG = []

    # --- Packet Structure ---------------------------------
    NUM_INFO_PACKET = 13  # Total number of fields per packet
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

    # --- RAN Parameters -----------------------------------
    # Number of control packets tracked at each layer per UE; impacts table sizes in simulation
    NUM_CONTROL_PACKET_ON_MAC = 10000
    NUM_CONTROL_PACKET_ON_RLC = 10000
    NUM_CONTROL_PACKET_ON_PDCP = 10000
    NUM_CONTROL_TB_ON_MAC = 1000  # Number of TBs tracked per UE at MAC layer
    # Number of TBs per MAC packet (granularity of segmentation)
    MAX_NUM_TB_per_MAC_PACKET = 250
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
    PROPOSED_SOLUTION_FACTOR = (
        0.8  # Scale factor applied to cwnd when coordinating with MPQUIC cwnd
    )
    ROBUFF_TIMEOUT_TI = int(  # Timeout threshold for reordering buffer [time_index]
        0.2 / TIME_SLOT_WINDOW  # Equivalent to 200 ms when TI = 1 ms
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
    # --- N6 Buffer Size -----------------------------------
    BUFF_MAX_VOLUME_N6_DL = int(1e9)
    BUFF_MAX_LENGTH_N6_DL = int(1e3)
    BUFF_MAX_VOLUME_N6_UL = int(1e9)
    BUFF_MAX_LENGTH_N6_UL = int(1e3)

    # --- Paths --------------------------------------------
    STR_5G, PROPAGATION_LOAD_PATH_LB = RAN_CONFIG_5G
    STR_6G, PROPAGATION_LOAD_PATH_HB = RAN_CONFIG_6G
    # Directory for storing large data (gitignored)
    HEAVY_DATA_PATH = "../heavy_data/"
    LOG_SAVE_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
    if RAN_FB_OPTION == "NONE":
        fb_option_str = ""
    elif RAN_FB_OPTION == "SINGLE":
        fb_option_str = "_PROPOSED"
    elif RAN_FB_OPTION == "BOTH":
        fb_option_str = "_IDEAL"
    # LOG_SAVE_PATH = f"{LOG_SAVE_ID}_ue={NUM_UE}_slot={NUM_SIMULATION_TIME_SLOTS}_D={N3_DELAY-1}_{'UDP' if UDP_MODE else 'QUIC'}{f"{int(UDP_RATE/1000000)}M" if UDP_MODE else ''}_5G={STR_5G}_6G={STR_6G}_{f'UDPon{MPQUIC_CC[:3]}' if UDP_MODE else f'{QUIC_CC[:3]}on{MPQUIC_CC[:3]}'}{fb_option_str}/"
    LOG_SAVE_PATH = (
        f"{LOG_SAVE_ID}_ue={NUM_UE}_slot={NUM_SIMULATION_TIME_SLOTS}_D={N3_DELAY-1}_"
        f"{'UDP' if UDP_MODE else 'QUIC'}_"
        f"{str(int(UDP_RATE/1000000)) + 'M' if UDP_MODE else ''}_"
        f"5G={STR_5G}_6G={STR_6G}_"
        f"{f'UDPon{MPQUIC_CC[:3]}' if UDP_MODE else f'{QUIC_CC[:3]}on{MPQUIC_CC[:3]}'}"
        f"{fb_option_str}/"
    )


class TimeManager:
    # Global simulation time index, shared across modules
    time_index = 0
