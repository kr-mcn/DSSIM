from enum import Enum
from dataclasses import dataclass

#-----------------------------------------------------------#
# Definition of BBR States                                  #
#-----------------------------------------------------------#

# Defining states
class BBRState(Enum):
    STARTUP = 1
    DRAIN = 2
    PROBEBW = 3
    PROBERTT = 4

#-----------------------------------------------------------#
# Definition of BBRs internal Packet Structure              #
#-----------------------------------------------------------#
    
# Defining data structure for bookkeeping of packets inflight
@dataclass
class PacketData:
    id: int
    delivered: int
    send_time: int
    size: int
    inflight: bool
    is_app_limited: bool
    is_retransmitted: bool
    ue_id : int

#-----------------------------------------------------------#
# Definition of BBRs internal Packet Manager                #
#-----------------------------------------------------------#

# Defining data structure for packet management (inflight data)
class PacketManager:

    # Initialize the empty dict
    def __init__(self):

        # seq_num -> PacketData
        self.packets = {}  

    # Add a packet or a list of packets to the dict
    def add_packet(self, seq_num, send_time, delivered, size, inflight, is_app_limited=False, is_retransmitted=False, ue_id=None):
        
        # Add a single packet to the dict
        self.packets[seq_num] = PacketData(
            id=seq_num,
            delivered=delivered,
            send_time=send_time,
            size=size,
            inflight=inflight,
            is_app_limited=is_app_limited,
            is_retransmitted=is_retransmitted,
            ue_id=ue_id
        )

    # Delete a single packet from the dict
    def delete_packet(self, seq_num):
        self.packets.pop(seq_num, None)

    # Get packet information safely
    def get_packet_info(self, seq_num):
        return self.packets.get(seq_num)
    
    # Get number of data inflight
    def inflight(self):
        result = sum(1 for packet in self.packets.values() if packet.inflight)
        return result
    
#-----------------------------------------------------------#
# Windowed Filter for BtlBw Estimation                      #
#-----------------------------------------------------------#
    
# Defining max windowed filter for BtlBw estimation
class WindowedFilter:

    # Initializing filter
    def __init__(self):
        self.window = []

    # Add value to filter
    def add_value(self, value, time, window_length):
        life_time = time + window_length
        self.window.append((value, time, window_length, life_time))

    # Remove old values with filter
    def remove_old_values(self, time):
        for i in range(len(self.window) - 1, -1, -1):
            if self.window[i][3] <= time: 
                self.window.pop(i)

    # Get max value
    def get_max(self):
        if not self.window:
            return 0.0
        return max(sample[0] for sample in self.window)