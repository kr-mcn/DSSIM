from param import ParameterClass, TimeManager
from dataclasses import dataclass, field


def get_time():
    time = round(TimeManager.time_index * ParameterClass.TIME_SLOT_WINDOW, 6)
    return time


@dataclass
class Packet:
    packet_number: int
    time_sent: int = field(default_factory=get_time)
    in_flight: bool = False
    sent_bits: int = 0
    retransmission: bool = False
    inner_packet_number: int = -1  # mpquic用


class Sent_packets(dict):  # Packets already sent.
    def __getitem__(self, packet_number):
        if packet_number not in self:
            self[packet_number] = Packet(packet_number=packet_number)
            # raise KeyError(f"Packet number {packet_number} not found.")
        return super().__getitem__(packet_number)
