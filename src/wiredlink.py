import numpy as np
from param import ParameterClass, TimeManager
import time


class WiredLink(ParameterClass, TimeManager):
    def __init__(self, packet_per_timeslot=10000, bits_per_timeslot=1e6):
        self.volume = 0  # Amount of data currently queued on the link (bits)
        self.length = 0  # Number of packets currently queued on the link
        # Maximum allowed queued data on the link (bits)
        self.max_volume = ParameterClass.WIREDLINK_MAX_VOLUME
        # Maximum allowed queued packets on the link
        self.max_length = ParameterClass.WIREDLINK_MAX_LENGTH
        # Link delay in time slots (unit: time_index)
        self.delay = self.N3_DELAY
        self.packet_per_timeslot = int(
            packet_per_timeslot
        )  # Max number of packets the link can process per time slot
        self.bits_per_timeslot = int(bits_per_timeslot)
        # [packet info fields]
        self.stay_packets = np.zeros(
            [self.max_length, self.NUM_INFO_PACKET], dtype=int)
        self.stay_times = np.zeros(
            [self.max_length], dtype=int
        )  # Residence time on the link, measured in time slots

    def dequeue(self):
        length_dequeued_packets_delay = np.sum(
            self.stay_times[: self.length] >= self.delay
        )

        cumulative_sum = np.cumsum(
            self.stay_packets[: self.length, self.INDEX_PAYLOAD_SIZE]
        )  # Cumulative sum of payload sizes
        length_dequeued_packets_thp = np.sum(
            cumulative_sum < self.bits_per_timeslot)
        length_dequeued_packets = np.min(
            [
                length_dequeued_packets_delay,
                length_dequeued_packets_thp,
                self.packet_per_timeslot,
                self.length,
            ]
        )
        length_dequeued_packets = np.max([length_dequeued_packets, 0])

        if length_dequeued_packets > 0:

            dequeued_packet = np.copy(
                self.stay_packets[:length_dequeued_packets])
            self.stay_packets[:length_dequeued_packets] = np.zeros_like(
                self.stay_packets[:length_dequeued_packets]
            )
            self.stay_times[:length_dequeued_packets] = np.zeros_like(
                self.stay_times[:length_dequeued_packets]
            )

            self.stay_packets[: self.length - length_dequeued_packets] = (
                self.stay_packets[length_dequeued_packets: self.length]
            )
            self.stay_packets[self.length - length_dequeued_packets: self.length] = (
                np.zeros_like(
                    self.stay_packets[
                        self.length - length_dequeued_packets: self.length
                    ]
                )
            )
            self.stay_times[: self.length - length_dequeued_packets] = self.stay_times[
                length_dequeued_packets: self.length
            ]

            self.stay_times[self.length - length_dequeued_packets: self.length] = (
                np.zeros(length_dequeued_packets, dtype=int)
            )

            self.length -= length_dequeued_packets
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])

        else:
            dequeued_packet = np.zeros([0, self.NUM_INFO_PACKET], dtype=int)

        return dequeued_packet

    def do_timeslot(self):
        array = np.zeros(self.max_length, dtype=int)
        array[: self.length] = 1
        self.stay_times = self.stay_times + array

    def enqueue(self, packets):
        """packets is an ndarray, shape = [num_packets, NUM_INFO_PACKET].
        Packets that do not fit the queue are dropped (tail-drop)."""
        if packets is None or len(packets) == 0:
            return 0

        length_enqueued_packet = len(packets)
        volume_enqueued_packet = np.sum(packets[:, self.INDEX_PAYLOAD_SIZE])
        if length_enqueued_packet + self.length > self.max_length:
            packets = packets[: self.max_length - self.length]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(
                packets[:, self.INDEX_PAYLOAD_SIZE])

        if volume_enqueued_packet + self.volume > self.max_volume:
            cumulative_sum = np.cumsum(
                packets[:, self.INDEX_PAYLOAD_SIZE]
            )  # Cumulative sum of payload sizes
            length_packets_canbe_enqueued = np.sum(
                cumulative_sum < self.bits_per_timeslot
            )
            packets = packets[:length_packets_canbe_enqueued]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(
                packets[:, self.INDEX_PAYLOAD_SIZE])

        self.stay_packets[self.length: length_enqueued_packet +
                          self.length] = packets

        self.length += length_enqueued_packet
        self.volume += volume_enqueued_packet

    def return_queue_statics(self):
        return self.length, self.volume

    def return_queue_content(self):
        return self.stay_packets
