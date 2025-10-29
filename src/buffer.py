import numpy as np
from param import ParameterClass, TimeManager


class Buffer(ParameterClass, TimeManager):
    def __init__(self, max_volume=ParameterClass.BUFF_MAX_VOLUME_DEFAULT, max_length=ParameterClass.BUFF_MAX_LENGTH_DEFAULT):
        self.volume = 0  # Current total bits stored in the buffer
        self.length = 0  # Current number of packets in the buffer
        self.max_volume = max_volume  # Maximum buffer capacity in bits
        # Maximum number of packets that can be stored
        self.max_length = int(max_length)
        # Each row = [packet id, packet size, ...] (NUM_INFO_PACKET fields)
        self.content = np.zeros(
            [self.max_length, self.NUM_INFO_PACKET], dtype=int)

    def enqueue(self, packets):
        """
        Add incoming packets to the buffer.
        packets: ndarray, shape = [num_packets, NUM_INFO_PACKET].
        If buffer capacity is exceeded, extra packets are dropped.
        """
        if packets is None:
            return

        length_enqueued_packet = len(packets)
        if length_enqueued_packet == 0:
            return

        volume_enqueued_packet = np.sum(packets[:, self.INDEX_PAYLOAD_SIZE])

        # If adding packets exceeds max_length, drop the excess
        if length_enqueued_packet + self.length > self.max_length:
            packets = packets[: self.max_length - self.length]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(
                packets[:, self.INDEX_PAYLOAD_SIZE])

        # If adding packets exceeds max_volume, truncate by cumulative sum
        if volume_enqueued_packet + self.volume > self.max_volume:
            # Compute cumulative sum of sizes
            cumulative_sum = np.cumsum(
                packets[:, self.INDEX_PAYLOAD_SIZE]
            )
            length_packets_canbe_enqueued = np.sum(
                cumulative_sum < self.bits_per_timeslot
            )
            packets = packets[:length_packets_canbe_enqueued]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(
                packets[:, self.INDEX_PAYLOAD_SIZE])

        self.content[self.length: length_enqueued_packet +
                     self.length] = packets
        self.length += length_enqueued_packet
        self.volume += volume_enqueued_packet

    def dequeue(self, dequeue_type="all", length=0, volume=0):
        """
        Remove and return packets from the buffer.
        dequeue_type can be:
            - "all": remove all packets
            - "length": remove a fixed number of packets
            - "volume": remove packets until reaching specified bit volume
        Returns: ndarray of dequeued packets.
        """
        if dequeue_type == "all":
            dequeued_packet = self.content[: self.length]
            self.length = 0
            self.volume = 0
            self.content = np.zeros_like(self.content)
            return dequeued_packet

        if dequeue_type == "length":
            if length >= self.length:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(self.NUM_INFO_PACKET):
                self.content[:, i] = np.roll(self.content[:, i], -length)

            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])
            return dequeued_packet

        if dequeue_type == "volume":
            if volume >= self.volume:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            cumulative_sum = np.cumsum(
                self.content[:, self.INDEX_PAYLOAD_SIZE])
            length = np.min([np.sum(cumulative_sum < volume) + 1, self.length])
            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(self.NUM_INFO_PACKET):
                self.content[:, i] = np.roll(self.content[:, i], -length)
            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])
            return dequeued_packet

    def return_queue_statics(self):
        return self.length, self.volume

    def return_queue_content(self):
        return self.content


class SaveBuffer(ParameterClass, TimeManager):
    def __init__(self, savefilename, max_length=ParameterClass.SVBUFF_MAX_LENGTH):
        self.volume = 0  # Current total bits stored
        self.length = 0  # Current number of packets stored
        self.max_length = int(max_length)
        self.content = np.zeros(
            [self.max_length, self.NUM_INFO_PACKET], dtype=int)
        self.save_count = 0
        self.savefilename = savefilename

    def enqueue(self, packets):
        length_enqueued_packet = len(packets)
        if length_enqueued_packet == 0:
            return
        packets[:, self.INDEX_UE_TIMESTAMP_ID] = self.time_index

        volume_enqueued_packet = np.sum(packets[:, self.INDEX_PAYLOAD_SIZE])
        if length_enqueued_packet + self.length > self.max_length:
            packet_splitting_position = self.max_length - self.length
            self.content[self.length:] = packets[:packet_splitting_position]
            self.save()
            self.dequeue()
            self.save_count += 1

            packets = packets[packet_splitting_position:]
            while len(packets) > self.max_length:
                packet_splitting_position = self.max_length
                self.content = packets[:packet_splitting_position]
                packets = packets[packet_splitting_position:]
                self.save()
                self.dequeue()
                self.save_count += 1

            self.length = len(packets)
            self.volume = np.sum(packets[:, self.INDEX_PAYLOAD_SIZE])
            self.content[: self.length] = packets
        else:
            self.content[self.length: length_enqueued_packet +
                         self.length] = packets
            self.length += length_enqueued_packet
            self.volume += volume_enqueued_packet

    def save(self):
        """Save current buffer content to .npy file."""
        definite_filepath = self.savefilename + str(self.save_count) + ".npy"
        np.save(definite_filepath, self.content[: self.length])

    def dequeue(self, dequeue_type="all", length=0, volume=0):
        if dequeue_type == "all":
            dequeued_packet = self.content[: self.length]
            self.length = 0
            self.volume = 0
            self.content = np.zeros_like(self.content)
            return dequeued_packet

        if dequeue_type == "length":
            if length >= self.length:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(self.NUM_INFO_PACKET):
                self.content[:, i] = np.roll(self.content[:, i], -length)

            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])
            return dequeued_packet

        if dequeue_type == "volume":
            if volume >= self.volume:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            cumulative_sum = np.cumsum(
                self.content[:, self.INDEX_PAYLOAD_SIZE])
            length = np.min([np.sum(cumulative_sum < volume) + 1, self.length])
            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(self.NUM_INFO_PACKET):
                self.content[:, i] = np.roll(self.content[:, i], -length)
            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])
            return dequeued_packet

    def return_queue_statics(self):
        return self.length, self.volume

    def return_queue_content(self):
        return self.content


class TBBuffer(ParameterClass, TimeManager):
    def __init__(self):
        self.volume = 0  # Current total bits
        self.length = 0  # Current number of packets
        self.max_length = ParameterClass.TBBUFF_MAX_LENGTH
        self.max_volume = ParameterClass.TBBUFF_MAX_VOLUME
        # Each row = [TB id, size]
        self.content = np.zeros([self.max_length, 2], dtype=int)

    def enqueue(self, packets):
        """
        Add TBs (transport blocks) to buffer. Excess packets are dropped.
        """
        length_enqueued_packet = len(packets)
        volume_enqueued_packet = np.sum(packets[:, 1])
        if length_enqueued_packet + self.length > self.max_length:
            packets = packets[: self.max_length - self.length]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(packets[:, 1])

        if volume_enqueued_packet + self.volume > self.max_volume:
            cumulative_sum = np.cumsum(packets[:, 1])
            length_packets_canbe_enqueued = np.sum(
                cumulative_sum < self.bits_per_timeslot)
            packets = packets[:length_packets_canbe_enqueued]
            length_enqueued_packet = len(packets)
            volume_enqueued_packet = np.sum(packets[:, 1])

        self.content[self.length: length_enqueued_packet +
                     self.length] = packets
        self.length += length_enqueued_packet
        self.volume += volume_enqueued_packet

    def dequeue(self, dequeue_type="all", length=0, volume=0):
        """
        Remove TBs from buffer. Similar to Buffer.dequeue().
        """
        if dequeue_type == "all":
            dequeued_packet = self.content[: self.length]
            self.length = 0
            self.volume = 0
            self.content = np.zeros_like(self.content)
            return dequeued_packet

        if dequeue_type == "length":
            if length >= self.length:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(2):
                self.content[:, i] = np.roll(self.content[:, i], -length)

            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, 1])
            return dequeued_packet

        if dequeue_type == "volume":
            if volume >= self.volume:
                dequeued_packet = self.content[: self.length]
                self.length = 0
                self.volume = 0
                self.content = np.zeros_like(self.content)
                return dequeued_packet

            cumulative_sum = np.cumsum(self.content[:, 1])
            length = np.min([np.sum(cumulative_sum < volume) + 1, self.length])
            dequeued_packet = np.copy(self.content[:length])
            self.content[:length] = np.zeros_like(self.content[:length])

            for i in range(2):
                self.content[:, i] = np.roll(self.content[:, i], -length)
            self.length -= length
            self.volume -= np.sum(dequeued_packet[:, self.INDEX_PAYLOAD_SIZE])

            return dequeued_packet

    def return_queue_statics(self):
        return self.length, self.volume

    def return_queue_content(self):
        return self.content


class ReorderingBuffer(ParameterClass, TimeManager):
    """
    Buffer used in MPQUIC for out-of-order packet handling (Outer L4 -> Inner L4).
    No need to specify dequeue length manually; packets are released to the upper
    layer as soon as a contiguous sequence becomes available.
    """

    def __init__(self, max_volume=ParameterClass.ROBUFF_MAX_VOLUME, max_length=ParameterClass.ROBUFF_MAX_LENGTH, ue_id=None, logger=None):
        self.max_volume = max_volume
        self.max_length = max_length
        self.length = 0
        self.volume = 0
        self.content = np.zeros(
            [self.max_length, self.NUM_INFO_PACKET], dtype=int)
        self.expected_packet_id = 0  # Next expected packet ID for in-order delivery
        self.missing_since = None  # Time when a gap was first detected
        self.ue_id = ue_id
        self.logger = logger
        self.buff_length_thresh = int(self.max_length * 0.9)

    def _detect_timeout(self):
        """
        Missing packet detection with timeout mechanism.
        If timeout expires, advance expected_packet_id to the smallest
        packet_id present in the buffer greater than the current expected.
        This prevents head-of-line blocking.
        """
        # If buffer is vacant, there are no missing packets.
        if self.length == 0:
            self.missing_since = None
            return

        # If expected_id is in buffer, assume that there are no missing packets.
        ids = self.content[:self.length, self.INDEX_STREAM_PACKET_ID]
        if self.expected_packet_id in ids:
            self.missing_since = None
            return

        # In other cases, there are missing packets.
        now = self.time_index
        if self.missing_since is None:
            # If timer has not yet been triggerd, trigger it.
            self.missing_since = now
            return

        if now - self.missing_since < self.ROBUFF_TIMEOUT_TI:
            # timeout is not yet.
            return

        # when timeout is detected
        # set expected_id to minimum packet_id in buff
        # next_candidates is list of packet_ids more than expected_id. Taking the minimum ID value directly risks the pointer backtracking due to late-arriving packets, hence this approach.
        self.logger.store("MPQUIC", f"UE{self.ue_id}", "RO_buff_event_log",
                          f"time={TimeManager.time_index}: timeout happens.")
        next_candidates = ids[ids > self.expected_packet_id]
        if next_candidates.size:
            self.expected_packet_id = int(next_candidates.min())
        self.missing_since = now

    def enqueue(self, packets):
        if packets is None or len(packets) == 0:
            self._detect_timeout()
            return

        for packet in packets:
            pkt_id = packet[self.INDEX_STREAM_PACKET_ID]
            pkt_size = packet[self.INDEX_PAYLOAD_SIZE]

            # Duplicate check with packets in the buffer (duplicates discarded)
            if pkt_id in self.content[:self.length, self.INDEX_STREAM_PACKET_ID]:
                self.logger.store("MPQUIC", f"UE{self.ue_id}", "RO_buff_loss_log",
                                            f"time={TimeManager.time_index}: {packet} [duplicate]")
                continue
            # Capacity Check (Discard if Over)
            if (self.length >= self.max_length or
                    self.volume + pkt_size > self.max_volume):
                self.logger.store("MPQUIC", f"UE{self.ue_id}", "RO_buff_loss_log",
                                            f"time={TimeManager.time_index}: {packet} [overflow]")
                continue

            # enqueue
            self.content[self.length] = packet
            self.length += 1
            self.volume += pkt_size

        # timeout prosessing after receiving
        self._detect_timeout()

        # Process to forcefully flush the buffer down to the threshold if it exceeds the threshold
        if self.length > self.buff_length_thresh:
            ids = self.content[:self.length, self.INDEX_STREAM_PACKET_ID]
            sorted_ids = np.sort(ids)
            index = self.length - self.buff_length_thresh
            new_expected = sorted_ids[index] + 1

            self.logger.store("MPQUIC", f"UE{self.ue_id}", "RO_buff_event_log",
                              f"time={TimeManager.time_index}: buffer over threshold ({self.length}). advance expected_packet_id from {self.expected_packet_id} to {new_expected}")

            self.expected_packet_id = new_expected

    def dequeue(self):
        """
        Sort buffer by packet_id, count contiguous packets starting from expected_packet_id,
        deliver them upward, and remove them from buffer.
        """
        if self.length == 0:
            return np.empty((0, self.NUM_INFO_PACKET), dtype=int)

        buf = self.content[:self.length]

        # 1) Packets smaller than expected_packet_id ---------------------------------
        # early_mask is an array of boolean values. e.g. early_mask = [True, False, ...]
        early_mask = buf[:,
                         self.INDEX_STREAM_PACKET_ID] < self.expected_packet_id
        early_packets = buf[early_mask]
        early_packets = early_packets[np.argsort(
            early_packets[:, self.INDEX_STREAM_PACKET_ID], kind="mergesort")]

        # 2) Packets equal to or larger than expected ---------------------------------------
        # ~early_mask is bit-inverted. True and False are swapped. This is retrieved using buf[].
        later_buf = buf[~early_mask]
        if later_buf.size == 0:
            # Only early_packets is passed to the upper layer
            self.length = 0
            self.volume = 0
            return early_packets.copy()

        # Sort IDs in ascending order and count consecutive intervals
        sort_idx = np.argsort(
            later_buf[:, self.INDEX_STREAM_PACKET_ID], kind="mergesort")
        later_buf = later_buf[sort_idx]

        cont = 0
        for pkt in later_buf:
            if pkt[self.INDEX_STREAM_PACKET_ID] == self.expected_packet_id + cont:
                cont += 1
            else:
                break

        contiguous_later = later_buf[:cont]
        remain = later_buf[cont:]

        # 3) Output and buffer update ----------------------------------------------
        dequeued = np.concatenate((early_packets, contiguous_later), axis=0)
        # Repack the buffer (leave only remain)
        self.length = len(remain)
        self.volume = int(
            np.sum(remain[:, self.INDEX_PAYLOAD_SIZE])) if self.length else 0
        self.content[:self.length] = remain
        self.expected_packet_id += cont

        goodput = len(dequeued) * ParameterClass.MTU_SIZE * \
            8 / ParameterClass.TIME_SLOT_WINDOW
        self.logger.store(
            "MPQUIC", f"UE{self.ue_id}", "MPQUIC-level_goodput", [TimeManager.time_index, goodput])

        return dequeued
