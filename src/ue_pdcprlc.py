import numpy as np
from param import ParameterClass, TimeManager


def light_rotate(array, count, largest_index, vacant_value=0):
    """
    input: array[N, M] or array[N] (e.g., array = [a_0,..., a_largest_index, 0,..., 0])
    output: array[N, M] or array[N]
    Shift elements in `array` forward by `count`, and fill the trailing region with `vacant_value`.
    For indices > largest_index in the input array, treat those elements as `vacant_value`.
    """
    if count == largest_index + 1:
        if array.ndim == 2:
            array[: largest_index +
                  1] = np.full_like(array[: largest_index + 1], vacant_value)
        if array.ndim == 1:
            array[: largest_index +
                  1] = np.full(len(array[: largest_index + 1]), vacant_value)
        return array
    array[: largest_index + 1 - count] = array[count: largest_index + 1]
    if array.ndim == 2:
        array[largest_index + 1 - count: largest_index + 1] = np.full_like(
            array[largest_index + 1 - count: largest_index + 1], vacant_value
        )
    if array.ndim == 1:
        array[largest_index + 1 - count: largest_index + 1] = np.full(
            len(array[largest_index + 1 - count: largest_index + 1]), vacant_value
        )

    return array


class PDCP_RLC_UE(ParameterClass, TimeManager):
    def __init__(self, ue_id, ran_name, class_type="rlc"):
        # Instance variables (per-instance)
        self.name = "ran_name:", ran_name, " PDCP_RLC_UE_class_" + \
            class_type + "_UEid_" + str(ue_id)
        self.ran_name = ran_name

        self.ue_id = ue_id
        self.class_type = class_type
        if class_type == "rlc":
            self.mother_number = self.NUM_CONTROL_PACKET_ON_RLC
            self.index_packet_id = self.INDEX_RLC_PACKET_ID
            self.waiting_count = 0
            # Give up after this many steps if the expected packet never arrives
            self.maximum_waiting_count = 13
        if class_type == "pdcp":
            self.mother_number = self.NUM_CONTROL_PACKET_ON_PDCP
            self.index_packet_id = self.INDEX_PDCP_PACKET_ID
            # Give up if the expected packet does not arrive within this timer
            self.maximum_waiting_timer = 300
            self.waiting_timer = 0
            self.last_recived_time_index = 0

        self.ordered_received_packet = np.full(
            self.mother_number, False, dtype=bool
        )  # Packets that have been received (possibly out of order) and stored

        self.num_incoming_packets = 0
        self.num_outgoing_packets = 0

        self.largest_packet_id = 0
        self.recived_queue = np.zeros(
            [self.mother_number, self.NUM_INFO_PACKET], dtype=int)
        # Packets that have been received (possibly out of order) and stored
        self.ordered_received_packet = np.full(
            self.mother_number, False, dtype=bool)
        self.first_packet_id = 0  # ID of the first packet in recived_queue

    def load_data(self, packet_mac2rlc):
        if len(packet_mac2rlc) > 0:
            self.num_incoming_packets += len(packet_mac2rlc)
            if np.max(packet_mac2rlc[:, self.index_packet_id]) - self.first_packet_id > self.mother_number - 1:
                self.TXTLOG.append(
                    f"{self.time_index}, {self.name}, error in load_data ue_pdcprlc.py, buffer overflow\n")
            elif np.min(packet_mac2rlc[:, self.index_packet_id]) < self.first_packet_id:
                self.TXTLOG.append(
                    f"{self.time_index}, {self.name}, error in load_data ue_pdcprlc.py, unexpected packets comes\n")
                packet_mac2rlc = packet_mac2rlc[
                    packet_mac2rlc[:,
                                   self.index_packet_id] >= self.first_packet_id
                ]

            else:
                self.recived_queue[packet_mac2rlc[:, self.index_packet_id] -
                                   self.first_packet_id] = packet_mac2rlc
                self.ordered_received_packet[
                    packet_mac2rlc[:, self.index_packet_id] -
                    self.first_packet_id
                ] = True
                self.largest_packet_id = np.max(
                    [
                        self.largest_packet_id,
                        np.max(packet_mac2rlc[:, self.index_packet_id]),
                    ]
                )

            if self.class_type == "rlc":
                # In this time index, if a packet arrived and the head of the waiting queue was received
                if self.ordered_received_packet[0]:
                    self.waiting_count = 0
                else:  # In this time index, only out-of-order packets arrived
                    self.waiting_count += 1

                if self.waiting_count > self.maximum_waiting_count:
                    # If there are packets buffered in the queue
                    if np.any(self.ordered_received_packet):
                        count = np.argmax(self.ordered_received_packet)
                        self.TXTLOG.append(
                            f"{self.time_index}, {self.name}, waiting count expired, packet id of waiting packet is {self.first_packet_id}, {count} waiting packets are wasted.\n"
                        )
                        self.recived_queue = light_rotate(
                            self.recived_queue, count, self.largest_packet_id - self.first_packet_id, vacant_value=0
                        )
                        self.ordered_received_packet = light_rotate(
                            self.ordered_received_packet, count, self.largest_packet_id - self.first_packet_id, vacant_value=False
                        )
                        self.first_packet_id = self.recived_queue[0,
                                                                  self.index_packet_id]
                        self.waiting_count = 0

        if self.class_type == "pdcp":  # Execute even when no packet is received in this time index
            # In this time index, if a packet arrived and the head of the waiting queue was received
            if self.ordered_received_packet[0]:
                self.waiting_timer = 0
                self.last_recived_time_index = self.time_index
            self.waiting_timer = self.time_index - self.last_recived_time_index

            if self.waiting_timer > self.maximum_waiting_timer:
                # If there are packets buffered in the queue
                if np.any(self.ordered_received_packet):
                    count = np.argmax(self.ordered_received_packet)
                    self.TXTLOG.append(
                        f"{self.time_index}, {self.name}, waiting count expired, packet id of waiting packet is {self.first_packet_id}, {count} waiting packets are wasted.\n"
                    )
                    self.recived_queue = light_rotate(
                        self.recived_queue, count, self.largest_packet_id - self.first_packet_id, vacant_value=0
                    )
                    self.ordered_received_packet = light_rotate(
                        self.ordered_received_packet, count, self.largest_packet_id - self.first_packet_id, vacant_value=False)
                    self.first_packet_id = self.recived_queue[0,
                                                              self.index_packet_id]

                    self.waiting_timer = 0
                    self.last_recived_time_index = self.time_index

    def reorder(self):
        # Count how many consecutive True from the head
        count = np.argmax(self.ordered_received_packet == False) if False in self.ordered_received_packet else len(
            self.ordered_received_packet)
        if self.ordered_received_packet[0]:
            packet_to_upper_layer = np.copy(self.recived_queue[:count])
            self.recived_queue = light_rotate(
                self.recived_queue, count, self.largest_packet_id - self.first_packet_id, vacant_value=0
            )
            self.ordered_received_packet = light_rotate(
                self.ordered_received_packet, count, self.largest_packet_id - self.first_packet_id, vacant_value=False
            )
            self.first_packet_id = packet_to_upper_layer[-1,
                                                         self.index_packet_id] + 1
        else:
            packet_to_upper_layer = []

        self.num_outgoing_packets += len(packet_to_upper_layer)
        return packet_to_upper_layer


def test():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
    array = np.array([True, True, True, False])
    count = 3
    largest_index = 2
    vacant_value = False
    print(array)
    array = light_rotate(array, count, largest_index,
                         vacant_value=vacant_value)
    print(array)
    print(array.shape)


def test_PDCP_RLC_UE():
    load_path = "../heavy_data/20250327_0/mn/"
    csv_load_path = load_path + "csv/"

    dl_rlc_packet_ue_0 = np.loadtxt(
        csv_load_path + "dl_rlc_packet_ue_0.csv", delimiter=",", skiprows=1, dtype=int
    )
    dl_pdcp_packet_ue_0 = np.loadtxt(
        csv_load_path + "dl_pdcp_packet_ue_0.csv", delimiter=",", skiprows=1, dtype=int
    )
    print(len(dl_rlc_packet_ue_0), len(dl_pdcp_packet_ue_0))
    INDEX_time_index = ParameterClass.INDEX_UE_TIMESTAMP_ID

    rlc_ue = PDCP_RLC_UE(ue_id=0, class_type="rlc")

    for _ in range(ParameterClass.NUM_SIMULATION_TIME_SLOTS):
        packet_mac2rlc = dl_rlc_packet_ue_0[
            dl_rlc_packet_ue_0[:, INDEX_time_index] == TimeManager.time_index
        ]
        rlc_ue.load_data(packet_mac2rlc)
        packet_RLC2PDCP = rlc_ue.reorder()
        TimeManager.time_index += 1


if __name__ == "__main__":
    test_PDCP_RLC_UE()
