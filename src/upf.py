import numpy as np
from param import ParameterClass, TimeManager
from buffer import Buffer


class UPF(ParameterClass, TimeManager):
    def __init__(self, routing_table=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), num_N3=2):
        self.N6_buffer = Buffer(max_volume=1e9, max_length=int(1000))
        self.num_N3 = num_N3
        self.N3_buffer_list = np.zeros(self.num_N3, dtype=Buffer)
        self.routing_table = routing_table

        for gNB_id in range(self.num_N3):
            self.N3_buffer_list[gNB_id] = Buffer(max_volume=1e9, max_length=int(1000))

    def update_routing_table(self):
        self.routing_table = self.routing_table

    def dl_enqueue(self, incoming_N6_packets):
        """incoming N6 packets = [packet#1.,packet#N]"""
        self.N6_buffer.enqueue(incoming_N6_packets)

    def route(self):
        """N6 -> N3"""
        dequeued = self.N6_buffer.dequeue(dequeue_type="all")
        if len(dequeued) == 0:
            return
        gNB_index_per_packet = self.routing_table[dequeued[:, self.INDEX_ue_id]]
        for N3_id in range(self.num_N3):
            dequeued_for_gNB = dequeued[gNB_index_per_packet == N3_id]
            self.N3_buffer_list[N3_id].enqueue(dequeued_for_gNB)

    def dl_dequeue(self, N3id):
        return self.N3_buffer_list[N3id].dequeue(dequeue_type="all")
