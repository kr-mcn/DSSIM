import numpy as np
from param import ParameterClass, TimeManager
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load():
    load_path = "../heavy_data/20250327_0/mn/"
    ParameterClass.HEAVY_DATA_PATH + ParameterClass.LOG_SAVE_PATH
    csv_save_path = load_path + "csv/"
    directory = Path(csv_save_path)
    if not directory.exists():
        directory.mkdir(parents=True)

    num_ue = ParameterClass.NUM_UE
    packet_data_list_ue = ["dl_ip_packet_ue_", "dl_pdcp_packet_ue_", "dl_rlc_packet_ue_"]
    packet_data_list_gnb_per_ue = ["dl_rlc_gnb_buffer_for_save", "dl_mac_gnb_buffer_for_save"]
    packet_data_list_gnb = ("dl_pdcp_gnb_buffer_for_save",)

    index = [
        "Packet ID",
        "Payload Size",
        "UE ID",
        "MAC Packet ID",
        "PDCP Packet ID",
        "Server Timestamp",
        "UE Timestamp(save time)",
        "RLC Incoming Timestamp",
        "Outer Packet ID",
        "Outer ACK Flag",
        "RLC Packet ID",
    ]

    for i in range(num_ue):
        for data_type in packet_data_list_ue + packet_data_list_gnb_per_ue:
            filename_start = data_type + str(i)
            files = [f for f in os.listdir(load_path) if f.startswith(filename_start)][::-1]

            for j, file in enumerate(files):
                if j == 0:
                    array = np.load(load_path + file)
                else:
                    array = np.vstack((array, np.load(load_path + file)))
            sorted_indices = np.argsort(array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID])  # arr[1] = [5, 15, 25]
            array = array[sorted_indices]
            np.savetxt(
                csv_save_path + data_type + str(i) + ".csv",
                array,
                delimiter=",",
                fmt="%d",
                header=",".join(index),
            )
            print(len(array))

    for data_type in packet_data_list_gnb:
        filename_start = data_type
        files = [f for f in os.listdir(load_path) if f.startswith(filename_start)][::-1]

        for j, file in enumerate(files):
            if j == 0:
                array = np.load(load_path + file)
            else:
                array = np.vstack((array, np.load(load_path + file)))
        sorted_indices = np.argsort(array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID])  # arr[1] = [5, 15, 25]
        array = array[sorted_indices]
        print(csv_save_path + data_type + ".csv")
        np.savetxt(
            csv_save_path + data_type + ".csv",
            array,
            delimiter=",",
            fmt="%d",
            header=",".join(index),
        )


if __name__ == "__main__":
    load()
