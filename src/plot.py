# import queue
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from param import ParameterClass


class TimeManager:
    # Class variable (shared across the class)
    time_index = 0


def thp_comparison():
    fileindexs = ["20250205_1/"]
    for fileindex in fileindexs:
        load_path = "./data/" + fileindex

        packet_payload = 1500 * 8  # bit

        # PDCP throughput
        pdcp_outgoing = np.load(load_path + "pdcp_outgoing.npy")
        pdcp_thp = (
            np.sum(pdcp_outgoing, axis=1)
            * packet_payload
            / (1e6 * 0.001 * np.arange(1, len(pdcp_outgoing[:, 0]) + 1))
        )
        plt.plot(pdcp_thp, label="cell pdcp thp " + fileindex)
        plt.ylabel("thp(Mbps)")

    single_cell = np.zeros(10)
    high_band_ue = [2, 3, 4, 5, 8]
    low_band_ue = [0, 1, 6, 7, 9]
    pdcp_outgoing_1 = np.load("./data/20250108_1/pdcp_outgoing.npy")
    single_cell[high_band_ue] = np.average(pdcp_outgoing_1, axis=0)

    pdcp_outgoing_2 = np.load("./data/20250108_2/pdcp_outgoing.npy")
    single_cell[low_band_ue] = np.average(pdcp_outgoing_2, axis=0)
    pdcp_outgoing = pdcp_outgoing_1 + pdcp_outgoing_2
    pdcp_thp = (
        np.sum(pdcp_outgoing, axis=1)
        * packet_payload
        / (1e6 * 0.001 * np.arange(1, len(pdcp_outgoing[:, 0]) + 1))
    )
    plt.plot(pdcp_thp, label="1+2")
    print(np.average(pdcp_thp), "1+2")
    print("single cell", single_cell)

    pdcp_outgoing_5 = np.load("./data/20250108_5/mn/pdcp_outgoing.npy")
    print("DC", np.average(pdcp_outgoing_5, axis=0))
    pdcp_thp = (
        np.sum(pdcp_outgoing_5, axis=1)
        * packet_payload
        / (1e6 * 0.001 * np.arange(1, len(pdcp_outgoing[:, 0]) + 1))
    )

    print(np.average(pdcp_thp), "5")
    plt.plot(pdcp_thp, label="5")

    fig_save_path = "./data/figure/20250108_1/"
    plt.legend()
    plt.savefig(fig_save_path + "thp_comparison_1_2_5.png")
    plt.clf()


def delay_comparison():
    load_path = "./data/20250108_4/"
    fig_save_path = "./data/figure/20250108_4/"
    num_ue = 10
    delay = []
    for i in range(num_ue):
        array = np.load(
            load_path + "concatenate_dl_ip_ue_buffer_" + str(i) + ".npy")
        ue_timestamp = array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID]
        rlc_timestamp = array[:,
                              ParameterClass.INDEX_RLC_INCOMING_TIMESTAMP_ID]
        ue_delay = (ue_timestamp - rlc_timestamp) * 0.001
        if i == 0:
            delay = ue_delay
        else:
            delay = np.concatenate([delay, ue_delay])

    plt.plot(np.sort(delay), np.arange(len(delay)) / len(delay))
    plt.savefig(fig_save_path + "CDF_delay.png")
    print(fig_save_path + "CDF_delay.png")
    plt.cla()


def main():
    # Set data load/save paths
    load_path = "../heavy_data/20250325_3/sn/"
    fig_save_path = "../heavy_data/figure/20250325_3/sn/"
    directory = Path(fig_save_path)
    # Create the folder if it does not exist
    if not directory.exists():
        directory.mkdir(parents=True)

    num_packets = np.zeros(10, dtype=int)
    mn_sn = "mn"
    packet_payload = 1500 * 8  # bit
    num_ue = 1

    if mn_sn == "mn":
        for i in range(num_ue):
            filename_start = "dl_ip_ue_buffer_" + str(i)
            # Gather files from the load path
            files = [f for f in os.listdir(load_path) if f.startswith(filename_start)][
                ::-1
            ]
            for j, file in enumerate(files):
                if j == 0:
                    array = np.load(load_path + file)
                else:
                    array = np.vstack((array, np.load(load_path + file)))
            array = array[np.argsort(array[:, 6])]
            print(array.shape)
            num_packets[i] = np.sum(array[:, 1] > 0)
            print(num_packets)
            array = array[-num_packets[i]:]
            print(array)
            print(array.shape)
            counts = np.bincount(array[:, -1])
            # Compute delay
            delay = (
                array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID]
                - array[:, ParameterClass.INDEX_RLC_INCOMING_TIMESTAMP_ID]
            ) * 0.001
            # Save results
            np.save(load_path + "concatenate_dl_ip_ue_buffer_" +
                    str(i) + ".npy", array)
            np.savetxt(
                load_path + "concatenate_dl_ip_ue_buffer_" + str(i) + ".csv",
                array,
                delimiter=",",
                fmt="%d",
            )
            plt.plot(delay)
        plt.savefig(fig_save_path + "delay.png")
        plt.clf()

        for i in range(num_ue):
            filename_start = "dl_rlc_ue_buffer_" + str(i)
            # Gather files from the load path
            files = [f for f in os.listdir(load_path) if f.startswith(filename_start)][
                ::-1
            ]
            for j, file in enumerate(files):
                if j == 0:
                    array = np.load(load_path + file)
                else:
                    array = np.vstack((array, np.load(load_path + file)))
            array = array[np.argsort(array[:, 6])]
            num_packets[i] = np.sum(array[:, 1] > 0)
            array = array[-num_packets[i]:]
            np.save(
                load_path + "concatenate_dl_rlc_ue_buffer_" +
                str(i) + ".npy", array
            )

        for i in range(num_ue):
            array = np.load(
                load_path + "concatenate_dl_ip_ue_buffer_" + str(i) + ".npy"
            )
            if i == 0:
                delay = (
                    array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID]
                    - array[:, ParameterClass.INDEX_RLC_INCOMING_TIMESTAMP_ID]
                ) * 0.001
            else:
                delay = np.concatenate(
                    [
                        delay,
                        (
                            array[:, ParameterClass.INDEX_UE_TIMESTAMP_ID]
                            - array[:, ParameterClass.INDEX_RLC_INCOMING_TIMESTAMP_ID]
                        )
                        * 0.001,
                    ]
                )

        plt.plot(np.sort(delay), np.arange(len(delay)) / len(delay))
        plt.savefig(fig_save_path + "CDF_delay.png")
        plt.cla()

        for i in range(num_ue):
            counts = np.bincount(array[:, -1])
            plt.plot(np.sort(counts), np.arange(len(counts)) / len(counts))
        plt.savefig(fig_save_path + "dist_num_packets_per_TS.png")
        plt.clf()

    plot = True
    if plot:
        # PDCP buffer size (on gNB)
        gnb_pdcp_buffer_size = np.load(
            load_path + "gnb_pdcp_buffer_length.npy")
        for i in range(num_ue):
            plt.plot(gnb_pdcp_buffer_size[:, i])
        plt.xlabel("time step")
        plt.ylabel("Buffer size on gNB PDCP layer (num. packets)")
        plt.savefig(fig_save_path + "gnb_pdcp_buffer_length.png")
        np.savetxt(
            load_path + "gnb_pdcp_buffer_length" + ".csv",
            gnb_pdcp_buffer_size[:, 0],
            delimiter=",",
            fmt="%d",
        )

        plt.clf()

        # RLC buffer size (on gNB)
        gnb_rlc_buffer_size = np.load(load_path + "gnb_rlc_buffer_length.npy")
        for i in range(num_ue):
            plt.plot(gnb_rlc_buffer_size[:, i])
        np.savetxt(
            load_path + "gnb_rlc_buffer_size" + ".csv",
            gnb_rlc_buffer_size[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.xlabel("time step")
        plt.ylabel("Buffer size on gNB RLC layer (num. packets)")
        plt.savefig(fig_save_path + "gnb_rlc_buffer_length.png")
        plt.clf()

        gnb_pdcp_num_incoming_packets = np.load(
            load_path + "gnb_pdcp_num_incoming_packets.npy"
        )
        for i in range(num_ue):
            plt.plot(gnb_pdcp_num_incoming_packets[:, i])
        np.savetxt(
            load_path + "gnb_pdcp_num_incoming_packets" + ".csv",
            gnb_pdcp_num_incoming_packets[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.xlabel("time step")
        plt.ylabel("gnb_pdcp_num_incoming_packets (num. packets)")
        plt.savefig(fig_save_path + "gnb_pdcp_num_incoming_packets.png")
        plt.clf()

        gnb_rlc_num_incoming_packets = np.load(
            load_path + "gnb_rlc_num_incoming_packets.npy"
        )
        for i in range(num_ue):
            plt.plot(gnb_rlc_num_incoming_packets[:, i])
        np.savetxt(
            load_path + "gnb_rlc_num_incoming_packets" + ".csv",
            gnb_rlc_num_incoming_packets[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.xlabel("time step")
        plt.ylabel("gnb_rlc_num_incoming_packets (num. packets)")
        plt.savefig(fig_save_path + "gnb_rlc_num_incoming_packets.png")
        plt.clf()

        # MAC buffer size (on gNB)
        total_buffer_size = np.load(load_path + "total_buffer_size.npy")
        for i in range(num_ue):
            plt.plot(total_buffer_size[:, i])
        np.savetxt(
            load_path + "total_buffer_size" + ".csv",
            total_buffer_size[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.xlabel("time step")
        plt.ylabel("Buffer size on gNB MAC layer (num. packets)")
        plt.savefig(fig_save_path + "gnb_mac_buffer_size.png")
        plt.clf()

        plt.plot(np.average(total_buffer_size, axis=1))
        plt.savefig(fig_save_path + "ave_total_buffer_size.png")
        plt.clf()

        assigned_TBS = np.load(load_path + "measured_TBS.npy")
        for i in range(num_ue):
            plt.plot(assigned_TBS[:, i])
        plt.savefig(fig_save_path + "assigned_TBS.png")
        plt.clf()

        plt.plot(np.average(assigned_TBS, axis=1))
        plt.savefig(fig_save_path + "ave_assigned_TBS.png")
        plt.clf()

        plt.plot(np.sum(assigned_TBS, axis=1) * 1e3 / 1e6)
        plt.savefig(fig_save_path + "PHY_cell_thp.png")
        plt.xlabel("time step")
        plt.ylabel("phy cell thp (Mbps)")
        plt.clf()

        estimated_channel_condition = np.load(
            load_path + "estimated_channel_condition.npy"
        )
        for i in range(num_ue):
            plt.plot(estimated_channel_condition[:, i])
        plt.savefig(fig_save_path + "estimated_channel_condition.png")
        plt.clf()

        plt.plot(np.average(estimated_channel_condition, axis=1))
        plt.savefig(fig_save_path + "ave_estimated_channel_condition.png")
        plt.clf()

        estimated_spectral_efficiency = np.load(
            load_path + "estimated_spectral_efficiency.npy",
        )
        for i in range(num_ue):
            plt.plot(estimated_spectral_efficiency[:, i])
        plt.savefig(fig_save_path + "estimated_spectral_efficiency.png")
        plt.clf()

        experienced_throughput = np.load(
            load_path + "experienced_throughput.npy")
        for i in range(num_ue):
            plt.plot(experienced_throughput[:, i] * 1e3 / 1e6)
        plt.savefig(fig_save_path + "experienced_throughput.png")
        plt.clf()

        plt.plot(np.average(experienced_throughput, axis=1) * 1e3 / 1e6)
        plt.savefig(fig_save_path + "ave_experienced_throughput.png")
        plt.clf()

        PF_metic = np.load(load_path + "PF_metic.npy")
        for i in range(num_ue):
            plt.plot(PF_metic[:, i])
        plt.yscale("log")
        plt.ylim([np.min(PF_metic), 1e-2])
        plt.savefig(fig_save_path + "PF_metic.png")
        plt.clf()

        assigned_bandwidth = np.load(load_path + "assigned_bandwidth.npy")
        for i in range(num_ue):
            plt.plot(assigned_bandwidth[:, i])
        plt.savefig(fig_save_path + "assigned_bandwidth.png")
        plt.clf()

        measured_spectrum_efficiency = np.load(
            load_path + "measured_spectrum_efficiency.npy"
        )
        for i in range(num_ue):
            plt.plot(measured_spectrum_efficiency[:, i])
        plt.savefig(fig_save_path + "measured_spectrum_efficiency.png")
        plt.clf()

        # Save UE-side MAC-related traces
        pass_or_drop = np.load(load_path + "UE_pass_or_drop_size.npy")
        for i in range(num_ue):
            plt.plot(pass_or_drop[:, i])
        plt.savefig(fig_save_path + "pass_or_drop.png")
        plt.clf()

        TBS = np.load(load_path + "UE_TBS.npy")
        for i in range(num_ue):
            plt.plot(TBS[:, i])
        plt.savefig(fig_save_path + "TBS.png")
        plt.clf()

        rlc_incoming = np.load(load_path + "ue_rlc_incoming.npy")
        print(rlc_incoming.shape)
        for i in range(num_ue):
            plt.plot(rlc_incoming[:, i])
        np.savetxt(
            load_path + "ue_rlc_incoming" + ".csv",
            rlc_incoming[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.savefig(fig_save_path + "ue_rlc_incoming.png")
        plt.clf()

        rlc_incoming = np.load(load_path + "ue_rlc_incoming.npy")
        for i in range(num_ue):
            mac_thp = (
                rlc_incoming[:, i]
                * packet_payload
                / (0.001 * np.arange(1, len(rlc_incoming[:, i]) + 1))
            )
            plt.plot(mac_thp / 1e6)
        plt.ylabel("thp(Mbps)")
        plt.savefig(fig_save_path + "mac_thp.png")
        plt.clf()

        plt.plot(
            np.average(rlc_incoming, axis=1)
            * packet_payload
            / (1e6 * (0.001 * np.arange(1, len(rlc_incoming[:, 0]) + 1)))
        )
        plt.ylabel("thp(Mbps)")
        plt.savefig(fig_save_path + "ave_mac_thp.png")
        plt.clf()

        rlc_outgoing = np.load(load_path + "ue_rlc_outgoing.npy")
        np.savetxt(
            load_path + "ue_rlc_outgoing" + ".csv",
            rlc_outgoing[:, 0],
            delimiter=",",
            fmt="%d",
        )
        print(rlc_outgoing.shape)
        for i in range(num_ue):
            plt.plot(rlc_outgoing[:, i])
        plt.savefig(fig_save_path + "ue_rlc_outgoing.png")
        plt.clf()

        for i in range(num_ue):
            plt.plot(rlc_incoming[:, i] - rlc_outgoing[:, i])
        plt.savefig(fig_save_path + "ue_rlc_buffer_length.png")
        plt.clf()

        pdcp_incoming = np.load(load_path + "ue_pdcp_incoming.npy")
        print(pdcp_incoming.shape)
        for i in range(num_ue):
            plt.plot(pdcp_incoming[:, i])
        np.savetxt(
            load_path + "ue_pdcp_incoming" + ".csv",
            pdcp_incoming[:, 0],
            delimiter=",",
            fmt="%d",
        )
        plt.savefig(fig_save_path + "ue_pdcp_incoming.png")
        plt.clf()

        rlc_outgoing = np.load(load_path + "ue_rlc_outgoing.npy")
        for i in range(num_ue):
            rlc_thp = (
                rlc_outgoing[:, i]
                * packet_payload
                / (0.001 * np.arange(1, len(rlc_outgoing[:, i]) + 1))
            )
            plt.plot(rlc_thp / 1e6)
        plt.ylabel("thp(Mbps)")
        plt.savefig(fig_save_path + "rlc_thp.png")
        plt.clf()

        plt.plot(
            np.average(rlc_outgoing, axis=1)
            * packet_payload
            / (1e6 * (0.001 * np.arange(1, len(rlc_outgoing[:, 0]) + 1)))
        )
        plt.ylabel("thp(Mbps)")
        plt.savefig(fig_save_path + "ave_rlc_thp.png")
        plt.clf()

        pdcp_outgoing = np.load(load_path + "ue_pdcp_outgoing.npy")
        print(pdcp_outgoing.shape)
        for i in range(num_ue):
            plt.plot(pdcp_outgoing[:, i])
        plt.savefig(fig_save_path + "ue_pdcp_outgoing.png")
        plt.clf()

        pdcp_outgoing = np.load(load_path + "ue_pdcp_outgoing.npy")
        np.savetxt(
            load_path + "ue_pdcp_outgoing" + ".csv",
            pdcp_outgoing[:, 0],
            delimiter=",",
            fmt="%d",
        )
        for i in range(num_ue):
            pdcp_thp = (
                pdcp_outgoing[:, i]
                * packet_payload
                / (0.001 * np.arange(1, len(pdcp_outgoing[:, i]) + 1))
            )
            plt.plot(pdcp_thp / 1e6)
        plt.ylabel("thp(Mbps)")
        plt.savefig(fig_save_path + "pdcp_thp.png")
        plt.clf()

        for i in range(num_ue):
            plt.plot(pdcp_incoming[:, i] - pdcp_outgoing[:, i], label=str(i))
        plt.legend()
        plt.savefig(fig_save_path + "ue_pdcp_buffer_length.png")
        plt.clf()
        np.savetxt(
            load_path + "ue_pdcp_buffer_length" + ".csv",
            pdcp_incoming[:, 0] - pdcp_outgoing[:, 0],
            delimiter=",",
            fmt="%d",
        )

        # PHY cell throughput
        assigned_TBS = np.load(load_path + "measured_TBS.npy")
        phy_thp = np.sum(assigned_TBS, axis=1) * 1e3 / 1e6
        plt.plot(phy_thp, label="cell phy thp")

        # MAC cell throughput
        rlc_incoming = np.load(load_path + "ue_rlc_incoming.npy")
        mac_thp = (
            np.sum(rlc_incoming, axis=1)
            * packet_payload
            / (1e6 * (0.001 * np.arange(1, len(rlc_incoming[:, 0]) + 1)))
        )
        plt.plot(mac_thp, label="cell mac thp")

        # RLC throughput
        rlc_outgoing = np.load(load_path + "ue_rlc_outgoing.npy")
        rlc_thp = (
            np.sum(rlc_outgoing, axis=1)
            * packet_payload
            / (1e6 * (0.001 * np.arange(1, len(rlc_outgoing[:, 0]) + 1)))
        )
        plt.plot(rlc_thp, label="cell rlc thp")

        # PDCP throughput
        pdcp_outgoing = np.load(load_path + "ue_pdcp_outgoing.npy")
        pdcp_thp = (
            np.sum(pdcp_outgoing, axis=1)
            * packet_payload
            / (1e6 * 0.001 * np.arange(1, len(pdcp_outgoing[:, 0]) + 1))
        )
        plt.plot(pdcp_thp, label="cell pdcp thp")
        plt.ylabel("thp(Mbps)")
        plt.legend()
        plt.savefig(fig_save_path + "thp_comparison.png")
        plt.clf()


if __name__ == "__main__":
    main()
