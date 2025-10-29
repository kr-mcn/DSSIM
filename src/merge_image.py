from pathlib import Path
import os
import glob
from PIL import Image
from param import ParameterClass
from collections import defaultdict
from math import isnan


def load_and_resize_images(image_paths, size=(1600, 1200)):
    return [Image.open(p).resize(size) for p in image_paths]


def get_latest_directory(base_dir):
    dirs = [d for d in glob.glob(os.path.join(
        base_dir, '*')) if os.path.isdir(d)]

    if not dirs:
        return None

    latest_dir = max(dirs, key=os.path.getctime)
    return os.path.basename(latest_dir)


def get_last_n_lines(filepath, n):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines[-n:]]


def extract_ran_loss_rates(filepath, ue_id):
    lines = []
    with open(filepath, 'r') as f:
        content = f.readlines()
        for idx, line in enumerate(content):
            if line.strip() == f"UE{ue_id}":
                lines = content[idx+1:idx+13]  # 12lines
                break
    return [line.strip() for line in lines]


def extract_values_from_lines(lines):
    values = []
    for line in lines:
        if not line.strip():
            continue
        try:
            number_str = line.split("=")[1].split("\t")[1].strip()
            values.append(float(number_str))
        except (IndexError, ValueError):
            pass
    return values


def extract_single_value(filepath):
    with open(filepath, 'r') as f:
        return f.readlines()[-1].strip()


def extract_txt_value(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()


def extract_line_by_number(filepath, line_number):
    if not os.path.exists(filepath):
        return "NaN"

    with open(filepath, 'r') as f:
        lines = f.readlines()
        if 0 <= line_number < len(lines):
            return lines[line_number].strip()
        else:
            return "NaN"


def culc_weighted_delay(delay_values, pkt_counts):
    """
    delay_values: Delay for each UE
    pkt_counts: Weight for each UE (number of received packets)
    return: Weighted average delay (returns 0.0 if denominator is 0)
    """
    assert len(delay_values) == len(pkt_counts), "length mismatch"
    num = 0.0
    den = 0.0
    for d, w in zip(delay_values, pkt_counts):
        if d is None or w is None:
            continue
        try:
            d = float(d)
            w = float(w)
        except (TypeError, ValueError):
            continue
        if isnan(d) or isnan(w) or w <= 0:
            continue
        num += w * d
        den += w
    return (num / den) if den > 0 else 0.0


def culc_jains_fairness_index(values):
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return 0.0
    n = len(values)
    return (sum(values) ** 2) / (n * sum(v ** 2 for v in values))


def summarize_results(base_dir, num_ue):
    base_dir = Path(base_dir)
    output_lines = []  # output data in txt format

    # --- Variables for calculating statistical values ---
    ran_loss_values_mn = [None] * num_ue
    ran_loss_values_sn = [None] * num_ue
    sf_data = defaultdict(lambda: {
        "sv_tx_thpt": [None]*num_ue,
        "ue_rx_thpt": [None]*num_ue,
        "txpkts_and_lossnum_values": [None]*num_ue
    })
    quic_level_rtt_values = []
    quic_level_rx_thpt_values = []
    one_way_delay_values = [None] * num_ue
    one_way_delay_values_p95 = [None] * num_ue
    one_way_delay_values_p99 = [None] * num_ue
    total_rx_pkts_for_each_UE = []

    for ue_id in range(num_ue):
        output_lines.append(f"UE{ue_id}")
        output_lines.append("[RAN-domain]")

        # MN
        mn_path = base_dir / "mn" / "csv" / "responsiveness_data.csv"
        output_lines.append("<mn>")
        lines = extract_ran_loss_rates(mn_path, ue_id)
        output_lines.extend(lines)
        ran_loss_values_mn[ue_id] = extract_values_from_lines(lines)

        # SN
        sn_path = base_dir / "sn" / "csv" / "responsiveness_data.csv"
        output_lines.append("<sn>")
        lines = extract_ran_loss_rates(sn_path, ue_id)
        output_lines.extend(lines)
        ran_loss_values_sn[ue_id] = extract_values_from_lines(lines)

        # MPQUIC
        ue_dir = base_dir / "L4_results" / "MPQUIC" / f"UE{ue_id}"
        rx_filepath = ue_dir / "MPQUIC-level_one_way_delay.txt"
        with rx_filepath.open("rb") as f:
            line_count = sum(1 for _ in f)
        total_rx_pkts_for_each_UE.append(line_count)
        output_lines.append("\n[MPQUIC-layer]")
        one_way_delay_values[ue_id] = float(extract_line_by_number(
            ue_dir / f"MPQUIC-level_one_way_delay_avg.txt", 0).split("\t")[0])
        output_lines.append(
            f"MPQUIC-level average one way delay (UPF-UE): \t{one_way_delay_values[ue_id]}")
        one_way_delay_values_p95[ue_id] = float(extract_line_by_number(
            ue_dir / f"MPQUIC-level_one_way_delay_avg.txt", 1).split("\t")[0])
        output_lines.append(
            f"MPQUIC-level one way delay (UPF-UE) [p95]: \t{one_way_delay_values_p95[ue_id]}")
        one_way_delay_values_p99[ue_id] = float(extract_line_by_number(
            ue_dir / f"MPQUIC-level_one_way_delay_avg.txt", 2).split("\t")[0])
        output_lines.append(
            f"MPQUIC-level one way delay (UPF-UE) [p95]: \t{one_way_delay_values_p99[ue_id]}")
        for sf in ["SF1", "SF2"]:
            output_lines.append(f"<{sf}>")
            send_thpt = extract_single_value(
                ue_dir / f"{sf}_server_send_throughput_avg_1000ms.csv")
            recv_thpt = extract_single_value(
                ue_dir / f"{sf}_UE_recv_throughput_avg_1000ms.csv")
            rtt = extract_line_by_number(
                ue_dir / f"{sf}_latest_rtt_change_log_avg.txt", 0)
            rtt_p95 = extract_line_by_number(
                ue_dir / f"{sf}_latest_rtt_change_log_avg.txt", 1)
            rtt_p99 = extract_line_by_number(
                ue_dir / f"{sf}_latest_rtt_change_log_avg.txt", 2)

            sf_data[sf]["sv_tx_thpt"][ue_id] = float(send_thpt.split("\t")[0])
            sf_data[sf]["ue_rx_thpt"][ue_id] = float(recv_thpt.split("\t")[0])

            output_lines.append(f"{sf}_server_send_throughput: \t{send_thpt}")
            output_lines.append(f"{sf}_UE_recv_throughput: \t{recv_thpt}")
            output_lines.append(f"{sf}_avg_rtt: \t{rtt}")
            output_lines.append(f"{sf}_rtt [p95]: \t{rtt_p95}")
            output_lines.append(f"{sf}_rtt [p99]: \t{rtt_p99}")
            lines = get_last_n_lines(ue_dir / f"{sf}/packet_loss_rate.txt", 3)
            output_lines.extend(lines)
            sf_data[sf]["txpkts_and_lossnum_values"][ue_id] = extract_values_from_lines(
                lines)

        if ParameterClass.UDP_MODE is True:
            # UDP
            output_lines.append("\n[UDP-layer]")
            ue_dir = base_dir / "L4_results" / "UDP" / f"UE{ue_id}"
            send_thpt = extract_single_value(
                ue_dir / "server_send_throughput_avg_1000ms.csv")
            output_lines.append(f"send_throughput: \t{send_thpt}")
            output_lines.append("")
        else:
            # QUIC
            output_lines.append("\n[QUIC-layer]")
            ue_dir = base_dir / "L4_results" / "QUIC" / f"UE{ue_id}"
            send_thpt = extract_single_value(
                ue_dir / "send_throughput_avg_1000ms.csv")
            recv_thpt = extract_single_value(
                ue_dir / "recv_throughput_avg_1000ms.csv")
            goodput = extract_single_value(ue_dir / "goodput_avg_1000ms.csv")
            rtt = extract_line_by_number(
                ue_dir / "latest_rtt_change_log_avg.txt", 0)
            rtt_p95 = extract_line_by_number(
                ue_dir / "latest_rtt_change_log_avg.txt", 1)
            rtt_p99 = extract_line_by_number(
                ue_dir / "latest_rtt_change_log_avg.txt", 2)
            quic_level_rtt_values.append(float(rtt.split("\t")[0]))
            quic_level_rx_thpt_values.append(float(recv_thpt.split("\t")[0]))

            output_lines.append(f"send_throughput: \t{send_thpt}")
            output_lines.append(f"recv_throughput: \t{recv_thpt}")
            output_lines.append(f"goodput: \t{goodput}")
            output_lines.append(f"avg_rtt: \t{rtt}")
            output_lines.append(f"avg_rtt [p95]: \t{rtt_p95}")
            output_lines.append(f"avg_rtt [p99]: \t{rtt_p99}")
            output_lines.extend(get_last_n_lines(
                ue_dir / f"packet_loss_rate.txt", 3))
            output_lines.append("")

    # SECTOR RESULTS
    output_lines.append("\n\n[SECTOR RESULTS]")
    all_ue_dir = base_dir / "L4_results" / "all_UE_results"

    # QUIC All UE
    if ParameterClass.UDP_MODE is True:
        pass
    else:
        output_lines.append("<UE QUIC>")
        output_lines.extend(get_last_n_lines(
            all_ue_dir / "all_UEs_thpt_all_UEs_thpt.txt", 3))
        # このセクターレベルPLRの部分は未完成。今はおそらくUE9の値を読み込んでいる。
        output_lines.extend(get_last_n_lines(
            ue_dir / f"packet_loss_rate.txt", 3))
        output_lines.append(
            f'Sector RTT [s]: \t{culc_weighted_delay(quic_level_rtt_values, total_rx_pkts_for_each_UE)}\t (Weighted by the num of rx packets for each UE)')
        print(quic_level_rtt_values)
        print(total_rx_pkts_for_each_UE)
        output_lines.append(
            f'QUIC-level Fairness: \t{culc_jains_fairness_index(quic_level_rx_thpt_values)}')

    # RAN loss results
    def append_ran_loss_values(output_lines, values_list, num_ue):
        def colsum(i): return sum(
            values_list[ue_id][i] for ue_id in range(num_ue))

        # PDCP
        num_tx_pdcp = colsum(0)
        num_loss_pdcp = colsum(1)
        amt_loss_pdcp = colsum(2)
        rate_pdcp = (num_loss_pdcp / num_tx_pdcp) if num_tx_pdcp else 0.0

        output_lines.append(
            f"[PDCP] Number of Transmission Opportunities: \t{num_tx_pdcp}")
        output_lines.append(
            f"[PDCP] Number of Transmission Opportunity Loss Occurrences: \t{num_loss_pdcp}")
        output_lines.append(
            f"[PDCP] Amount of Transmission Opportunity Loss [packets]: \t{amt_loss_pdcp}")
        output_lines.append(
            f"[PDCP] Transmission Opportunity Loss Rate: \t{rate_pdcp}")

        # MAC
        num_tx_mac = colsum(4)
        num_loss_mac = colsum(5)
        amt_loss_mac = colsum(6)
        rate_mac = (num_loss_mac / num_tx_mac) if num_tx_mac else 0.0

        output_lines.append(
            f"[MAC] Number of Transmission Opportunities: \t{num_tx_mac}")
        output_lines.append(
            f"[MAC] Number of Transmission Opportunity Loss Occurrences: \t{num_loss_mac}")
        output_lines.append(
            f"[MAC] Amount of Transmission Opportunity Loss [bits]: \t{amt_loss_mac}")
        output_lines.append(
            f"[MAC] Transmission Opportunity Loss Rate: \t{rate_mac}")

        # Over-reception
        output_lines.append(
            f"[PDCP] Number of Over-reception Occurrences: \t{colsum(8)}")
        output_lines.append(
            f"[PDCP] Amount of Over-reception [packets]: \t{colsum(9)}")
        output_lines.append(
            f"[MAC] Number of Over-reception Occurrences: \t{colsum(10)}")
        output_lines.append(
            f"[MAC] Amount of Over-reception [bits]: \t{colsum(11)}")

    output_lines.append("\n<6G RAN>")
    append_ran_loss_values(output_lines, ran_loss_values_sn, num_ue)
    output_lines.append("\n<5G RAN>")
    append_ran_loss_values(output_lines, ran_loss_values_mn, num_ue)
    output_lines.append("\n<6G + 5G>")
    ran_loss_values_total = []
    for row_mn, row_sn in zip(ran_loss_values_mn, ran_loss_values_sn):
        ran_loss_values_total.append(
            [mn_val + sn_val for mn_val, sn_val in zip(row_mn, row_sn)])
    append_ran_loss_values(output_lines, ran_loss_values_total, num_ue)
    output_lines.append("\n<MPQUIC-layer>")
    for sf in ["SF1", "SF2"]:
        output_lines.append(f"\n<{sf}>")
        output_lines.append(
            f'Total Server Tx Thpt [Mbps]: \t{sum(sf_data[sf]["sv_tx_thpt"])}')
        output_lines.append(
            f'Total UE Rx Thpt [Mbps]: \t{sum(sf_data[sf]["ue_rx_thpt"])}')
        total_tx_pkts = sum(row[0] for row in sf_data[sf]
                            ["txpkts_and_lossnum_values"] if row is not None)
        total_loss = sum(row[1] for row in sf_data[sf]
                         ["txpkts_and_lossnum_values"] if row is not None)
        output_lines.append(f'Total Tx Packets: \t{total_tx_pkts}')
        output_lines.append(f'Total Packet Losses: \t{total_loss}')
        output_lines.append(f'Packet Loss Rate: \t{total_loss/total_tx_pkts}')

    output_lines.append("\n<SF1+SF2>")
    output_lines.append(
        f'Total Server Tx Thpt [Mbps]: \t{sum(sf_data["SF1"]["sv_tx_thpt"])+sum(sf_data["SF2"]["sv_tx_thpt"])}')
    output_lines.append(
        f'Total UE Rx Thpt [Mbps]: \t{sum(sf_data["SF1"]["ue_rx_thpt"])+sum(sf_data["SF2"]["ue_rx_thpt"])}')
    total_tx_pkts = sum(row[0] for row in sf_data["SF1"]["txpkts_and_lossnum_values"] if row is not None) + \
        sum(row[0] for row in sf_data["SF2"]
            ["txpkts_and_lossnum_values"] if row is not None)
    total_loss = sum(row[1] for row in sf_data["SF1"]["txpkts_and_lossnum_values"] if row is not None) + \
        sum(row[1] for row in sf_data["SF2"]
            ["txpkts_and_lossnum_values"] if row is not None)
    output_lines.append(f'Total Tx Packets: \t{total_tx_pkts}')
    output_lines.append(f'Total Packet Losses: \t{total_loss}')
    output_lines.append(f'Packet Loss Rate: \t{total_loss/total_tx_pkts}')

    output_lines.append("\n<Sector Delay>")
    output_lines.append(
        f'MPQUIC one-way-delay Overall [s]: \t{culc_weighted_delay(one_way_delay_values, total_rx_pkts_for_each_UE)}\t (Weighted by the num of rx packets for each UE)')
    output_lines.append(
        f'Average p95 one-way-delay [s]: \t{sum(one_way_delay_values_p95)/len(one_way_delay_values_p95)}')
    output_lines.append(
        f'Average p99 one-way-delay [s]: \t{sum(one_way_delay_values_p99)/len(one_way_delay_values_p99)}')

    output_lines.append("\n<Rx Thpt Fairness>")

    output_lines.append(
        f'SF1 (6G): \t{culc_jains_fairness_index(sf_data["SF1"]["ue_rx_thpt"])}')
    output_lines.append(
        f'SF2 (5G): \t{culc_jains_fairness_index(sf_data["SF2"]["ue_rx_thpt"])}')
    rx_thpt_sum = [
        a + b for a, b in zip(sf_data["SF1"]["ue_rx_thpt"], sf_data["SF2"]["ue_rx_thpt"])]
    output_lines.append(
        f'MPQUIC (5G+6G): \t{culc_jains_fairness_index(rx_thpt_sum)}')

    out_path = base_dir / "summary_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Summary saved to: {out_path}")


def main():
    dir1 = ParameterClass.HEAVY_DATA_PATH

    """
    # Batch Execution
    dir2_list = [
        "Exp_for_access_UDPfullbuff_3scheme_comparison/20250919-150908_ue=10_slot=30000_D=10_UDP_400M_5G=train5_6G=train5_UDPonCUB",
        "Exp_for_access_UDPfullbuff_3scheme_comparison/20250919-150917_ue=10_slot=30000_D=10_UDP_400M_5G=train5_6G=train5_UDPonCUB_PROPOSED",
        "Exp_for_access_UDPfullbuff_3scheme_comparison/20250919-150926_ue=10_slot=30000_D=10_UDP_400M_5G=train5_6G=train5_UDPonCUB_IDEAL",
        ...
    ]
    """
    # Latest Directory Execution
    dir2 = get_latest_directory(dir1)
    dir2_list = [dir2]

    for dir2 in dir2_list:
        pathname = "/summary"
        mkdir = f"{dir1}{dir2}{pathname}"
        os.makedirs(mkdir, exist_ok=True)
        basedir = f"{dir1}{dir2}"
        dir_l4 = "L4_results"
        dir_ran_mn = "mn/figure"
        dir_ran_sn = "sn/figure"

        for ue_id in range(ParameterClass.NUM_UE):
            ue = f"MPQUIC/UE{ue_id}"
            image_files = [
                # sn (6G Path)
                os.path.join(basedir, dir_ran_sn,
                             f"estimated_channel_condition{ue_id}.png"),
                os.path.join(basedir, dir_ran_sn,
                             f"gNB_buffer_length_{ue_id}.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF1_server_send_throughput_avg_1000ms.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF1_UE_recv_throughput_avg_1000ms.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF1_cwnd_size_log.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF1_latest_RTT.png"),
                # mn (5G path)
                os.path.join(basedir, dir_ran_mn,
                             f"estimated_channel_condition{ue_id}.png"),
                os.path.join(basedir, dir_ran_mn,
                             f"gNB_buffer_length_{ue_id}.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF2_server_send_throughput_avg_1000ms.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF2_UE_recv_throughput_avg_1000ms.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF2_cwnd_size_log.png"),
                os.path.join(basedir, dir_l4, ue,
                             "SF2_latest_RTT.png"),
            ]
            images = load_and_resize_images(image_files)

            cols = 6
            rows = 2
            width, height = images[0].size
            combined = Image.new("RGB", (cols * width, rows * height))

            for idx, img in enumerate(images):
                x = (idx % cols) * width
                y = (idx // cols) * height
                combined.paste(img, (x, y))

            output_path = os.path.join(
                f"{basedir}{pathname}", f"combined_image_UE{ue_id}.png")
            combined.save(output_path)
            print(f"Saved to: {output_path}")

        summarize_results(basedir, ParameterClass.NUM_UE)


if __name__ == "__main__":
    main()
