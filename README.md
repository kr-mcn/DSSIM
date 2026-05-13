# DSSIM

DualSteer Simulator (DSSIM) is a packet-level simulation framework for
core-network-level multi-access communication, known as DualSteer in 3GPP.

This repository contains the simulation code and configuration data used in our
research on DualSteer, including cross-layer traffic control, MASQUE-based
tunneling, MPQUIC-based multi-access aggregation, and related transport-control
mechanisms.

## Paper-specific versions

The `main` branch contains the latest implementation.

For exact reproducibility, please use the release/tag corresponding to each
paper instead of the latest `main` branch.

| Paper | Status | Release / Tag | Notes |
|---|---|---|---|
| Multi-Generation DualSteer: Cross-Layer Traffic Control With Asymmetric RAN Feedback | Accepted, IEEE Access 2026 | `ieee-access-2026-v1.0.0` | Version used for the accepted IEEE Access paper |
| GC26 submission | Under review | `gc-2026-v1.0.0` | Version used for the GC26 submission |

The `main` branch may include implementation changes introduced after each
release. For reproducing a specific paper, please use the corresponding
release/tag.

## Overview

DualSteer is a mobile-core multi-access aggregation architecture. It hosts a
MASQUE proxy in the mobile core and aggregates traffic over multiple RANs
through a proxy–UE MPQUIC tunnel, without requiring modifications to legacy base
stations.

This simulator models downlink communication in DualSteer, including:

- an application server,
- a UPF-side MASQUE proxy,
- a UE-side MASQUE client,
- an E2E QUIC connection,
- a proxy–UE MPQUIC tunnel over multiple RANs, and
- wireless traces generated from pre-generated radio propagation data.

## Features

### Common DualSteer simulation features

- **Core-network-level multi-access aggregation**  
  Simulates downlink traffic aggregation at the mobile core, with traffic
  steering at the UPF and packet delivery over multiple RANs.

- **MASQUE-based proxy architecture**  
  Models an E2E QUIC connection carried over a MASQUE-style proxy–UE tunnel.

- **MPQUIC tunnel over multiple RANs**  
  Supports a proxy–UE MPQUIC tunnel for multi-access communication.

- **Packet-level simulation**  
  Operates with a 1 ms time step for fine-grained analysis of throughput,
  delay, queueing, and loss.

- **Wireless trace-based evaluation**  
  Uses pre-generated radio propagation configurations under `heavy_data/conf/`.

### Features for the IEEE Access version

The IEEE Access version focuses on cross-layer traffic control with asymmetric
RAN feedback.

It supports:

- No-RAN feedback,
- Single-RAN feedback, where only the 6G RAN provides feedback, and
- Dual-RAN feedback as an upper-bound reference.

For exact reproduction of this version, use:

```text
ieee-access-2026-v1.0.0
```

### Features for the GC26 version

The GC26 version includes updates to the simulator for the submitted GC26
manuscript.

Compared with the IEEE Access version, this version updates files under `src/`
for the GC26 experiments. The detailed method, evaluation settings, and results
are described in the submitted manuscript.

For reproducing the submitted GC26 results, use:

```text
gc-2026-v1.0.0
```

The GC26 manuscript is currently under review. This README will be updated with
citation information and additional details after the paper becomes publicly
available.

## Repository structure

```text
DSSIM/
├── heavy_data/
│   └── conf/        Pre-generated radio propagation configuration data
├── src/             Simulator source code
├── README.md
└── LICENSE
```

The directory layout is intentionally kept simple. In particular, the simulator
source code is kept under `src/` for all versions because some scripts rely on
relative paths.

Different paper versions are managed by Git tags/releases, not by placing
multiple copies such as `src/paper-A/` and `src/paper-B/` in the same branch.

## Installation

Clone this repository and install the required dependencies.

```bash
git clone https://github.com/kr-mcn/DSSIM.git
cd DSSIM
```

Required packages:

- Python 3.10.12 or later
- NumPy 2.2.4 or later

## Usage

Run the simulator from the `src/` directory:

```bash
cd src
python main.py
```

The simulator reads configuration settings from `src/param.py`.

## Configuration

All simulation parameters are defined in:

```text
src/param.py
```

The active parameters depend on the release/tag being used.  
Some parameters are common across the IEEE Access and GC26 versions, while
others are used only in a specific version.

For exact reproduction, please use the corresponding release/tag and follow the
configuration described for that version.

```text
IEEE Access paper:
  use ieee-access-2026-v1.0.0

GC26 submission:
  use gc-2026-v1.0.0
```

### Common parameters

The following parameters are commonly used across versions.

| Category | Key Parameters | Description |
|---|---|---|
| Simulation time | `NUM_SIMULATION_TIME_SLOTS`, `TIME_SLOT_WINDOW` | Define the simulation duration and time resolution. A value of 0.001 for `TIME_SLOT_WINDOW` corresponds to 1 ms. |
| Radio configurations | `RAN_CONFIG_5G`, `RAN_CONFIG_6G` | Specify which propagation configuration under `heavy_data/conf/` to use for each RAN. |
| Congestion control | `QUIC_CC`, `MPQUIC_CC` | Specify congestion control algorithms for E2E QUIC and MPQUIC, when applicable. |
| Logging | `LOG_SAVE_PATH` | Specifies the output directory for simulation logs. |

Each configuration directory under `heavy_data/conf/` contains pre-generated
radio propagation data. Each file represents one dataset, where each line
corresponds to a single time index.

### IEEE Access version-specific parameters

The IEEE Access version focuses on cross-layer traffic control with asymmetric
RAN feedback. The following parameters are mainly used for reproducing the IEEE
Access paper.

| Category | Key Parameters | Description |
|---|---|---|
| RAN feedback mode | `RAN_FB_OPTION` | Selects the RAN feedback configuration, such as no feedback, single-RAN feedback, or dual-RAN feedback. |
| Transport protocol mode | `UDP_MODE` | Selects the application-level transport mode used in the IEEE Access experiments. |
| RAN feedback interval | `RAN_FB_CYCLE` | Sets the feedback reporting interval for RAN feedback. |
| Experienced throughput window | `EXP_THPT_RANGE_SEC` | Sets the time window used to calculate experienced throughput for feedback. |

These parameters are intended for the IEEE Access version.  
They may not be used, or may have different meanings, in the GC26 version.

### GC26 version configuration

For GC26, the table below lists the parameter values that should be changed
from the default settings in `src/param.py`.

Parameters not listed here can be left at their default values unless otherwise
specified.


#### GC26 evaluation-specific settings

The following table summarizes the parameter values that should be changed from
the default settings in `src/param.py` for each GC26 evaluation case.

| Evaluation | Parameter values changed from default |
|---|---|
| B-1 | The default settings can be used for Vanilla DS. For TECC, set `TECC_OPTION = True`. For CAMF, set `CAMF_OPTION = True`. For wireless traces, use 20 patterns from `trace0` to `trace19` in `RAN_CONFIG`. Uncomment only the trace used for each simulation run. |
| B-2 | Set `NUM_SIMULATION_TIME_SLOTS = 60000`, `N6_BANDWIDTH_BPS = 50 * 1e6`, `N6_BANDWIDTH_SCHEDULE = [(10000, 150 * 1e6)]`, and `RAN_CONFIG = "stable1"`. For TECC, set `TECC_OPTION = True`. For CAMF, set `CAMF_OPTION = True`. |
| C | Set `N6_BANDWIDTH_BPS = 50 * 1e6` and `RAN_CONFIG = "btleval"`. For TECC, set `TECC_OPTION = True`. For CAMF, set `CAMF_OPTION = True`. For CAMF without the cwnd guard, set both `CAMF_OPTION = True` and `CAMF_NOCAP_OPTION = True`. |
| D | Set `NUM_UE = 10`. For wireless traces, use 20 patterns from `trace0` to `trace19` in `RAN_CONFIG`. For Vanilla DS, no additional option is required. For TECC, set `TECC_OPTION = True`. For CAMF, set `CAMF_OPTION = True`. |

Parameters not listed here can be left at their default values unless otherwise
specified.


## License

This project is licensed under the [MIT License](./LICENSE).

## Citation

If you use the IEEE Access version of DSSIM, please cite:

```bibtex
@ARTICLE{11357890,
  author={Suzuki, Akito and Itahara, Sohei and Ogawara, Takeo and Suzuki, Masaki},
  journal={IEEE Access},
  title={Multi-Generation DualSteer: Cross-Layer Traffic Control With Asymmetric RAN Feedback},
  year={2026},
  volume={14},
  number={},
  pages={12594-12604},
  keywords={6G mobile communication;Cross layer design;Throughput;Protocols;Downlink;Costs;5G mobile communication;Computer architecture;3GPP;Packet loss;Cellular network;mobile core;multi-access aggregation;DualSteer;cross-layer control;3GPP;6G network},
  doi={10.1109/ACCESS.2026.3655089}
}
```

## Contact

kr-mcn@kddi.com
