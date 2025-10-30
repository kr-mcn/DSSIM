# DSSIM
DualSteer Simulator (DSSIM) is a simulation framework for core-network-level multi-access communication, known as DualSteer in 3GPP. This organization hosts simulation code used in our research, promoting reproducibility and enabling comparative studies on transport protocols, congestion control, and cross-layer design.

## Features
- **DualSteer architecture implementation**  
  Simulates core-network–level multi-access aggregation with traffic splitting at the UPF and reassembly at the UE.

- **Asymmetric RAN feedback control**  
  Implements the proposed Single-RAN FB scheme, where only the 6G RAN provides feedback (experienced throughput and buffer occupancy) to the UPF.

- **Configurable feedback modes**  
  Supports three modes: No-RAN FB, Single-RAN FB (proposed), and Dual-RAN FB (upper bound).

- **5G/6G wireless modeling with [PyWiCh](https://github.com/PyWiCh/PyWiCh)**  
  Includes sample propagation configurations for different mobility patterns (train, car, and foot).

- **Millisecond-level simulation**  
  Operates on a 1 ms time step, enabling fine-grained analysis of throughput, delay, and loss.


## Installation
Clone this repository and install the below:
- Python 3.10.12 or later
- NumPy 2.2.4 or later


## Usage
Run the main.py:

```bash
python main.py
```
The simulator reads all configuration settings from the param.py.
You can modify this file to select which configuration under heavy_data/conf/ to use and adjust simulation parameters such as duration or logging options.


## Configuration

All simulation parameters are defined in `src/param.py`.

Only this file needs to be modified to change simulation settings.  
The main categories of parameters are:

| Category | Key Parameters | Description |
|-----------|----------------|--------------|
| **Simulation Time** | `NUM_SIMULATION_TIME_SLOTS`, `TIME_SLOT_WINDOW` | `TIME_SLOT_WINDOW` defines the time resolution, and 0.001 (1ms) is recommended. The total simulation time is calculated as `NUM_SIMULATION_TIME_SLOTS × TIME_SLOT_WINDOW`. |
| **Feedback Mode** | `RAN_FB_OPTION` | Selects the feedback configuration: `NONE` (no feedback), `SINGLE` (feedback from 6G only), or `BOTH` (feedback from both 5G and 6G). |
| **Transport Protocol** | `UDP_MODE` | Determines the transport layer protocol at the application level. If `True`, UDP is used (no congestion control); if `False`, QUIC is used. |
| **Congestion Control** | `QUIC_CC`, `MPQUIC_CC` | Specifies congestion control algorithms for the QUIC and MPQUIC layers (default: `CUBIC`). |
| **Feedback Behavior** | `RAN_FB_CYCLE`| Sets the feedback reporting interval. |
| **Experienced Throughput** | `EXP_THPT_RANGE_SEC` | Time window (in seconds) for the moving average of experienced throughput. |
| **Radio Configurations** | `RAN_CONFIG_5G`, `RAN_CONFIG_6G` | Specifies which propagation configuration under `heavy_data/conf/` to use for the 5G and 6G RANs. |
| **Logging** | `LOG_SAVE_PATH` | Automatically generated log directory name, including key parameters and timestamps. |


Each config directory (e.g., `heavy_data/conf/large_2GHz_10UEs_.../`) contains **pre-generated radio propagation data**.  
Each file represents one dataset, where **each line corresponds to a single time index** and the value indicates the achievable spectral efficiency in **bit/Hz**.  
The simulator loads the value corresponding to the current time index — that is, **line *n+1* is used when `time_index = n`**.
You only need to modify `param.py` to switch between configurations.




## License
This project is licensed under the [MIT License](./LICENSE).


## Citation
TBD

## Contact
kr-mcn@kddi.com
