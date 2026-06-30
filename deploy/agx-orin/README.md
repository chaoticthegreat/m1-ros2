# `deploy/agx-orin/` — AGX Orin deployment assets

Everything here supports deploying the M1 control brain on an **NVIDIA Jetson
AGX Orin** (real-hardware, no Isaac Sim). **Start with the runbook:**
[`../../DEPLOY_AGX_ORIN.md`](../../DEPLOY_AGX_ORIN.md).

| File | What it does | Phase in the runbook |
|---|---|---|
| `setup_agx_orin.sh` | Idempotent software bootstrap: verifies the host, installs ROS 2 Jazzy + `ros2_control` + Drake + CAN deps, runs `rosdep`, builds the workspace, fast preflight. **No hardware touched.** | 2–5 |
| `can_up.sh` | Bring up `can0` (CAN-FD 1M/5M) with Jetson-aware detection (missing `gs_usb`, unmuxed native pins). | 6 |
| `Dockerfile` | Jazzy-in-Docker image for a JetPack 6 (Ubuntu 22.04) host that can't run native Jazzy. | 2 (Path B) |
| `systemd/m1.env` | Env file (paths, CAN mode, motor-map path, launch args) sourced by the units. **Edit before installing.** | 11 |
| `systemd/can0.service` | Bring up CAN at boot, before the stack. | 11 |
| `systemd/jetson-clocks.service` | Lock Jetson clocks at max each boot (Jetson only). | 10/11 |
| `systemd/m1-bringup.service` | Launch the real-hardware control stack on boot, after CAN. | 11 |
| `systemd/60-ros2-dds.conf` | Kernel UDP-buffer / DDS tuning. | 10 |

## Quick start

```bash
# from the repo root, on the Orin:
./deploy/agx-orin/setup_agx_orin.sh --check     # inspect the host, change nothing
./deploy/agx-orin/setup_agx_orin.sh             # install + build (native-Noble host)
# then follow DEPLOY_AGX_ORIN.md from Phase 5 (verify) onward.
```

## Safety reminders (full detail in the runbook)

- **Interpreter:** ROS/solver scripts run under `/usr/bin/python3`, never conda's.
- **Bus ownership is exclusive:** `m1_hwconfig` (config) XOR `ros2_control` (run).
- **E-stop must be hardware.** `m1-bringup.service` energizes motors on boot.
- The `kp==0` guard refuses to open the bus on an empty/bad motor map — that's by
  design (prevents limp/mis-scaled arms).
- Cleaning up a launch: SIGINT the launch, sweep by PID — **never** `pkill -f`.
