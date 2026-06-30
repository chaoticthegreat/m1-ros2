# Deploying M1 on an NVIDIA Jetson AGX Orin — agent runbook

**Audience: the AI agent that just cloned this repo onto an AGX Orin.** This is
your end-to-end runbook for turning a bare Orin into a working M1 control box.
It is written to be executed top to bottom, with explicit *detect → decide →
do → verify* gates. **Do not skip the verification gates** — each phase proves
the previous one before you touch anything more expensive (and, eventually,
before you energize motors).

> **You don't need physical hardware for Phases 0–5.** The entire software stack
> (the Drake IK brain + the `ros2_control` mock) is validated offline. Get all of
> that green first; only Phases 6+ need the CAN bus and motors.

If anything in here disagrees with what you observe on the actual device, **trust
the device and say so** — the Jetson firmware/kernel landscape moves fast and
this guide was written 2026-06-30. Verify, don't assume.

---

## 0. Orient (read this first — 5 minutes)

### What you are deploying

This robot is an AgileX swerve base + prismatic lift + dual 7-DOF OpenArm arms
(27 DOF). The control "brain" gives an arm a **Cartesian target point** and the
arm joints + shared lift reach toward it (position-only — orientation is ignored
by design); the base is driven by a velocity command. On the dev machine this
runs against Isaac Sim. **On the Orin there is no sim** — you replace the sim
driver (`isaac/ros_sim.py`) with a `ros2_control` stack that talks to the real
Damiao CAN motors. Everything else is identical.

```
operators ──/m1/*──► m1_controller (Drake IK + swerve, 60 Hz) ──► /m1/joint_command
   (web/quest/teleop)                                                     │
                          ┌────────────────────────────────────────────────┤
                          ▼ upper body (lift+arms+grippers)                ▼ base (optional)
              m1_joint_bridge ─► ros2_control                    m1_base_bridge ─Twist─► AgileX driver
              arm_position_controller (forward position)                    │
                          ▼                                                 ▼ CAN feedback
              m1_hardware/M1SystemInterface (Damiao MIT, SocketCAN)  m1_ranger_shim ─► /joint_states
```

The two architecture docs you must read before non-trivial work:
- **`AGENTS.md`** — full architecture, every node, what's verified, gotchas.
- **`ros2_ws/HARDWARE.md`** — the real-hardware seam (this is the doc this guide
  operationalizes for the Orin specifically).
- `CLAUDE.md` — the hard rules. The most-broken ones are repeated below.

### Hard invariants (do not violate)

1. **Interpreter.** Run ROS / solver scripts with **`/usr/bin/python3`** (Jazzy,
   Python 3.12). A bare `python3` on `PATH` may be a conda 3.13 — wrong numpy, no
   `rclpy`, no Jazzy messages. `ros2 run` / `ros2 launch` pick the right
   interpreter via their shebang; standalone scripts do not. Standalone
   `m1_control` modules need `PYTHONPATH=ros2_ws/src/m1_control`; `m1_can_tools`
   modules need `PYTHONPATH=ros2_ws/src/m1_can_tools`.
2. **The arm reach is POSITION-ONLY.** A target is a 3D point. Don't reintroduce
   orientation.
3. **60 Hz is a soft goal, not a hard cutoff.** Accuracy first.
4. **The `/m1/*` + `/joint_states` contract is invariant** (27-DOF name order;
   wheels = velocity, everything else = position; the two `*_finger_joint2` are
   mimic-coupled state-only). The brain and all operator nodes are unchanged from
   sim — you are only swapping what produces `/joint_states` and consumes
   `/m1/joint_command`.
5. **Bus ownership is exclusive:** the `m1_hwconfig` maintenance tool **XOR** the
   `ros2_control` run stack owns the CAN bus. Never both at once.
6. **E-stop must be a hardware mechanism.** Never route e-stop through a ROS topic.
7. **Cleaning up a launched stack:** SIGINT the `ros2 launch` process, then sweep
   leftover PIDs **by PID**. **Never `pkill -f <node-name>`** — the pattern
   matches your own shell's command line and SIGKILLs your shell.

### The reference environment (this stack is already proven on ARM64)

The dev machine is **also `aarch64` + Ubuntu 24.04 Noble + ROS 2 Jazzy + Python
3.12**, so the entire software stack is known-good on ARM64. Reproduce this exact
set on the Orin:

| Component | Proven value | Source |
|---|---|---|
| Arch / OS | `aarch64` / Ubuntu 24.04 Noble | `uname -m`, `/etc/os-release` |
| ROS | Jazzy (`/opt/ros/jazzy`), Python 3.12.x | apt |
| Drake | `drake` 1.54.0, wheel `cp312 manylinux_2_34_aarch64` | pip (`--user --break-system-packages`) |
| `ros2_control` | `controller_manager`, `ros2_controllers`, `joint_state_broadcaster`, `forward_command_controller`, `joint_trajectory_controller`, mock components | apt |
| CAN userspace | `can-utils`, `python-can` 4.6.1 | apt + pip |

The **only** things that differ on the Orin: the host OS comes from JetPack (see
Phase 2), the CAN bus is real hardware (Phase 6), and you should apply Jetson
power/clock tuning (Phase 10).

### The fast path

There is a bootstrap script that does Phases 2–4 (software install + build) on a
Noble host. Use it, then verify:

```bash
# from the repo root, on the Orin:
./deploy/agx-orin/setup_agx_orin.sh --check     # detect & print the plan, change nothing
./deploy/agx-orin/setup_agx_orin.sh             # install ROS deps + Drake + build the workspace
```

It is idempotent (safe to re-run) and refuses to do the wrong thing on a non-Noble
host (it points you at the Docker path instead). Read the rest of this guide to
understand what it does and to do the hardware phases it deliberately does *not*
do.

---

## 1. Fingerprint the device

Before choosing anything, learn what you're on. Run all of these and record the
answers:

```bash
uname -m                                   # expect: aarch64
. /etc/os-release && echo "$VERSION_CODENAME ($VERSION_ID)"   # noble (24.04) ? jammy (22.04) ?
dpkg --print-architecture                  # arm64
cat /sys/firmware/devicetree/base/model 2>/dev/null; echo     # e.g. "NVIDIA Jetson AGX Orin ..."
cat /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release (not an L4T/Jetson image)"
dpkg-query -W -f='${Version}\n' nvidia-l4t-core 2>/dev/null || true
uname -r                                    # kernel; L4T BSP is 5.15-tegra (JP6) or 6.8 (JP7)
ldd --version | head -1                      # glibc; need >= 2.34 for the Drake wheel (Noble = 2.39)
/usr/bin/python3 --version                   # need 3.12.x for the ROS + Drake stack
nproc                                        # AGX Orin = 12 cores
```

**Interpret `nv_tegra_release` / `os-release`:**

| You see | Means | Path |
|---|---|---|
| `R36` + `jammy` (22.04) | **JetPack 6.x** (L4T r36, kernel 5.15) | ROS Jazzy is **not** natively supported on 22.04 → **Path B (Docker)** or re-flash to JetPack 7.2 |
| `R38`/`R39` + `noble` (24.04) | **JetPack 7.x** (L4T 38/39, kernel 6.8) | **Path A (native Jazzy)** |
| no `nv_tegra_release`, `noble`, `aarch64` | A generic ARM64 Noble box (like the dev DGX Spark) | **Path A (native Jazzy)** — the Jetson-specific tuning in Phase 10 won't apply |

> **Why this matters:** ROS 2 Jazzy's only Tier-1 Ubuntu is **24.04 Noble**
> (per REP-2000). 22.04 Jammy is source-build-only (no apt binaries). And Drake's
> official ARM64 wheel targets Noble + Python 3.12. So the whole stack wants
> **Noble**. JetPack decides whether you get Noble natively.

---

## 2. Get to Ubuntu 24.04 Noble + ROS 2 Jazzy

Pick **one** path based on Phase 1.

### Path A — native Jazzy on JetPack 7.2 (Noble)  ✅ preferred when available

JetPack **7.2** (Jetson Linux 39.2, Ubuntu 24.04, kernel 6.8, CUDA 13.0; GA
~June 2026) is the **first** JetPack 7 release that supports the **Orin** family
(7.0/7.1 were Thor-only). If `os-release` already says `noble`, you're here. If
you're on JetPack 6 and want this path, it's a **full re-flash** (SDK Manager or
the unified-ISO/USB installer) — *not* an apt dist-upgrade; back up first and
confirm your carrier board is supported in the Jetson Linux 39.2 release notes.

> ⚠️ JetPack 7.2 Orin support is new. The CUDA-13/kernel-6.8 jump can break
> JetPack-6-era CUDA/vendor driver stacks. **This control box needs no CUDA**
> (Drake IK is CPU-only), so that's not a blocker for *this* app — but if other
> things on the Orin depend on a JetPack-6 driver stack, prefer Path B.

Install native Jazzy (the bootstrap script automates this):

```bash
# locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8 && sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# universe + the ROS 2 apt source (current .deb method; the old apt-key way is deprecated)
sudo apt install -y software-properties-common curl && sudo add-apt-repository -y universe
export RAS=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${RAS}/ros2-apt-source_${RAS}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb && sudo apt update && sudo apt upgrade -y

# ROS itself — ros-base (headless control box), NOT desktop. Plus the bits this
# repo uses at runtime that are NOT pulled by rosdep (they aren't in any package.xml):
sudo apt install -y \
  ros-jazzy-ros-base ros-dev-tools \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
  python3-colcon-common-extensions python3-rosdep
# Optional (only for planned collision-aware moves, Phase 3 of HARDWARE.md):
# sudo apt install -y ros-jazzy-moveit

sudo rosdep init || true   # ignore "already initialized"
rosdep update
```

Then go to **Phase 3**.

### Path B — Jazzy in Docker on JetPack 6 (Jammy)

Choose this if the Orin must stay on the mature, long-proven JetPack 6.x firmware
(safest for peripheral/driver stability), or if re-flashing to 7.2 isn't
acceptable. The container gives you the exact Noble+Jazzy userspace the app needs;
only the host OS differs, and the `/m1/*` contract is unchanged.

```bash
sudo apt install -y docker.io
sudo usermod -aG docker "$USER"     # then re-login
docker pull osrf/ros:jazzy-ros-base # confirm it resolves an arm64/aarch64 manifest:
docker run --rm osrf/ros:jazzy-ros-base uname -m   # expect aarch64
```

Build a thin image with this repo's extra deps (write this `Dockerfile` next to
the repo, or use `deploy/agx-orin/Dockerfile` if present):

```dockerfile
FROM osrf/ros:jazzy-ros-base
RUN apt-get update && apt-get install -y \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
      python3-pip can-utils libyaml-cpp-dev \
    && pip3 install --break-system-packages drake python-can \
    && rm -rf /var/lib/apt/lists/*
```

```bash
docker build -t m1-jazzy -f deploy/agx-orin/Dockerfile .
```

Run it with **host networking + device passthrough** so SocketCAN (`can0`, a host
kernel interface) and any serial dongle reach the container:

```bash
docker run -it --rm --network host --cap-add NET_ADMIN \
  -v /dev:/dev -v "$PWD":/work -w /work m1-jazzy
# inside the container: build + run exactly as Path A from Phase 4 on.
# NB: bring up can0 on the HOST (Phase 6), not in the container.
```

> Inside the container, `python3` *is* Jazzy's 3.12 — no conda conflict — but the
> rest of this guide still writes `/usr/bin/python3` for consistency; both resolve
> to the same interpreter there.

**Do not** use conda/RoboStack (forbidden by this repo), and **do not** pull an
Isaac-ROS Jazzy image expecting Orin support (those target x86 + Jetson Thor).
Plain `osrf/ros:jazzy-ros-base` is the right base; no Isaac is needed at runtime.

---

## 3. Install Drake (the IK backend) + system deps

Drake (`pydrake`) is a **hard runtime dependency** of the brain — the Cartesian
reach IK won't construct without it — but it's deliberately not in any
`package.xml` (it's a pip wheel, not a rosdep). Install it for the **Jazzy
interpreter**:

```bash
/usr/bin/python3 -m pip install --user --break-system-packages -U pip
/usr/bin/python3 -m pip install --user --break-system-packages drake
# pin for reproducibility if you like: ... drake==1.54.0
```

This pulls the official `drake-*-cp312-cp312-manylinux_2_34_aarch64.whl` (~41 MB,
needs glibc ≥ 2.34; Noble has 2.39). Linux ARM64 is an **officially supported**
Drake platform (not experimental) and the public wheel **includes SNOPT** (the
solver the IK prefers; it falls back to IPOPT) with no license needed.

Verify Drake under the **right** interpreter:

```bash
/usr/bin/python3 -c 'import pydrake; print("drake", pydrake.__version__)'
/usr/bin/python3 -c 'from pydrake.solvers import SnoptSolver, IpoptSolver; \
  print("snopt", SnoptSolver().available(), "ipopt", IpoptSolver().available())'
/usr/bin/python3 -c 'import pydrake.all; print("pydrake.all OK")'
```

**If `import pydrake.all` fails on a missing shared lib** (commonly
`libOpenGL.so.0` / `libGL`), install the runtime libs and retry:

```bash
sudo apt-get install -y libgl1 libopengl0 libglib2.0-0 libx11-6
```

If `pip install drake` reports "no matching distribution", the cause is almost
always the **wrong interpreter** (conda `python3`, not `/usr/bin/python3`) or an
**old pip** that doesn't recognize the `manylinux_2_34` tag — not a missing wheel.
Re-check `which -a python3` and upgrade pip. Building Drake from source is a
multi-hour, RAM-hungry last resort; you should not need it.

Other system/CAN userspace deps (the bootstrap script installs these too):

```bash
sudo apt install -y can-utils                  # candump/cansend for CAN bring-up + debug
/usr/bin/python3 -m pip install --user --break-system-packages python-can pyserial
# python-can: real SocketCAN path of m1_hwconfig. pyserial: the DAMIAO USB2CAN serial dongle.
# Both are lazily imported and only needed on the real CAN path.
```

---

## 4. Build the ROS workspace

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash

# Pull the rosdep-resolvable deps for all packages (this does NOT pull
# ros2_control / controllers / drake — those were installed explicitly above):
rosdep install --from-paths src --ignore-src -y --rosdistro jazzy

colcon build --symlink-install
source install/setup.bash
```

The workspace has **6 packages**: `m1_control` + `m1_bringup` (ament_python),
`m1_can_tools` (ament_python), `m1_hardware` (ament_cmake **C++** ros2_control
plugin — Damiao CAN driver with vendored `openarm_can`), and the two description
packages `ranger_air_description` + `openarm_description` (ament_cmake, URDF +
meshes). A clean build finishes with no errors.

Notes:
- `m1_hardware` builds and **loads with no CAN bus present** (it falls back to
  mock/echo I/O), so the build succeeds on a bench with nothing plugged in.
- It links `yaml-cpp` via `yaml_cpp_vendor` (pulled by rosdep); if a standalone
  build complains, `sudo apt install -y libyaml-cpp-dev`.
- It does **not** build `openarm_can`'s own CMake/CLI/python targets — only the
  vendored C++ sources — so you do **not** need CLI11 / nanobind / scikit-build.

---

## 5. Verify the software offline (the big gate — do this before any hardware)

This proves the brain + the `ros2_control` mock work on the Orin's silicon with
**zero hardware risk**. Run from the **repo root** with `/usr/bin/python3`.

### 5a. The gated solver/robot regression suites

```bash
/usr/bin/python3 _solver_test.py            # 15/15  single/dual reach, tracking, latency
/usr/bin/python3 _solver_test_positions.py  # 21/21  300 single + 200 dual reachable points
/usr/bin/python3 _solver_test_tracking.py   # 21/21  continuous sweeps / singularity crossings
/usr/bin/python3 _solver_test_pathing.py    # 23/23  plan-and-track A→B landing accuracy
/usr/bin/python3 _accuracy_bench.py         # 10/10  accuracy regression gates
/usr/bin/python3 _swerve_test.py            # 16/16  swerve IK/FK + odometry
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.collision    # 9/9
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.trajectory   # 4/4

# Hardware-bridge + CAN codec tests (no hardware needed):
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 ros2_ws/src/m1_control/_bridge_test.py    # 15/15
PYTHONPATH=ros2_ws/src/m1_can_tools /usr/bin/python3 -m pytest ros2_ws/src/m1_can_tools/test -v  # 34/34
```

Each script prints `[PASS]/[FAIL]` lines, an `N/N gates passed` total, and exits 0
only if all gates pass. (There's a `/run-solver-suite` skill that runs the brain
suites and reports a metrics table.) Allow generous timeouts — the cold
multi-seed IK suites are slow.

**Also read the printed worst-case `solve_step` ms.** On the dev silicon it's
~20–21 ms (median ~1–2 ms; 60 Hz is a soft goal). The Orin's Cortex-A78AE cores
are different — note the number now, then re-check it **after** the Phase 10
power/clock tuning to confirm the budget holds on this board.

### 5b. The `ros2_control` mock bring-up (full stack, no motors)

This is the dress rehearsal for real hardware: it runs the **exact** launch you'll
use on hardware, but with `mock_components` instead of the Damiao plugin.

```bash
source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
ros2 launch m1_bringup hardware.launch.py use_mock:=true
```

In another shell, confirm the stack is live and the brain → bridge → controller
path flows:

```bash
ros2 control list_controllers        # joint_state_broadcaster + arm_position_controller = active;
                                     # left_arm_jtc / right_arm_jtc = inactive (planned moves only)
ros2 topic hz /joint_states          # ~publishing
ros2 run m1_control m1_send_pose --arm left --xyz 0.30 0.20 0.95   # command a reach
ros2 topic echo /m1/joint_command --once                          # the brain is emitting commands
```

Clean up: **SIGINT (Ctrl-C) the `ros2 launch`**, then sweep leftover PIDs by PID
(never `pkill -f`).

> ✅ **Gate:** do not proceed to hardware until 5a is all-green and 5b comes up
> with both controllers active and a reach command flowing. If a suite *errored*
> (import/dep problem) vs *failed a gate*, fix the dep first (almost always Drake
> under the wrong interpreter).

---

## 6. CAN bus bring-up (first hardware step)

The Damiao DM motors (arms + lift) speak **SocketCAN CAN-FD at 1 Mbps nominal /
5 Mbps data**. You need a `can0` interface up at those rates before the real
launch. There are two host transports; **detect what you have first**, because the
"obvious easy" choice (a USB CANable) is often the *hardest* on a stock Jetson
kernel.

### 6a. Detect

```bash
ls /sys/class/net | grep -i can            # already-present can interfaces
ip -d link show type can 2>/dev/null       # their type/state if any
lsmod | grep -E 'can|mttcan|gs_usb|peak_usb'
modinfo can can_raw can_dev mttcan gs_usb peak_usb 2>&1 | grep -E 'filename|not found'
lsusb                                      # is a USB-CAN adapter enumerated?
dmesg | grep -iE 'mttcan|can[0-9]|gs_usb|peak'
ls /opt/nvidia/jetson-io/ 2>/dev/null      # Jetson-IO pin tool present?
```

### 6b. Choose the transport

| Transport | Stock-JetPack status | Use when |
|---|---|---|
| **Native `mttcan`** (Orin's on-SoC CAN controllers, 40-pin header) | `can`/`can_raw`/`mttcan` modules **ship** in stock JetPack, but the controller pins need a **pinmux/device-tree enable** and an **external 3.3 V transceiver** | the default, lowest-friction path on a bare Orin |
| **USB PEAK PCAN-USB FD** (`peak_usb`) | `peak_usb` is mainline → most likely present (`modinfo peak_usb`); transceiver is built into the dongle | you have a PEAK dongle and want no device-tree work |
| **USB CANable / candleLight** (`gs_usb`) | ⚠️ `gs_usb` is **NOT** prebuilt in stock JetPack 6 (`modprobe gs_usb` → "Module not found") → needs an out-of-tree kernel-module build | only if you accept building the module |
| **DAMIAO USB2CAN serial dongle** (`/dev/ttyACM0`) | needs no kernel CAN driver at all (it's vendor serial framing, not SocketCAN) | **bench/config fallback** — works for `m1_hwconfig` only, **not** for the `ros2_control` run path (the C++ plugin is SocketCAN-only) |

> The adapter named in `HARDWARE.md` (CANable/candleLight) is `gs_usb`-class, so on
> a stock AGX Orin it will fail until you build `gs_usb`. **Prefer native `mttcan`
> or a PEAK dongle.** On JetPack 7 (kernel 6.8) module availability may differ —
> trust your `modinfo`/`lsmod` output over this table.

### 6c. Wire it (native `mttcan` only)

- Add a **3.3 V CAN transceiver** (e.g. WaveShare SN65HVD230) — the Orin CAN pins
  are 3.3 V logic with **no on-board transceiver**. (USB dongles have one built in.)
- `CAN_H↔CAN_H`, `CAN_L↔CAN_L`, and a **common ground** between Orin, transceiver,
  and the motor bus (mandatory — without it frames corrupt / NACK).
- **120 Ω termination at BOTH physical ends** of the bus (≈60 Ω measured across
  CAN_H/CAN_L, bus unpowered). Some Damiao/reComputer boards have switchable
  on-board terminators — enable only at the ends.

### 6d. Enable native pins (native `mttcan` only)

```bash
sudo modprobe can && sudo modprobe can_raw && sudo modprobe mttcan
dmesg | grep -i mttcan      # want: "net can0: mttcan device registered"
```

If no `canX` registers, the controller pins aren't muxed to CAN. Enable them with
**Jetson-IO** (preferred, persistent) and reboot:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py   # enable the CAN function on the 40-pin header
```

(Alternatives: a DTB overlay setting `mttcan@c310000`/`@c320000` `status="okay"` +
the pinmux, or, only to *validate wiring* without a DTB rebuild, the documented
`busybox devmem` pinmux register writes — **addresses are board/pin specific; read
them from the NVIDIA CAN guide for your exact pins, never copy a forum's values
blindly.**)

### 6e. Bring the link up (all transports, once the device exists)

There's a helper that does this idempotently with the right rates:

```bash
sudo ./deploy/agx-orin/can_up.sh            # defaults: can0, CAN-FD 1M/5M
# or do it by hand:
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on restart-ms 100
sudo ip link set can0 txqueuelen 1000       # avoid ENOBUFS under bursty multi-motor traffic
```

Verify and smoke-test:

```bash
ip -d link show can0 | grep -iE 'state|fd|bitrate'   # state UP, "fd on", bitrate 1000000, dbitrate 5000000
candump can0                                          # in one shell; should show frames when motors are powered
/usr/bin/python3 -c 'import can; print("python-can", can.__version__)'
```

For **classic** CAN instead of FD (if a motor/adapter can't do 5 Mbps):
`sudo ip link set can0 up type can bitrate 1000000 restart-ms 100` and launch with
`can_fd:=false`.

(Persisting CAN at boot is Phase 11.)

---

## 7. Configure the motors (`m1_hwconfig`) — maintenance mode

**Bus ownership rule:** the config tool and `ros2_control` must never own the bus
at once. Do all of this with the run stack **down**.

```bash
# dry run first (fake transport, no bus needed) to learn the page at :8090:
ros2 run m1_can_tools m1_hwconfig
# against the real bus:
ros2 run m1_can_tools m1_hwconfig --ros-args -p transport:=socketcan -p can_channel:=can0
```

Open `http://<orin-ip>:8090`. Use the page to **scan** the bus, set/verify each
motor's **CAN ID + master ID**, **map** each motor → logical joint, edit per-joint
**soft limits**, **set-zero**, and **jog/test** a motor (clamped slider + dead-man).
Save the map; it writes a YAML in the schema the C++ plugin reads.

The map schema (see the committed template `ros2_ws/src/m1_can_tools/config/motor_map.example.yaml`):

```yaml
openarm_left_joint1:
  id: 0x02            # CAN slave id
  master_id: 0x12     # host/master id (feedback arb id; defaults to id + 0x10; never 0)
  model: DM4340       # DM model -> per-model [P,V,T]MAX quantization (DM3507/4310/4340/6006/8009/...)
  kp: 70.0            # MIT-mode stiffness (the brain's setpoint is impedance-tracked)
  kd: 2.0             # MIT-mode damping
  dir: 1              # +1/-1 joint-direction sign vs the motor
  offset: 0.0         # zero offset (rad/m) applied to feedback
  soft_limits: {pos: [-3.10, 3.10], vel: 6.0, effort: 20.0}
```

Save the working map where the launch will read it (convention):
`$HOME/.config/m1/motor_map.yaml`.

> ⚠️ The example part numbers/IDs are **placeholders**. These arms are customized
> OpenArm — confirm each motor's real DM model and assign IDs on the bench. The
> reasonable default gains are in `ros2_ws/src/m1_hardware/config/control_gains.yaml`
> (lift DM8009 kp120/kd3; proximal arm kp70; wrist kp10; grippers kp5) — expect to
> tune for sag/overshoot on hardware.

**The kp==0 limp-arm safety guard:** if any of the **17 commanded** joints resolves
to `kp == 0` (e.g. a missing/empty `motor_map` on a real launch), the plugin logs
`UNSAFE CONFIG` and **refuses to open the CAN bus** (motors never enabled). So a
complete, non-zero-gain motor map is mandatory before live driving. (There are 17
commanded motors, not 19 — the two `*_finger_joint2` are state-only mimics.)

---

## 8. Real-hardware bring-up

With `can0` up (Phase 6), a valid `motor_map.yaml` (Phase 7), the config tool
**stopped**, and **a hardware e-stop within reach**:

```bash
source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
ros2 launch m1_bringup hardware.launch.py \
  use_mock:=false can_interface:=can0 can_fd:=true \
  motor_map:=$HOME/.config/m1/motor_map.yaml
```

`hardware.launch.py` arguments (defaults in parentheses): `use_mock` (`true`),
`use_rviz` (`false`), `use_base` (`false`), `can_interface` (`can0`), `can_fd`
(`true`), `motor_map` (`""`). It starts: `robot_state_publisher`, the
`controller_manager` (hosting the `m1_hardware/M1SystemInterface` plugin),
spawners for `joint_state_broadcaster` + `arm_position_controller` (+ inactive
per-arm JTCs), `m1_joint_bridge`, and the `m1_controller` brain.

### Live-validation checkpoints (the offline suites cannot catch these)

The offline suites use perfect feedback and zeroed fingers. On real motors, verify
in order, **cautiously, with the e-stop ready**:

1. **Per-joint sign + offset.** Jog each joint a small amount (via `m1_hwconfig`
   *before* this launch, or watch `/joint_states` here) and confirm the joint
   moves the **correct direction** and reads a sane zero. Fix `dir`/`offset` in the
   motor map. A wrong sign on an impedance-controlled motor is dangerous — check at
   low gain first.
2. **The live closed loop.** Command a reach and confirm the controller's
   **command-fingertip** matches the **measured** `/joint_states` fingertip to
   ~0 mm. A persistent offset means a mimic/sign/scale bug. (This is the discipline
   the brain's offline gates can't exercise.)
3. **Gains.** Expect gravity-comp tuning — watch for sag (raise kp/kd) or
   overshoot (lower) per joint, starting from `control_gains.yaml`.

Operator interfaces are **identical to sim** (they speak only `/m1/*` +
`/joint_states`):

```bash
ros2 run m1_control m1_web        # browser panel at http://<orin-ip>:8080
ros2 run m1_control m1_teleop     # terminal keyboard console
ros2 run m1_control m1_quest      # Meta Quest WebXR teleop (HTTPS, self-signed)
ros2 run m1_control m1_send_pose --arm left --xyz 0.30 0.20 0.95
```

> If "nothing moves": the controller stays idle until `/joint_states` arrives (it
> seeds its command pose from feedback first). Confirm the broadcaster is
> publishing and the plugin actually opened the bus (it logs whether it's driving
> motors vs running mock/echo I/O — a `UNSAFE CONFIG` line means a kp==0 in the
> map).

---

## 9. (Optional) AgileX base

The base path is **wired but the AgileX driver is not vendored** — it's a
bring-up TODO. The ROS-side bridges are implemented and unit-tested:
`m1_base_bridge` (`/m1/cmd_vel` → body `Twist` + motion-mode) and `m1_ranger_shim`
(AgileX wheel feedback → `/joint_states`). Launch them with `use_base:=true`.

To finish on hardware (see `ros2_ws/HARDWARE.md` "AgileX base integration"):
1. Clone + build `ranger_ros2` (`air_delta` branch for the Ranger Air) + `ugv_sdk`
   on Jazzy (no official Jazzy branch — budget a small rclcpp/tf2 port).
2. Bring up a **separate** base CAN adapter at **500 kbps** (AgileX
   `setup_can2usb`) — distinct from the arms' `can0`.
3. Point `m1_base_bridge`'s `/cmd_vel` at the driver and set `m1_ranger_shim`'s
   `steer_topic`/`wheel_topic` to the driver's feedback topics; map the motion-mode
   `Int8` to the driver's `SetMotionMode`.
4. Stock Ranger firmware is **mode-switched** (PARALLEL/SPINNING/DUAL_ACKERMANN),
   not free-holonomic — `swerve.py`'s per-module output can't drive it. Confirm
   whether your base is stock (Twist-only) or exposes per-module control.

---

## 10. Jetson runtime tuning (skip on a non-Jetson box)

For a 60 Hz (16.7 ms) **soft**-real-time loop the dominant win is removing
clock/frequency jitter, not chasing hard RT. Apply in tiers, lowest-effort-first,
and **re-run `_solver_test.py` before/after** to see the worst-case `solve_step`
ms improve.

**Tier 1 — power + clocks (do this; ~5 min, biggest win):**

```bash
sudo nvpmodel -q                 # check current mode FIRST — numbering is board-specific
sudo nvpmodel -m 0               # MAXN on AGX Orin (persists across reboot). Verify the number with -q.
sudo jetson_clocks               # lock CPU/GPU/EMC to max, disabling DVFS (NOT persistent → systemd, Phase 11)
# confirm clocks actually sit at max (governor name may still read schedutil):
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
```

Why: with DVFS active a Drake solve landing on a down-clocked core eats a ramp-up
first → variable latency. Locking clocks makes per-tick compute flat. **Caveat:**
MAXN is *unconstrained power*, not *no-throttle* — sustained load will thermally
throttle (and jitter returns) without cooling. Keep `nvfancontrol` running and
watch `sudo tegrastats`.

**Tier 2 — CPU isolation + RT priority for the control loop (cheap, high value):**

```bash
# /boot/extlinux/extlinux.conf APPEND line (Jetson uses extlinux, NOT GRUB): isolate 1-2 of 12 cores
#   ... isolcpus=10,11 nohz_full=10,11 rcu_nocbs=10,11
# then reboot and confirm:  cat /sys/devices/system/cpu/isolated
# run the control node pinned + SCHED_FIFO (needs rtprio ulimit — LimitRTPRIO in the unit or limits.conf):
taskset -c 10 chrt -f 80 <command>
```

**Tier 3 — PREEMPT_RT kernel: OVERKILL for soft 60 Hz. Skip it.** NVIDIA ships no
prebuilt RT kernel for JetPack; you'd patch+build L4T yourself. Only warranted if
you later need kHz hard-RT.

**DDS (single host / small LAN):**

```bash
# If everything runs on the Orin, kill cross-host discovery jitter:
export ROS_LOCALHOST_ONLY=1
# If off-board viz (a laptop RViz, the Quest over LAN) must reach the Orin, use instead:
# export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
# Raise kernel UDP buffers (the documented Fast DDS first-message-latency fix):
sudo tee /etc/sysctl.d/60-ros2-dds.conf >/dev/null <<'EOF'
net.core.rmem_max=2147483647
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
EOF
sudo sysctl -p /etc/sysctl.d/60-ros2-dds.conf
# Optional lower-latency RMW: sudo apt install ros-jazzy-rmw-cyclonedds-cpp; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

(Don't bother with iceoryx/shared-memory zero-copy — the joint msgs are tiny; it
only helps for images/point clouds.)

---

## 11. Autostart on boot (systemd)

Templates live in `deploy/agx-orin/systemd/`. They are **templates** — edit the
placeholder paths/user, then install. The dependency order is: bring up CAN →
(Jetson) lock clocks → launch the stack.

```bash
# 1) edit the env file with your paths/launch args:
$EDITOR deploy/agx-orin/systemd/m1.env
# 2) install (copy units, reload, enable):
sudo cp deploy/agx-orin/systemd/can0.service          /etc/systemd/system/
sudo cp deploy/agx-orin/systemd/jetson-clocks.service /etc/systemd/system/   # Jetson only
sudo cp deploy/agx-orin/systemd/m1-bringup.service    /etc/systemd/system/
sudo mkdir -p /etc/m1 && sudo cp deploy/agx-orin/systemd/m1.env /etc/m1/m1.env
sudo systemctl daemon-reload
sudo systemctl enable --now can0.service jetson-clocks.service m1-bringup.service
# auto-load CAN modules at boot too (native mttcan shown; add gs_usb/peak_usb if USB):
echo -e 'can\ncan_raw\nmttcan' | sudo tee /etc/modules-load.d/can.conf
```

> **Known Orin quirk:** a `jetson_clocks` service that runs too early in boot
> silently no-ops — the unit includes `ExecStartPre=/bin/sleep 90`. Also ensure the
> bring-up service does **not** race the config tool for the bus (bus ownership is
> exclusive); don't enable `m1-bringup.service` while you're still using
> `m1_hwconfig`.

Verify after a reboot:

```bash
systemctl is-active can0.service jetson-clocks.service m1-bringup.service
ip -d link show can0
ros2 control list_controllers
```

---

## 12. Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install drake` → "no matching distribution" | wrong interpreter (conda `python3`) or old pip | use `/usr/bin/python3`; `pip install -U pip`; confirm `aarch64` + glibc ≥ 2.34 + py3.12 |
| `import pydrake.all` → `libOpenGL.so.0` missing | headless image lacks GL libs | `sudo apt install libgl1 libopengl0 libglib2.0-0` |
| solver suite *errors* (not fails a gate) | Drake not importable under `/usr/bin/python3` | redo Phase 3 verification |
| `rclpy` not found / wrong numpy | ran a script with conda `python3` | use `/usr/bin/python3`; `source /opt/ros/jazzy/setup.bash` |
| `colcon` can't find `ros2_control` controllers at launch | not in any `package.xml`; rosdep won't pull them | `apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager` |
| `modprobe gs_usb` → "Module not found" | `gs_usb` not in stock JetPack kernel | use native `mttcan` or a PEAK (`peak_usb`) dongle, or build `gs_usb` out-of-tree |
| `mttcan` loads but no `can0` | CAN pins not muxed | `sudo /opt/nvidia/jetson-io/jetson-io.py` → enable CAN → reboot |
| `ip link ... fd on` works but no frames on the bus | missing transceiver / termination / common ground | add 3.3 V transceiver, 120 Ω at both ends, tie grounds |
| plugin logs `UNSAFE CONFIG`, won't drive | a commanded joint has `kp == 0` (bad/empty motor_map) | provide a complete motor_map with non-zero gains for all 17 |
| arms move but command-fingertip ≠ measured | per-joint `dir`/`offset` or mimic/sign bug | fix `dir`/`offset` in the map; verify the live loop (checkpoint 8.2) |
| robot tips / "solver fails to converge" on high/forward targets | not the solver — a **dynamics** issue (base too light) on real hw too; check joint efforts / physical stability | this was the sim base-mass bug; on hardware verify the base is stable and joints aren't saturating |
| viz freezes for seconds (Quest) | network stall, not the planner | known + fixed (request deadline); see AGENTS.md "Performance pass 3" |
| my shell got killed during cleanup | you ran `pkill -f <node-name>` | **never** do that; SIGINT the launch + sweep by PID |

---

## What "done" looks like

- Phase 5 (offline) all green on the Orin; worst-case `solve_step` within a sane
  multiple of 16.7 ms after Phase 10 tuning.
- `ros2 launch m1_bringup hardware.launch.py use_mock:=false ...` brings up the
  stack, the plugin opens the bus, `/joint_states` reflects real encoders, and a
  reach command drives the arms with command-fingertip ≈ measured.
- (If base) `use_base:=true` drives the chassis via the AgileX driver.
- Autostart units bring it all up on boot, CAN-first.

When you finish a phase, note what you verified (and any deviation from this guide
— firmware moves) so the next agent inherits ground truth, not assumptions.
