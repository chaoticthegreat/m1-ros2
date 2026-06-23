# M1 Real-Hardware Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the real M1 robot (Damiao CAN motors for dual arms + lift, AgileX base) with the existing Drake brain, plus a hardware config/test web page — without changing the brain or its gated tests.

**Architecture:** Hybrid. Reuse OpenArm's Damiao CAN/ros2_control hardware layer; keep the Drake position-only IK as the reactive brain; MoveIt optional/later. The `/m1/*` + `/joint_states` contract is preserved, so the new code is bridge nodes + a Python CAN tool layer + a C++ ros2_control SystemInterface + an AgileX Twist base path + a config web page. See `docs/superpowers/specs/2026-06-23-real-hardware-deployment-design.md`.

**Tech Stack:** ROS 2 Jazzy (Python 3.12 via `/usr/bin/python3`), ros2_control + ros2_controllers + mock_components (installed), MoveIt 2 + OMPL (installed), `pydrake` (installed), `python-can` (install for the real SocketCAN path; codec is pure-python), OpenArm `openarm_can`/`openarm_hardware` (clones in `/tmp`, Apache-2.0), AgileX `ranger_ros2`/`ugv_sdk`.

## Global Constraints

- **Interpreter:** standalone Python tests/scripts run with **`/usr/bin/python3`** (Jazzy 3.12). `python3` on PATH is conda 3.13 — wrong. Standalone solver/tool modules need `PYTHONPATH=ros2_ws/src/m1_control` (and `:ros2_ws/src/m1_can_tools` for the new package).
- **Reach is POSITION-ONLY.** Do not reintroduce orientation rows.
- **60 Hz is a soft goal**, not a hard cutoff — accuracy first.
- **The 27-DOF contract is invariant:** name order `fl/fr/rr/rl_steering_joint, lift_joint, openarm_left_joint1..7, openarm_right_joint1..7, openarm_left_finger_joint1/2, openarm_right_finger_joint1/2, fl/fr/rr/rl_wheel_joint`. Wheels=velocity, rest=position; fingers mimic-coupled (right negated); messages NaN-free.
- **DM CAN constants (verbatim from research, use exactly):** per-model `[P_MAX(rad), V_MAX(rad/s), T_MAX(Nm)]` = DM4310 `[12.5,30,10]`, DM4310_48V `[12.5,50,10]`, DM4340 `[12.5,8,28]`, DM4340_48V `[12.5,10,28]`, DM6006 `[12.5,45,20]`, DM8006 `[12.5,45,40]`, DM8009 `[12.5,45,54]`, DM10010L `[12.5,25,200]`, DM10010 `[12.5,20,200]`, DMH3510 `[12.5,280,1]`, DMH6215 `[12.5,45,10]`, DMG6220 `[12.5,45,10]`. KP range `0..500`, KD range `0..5`. Mode ID offsets: MIT=`+0x000`, POS_VEL=`+0x100`, VEL=`+0x200`, FORCE_POS=`+0x300`; param r/w=`0x7FF`. Special frames (send to slave id, 8 bytes): enable=`FF FF FF FF FF FF FF FC`, disable=`...FD`, set-zero=`...FE`, clear-error=`...FB`. Master ID convention = slave + 0x10, never 0.
- **Don't commit build artifacts** (`__pycache__`, `*.pyc`, `ros2_ws/{build,install,log}/`).
- **License:** new packages Apache-2.0 (match repo); vendored `openarm_can`/`openarm_hardware` keep their Apache-2.0 + NOTICE.

---

## File Structure

```
ros2_ws/src/
  m1_can_tools/                          (NEW ament_python pkg)
    m1_can_tools/
      __init__.py
      dm_protocol.py        # pure-python DM codec + limit tables (NO ros/can deps)
      transport.py          # Transport ABC, FakeTransport, SocketCanTransport, SerialTransport
      motor_bus.py          # maintenance-mode bus owner (scan/enable/jog/zero/telemetry) + ID->joint YAML
      hwconfig_node.py       # m1_hwconfig web node (stdlib HTTP + embedded HTML/JS)
      web/hwconfig.html      # the config page (served by hwconfig_node)
    config/motor_map.example.yaml
    test/                    # pytest, hardware-free
      test_dm_protocol.py
      test_transport.py
      test_motor_bus.py
      test_hwconfig_datapath.py
    package.xml  setup.py  setup.cfg  resource/m1_can_tools
  m1_control/                            (EXISTS — add only)
    m1_control/joint_command_bridge.py   # /m1/joint_command -> forward_position_controller
    m1_control/base_bridge.py            # /m1/cmd_vel -> AgileX Twist + motion-mode
    m1_control/ranger_state_shim.py      # AgileX wheel feedback -> /joint_states (8 base joints)
    _bridge_test.py                      # hardware-free unit tests for the 3 bridges (pure mappers)
  m1_hardware/                           (NEW ament_cmake pkg, Phase 2)
    src/m1_system_interface.cpp  include/m1_hardware/m1_system_interface.hpp
    m1_hardware.xml  CMakeLists.txt  package.xml
    vendor/openarm_can/                  # vendored Apache-2.0 CAN lib
    config/control_gains.yaml
  m1_bringup/                            (EXISTS — add only)
    launch/hardware.launch.py            # ros2_control_node + spawners + RSP + bridges (mock<->real arg)
    config/m1_controllers.yaml
    config/m1_joint_limits.yaml
    urdf/m1.ros2_control.xacro           # <ros2_control> tag (mock default, real by arg)
    urdf/m1_hardware.urdf.xacro          # includes ranger_air URDF + the ros2_control tag
    moveit/                              # Phase 3 MoveIt config (optional)
```

---

## Phase 0 — Damiao bring-up driver + config/test web page (no hardware, no ros2_control)

### Task 0.1: `m1_can_tools` package + DM limit tables & quantization

**Files:**
- Create: `ros2_ws/src/m1_can_tools/{package.xml,setup.py,setup.cfg,resource/m1_can_tools}`
- Create: `ros2_ws/src/m1_can_tools/m1_can_tools/__init__.py`, `dm_protocol.py`
- Test: `ros2_ws/src/m1_can_tools/test/test_dm_protocol.py`

**Interfaces — Produces:**
- `dm_protocol.LIMITS: dict[str, tuple[float,float,float]]` (model → (P_MAX,V_MAX,T_MAX), values from Global Constraints).
- `dm_protocol.float_to_uint(x: float, lo: float, hi: float, bits: int) -> int`
- `dm_protocol.uint_to_float(u: int, lo: float, hi: float, bits: int) -> float`

- [ ] **Step 1: failing test** — `test/test_dm_protocol.py`:
```python
from m1_can_tools import dm_protocol as dm
def test_limits_table_values():
    assert dm.LIMITS["DM4310"] == (12.5, 30.0, 10.0)
    assert dm.LIMITS["DM8009"] == (12.5, 45.0, 54.0)
    assert dm.LIMITS["DM4340"] == (12.5, 8.0, 28.0)
def test_quantization_roundtrip():
    for bits in (12, 16):
        hi = 12.5
        for x in (-hi, -1.0, 0.0, 3.3, hi):
            u = dm.float_to_uint(x, -hi, hi, bits)
            assert 0 <= u < (1 << bits)
            back = dm.uint_to_float(u, -hi, hi, bits)
            assert abs(back - x) <= (2*hi)/(1<<bits) + 1e-9
def test_float_to_uint_endpoints():
    assert dm.float_to_uint(-12.5, -12.5, 12.5, 16) == 0
    assert dm.float_to_uint(12.5, -12.5, 12.5, 16) == (1<<16) - 1
```
- [ ] **Step 2: run, expect fail** — `PYTHONPATH=ros2_ws/src/m1_can_tools /usr/bin/python3 -m pytest ros2_ws/src/m1_can_tools/test/test_dm_protocol.py -v` → ImportError.
- [ ] **Step 3: implement** `dm_protocol.py` LIMITS + float_to_uint (clamp to [lo,hi], `int((x-lo)*((1<<bits)-1)/(hi-lo))`) + uint_to_float inverse.
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: scaffold package.xml (ament_python, deps: rclpy, pyyaml; exec dep python3-can optional), setup.py (entry point `m1_hwconfig = m1_can_tools.hwconfig_node:main`), setup.cfg, resource file.** Build check: `source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon build --symlink-install --packages-select m1_can_tools` → clean.
- [ ] **Step 6: commit** `feat(can): m1_can_tools pkg + DM limit tables & quantization`.

### Task 0.2: DM CAN frame codec (the correctness-critical task)

**Files:** Modify `dm_protocol.py`; Test `test/test_dm_protocol.py`.

**Interfaces — Produces:**
- `encode_mit(p,v,kp,kd,tau, model) -> bytes` (8 bytes; packing per Global Constraints: q 16b, dq 12b, kp 12b[0..500], kd 12b[0..5], tau 12b).
- `encode_pos_vel(pos: float, vel: float) -> bytes` (two LE float32).
- `encode_vel(vel: float) -> bytes` (one LE float32).
- `special_frame(kind: str) -> bytes` where kind ∈ {"enable","disable","set_zero","clear_error"}.
- `arb_id(slave_id: int, mode: str) -> int` (mode ∈ {"mit","pos_vel","vel","force_pos"}).
- `decode_feedback(data: bytes, model: str) -> dict` → `{id, err, pos, vel, torque, t_mos, t_rotor}`.

- [ ] **Step 1: failing tests** with exact byte expectations:
```python
def test_special_frames():
    assert dm.special_frame("enable")      == bytes([0xFF]*7+[0xFC])
    assert dm.special_frame("disable")     == bytes([0xFF]*7+[0xFD])
    assert dm.special_frame("set_zero")    == bytes([0xFF]*7+[0xFE])
    assert dm.special_frame("clear_error") == bytes([0xFF]*7+[0xFB])
def test_arb_ids():
    assert dm.arb_id(0x01,"mit")==0x01
    assert dm.arb_id(0x01,"pos_vel")==0x101
    assert dm.arb_id(0x05,"vel")==0x205
def test_mit_zero_packing():
    # p=0,v=0,kp=0,kd=0,tau=0 over symmetric ranges -> midpoints
    b = dm.encode_mit(0,0,0,0,0,"DM4310")
    assert len(b)==8
    # q midpoint of 16b = 0x7FFF -> data[0]=0x7F data[1]=0xFF
    assert b[0]==0x7F and b[1]==0xFF
def test_mit_kp_kd_ranges():
    # kp uses 0..500, kd 0..5 (NOT symmetric) -> kp=0 -> 0, kp=500 -> 0xFFF
    b_lo = dm.encode_mit(0,0,0,0,0,"DM4310"); b_hi = dm.encode_mit(0,0,500,5,0,"DM4310")
    assert (((b_lo[3]&0xf)<<8)|b_lo[4]) == 0
    assert (((b_hi[3]&0xf)<<8)|b_hi[4]) == 0xFFF
def test_pos_vel_le_float():
    import struct
    assert dm.encode_pos_vel(1.0, 2.0) == struct.pack("<ff",1.0,2.0)
def test_decode_roundtrip_pos():
    # encode a feedback-style buffer and decode pos within quantization
    # build: id=1,err=0; q=0 -> 0x7FFF; vel=0,torque=0; t_mos=40,t_rotor=45
    data = bytes([0x01, 0x7F,0xFF, 0x7F,0xF0, 0x00, 40, 45])
    fb = dm.decode_feedback(data,"DM4310")
    assert fb["id"]==1 and fb["err"]==0
    assert abs(fb["pos"]) < 0.001 and fb["t_mos"]==40 and fb["t_rotor"]==45
```
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** the MIT 16/12/12/12/12 packing exactly as in Global Constraints (data[0..7] formula), pos_vel/vel via `struct.pack("<f"/"<ff")`, special_frame table, arb_id offsets, decode_feedback (D[0] low nibble=id, high=err; pos 16b D[1:3]; vel 12b `(D[3]<<4)|(D[4]>>4)`; torque 12b `((D[4]&0xF)<<8)|D[5]`; t_mos=D[6]; t_rotor=D[7]; map via uint_to_float with model limits).
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(can): DM MIT/pos-vel/vel codec + feedback decode (byte-exact, tested)`.

### Task 0.3: Transport abstraction (Fake / SocketCAN / Serial)

**Files:** Create `transport.py`; Test `test/test_transport.py`.

**Interfaces — Produces:**
- `transport.Transport` (ABC): `send(arb_id:int, data:bytes)->None`, `recv(timeout:float)->tuple[int,bytes]|None`, `close()->None`.
- `transport.FakeTransport(Transport)`: records sent frames in `.sent: list[tuple[int,bytes]]`; `.inject(arb_id,data)` queues a frame for `recv`.
- `transport.SocketCanTransport(channel="can0", fd=False)`: **lazy** `import can`; raises a clear error if python-can missing.
- `transport.SerialTransport(dev="/dev/ttyACM0", baud=921600)`: **lazy** `import serial`; vendor 0xAA/0x55 framing (`[0x55,0xAA, ...id LE..., 8 data], tail 0x55`, 16-byte frame).
- `transport.make_transport(spec: dict) -> Transport` (spec `{"kind":"fake|socketcan|serial", ...}`).

- [ ] **Step 1: failing test** — FakeTransport send/recv/inject; `make_transport({"kind":"fake"})` returns FakeTransport; `make_transport({"kind":"socketcan"})` does NOT import `can` until used (monkeypatch to assert lazy).
- [ ] **Step 2: run fail. Step 3: implement** (lazy imports mirror `kinematics.py` pydrake pattern). **Step 4: pass.**
- [ ] **Step 5: commit** `feat(can): pluggable Transport (fake/socketcan/serial), lazy backends`.

### Task 0.4: `MotorBus` maintenance-mode owner + ID→joint map

**Files:** Create `motor_bus.py`, `config/motor_map.example.yaml`; Test `test/test_motor_bus.py`.

**Interfaces — Consumes:** `dm_protocol`, `transport`. **Produces:**
- `motor_bus.MotorBus(transport, motor_map: dict)` with: `scan(ids: range)->list[dict]` (enable→read→list responders), `enable(joint)/disable(joint)`, `jog(joint, pos, vel=0, kp=..., kd=...)` (clamps pos/vel/tau to the joint's model limits AND to the configured soft limits), `set_zero(joint)`, `telemetry(joint)->dict`, `enable_all()/disable_all()`.
- `motor_bus.load_map(path)->dict` / `save_map(path, m)->None`. Map schema: `{joint_name: {id, master_id, model, soft_limits:{pos:[lo,hi],vel,effort}, dir:+1/-1, offset}}`.

- [ ] **Step 1: failing tests** (FakeTransport): enable(joint) sends `arb_id(id,"mit")` + enable special frame; jog beyond soft limit is clamped (assert the decoded MIT pos == clamped value); set_zero sends `...FE`; telemetry decodes an injected feedback frame; load/save map round-trips YAML.
- [ ] **Step 2-4: implement to green.**
- [ ] **Step 5: commit** `feat(can): MotorBus maintenance-mode owner (scan/enable/jog/zero/telemetry) + ID->joint map`.

### Task 0.5: `m1_hwconfig` web node + page

**Files:** Create `hwconfig_node.py`, `web/hwconfig.html`; Test `test/test_hwconfig_datapath.py`.

**Interfaces — Consumes:** `MotorBus`, `transport.make_transport`. **Produces (HTTP JSON API):**
- `GET /api/state` → `{mode:"maintenance|run", motors:[{joint,id,model,pos,vel,torque,t_mos,t_rotor,err,enabled}], bus_ok, map}`
- `POST /api/scan` `{from,to}` ; `POST /api/assign` `{old_id,new_id,master_id}` ; `POST /api/map` `{joint,id,model,...}` (persist) ; `POST /api/limits` `{joint,pos:[lo,hi],vel,effort}` (writes `m1_joint_limits.yaml`) ; `POST /api/jog` `{joint,pos,vel,kp,kd}` (deadman: requires `hold:true` refresh) ; `POST /api/enable` `{joint,on}` ; `POST /api/zero` `{joint}` ; `POST /api/mode` `{mode}`.
- The node refuses write/jog endpoints unless `mode=="maintenance"`; in run mode `/api/state` reads `/joint_states` (subscribe) instead of owning the bus.

**Pattern:** copy the structure of `m1_control/web_node.py` (stdlib `ThreadingHTTPServer`, embedded HTML served from `web/`, rclpy node in a thread, deadman like `BASE_HOLD`). Theme to match (warm cream / clay).

- [ ] **Step 1: failing test** `test_hwconfig_datapath.py` — construct the handler logic against a `MotorBus(FakeTransport)`; assert `/api/scan` lists injected motors; `/api/jog` while `mode!=maintenance` is rejected (403); `/api/jog` in maintenance sends a clamped MIT frame; `/api/limits` writes a YAML with the new values; deadman zeroes after timeout. (Drive the request-handling functions directly like `_quest_position_test.py` drives `on_xr_frame`.)
- [ ] **Step 2-4: implement to green.** Build: `colcon build --packages-select m1_can_tools`.
- [ ] **Step 5: commit** `feat(hwconfig): motor assign/limits/jog/zero/telemetry web page (maintenance-mode guarded)`.

**Phase 0 gate:** `PYTHONPATH=ros2_ws/src/m1_can_tools /usr/bin/python3 -m pytest ros2_ws/src/m1_can_tools/test -v` all green; package builds clean. No hardware used.

---

## Phase 1 — ros2_control on mock_components + bridges + control re-wire (offline)

### Task 1.1: URDF `<ros2_control>` xacro (mock default, real by arg)

**Files:** Create `m1_bringup/urdf/m1.ros2_control.xacro`, `m1_bringup/urdf/m1_hardware.urdf.xacro`; Test: a parse check.

**Interfaces — Produces:** a xacro that, given `use_mock:=true|false`, emits a `<ros2_control name="m1_arms" type="system">` block with `<hardware>` = `mock_components/GenericSystem` (mock) or `m1_hardware/M1SystemInterface` (real), and for the **19 upper-body joints** (lift + 14 arm + 4 finger) a `<command_interface name="position"/>` + `<state_interface name="position"/><state_interface name="velocity"/>` (+ `effort` state). Fingers: include the mimic note (command joint1; joint2 state mirrors). `m1_hardware.urdf.xacro` `<xacro:include>`s the existing `ranger_air_description.urdf` content (or its xacro) and the ros2_control xacro.

- [ ] **Step 1:** write the xacro. **Step 2: test** — `xacro m1_bringup/urdf/m1_hardware.urdf.xacro use_mock:=true | grep -c 'ros2_control'` ≥1 and the 19 joint names present; `check_urdf` parses the expansion. **Step 3: commit** `feat(bringup): ros2_control xacro for lift+arms+fingers (mock default, real by arg)`.

### Task 1.2: controllers.yaml + joint_limits.yaml

**Files:** Create `m1_bringup/config/m1_controllers.yaml`, `m1_bringup/config/m1_joint_limits.yaml`.

**Produces:**
- `controller_manager` `update_rate: 200` with: `joint_state_broadcaster`; `arm_position_controller` = `forward_command_controller/ForwardCommandController` (interface `position`, `joints:` = the 19 upper-body joints in canonical order); `left_arm_jtc`/`right_arm_jtc` = `joint_trajectory_controller` (loaded, inactive, for planned moves); gripper handled within the position controller (fingers are in the 19).
- `m1_joint_limits.yaml`: per-joint has_position/velocity/effort limits from the URDF table (e.g. `lift_joint` 0..0.85, vel 0.25; arm joints from the limit table in the spec), `has_acceleration_limits` modest defaults.

- [ ] **Step 1:** write both YAMLs. **Step 2: test** — `/usr/bin/python3 -c "import yaml,sys; d=yaml.safe_load(open('ros2_ws/src/m1_bringup/config/m1_controllers.yaml')); assert 'controller_manager' in d; ..."` validates joint sets/order. **Step 3: commit** `feat(bringup): controllers.yaml (forward position + JTC + JSB) and joint_limits.yaml`.

### Task 1.3: `m1_joint_bridge` node

**Files:** Create `m1_control/joint_command_bridge.py`; add entry point `m1_joint_bridge` in `m1_control/setup.py`; Test `m1_control/_bridge_test.py`.

**Interfaces — Produces:**
- `joint_command_bridge.map_command(js_name: list, js_pos: list, order: list) -> list[float]` — pure: pick positions for `order` (the 19 upper-body joints) by name; missing → last value / 0. (Unit-testable, no ROS.)
- Node `JointCommandBridge`: sub `/m1/joint_command` (JointState) → pub `/arm_position_controller/commands` (std_msgs/Float64MultiArray) using `map_command`.

- [ ] **Step 1: failing test** in `_bridge_test.py`: `map_command` reorders correctly, drops steer/wheel, preserves canonical 19-order; out-of-order input handled. **Step 2-4: implement to green** (`PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 _bridge_test.py`). **Step 5: commit** `feat(bridge): /m1/joint_command -> forward_position_controller`.

### Task 1.4: `m1_base_bridge` node

**Files:** Create `m1_control/base_bridge.py`; entry point `m1_base_bridge`; extend `_bridge_test.py`.

**Interfaces — Produces:**
- `base_bridge.select_motion_mode(vx,vy,yaw) -> tuple[str, Twist-fields]` — pure: implements the AgileX mode-switch logic (vy≠0 → PARALLEL/strafe, yaw ignored; small radius → SPINNING, linear=0; else DUAL_ACKERMANN). Returns the mode + the body Twist to send.
- Node `BaseBridge`: sub `/m1/cmd_vel` → pub `/cmd_vel` (Twist) + the motion-mode topic the AgileX driver expects; keep a `BASE_HOLD` deadman.

- [ ] **Step 1: failing test** — `select_motion_mode` returns PARALLEL when vy≠0 (and zeroes yaw), SPINNING when only yaw (linear=0), DUAL_ACKERMANN for vx+small-yaw. **Step 2-4: green. Step 5: commit** `feat(bridge): /m1/cmd_vel -> AgileX body Twist + motion-mode`.

### Task 1.5: `ranger_state_shim` node

**Files:** Create `m1_control/ranger_state_shim.py`; entry point `m1_ranger_shim`; extend `_bridge_test.py`.

**Interfaces — Produces:**
- `ranger_state_shim.steer_wheel_to_jointstate(steer: list[float], wheel: list[float]) -> (names, pos, vel)` — pure: maps 4 steering angles + 4 wheel speeds to the 8 base joint names (canonical), applying the URDF sign conventions from `swerve.py` (`STEER_DIR`, `WHEEL_DIR`).
- Node `RangerStateShim`: sub the AgileX feedback topics → pub `/joint_states` (8 base joints).

- [ ] **Step 1: failing test** — names/order correct, sign conventions applied. **Step 2-4: green. Step 5: commit** `feat(bridge): AgileX wheel feedback -> /joint_states base joints`.

### Task 1.6: hardware bringup launch + mock end-to-end + re-gate

**Files:** Create `m1_bringup/launch/hardware.launch.py`; update `m1_bringup/setup.py` data_files (install urdf/, config/, launch/).

**Produces:** a launch with args `use_mock:=true` (default), `use_rviz`, `use_base:=false`: starts RSP (from `m1_hardware.urdf.xacro`), `ros2_control_node` (controllers.yaml), spawners (`joint_state_broadcaster`, `arm_position_controller`; JTC inactive), `m1_joint_bridge`, the existing `m1_controller`, and (if `use_base`) `m1_base_bridge` + AgileX driver + `m1_ranger_shim`.

- [ ] **Step 1:** write the launch. **Step 2: build** `colcon build --packages-select m1_bringup m1_control m1_can_tools`. **Step 3: mock smoke** — `ros2 launch m1_bringup hardware.launch.py use_mock:=true use_rviz:=false` (with `ROS_LOG_DIR` writable); in another shell publish a target and assert `/joint_states` moves toward it and `/arm_position_controller/commands` is populated (a scripted check like `_ros_reach_check.py`, driven from ROS timers). Mark live-hardware reach as deferred.
- [ ] **Step 4: RE-GATE the brain** — run every existing suite, confirm still green:
```
/usr/bin/python3 _solver_test.py && /usr/bin/python3 _solver_test_positions.py && /usr/bin/python3 _solver_test_tracking.py && /usr/bin/python3 _solver_test_pathing.py && /usr/bin/python3 _accuracy_bench.py && /usr/bin/python3 _swerve_test.py && PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.collision && PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.trajectory
```
- [ ] **Step 5: commit** `feat(bringup): hardware launch (mock<->real) + mock e2e reach; brain suites still green`.

**Phase 1 gate:** mock ros2_control round-trips the contract; all bridge unit tests green; all existing brain suites still green; no hardware used.

---

## Phase 2 — Real C++ SystemInterface (`m1_hardware`) + AgileX base (offline build; live deferred)

### Task 2.1: vendor `openarm_can`, build on Jazzy

**Files:** Create `m1_hardware/` (ament_cmake), copy `/tmp/openarm_can` → `m1_hardware/vendor/openarm_can` (keep LICENSE/NOTICE).

- [ ] **Step 1:** vendor the lib; write `m1_hardware/CMakeLists.txt` that builds `openarm_can` as a subdir/dependency. **Step 2: build** `colcon build --packages-select m1_hardware` on Jazzy; fix any toolchain breaks. **Step 3:** a tiny C++ gtest that constructs the CAN encode for a known MIT command and asserts the bytes equal Python `dm_protocol.encode_mit` (cross-check the two codecs). **Step 4: commit** `feat(hw): vendor openarm_can, builds on Jazzy + cross-checks Python codec`.

### Task 2.2: `M1SystemInterface` (fork openarm_hardware, port Humble→Jazzy, add lift)

**Files:** Create `m1_hardware/{include/m1_hardware/m1_system_interface.hpp, src/m1_system_interface.cpp, m1_hardware.xml, config/control_gains.yaml}`, adapting `/tmp/openarm_ros2/openarm_hardware`.

**Interfaces — Produces:** pluginlib class `m1_hardware/M1SystemInterface : hardware_interface::SystemInterface` exporting per-joint `position` command + `position/velocity/effort` state; MIT-mode `write()`; `on_activate` enable / `on_deactivate` disable; SocketCAN (CAN-FD arg); motor IDs + models read from `<param>`/the motor_map YAML (NOT hardcoded — generalize OpenArm's fixed 7-DOF/dual-bus); supports an arbitrary joint count incl. the **lift**.

- [ ] **Step 1:** port the class; replace hardcoded `ARM_DOF=7`/CAN-ID arrays with params; add lift joint handling; apply the Jazzy interface-export migration (state/command interfaces auto-exported from URDF; lifecycle types). **Step 2: build clean** on Jazzy. **Step 3:** load via `mock`-style URDF arg flip and confirm `controller_manager` loads the plugin (`ros2 control list_hardware_interfaces` lists the 19 joints) — bus I/O can no-op without hardware (guard so it loads with no CAN device, logging "no bus"). **Step 4: commit** `feat(hw): M1SystemInterface (Damiao MIT, SocketCAN, generalized DOF + lift), Jazzy`.

### Task 2.3: wire real plugin + transport selection + live-loop checkpoint

- [ ] **Step 1:** make `m1.ros2_control.xacro use_mock:=false` select `m1_hardware/M1SystemInterface` with params (can_interface, can_fd, motor_map path). **Step 2: build.** **Step 3 (HARDWARE, deferred until available):** bring up `can0`, run `hardware.launch.py use_mock:=false`, validate the **live closed loop** — controller command-fingertip vs measured `/joint_states` ≈ 0 mm (the discipline from memory `drake-solver-backend`); confirm enable/limit/e-stop. Document the exact bring-up steps in `ros2_ws/README.md`. **Step 4: commit** `feat(hw): real plugin wired + transport params + live bring-up doc`.

### Task 2.4: AgileX base integration (Twist path) + finish shim

**Files:** add AgileX driver as a vcs dep or vendor `ranger_ros2` (air_delta) + `ugv_sdk`; wire into `hardware.launch.py` under `use_base:=true`.

- [ ] **Step 1:** clone/vendor `ranger_ros2` (air_delta) + `ugv_sdk`; port to Jazzy (recompile, fix API). **Step 2: build.** **Step 3:** connect `m1_base_bridge` → driver `/cmd_vel`; `m1_ranger_shim` ← driver feedback. Sim stays on `swerve.py`/Isaac (unchanged). **Step 4 (HARDWARE deferred):** confirm Twist drives the base; per-module access check. **Step 5: commit** `feat(base): AgileX ranger Twist driver + state shim, Jazzy`.

**Phase 2 gate:** `m1_hardware` + base build clean on Jazzy; C++/Python codecs cross-check; plugin loads in controller_manager without a bus. Live motor validation is the on-hardware checkpoint (flagged, not blocking the offline build).

---

## Phase 3 — MoveIt for planned collision-aware moves (optional, additive)

### Task 3.1: MoveIt config (lift as a joint-space planning group)

**Files:** Create `m1_bringup/moveit/` (SRDF with `left_arm`,`right_arm` chains + a `both_arms_lift` **joint-space** group containing lift+14 for planned moves; `kinematics.yaml` KDL per arm; `ompl_planning.yaml`; `moveit_controllers.yaml` → the JTCs; collision matrix incl. cross-arm).

- [ ] **Step 1:** generate/adapt from `/tmp/openarm_ros2/openarm_bimanual_moveit_config` + add the lift. **Step 2:** a mock planned-move test — plan a joint-space goal for `both_arms_lift` in mock and execute via JTC (hot-swap from the forward controller). Reactive Drake path unaffected. **Step 3: commit** `feat(moveit): optional planned-move config (per-arm KDL + joint-space lift group)`.

**Phase 3 gate:** a planned move executes in mock; reactive teleop/shared-lift unchanged.

---

## Self-Review (done at write time)

- **Spec coverage:** §3 contract→Tasks 1.1/1.3; §4 seam→1.3/1.4/1.5/1.6; §5 packages→all; §6 phases→Phase 0/1/2/3 1:1; §7 config page→0.5; §8 safety→1.2 limits + 2.2 enable/disable + bridge deadmans; §9 validation→every gate + 1.6 re-gate; §10 risks→2.3/2.4 hardware checkpoints. No gaps.
- **Placeholder scan:** none — every code-touching step names exact files/functions/byte values; hardware-only steps are explicitly flagged deferred (not placeholders).
- **Type consistency:** `encode_mit/decode_feedback/arb_id/special_frame`, `Transport.send/recv`, `MotorBus.jog/telemetry`, `map_command`, `select_motion_mode`, `steer_wheel_to_jointstate`, `M1SystemInterface` used consistently across tasks.
