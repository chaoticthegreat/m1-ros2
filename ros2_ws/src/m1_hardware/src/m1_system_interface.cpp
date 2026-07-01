// Copyright 2026 M1 Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Forked from openarm_hardware (Copyright 2025 Enactic, Inc., Apache-2.0),
// ported Humble -> Jazzy and generalized (arbitrary DOF + lift, URDF-driven).

#include "m1_hardware/m1_system_interface.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_control.hpp>

#include <yaml-cpp/yaml.h>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{
constexpr const char * kLogger = "M1SystemInterface";

std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Parse a CAN id written as "0x07", "7", "0X07" etc.
uint32_t parse_id(const std::string & s, uint32_t fallback)
{
  if (s.empty())
  {
    return fallback;
  }
  try
  {
    return static_cast<uint32_t>(std::stoul(s, nullptr, 0));  // base 0 -> honors 0x prefix
  }
  catch (const std::exception &)
  {
    return fallback;
  }
}

double parse_double(const std::string & s, double fallback)
{
  if (s.empty())
  {
    return fallback;
  }
  try
  {
    return std::stod(s);
  }
  catch (const std::exception &)
  {
    return fallback;
  }
}

bool parse_bool(const std::string & s, bool fallback)
{
  if (s.empty())
  {
    return fallback;
  }
  const std::string v = to_lower(s);
  return v == "true" || v == "1" || v == "yes" || v == "on";
}

// Look a key up in a per-joint param map, then fall back to a hardware-level
// param map, then to a default string.
std::string lookup(
  const std::unordered_map<std::string, std::string> & joint_params,
  const std::unordered_map<std::string, std::string> & hw_params, const std::string & key,
  const std::string & def = "")
{
  auto it = joint_params.find(key);
  if (it != joint_params.end())
  {
    return it->second;
  }
  it = hw_params.find(key);
  if (it != hw_params.end())
  {
    return it->second;
  }
  return def;
}

// Read a scalar YAML node as a string ("" if missing/null). Lets the existing
// string-based parse_id/parse_double/parse_bool reuse the same coercion rules
// for both URDF <param> strings and motor_map YAML values.
std::string yaml_str(const YAML::Node & node)
{
  if (!node || node.IsNull() || !node.IsScalar())
  {
    return "";
  }
  return node.as<std::string>();
}

// Reverse of model_from_string for logging (MotorType -> model name). Falls back
// to the numeric enum value for any unexpected type.
std::string model_string_of(openarm::damiao_motor::MotorType m)
{
  using MT = openarm::damiao_motor::MotorType;
  switch (m)
  {
    case MT::DM3507: return "DM3507";
    case MT::DM4310: return "DM4310";
    case MT::DM4310_48V: return "DM4310_48V";
    case MT::DM4340: return "DM4340";
    case MT::DM4340_48V: return "DM4340_48V";
    case MT::DM6006: return "DM6006";
    case MT::DM8006: return "DM8006";
    case MT::DM8009: return "DM8009";
    case MT::DM10010L: return "DM10010L";
    case MT::DM10010: return "DM10010";
    case MT::DMH3510: return "DMH3510";
    case MT::DMH6215: return "DMH6215";
    case MT::DMG6220: return "DMG6220";
    default: return "DM(" + std::to_string(static_cast<int>(m)) + ")";
  }
}
}  // namespace

namespace m1_hardware
{

using openarm::damiao_motor::MotorType;

M1SystemInterface::M1SystemInterface() = default;
M1SystemInterface::~M1SystemInterface() = default;

MotorType M1SystemInterface::model_from_string(const std::string & s, bool & ok)
{
  // Map the documented DM model strings -> openarm_can MotorType. The
  // openarm_can MOTOR_LIMIT_PARAMS table carries the [P,V,T]MAX scaling and
  // matches the project's Global-Constraints values verbatim.
  static const std::unordered_map<std::string, MotorType> kMap = {
    {"DM3507", MotorType::DM3507},        {"DM4310", MotorType::DM4310},
    {"DM4310_48V", MotorType::DM4310_48V}, {"DM4340", MotorType::DM4340},
    {"DM4340_48V", MotorType::DM4340_48V}, {"DM6006", MotorType::DM6006},
    {"DM8006", MotorType::DM8006},        {"DM8009", MotorType::DM8009},
    {"DM10010L", MotorType::DM10010L},     {"DM10010", MotorType::DM10010},
    {"DMH3510", MotorType::DMH3510},       {"DMH6215", MotorType::DMH6215},
    {"DMG6220", MotorType::DMG6220},
  };
  std::string key = s;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return std::toupper(c);
  });
  auto it = kMap.find(key);
  if (it == kMap.end())
  {
    ok = false;
    return MotorType::DM4310;
  }
  ok = true;
  return it->second;
}

void M1SystemInterface::parse_bus_params(const hardware_interface::HardwareInfo & info)
{
  const auto & hw = info.hardware_parameters;
  auto it = hw.find("can_interface");
  can_interface_ = (it != hw.end()) ? it->second : "can0";
  it = hw.find("can_fd");
  can_fd_ = (it != hw.end()) ? parse_bool(it->second, false) : false;
  it = hw.find("motor_map");
  motor_map_path_ = (it != hw.end()) ? it->second : "";

  RCLCPP_INFO(
    rclcpp::get_logger(kLogger), "Bus config: can_interface=%s can_fd=%s motor_map=%s",
    can_interface_.c_str(), can_fd_ ? "true" : "false",
    motor_map_path_.empty() ? "(none)" : motor_map_path_.c_str());
}

std::unordered_map<std::string, JointMotor> M1SystemInterface::load_motor_map()
{
  // Schema (exactly what m1_can_tools.motor_bus.save_map writes), per joint:
  //   joint_name:
  //     id:         <int>        # CAN slave/send id
  //     master_id:  <int>        # host/recv id (= id + 0x10)
  //     model:      <str>        # DM model string
  //     kp: <float>  kd: <float>  dir: +1|-1  offset: <float>
  //     soft_limits: {pos:[lo,hi], vel:<float>, effort:<float>}  # not consumed here
  std::unordered_map<std::string, JointMotor> out;
  if (motor_map_path_.empty())
  {
    return out;
  }

  YAML::Node root;
  try
  {
    root = YAML::LoadFile(motor_map_path_);
  }
  catch (const std::exception & e)
  {
    RCLCPP_WARN(
      rclcpp::get_logger(kLogger),
      "motor_map '%s' could not be loaded (%s) -- falling back to URDF/defaults.",
      motor_map_path_.c_str(), e.what());
    return out;
  }
  if (!root || !root.IsMap())
  {
    RCLCPP_WARN(
      rclcpp::get_logger(kLogger),
      "motor_map '%s' is empty or not a joint map -- falling back to URDF/defaults.",
      motor_map_path_.c_str());
    return out;
  }

  for (const auto & kv : root)
  {
    const std::string joint = kv.first.as<std::string>();
    const YAML::Node & cfg = kv.second;
    if (!cfg || !cfg.IsMap())
    {
      continue;
    }
    JointMotor m;
    m.name = joint;
    // id / master_id (mirror motor_bus: master defaults to id + 0x10).
    const std::string id_s = yaml_str(cfg["id"]);
    m.can_id = parse_id(id_s, 0);
    m.master_id = parse_id(yaml_str(cfg["master_id"]), m.can_id + 0x10);

    bool model_ok = true;
    const std::string model_str = yaml_str(cfg["model"]);
    if (!model_str.empty())
    {
      m.model = model_from_string(model_str, model_ok);
      if (!model_ok)
      {
        RCLCPP_WARN(
          rclcpp::get_logger(kLogger),
          "motor_map joint '%s': unknown model '%s', defaulting to DM4310",
          joint.c_str(), model_str.c_str());
      }
    }
    m.kp = parse_double(yaml_str(cfg["kp"]), 0.0);
    m.kd = parse_double(yaml_str(cfg["kd"]), 0.0);
    m.direction = parse_double(yaml_str(cfg["dir"]), 1.0);
    m.offset = parse_double(yaml_str(cfg["offset"]), 0.0);
    out[joint] = m;
  }

  RCLCPP_INFO(
    rclcpp::get_logger(kLogger), "Loaded motor_map '%s' (%zu joint(s)).",
    motor_map_path_.c_str(), out.size());
  return out;
}

bool M1SystemInterface::parse_joints(const hardware_interface::HardwareInfo & info)
{
  motors_.clear();
  state_names_.clear();
  limp_config_ = false;
  const auto & hw = info.hardware_parameters;

  // Source of per-joint config: URDF <param> (highest precedence) > motor_map
  // YAML > built-in default. Load the YAML once up front (FIX 1).
  const std::unordered_map<std::string, JointMotor> ymap = load_motor_map();

  uint32_t auto_id = 1;  // fallback CAN id allocator if none specified
  for (const auto & joint : info.joints)
  {
    // FIX 3: only joints with a `position` COMMAND interface have a physical
    // motor. The state-only mimic *_finger_joint2 (no command interface) get a
    // storage slot + STATE interfaces but NO motor / command interface.
    const bool commanded = std::any_of(
      joint.command_interfaces.begin(), joint.command_interfaces.end(),
      [](const hardware_interface::InterfaceInfo & ci) {
        return ci.name == hardware_interface::HW_IF_POSITION;
      });

    const size_t state_index = state_names_.size();
    state_names_.push_back(joint.name);

    if (!commanded)
    {
      RCLCPP_INFO(
        rclcpp::get_logger(kLogger),
        "Joint '%s': no position command interface -> state-only (mimic), no motor.",
        joint.name.c_str());
      continue;
    }

    JointMotor m;
    m.name = joint.name;
    m.state_index = state_index;

    // Per-joint values: URDF <param> if present, else motor_map YAML, else default.
    const auto & jp = joint.parameters;
    auto yit = ymap.find(joint.name);
    const bool have_yaml = (yit != ymap.end());
    const JointMotor ym = have_yaml ? yit->second : JointMotor{};

    // Precedence per key: URDF <param> (joint- or hardware-level) > motor_map
    // YAML > built-in default.
    auto pick_id = [&](const char * key, uint32_t yaml_val, uint32_t def) {
      const std::string s = lookup(jp, hw, key);
      if (!s.empty()) return parse_id(s, def);
      if (have_yaml) return yaml_val;
      return def;
    };
    auto pick_double = [&](const char * key, double yaml_val, double def) {
      const std::string s = lookup(jp, hw, key);
      if (!s.empty()) return parse_double(s, def);
      if (have_yaml) return yaml_val;
      return def;
    };

    m.can_id = pick_id("can_id", ym.can_id, auto_id);
    // Master/recv id convention from the spec: slave + 0x10 (never 0).
    m.master_id = pick_id("master_id", ym.master_id, m.can_id + 0x10);

    bool model_ok = true;
    const std::string model_str = lookup(jp, hw, "motor_model");  // URDF param only
    if (!model_str.empty())
    {
      m.model = model_from_string(model_str, model_ok);
      if (!model_ok)
      {
        RCLCPP_WARN(
          rclcpp::get_logger(kLogger),
          "Joint '%s': unknown motor_model '%s', defaulting to DM4310", m.name.c_str(),
          model_str.c_str());
      }
    }
    else if (have_yaml)
    {
      m.model = ym.model;  // already validated in load_motor_map()
    }
    // else: built-in default DM4310 (JointMotor default).

    m.kp = pick_double("kp", ym.kp, 0.0);
    m.kd = pick_double("kd", ym.kd, 0.0);
    m.direction = pick_double("dir", ym.direction, 1.0);
    m.offset = pick_double("offset", ym.offset, 0.0);

    motors_.push_back(m);
    auto_id = m.can_id + 1;

    if (m.kp == 0.0)
    {
      limp_config_ = true;
    }
    if (m.direction == 0.0)
    {
      // dir==0 would freeze read() feedback at `offset` (0*motor+offset) while
      // write() still drives the motor (its (dir==0?1:dir) guard) -- an open-loop
      // runaway. Treat it as unsafe, like kp==0, and refuse the bus.
      limp_config_ = true;
      RCLCPP_ERROR(
        rclcpp::get_logger(kLogger),
        "Joint '%s' has dir==0 (illegal direction; must be +1 or -1) -- refusing the bus.",
        m.name.c_str());
    }

    RCLCPP_INFO(
      rclcpp::get_logger(kLogger),
      "Motor[%zu] '%s': model=%s can_id=0x%02X master_id=0x%02X kp=%.3f kd=%.3f dir=%+.0f "
      "offset=%.4f (motor_map=%s)",
      motors_.size() - 1, m.name.c_str(),
      model_string_of(m.model).c_str(), m.can_id, m.master_id, m.kp, m.kd, m.direction, m.offset,
      have_yaml ? "yes" : "no");
  }

  // Record URDF mimic joints (state-only, e.g. *_finger_joint2 -> *_finger_joint1)
  // so read() can propagate their state from the leader. joint_index /
  // mimicked_joint_index are indices into info.joints, which is exactly the order
  // state_names_ was built in (one entry per joint, in info.joints order).
  mimics_ = info.mimic_joints;

  if (state_names_.empty())
  {
    RCLCPP_ERROR(rclcpp::get_logger(kLogger), "No joints found in HardwareInfo");
    return false;
  }
  if (motors_.empty())
  {
    RCLCPP_ERROR(rclcpp::get_logger(kLogger), "No COMMANDED joints found in HardwareInfo");
    return false;
  }

  RCLCPP_INFO(
    rclcpp::get_logger(kLogger), "Parsed %zu joints (%zu commanded motors, %zu state-only).",
    state_names_.size(), motors_.size(), state_names_.size() - motors_.size());

  // FIX 1 safety guard: a COMMANDED motor with kp==0 is limp/unsafe. Warn here;
  // the bus is refused in try_open_bus() so the live arms can never silently go
  // limp. A no-bus mock load still succeeds (for offline I/O).
  if (limp_config_)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger(kLogger),
      "UNSAFE CONFIG: at least one commanded joint has kp==0 (limp / mis-scaled motor). "
      "The CAN bus will be REFUSED (motors not enabled). Provide kp via the motor_map YAML "
      "or URDF <param name=\"kp\">. (Mock/no-bus load still proceeds.)");
  }
  return true;
}

void M1SystemInterface::try_open_bus()
{
  bus_ok_ = false;
  openarm_.reset();
  for (auto & m : motors_)
  {
    m.device_index = -1;
  }

  // FIX 1 safety guard: refuse to drive a live bus with a limp (kp==0) commanded
  // motor. Stay LOADED in no-bus/mock mode (read() echoes commands) so offline
  // validation still works -- we just never enable real motors.
  if (limp_config_)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger(kLogger),
      "Refusing to open CAN bus '%s': a commanded joint has kp==0 (would drive limp/mis-scaled "
      "motors). Running with NO BUS / mock I/O. Fix the motor_map kp before live control.",
      can_interface_.c_str());
    return;
  }

  try
  {
    openarm_ = std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

    std::vector<MotorType> models;
    std::vector<uint32_t> send_ids;
    std::vector<uint32_t> recv_ids;
    models.reserve(motors_.size());
    send_ids.reserve(motors_.size());
    recv_ids.reserve(motors_.size());
    for (const auto & m : motors_)
    {
      models.push_back(m.model);
      send_ids.push_back(m.can_id);
      recv_ids.push_back(m.master_id);
    }
    // All M1 motors are treated as one MIT-mode group (no gripper-split: the
    // fingers are ordinary DM motors here, generalized away from OpenArm's
    // dedicated gripper component).
    openarm_->init_arm_motors(models, send_ids, recv_ids);

    // FIX 2: the device collection is a std::map keyed by recv_can_id (master_id),
    // so get_motors()[i] / mit_control_one(i) iterate in ASCENDING-master_id order,
    // NOT URDF/motors_ order. Resolve each motor's device index by matching its
    // master_id to the device's recv_can_id, so we command/read each motor on ITS
    // own device regardless of map order.
    const auto motors_by_dev = openarm_->get_arm().get_motors();  // map (ascending) order
    if (!resolve_device_indices(motors_by_dev))
    {
      // Configured master_ids don't match the device set: unsafe to command by a
      // guessed index. Drop to no-bus so we never cross-wire a live motor.
      openarm_.reset();
      bus_ok_ = false;
      return;
    }

    bus_ok_ = true;
    RCLCPP_INFO(
      rclcpp::get_logger(kLogger), "CAN bus '%s' open; %zu motors initialised",
      can_interface_.c_str(), motors_.size());
  }
  catch (const std::exception & e)
  {
    // No CAN device on this machine (e.g. controller_manager smoke-load with
    // no motors). Stay loaded and run mock I/O instead of failing the plugin.
    openarm_.reset();
    bus_ok_ = false;
    RCLCPP_WARN(
      rclcpp::get_logger(kLogger),
      "CAN bus '%s' unavailable (%s) -- running with no bus / mock I/O. "
      "Live motor I/O will be validated on hardware.",
      can_interface_.c_str(), e.what());
  }
}

bool M1SystemInterface::resolve_device_indices(
  const std::vector<openarm::damiao_motor::Motor> & devices)
{
  // Build recv_can_id (master_id) -> device-collection index. The collection is
  // returned in std::map(recv_can_id) ascending order; this map lets each
  // URDF-ordered motors_[k] find ITS device by master_id (FIX 2).
  std::unordered_map<uint32_t, int> by_master;
  by_master.reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i)
  {
    by_master[devices[i].get_recv_can_id()] = static_cast<int>(i);
  }

  bool all_ok = true;
  for (auto & m : motors_)
  {
    auto it = by_master.find(m.master_id);
    if (it == by_master.end())
    {
      RCLCPP_ERROR(
        rclcpp::get_logger(kLogger),
        "Device set mismatch: no CAN device with master_id=0x%02X for joint '%s'.",
        m.master_id, m.name.c_str());
      all_ok = false;
      continue;
    }
    m.device_index = it->second;
  }

  if (devices.size() != motors_.size())
  {
    RCLCPP_ERROR(
      rclcpp::get_logger(kLogger),
      "Device count (%zu) != configured motor count (%zu).", devices.size(), motors_.size());
    all_ok = false;
  }
  if (!all_ok)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger(kLogger),
      "Configured master_ids do not match the CAN device set -- refusing to drive (no bus).");
  }
  else
  {
    RCLCPP_INFO(
      rclcpp::get_logger(kLogger),
      "Matched %zu motors to CAN devices by master_id (order-independent).", motors_.size());
  }
  return all_ok;
}

hardware_interface::CallbackReturn M1SystemInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const hardware_interface::HardwareInfo & info = get_hardware_info();

  parse_bus_params(info);
  if (!parse_joints(info))
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // One storage slot per JOINT (incl. state-only mimics) so /joint_states is
  // complete; commands are written per-motor by state_index.
  const size_t n = state_names_.size();
  pos_states_.assign(n, 0.0);
  vel_states_.assign(n, 0.0);
  tau_states_.assign(n, 0.0);
  pos_commands_.assign(n, 0.0);

  RCLCPP_INFO(
    rclcpp::get_logger(kLogger),
    "M1SystemInterface initialised with %zu joints (%zu commanded motors)", n, motors_.size());
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface::ConstSharedPtr>
M1SystemInterface::on_export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> out;
  pos_state_ifaces_.clear();
  vel_state_ifaces_.clear();
  tau_state_ifaces_.clear();

  // STATE interfaces for EVERY joint, including the state-only mimic
  // finger_joint2 (FIX 3: they have no motor but DO appear in /joint_states).
  for (size_t i = 0; i < state_names_.size(); ++i)
  {
    auto make = [&](const std::string & iface, double * ptr) {
      hardware_interface::InterfaceInfo ii;
      ii.name = iface;
      hardware_interface::InterfaceDescription desc(state_names_[i], ii);
      auto si = std::make_shared<hardware_interface::StateInterface>(desc);
      // Seed the handle from our owned storage; read() updates it each cycle.
      std::ignore = si->set_value(*ptr);
      return si;
    };
    auto p = make(hardware_interface::HW_IF_POSITION, &pos_states_[i]);
    auto v = make(hardware_interface::HW_IF_VELOCITY, &vel_states_[i]);
    auto t = make(hardware_interface::HW_IF_EFFORT, &tau_states_[i]);
    pos_state_ifaces_.push_back(p);
    vel_state_ifaces_.push_back(v);
    tau_state_ifaces_.push_back(t);
    out.push_back(p);
    out.push_back(v);
    out.push_back(t);
  }
  return out;
}

std::vector<hardware_interface::CommandInterface::SharedPtr>
M1SystemInterface::on_export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface::SharedPtr> out;
  pos_cmd_ifaces_.clear();

  // COMMAND interfaces only for commanded motors (FIX 3: no command interface for
  // the state-only mimic joints -- ros2_control forbids it on a mimic and there
  // is no physical motor). pos_cmd_ifaces_[k] is parallel to motors_[k].
  for (size_t k = 0; k < motors_.size(); ++k)
  {
    const size_t si = motors_[k].state_index;
    hardware_interface::InterfaceInfo ii;
    ii.name = hardware_interface::HW_IF_POSITION;
    hardware_interface::InterfaceDescription desc(motors_[k].name, ii);
    auto ci = std::make_shared<hardware_interface::CommandInterface>(desc);
    std::ignore = ci->set_value(pos_commands_[si]);
    pos_cmd_ifaces_.push_back(ci);
    out.push_back(ci);
  }
  return out;
}

hardware_interface::CallbackReturn M1SystemInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Open the bus (or fall back to mock). Done here (INACTIVE) so states can be
  // read before activation; never fails -- a missing bus is a warning.
  try_open_bus();

  // Seed command = current state so the first activation doesn't jump.
  if (bus_ok_ && openarm_)
  {
    try
    {
      openarm_->refresh_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      openarm_->recv_all();
    }
    catch (const std::exception & e)
    {
      RCLCPP_WARN(rclcpp::get_logger(kLogger), "on_configure refresh failed: %s", e.what());
    }
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn M1SystemInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger(kLogger), "Activating M1SystemInterface...");
  if (bus_ok_ && openarm_)
  {
    try
    {
      openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
      openarm_->enable_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      openarm_->recv_all();
    }
    catch (const std::exception & e)
    {
      // Transient CAN fault on enable: do NOT return ERROR (that finalizes the
      // component). Stay active; read/write will keep retrying.
      RCLCPP_WARN(
        rclcpp::get_logger(kLogger), "enable_all failed (transient?): %s -- staying active",
        e.what());
    }
  }
  else
  {
    RCLCPP_WARN(
      rclcpp::get_logger(kLogger), "Activating with no bus / mock I/O (no motors enabled).");
  }

  // Copy the freshly-decoded motor feedback into pos_states_ BEFORE seeding the
  // command interfaces, so "hold position on activate" actually holds the
  // MEASURED pose. read() -- the only other pos_states_ writer -- does not run
  // until after activation, so without this the seed below would be the on_init
  // ZERO, commanding every joint to 0 rad at full kp: a violent lurch to the
  // folded all-zeros posture on a real arm. (No-bus/mock legitimately stays 0.)
  if (bus_ok_ && openarm_)
  {
    try
    {
      auto & arm = openarm_->get_arm();
      for (const auto & m : motors_)
      {
        if (m.device_index < 0)
        {
          continue;
        }
        const auto motor = arm.get_motor(m.device_index);
        pos_states_[m.state_index] = m.direction * motor.get_position() + m.offset;
        vel_states_[m.state_index] = m.direction * motor.get_velocity();
        tau_states_[m.state_index] = m.direction * motor.get_torque();
      }
    }
    catch (const std::exception & e)
    {
      RCLCPP_WARN(
        rclcpp::get_logger(kLogger),
        "on_activate state seed failed: %s (seeding command from last known state)", e.what());
    }
  }

  // Seed command interfaces from current state so we hold position on activate.
  // pos_commands_/pos_states_ are indexed by state_index; pos_cmd_ifaces_[k] is
  // parallel to motors_[k] (commanded joints only).
  for (size_t k = 0; k < motors_.size(); ++k)
  {
    const size_t si = motors_[k].state_index;
    pos_commands_[si] = pos_states_[si];
    if (k < pos_cmd_ifaces_.size())
    {
      std::ignore = pos_cmd_ifaces_[k]->set_value(pos_commands_[si]);
    }
  }
  RCLCPP_INFO(rclcpp::get_logger(kLogger), "M1SystemInterface activated.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn M1SystemInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger(kLogger), "Deactivating M1SystemInterface...");
  if (bus_ok_ && openarm_)
  {
    try
    {
      for (int i = 0; i < 3; ++i)
      {
        openarm_->disable_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        openarm_->recv_all();
      }
    }
    catch (const std::exception & e)
    {
      RCLCPP_WARN(rclcpp::get_logger(kLogger), "disable_all failed: %s", e.what());
    }
  }
  RCLCPP_INFO(rclcpp::get_logger(kLogger), "M1SystemInterface deactivated.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn M1SystemInterface::on_error(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_ERROR(rclcpp::get_logger(kLogger), "on_error: disabling motors.");
  if (bus_ok_ && openarm_)
  {
    try
    {
      openarm_->disable_all();
      openarm_->recv_all();
    }
    catch (const std::exception & e)
    {
      RCLCPP_WARN(rclcpp::get_logger(kLogger), "on_error disable_all failed: %s", e.what());
    }
  }
  // Recoverable: allow the component to be re-configured/re-activated.
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type M1SystemInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (bus_ok_ && openarm_)
  {
    try
    {
      openarm_->refresh_all();
      openarm_->recv_all();
      // FIX 2: fetch each motor's feedback by ITS device index (master_id-matched),
      // not by map iteration order. get_motor(i) indexes get_dm_devices()[i].
      auto & arm = openarm_->get_arm();
      for (const auto & m : motors_)
      {
        if (m.device_index < 0)
        {
          continue;
        }
        const auto motor = arm.get_motor(m.device_index);
        pos_states_[m.state_index] = m.direction * motor.get_position() + m.offset;
        vel_states_[m.state_index] = m.direction * motor.get_velocity();
        tau_states_[m.state_index] = m.direction * motor.get_torque();
      }
    }
    catch (const std::exception & e)
    {
      // Transient CAN read fault: hold last state, do NOT escalate to ERROR.
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger(kLogger), *get_clock(), 1000, "read() CAN fault: %s (holding state)",
        e.what());
    }
  }
  else
  {
    // Mock I/O: reflect the commanded position so the loop / RViz stays sane.
    // Over ALL joints (incl. state-only mimics, whose command stays at its seed).
    for (size_t i = 0; i < state_names_.size(); ++i)
    {
      pos_states_[i] = pos_commands_[i];
      vel_states_[i] = 0.0;
      tau_states_[i] = 0.0;
    }
  }

  // Propagate mimic (state-only) joints from their leader, in BOTH the real-bus
  // and mock branches: a mimic has no motor, so its slot is otherwise never
  // updated and would report a constant 0 (e.g. finger_joint2 stuck at 0 while
  // finger_joint1 opens). state = offset + multiplier * leader, per the URDF.
  for (const auto & mj : mimics_)
  {
    if (mj.joint_index < pos_states_.size() &&
        mj.mimicked_joint_index < pos_states_.size())
    {
      pos_states_[mj.joint_index] =
        mj.offset + mj.multiplier * pos_states_[mj.mimicked_joint_index];
      vel_states_[mj.joint_index] = mj.multiplier * vel_states_[mj.mimicked_joint_index];
      tau_states_[mj.joint_index] = 0.0;
    }
  }

  // Publish into the exported state handles (one per joint, all joints).
  for (size_t i = 0; i < state_names_.size(); ++i)
  {
    if (i < pos_state_ifaces_.size())
    {
      std::ignore = pos_state_ifaces_[i]->set_value(pos_states_[i]);
      std::ignore = vel_state_ifaces_[i]->set_value(vel_states_[i]);
      std::ignore = tau_state_ifaces_[i]->set_value(tau_states_[i]);
    }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type M1SystemInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Pull the latest position commands from the exported handles into each
  // motor's storage slot (state_index). pos_cmd_ifaces_[k] is parallel to motors_.
  for (size_t k = 0; k < motors_.size(); ++k)
  {
    if (k < pos_cmd_ifaces_.size())
    {
      const auto v = pos_cmd_ifaces_[k]->get_optional();
      if (v.has_value())
      {
        pos_commands_[motors_[k].state_index] = v.value();
      }
    }
  }

  if (bus_ok_ && openarm_)
  {
    try
    {
      // FIX 2: command each motor on ITS device index (master_id-matched), via
      // mit_control_one, so a non-monotonic URDF<->master_id mapping never
      // cross-wires commands. (mit_control_all would re-impose map order.)
      auto & arm = openarm_->get_arm();
      for (const auto & m : motors_)
      {
        if (m.device_index < 0)
        {
          continue;
        }
        const double dir = m.direction;
        const double off = m.offset;
        // joint = dir*motor + off  =>  motor = (joint - off)/dir
        const double motor_q = (pos_commands_[m.state_index] - off) / (dir == 0.0 ? 1.0 : dir);
        // MIT: kp, kd from gains; position setpoint; vel=0; tau=0.
        const openarm::damiao_motor::MITParam param{m.kp, m.kd, motor_q, 0.0, 0.0};
        arm.mit_control_one(m.device_index, param);
      }
      openarm_->recv_all(100);
    }
    catch (const std::exception & e)
    {
      // Transient CAN write fault: skip this cycle, do NOT escalate to ERROR.
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger(kLogger), *get_clock(), 1000, "write() CAN fault: %s (skipping cycle)",
        e.what());
    }
  }
  // Mock mode: nothing to send; read() echoes the command into state.
  return hardware_interface::return_type::OK;
}

}  // namespace m1_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(m1_hardware::M1SystemInterface, hardware_interface::SystemInterface)
