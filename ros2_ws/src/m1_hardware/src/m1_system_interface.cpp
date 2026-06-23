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

bool M1SystemInterface::parse_joints(const hardware_interface::HardwareInfo & info)
{
  motors_.clear();
  const auto & hw = info.hardware_parameters;

  uint32_t auto_id = 1;  // fallback CAN id allocator if none specified
  for (const auto & joint : info.joints)
  {
    JointMotor m;
    m.name = joint.name;

    const auto & jp = joint.parameters;
    m.can_id = parse_id(lookup(jp, hw, "can_id"), auto_id);
    // Master/recv id convention from the spec: slave + 0x10 (never 0).
    m.master_id = parse_id(lookup(jp, hw, "master_id"), m.can_id + 0x10);

    bool model_ok = true;
    const std::string model_str = lookup(jp, hw, "motor_model", "DM4310");
    m.model = model_from_string(model_str, model_ok);
    if (!model_ok)
    {
      RCLCPP_WARN(
        rclcpp::get_logger(kLogger),
        "Joint '%s': unknown motor_model '%s', defaulting to DM4310", m.name.c_str(),
        model_str.c_str());
    }

    // Gains: per-joint <param kp/kd>, else hardware-level kp/kd, else 0.
    m.kp = parse_double(lookup(jp, hw, "kp"), 0.0);
    m.kd = parse_double(lookup(jp, hw, "kd"), 0.0);
    m.direction = parse_double(lookup(jp, hw, "dir"), 1.0);
    m.offset = parse_double(lookup(jp, hw, "offset"), 0.0);

    motors_.push_back(m);
    auto_id = m.can_id + 1;

    RCLCPP_INFO(
      rclcpp::get_logger(kLogger),
      "Joint[%zu] '%s': model=%s can_id=0x%02X master_id=0x%02X kp=%.3f kd=%.3f dir=%+.0f",
      motors_.size() - 1, m.name.c_str(), model_str.c_str(), m.can_id, m.master_id, m.kp, m.kd,
      m.direction);
  }

  if (motors_.empty())
  {
    RCLCPP_ERROR(rclcpp::get_logger(kLogger), "No joints found in HardwareInfo");
    return false;
  }
  return true;
}

void M1SystemInterface::try_open_bus()
{
  bus_ok_ = false;
  openarm_.reset();
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

  const size_t n = motors_.size();
  pos_states_.assign(n, 0.0);
  vel_states_.assign(n, 0.0);
  tau_states_.assign(n, 0.0);
  pos_commands_.assign(n, 0.0);

  RCLCPP_INFO(
    rclcpp::get_logger(kLogger), "M1SystemInterface initialised with %zu joints", n);
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface::ConstSharedPtr>
M1SystemInterface::on_export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> out;
  pos_state_ifaces_.clear();
  vel_state_ifaces_.clear();
  tau_state_ifaces_.clear();

  for (size_t i = 0; i < motors_.size(); ++i)
  {
    auto make = [&](const std::string & iface, double * ptr) {
      hardware_interface::InterfaceInfo ii;
      ii.name = iface;
      hardware_interface::InterfaceDescription desc(motors_[i].name, ii);
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

  for (size_t i = 0; i < motors_.size(); ++i)
  {
    hardware_interface::InterfaceInfo ii;
    ii.name = hardware_interface::HW_IF_POSITION;
    hardware_interface::InterfaceDescription desc(motors_[i].name, ii);
    auto ci = std::make_shared<hardware_interface::CommandInterface>(desc);
    std::ignore = ci->set_value(pos_commands_[i]);
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

  // Seed command interfaces from current state so we hold position on activate.
  for (size_t i = 0; i < motors_.size(); ++i)
  {
    pos_commands_[i] = pos_states_[i];
    if (i < pos_cmd_ifaces_.size())
    {
      std::ignore = pos_cmd_ifaces_[i]->set_value(pos_commands_[i]);
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
      const auto & arm_motors = openarm_->get_arm().get_motors();
      for (size_t i = 0; i < motors_.size() && i < arm_motors.size(); ++i)
      {
        const double dir = motors_[i].direction;
        const double off = motors_[i].offset;
        pos_states_[i] = dir * arm_motors[i].get_position() + off;
        vel_states_[i] = dir * arm_motors[i].get_velocity();
        tau_states_[i] = dir * arm_motors[i].get_torque();
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
    for (size_t i = 0; i < motors_.size(); ++i)
    {
      pos_states_[i] = pos_commands_[i];
      vel_states_[i] = 0.0;
      tau_states_[i] = 0.0;
    }
  }

  // Publish into the exported state handles.
  for (size_t i = 0; i < motors_.size(); ++i)
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
  // Pull the latest position commands from the exported handles.
  for (size_t i = 0; i < motors_.size(); ++i)
  {
    if (i < pos_cmd_ifaces_.size())
    {
      const auto v = pos_cmd_ifaces_[i]->get_optional();
      if (v.has_value())
      {
        pos_commands_[i] = v.value();
      }
    }
  }

  if (bus_ok_ && openarm_)
  {
    try
    {
      std::vector<openarm::damiao_motor::MITParam> params;
      params.reserve(motors_.size());
      for (size_t i = 0; i < motors_.size(); ++i)
      {
        const double dir = motors_[i].direction;
        const double off = motors_[i].offset;
        // joint = dir*motor + off  =>  motor = (joint - off)/dir
        const double motor_q = (pos_commands_[i] - off) / (dir == 0.0 ? 1.0 : dir);
        // MIT: kp, kd from gains; position setpoint; vel=0; tau=0.
        params.push_back({motors_[i].kp, motors_[i].kd, motor_q, 0.0, 0.0});
      }
      openarm_->get_arm().mit_control_all(params);
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
