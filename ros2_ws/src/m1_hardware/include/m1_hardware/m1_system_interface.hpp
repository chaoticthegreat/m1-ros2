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
// ----------------------------------------------------------------------------
// Forked and adapted from openarm_hardware (openarm_simple_hardware.hpp),
// Copyright 2025 Enactic, Inc., Apache-2.0. Ported Humble -> Jazzy and
// generalized away from the hardcoded 7-DOF / fixed CAN-id arrays so an
// arbitrary joint set (incl. a prismatic lift) can be driven from the URDF
// ros2_control <param>s. See vendor/README.vendored.md.
// ----------------------------------------------------------------------------

#ifndef M1_HARDWARE__M1_SYSTEM_INTERFACE_HPP_
#define M1_HARDWARE__M1_SYSTEM_INTERFACE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include <openarm/damiao_motor/dm_motor_constants.hpp>
#include <openarm/damiao_motor/dm_motor.hpp>

// Forward-declare so the heavy SocketCAN header only lives in the .cpp.
namespace openarm::can::socket
{
class OpenArm;
}

namespace m1_hardware
{

/// Per-joint Damiao motor configuration, parsed from the URDF ros2_control
/// <joint> <param>s (or the motor_map YAML). No fixed DOF / id arrays.
struct JointMotor
{
  std::string name;                            // ros2_control joint name
  uint32_t can_id = 0;                         // Damiao send/slave id (e.g. 0x01)
  uint32_t master_id = 0;                      // recv/master id (convention: can_id + 0x10)
  openarm::damiao_motor::MotorType model =     // motor model -> [P,V,T]MAX scaling
    openarm::damiao_motor::MotorType::DM4310;
  double kp = 0.0;                             // MIT impedance proportional gain
  double kd = 0.0;                             // MIT impedance derivative gain
  double direction = 1.0;                      // +1 / -1 URDF sign convention
  double offset = 0.0;                         // joint = direction * motor + offset (rad / m)

  // Index into the per-joint state/command storage vectors (state_names_ order,
  // == URDF joint order). A commanded joint owns one storage slot; this lets a
  // motor read/write its own slot even though motors_ is a SUBSET of all joints
  // (the state-only mimic finger_joint2 have a storage slot but no motor).
  size_t state_index = 0;

  // Index into the openarm device collection (get_dm_devices() / mit_control_one
  // ascending-master_id order), resolved once in try_open_bus() by matching this
  // motor's master_id to the device's recv_can_id. -1 until resolved / no bus.
  // Decouples motor command/feedback from std::map iteration order (FIX 2).
  int device_index = -1;
};

/**
 * @brief ros2_control SystemInterface for the M1 Damiao CAN motors.
 *
 * Exports, per joint, a `position` command interface and `position`,
 * `velocity`, `effort` state interfaces. write() issues MIT-mode commands
 * (kp/kd from gains, the position setpoint, vel=0, tau=0); read() pulls
 * position/velocity/torque feedback.
 *
 * The joint set comes from the URDF; every per-motor parameter (CAN id, master
 * id, model, gains, dir, offset) is taken from the URDF <param> if present, else
 * the motor_map YAML (the schema m1_can_tools writes), else a built-in default —
 * see parse_joints()/load_motor_map(). The plugin LOADS and runs even with no CAN
 * device present (logs "no bus / mock I/O"); a commanded joint with kp==0 refuses
 * to drive the live bus (limp-arm guard). Live motor I/O is validated on hardware.
 */
class M1SystemInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(M1SystemInterface)

  M1SystemInterface();
  ~M1SystemInterface() override;

  // --- Jazzy lifecycle / init ------------------------------------------------
  // Jazzy hands a HardwareComponentInterfaceParams; the legacy on_init(HardwareInfo)
  // is deprecated. We override the params overload and pull hardware_info from it.
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_error(
    const rclcpp_lifecycle::State & previous_state) override;

  // --- Jazzy interface export (shared-ptr, non-deprecated path) --------------
  std::vector<hardware_interface::StateInterface::ConstSharedPtr>
  on_export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface::SharedPtr>
  on_export_command_interfaces() override;

  // --- realtime loop ---------------------------------------------------------
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Parse joints + per-joint motor params from the HardwareInfo (URDF).
  bool parse_joints(const hardware_interface::HardwareInfo & info);
  // Parse bus-level params (can_interface, can_fd, motor_map) from the HardwareInfo.
  void parse_bus_params(const hardware_interface::HardwareInfo & info);
  // Load per-joint motor config from the motor_map YAML (if motor_map_path_ is a
  // non-empty, readable file). Returns the joint->config map; empty on miss/error.
  std::unordered_map<std::string, JointMotor> load_motor_map();
  // Try to open the CAN bus + init motors. Sets bus_ok_; never throws.
  void try_open_bus();
  // Resolve each motors_[k].device_index by matching its master_id to the
  // device collection's recv_can_id (FIX 2: order-independent motor<->device
  // wiring). Returns false (and refuses the bus) if the configured master_ids
  // don't match the device set.
  bool resolve_device_indices(const std::vector<openarm::damiao_motor::Motor> & devices);

  static openarm::damiao_motor::MotorType model_from_string(const std::string & s, bool & ok);

  // --- configuration ---------------------------------------------------------
  std::string can_interface_ = "can0";
  bool can_fd_ = false;
  std::string motor_map_path_;  // optional YAML of per-joint motor config (FIX 1)

  // Every ros2_control joint, in URDF order, with its storage index. State
  // interfaces are exported for ALL of these (incl. the state-only mimic
  // finger_joint2). Names index pos_states_/vel_states_/tau_states_.
  std::vector<std::string> state_names_;

  // The COMMANDED motors (subset of state_names_): only joints that have a
  // `position` command interface. The state-only mimic joints are excluded
  // (no physical motor) -- FIX 3. Each entry's state_index points back into the
  // per-joint storage vectors / state_names_.
  std::vector<JointMotor> motors_;

  // True once parse_joints found a COMMANDED joint with kp==0 (or dir==0) -- a
  // limp/unsafe config. The bus is then refused (never enabled) so the arms can't
  // silently go limp / run open-loop on hardware; mock/no-bus loads still succeed
  // (FIX 1 safety guard).
  bool limp_config_ = false;

  // Mimic (state-only) joints copied from the URDF, e.g. *_finger_joint2 mimicking
  // *_finger_joint1. joint_index / mimicked_joint_index are state_names_ indices
  // (== URDF joint order). read() propagates state = offset + multiplier * leader,
  // since a mimic has no motor and would otherwise report a constant 0.
  std::vector<hardware_interface::MimicJoint> mimics_;

  // --- CAN bus (lazily opened; null => mock I/O) -----------------------------
  std::unique_ptr<openarm::can::socket::OpenArm> openarm_;
  bool bus_ok_ = false;  // true once a real bus is open and motors are initialised

  // --- state / command storage (owned; pointed at by exported interfaces) ----
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;
  std::vector<double> pos_commands_;

  // exported handles (Jazzy shared-ptr interfaces) kept for realtime read/write
  std::vector<hardware_interface::StateInterface::SharedPtr> pos_state_ifaces_;
  std::vector<hardware_interface::StateInterface::SharedPtr> vel_state_ifaces_;
  std::vector<hardware_interface::StateInterface::SharedPtr> tau_state_ifaces_;
  std::vector<hardware_interface::CommandInterface::SharedPtr> pos_cmd_ifaces_;
};

}  // namespace m1_hardware

#endif  // M1_HARDWARE__M1_SYSTEM_INTERFACE_HPP_
