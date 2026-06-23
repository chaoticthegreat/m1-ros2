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
};

/**
 * @brief ros2_control SystemInterface for the M1 Damiao CAN motors.
 *
 * Exports, per joint, a `position` command interface and `position`,
 * `velocity`, `effort` state interfaces. write() issues MIT-mode commands
 * (kp/kd from gains, the position setpoint, vel=0, tau=0); read() pulls
 * position/velocity/torque feedback.
 *
 * The joint set and every motor parameter (CAN id, master id, model, gains)
 * are read from the URDF — see parse_joints(). The plugin LOADS and runs even
 * with no CAN device present (logs "no bus / mock I/O"); live motor I/O is
 * validated later on hardware.
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
  // Parse bus-level params (can_interface, can_fd) from the HardwareInfo.
  void parse_bus_params(const hardware_interface::HardwareInfo & info);
  // Try to open the CAN bus + init motors. Sets bus_ok_; never throws.
  void try_open_bus();

  static openarm::damiao_motor::MotorType model_from_string(const std::string & s, bool & ok);

  // --- configuration ---------------------------------------------------------
  std::string can_interface_ = "can0";
  bool can_fd_ = false;
  std::string motor_map_path_;  // optional, informational (URDF is source of truth)

  std::vector<JointMotor> motors_;  // one entry per ros2_control joint, URDF order

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
