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
//
// FIX 2 regression: motors must be matched to CAN devices by master_id
// (recv_can_id), NOT by std::map iteration order. The openarm device collection
// is a std::map keyed by recv_can_id, so get_motors()/mit_control_one(i) iterate
// in ASCENDING-master_id order. When the URDF joint order is NOT monotonic in
// master_id (operators reassign ids via the config page), indexing motors_[i]
// against device[i] cross-wires commands/feedback. This test locks in the
// by-master_id resolution against the real openarm Motor API.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include <openarm/damiao_motor/dm_motor.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

using openarm::damiao_motor::Motor;
using openarm::damiao_motor::MotorType;

namespace
{
// Mirror of M1SystemInterface::resolve_device_indices' core: build
// master_id(recv_can_id) -> device index over the device-collection order, then
// look each configured master_id up. Returns per-config device indices (-1 if a
// configured master_id has no device).
std::vector<int> resolve(
  const std::vector<uint32_t> & configured_master_ids, const std::vector<Motor> & devices)
{
  std::unordered_map<uint32_t, int> by_master;
  for (size_t i = 0; i < devices.size(); ++i)
  {
    by_master[devices[i].get_recv_can_id()] = static_cast<int>(i);
  }
  std::vector<int> out;
  out.reserve(configured_master_ids.size());
  for (uint32_t mid : configured_master_ids)
  {
    auto it = by_master.find(mid);
    out.push_back(it == by_master.end() ? -1 : it->second);
  }
  return out;
}

// The device collection is a std::map keyed by recv_can_id: replicate that
// ascending ordering to build the "as openarm returns them" device vector from a
// set of (send_id, recv_id, model) the same way init_arm_motors would.
std::vector<Motor> devices_in_map_order(
  const std::vector<std::tuple<uint32_t, uint32_t, MotorType>> & spec)
{
  std::map<uint32_t, Motor> by_recv;  // ascending recv_can_id == device map order
  for (const auto & [send, recv, model] : spec)
  {
    by_recv.emplace(recv, Motor(model, send, recv));
  }
  std::vector<Motor> out;
  for (const auto & [recv, motor] : by_recv)
  {
    out.push_back(motor);
  }
  return out;
}
}  // namespace

// URDF order that is NON-monotonic in master_id (the scrambled-id case): the
// device collection sorts by master_id, so a naive motors_[i] vs device[i] would
// mis-wire. By-master_id resolution must recover the correct per-motor index.
TEST(DeviceMatching, NonMonotonicMasterIdsResolveCorrectly)
{
  // URDF/motors_ order (as parsed): name -> (send_id, master_id, model).
  // master_ids deliberately out of ascending order vs URDF order.
  struct Cfg { uint32_t send, master; MotorType model; };
  const std::vector<Cfg> urdf = {
    {0x05, 0x15, MotorType::DM4340},      // lift_joint
    {0x02, 0x12, MotorType::DM8009},      // left_joint1  (master 0x12 < lift's 0x15)
    {0x03, 0x13, MotorType::DM4340_48V},  // left_joint2
    {0x11, 0x21, MotorType::DM3507},      // right_finger (highest master)
    {0x04, 0x14, MotorType::DM4310},      // left_joint3
  };

  std::vector<std::tuple<uint32_t, uint32_t, MotorType>> spec;
  std::vector<uint32_t> masters;
  for (const auto & c : urdf)
  {
    spec.emplace_back(c.send, c.master, c.model);
    masters.push_back(c.master);
  }

  const std::vector<Motor> devices = devices_in_map_order(spec);
  // Sanity: the device vector IS in ascending-master order (0x12,0x13,0x14,0x15,0x21).
  ASSERT_EQ(devices.size(), 5u);
  EXPECT_EQ(devices[0].get_recv_can_id(), 0x12u);
  EXPECT_EQ(devices[1].get_recv_can_id(), 0x13u);
  EXPECT_EQ(devices[2].get_recv_can_id(), 0x14u);
  EXPECT_EQ(devices[3].get_recv_can_id(), 0x15u);
  EXPECT_EQ(devices[4].get_recv_can_id(), 0x21u);

  const std::vector<int> idx = resolve(masters, devices);
  ASSERT_EQ(idx.size(), urdf.size());

  // Each URDF-order motor must resolve to the device with ITS master_id, and the
  // device at that index must carry the matching send id + model (no cross-wire).
  for (size_t i = 0; i < urdf.size(); ++i)
  {
    ASSERT_GE(idx[i], 0) << "motor " << i << " master 0x" << std::hex << urdf[i].master
                         << " had no device";
    const Motor & dev = devices[idx[i]];
    EXPECT_EQ(dev.get_recv_can_id(), urdf[i].master);
    EXPECT_EQ(dev.get_send_can_id(), urdf[i].send);
    EXPECT_EQ(dev.get_motor_type(), urdf[i].model);
  }

  // Specifically: a naive motors_[i] vs device[i] WOULD be wrong (lift is URDF
  // index 0 but device index 3). The resolver fixes that.
  EXPECT_EQ(idx[0], 3);  // lift_joint (master 0x15) -> 4th device
  EXPECT_EQ(idx[1], 0);  // left_joint1 (master 0x12) -> 1st device
  EXPECT_EQ(idx[3], 4);  // right_finger (master 0x21) -> 5th device
}

// A configured master_id with no matching device must be flagged (-1), which the
// plugin treats as a device-set mismatch and refuses to drive.
TEST(DeviceMatching, MissingDeviceFlagged)
{
  const std::vector<Motor> devices = devices_in_map_order({
    {0x02, 0x12, MotorType::DM8009},
    {0x03, 0x13, MotorType::DM4340},
  });
  const std::vector<int> idx = resolve({0x12, 0x99}, devices);  // 0x99 absent
  EXPECT_EQ(idx[0], 0);
  EXPECT_EQ(idx[1], -1);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
