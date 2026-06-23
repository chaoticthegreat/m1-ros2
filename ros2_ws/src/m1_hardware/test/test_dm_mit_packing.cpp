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
// Cross-checks the vendored openarm_can DM constants + the MIT byte packing
// against the documented values in the project's Global Constraints (and the
// Python m1_can_tools.dm_protocol.encode_mit semantics).

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include <openarm/damiao_motor/dm_motor_constants.hpp>

using openarm::damiao_motor::MotorType;
using openarm::damiao_motor::MOTOR_LIMIT_PARAMS;

namespace
{
// Mirror of the documented float->uint quantization (clamp, then linear map
// lo->0, hi->(1<<bits)-1, truncating). Byte-identical to both the Python
// dm_protocol.float_to_uint and openarm_can's private double_to_uint.
uint16_t float_to_uint(double x, double lo, double hi, int bits)
{
  const double span = hi - lo;
  if (span <= 0.0)
  {
    return 0;
  }
  if (x < lo)
  {
    x = lo;
  }
  else if (x > hi)
  {
    x = hi;
  }
  return static_cast<uint16_t>((x - lo) * ((1 << bits) - 1) / span);
}

// Re-implementation of the documented MIT 16/12/12/12/12 packing, using the
// openarm_can limit table for [P,V,T]MAX. This is the exact byte layout the
// hardware interface relies on openarm_can to produce on the wire.
std::array<uint8_t, 8> encode_mit(
  double p, double v, double kp, double kd, double tau, MotorType model)
{
  const auto & lim = MOTOR_LIMIT_PARAMS[static_cast<int>(model)];
  const uint16_t q_u = float_to_uint(p, -lim.pMax, lim.pMax, 16);
  const uint16_t dq_u = float_to_uint(v, -lim.vMax, lim.vMax, 12);
  const uint16_t kp_u = float_to_uint(kp, 0.0, 500.0, 12);
  const uint16_t kd_u = float_to_uint(kd, 0.0, 5.0, 12);
  const uint16_t tau_u = float_to_uint(tau, -lim.tMax, lim.tMax, 12);
  return {
    static_cast<uint8_t>((q_u >> 8) & 0xFF),
    static_cast<uint8_t>(q_u & 0xFF),
    static_cast<uint8_t>((dq_u >> 4) & 0xFF),
    static_cast<uint8_t>(((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)),
    static_cast<uint8_t>(kp_u & 0xFF),
    static_cast<uint8_t>((kd_u >> 4) & 0xFF),
    static_cast<uint8_t>(((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)),
    static_cast<uint8_t>(tau_u & 0xFF)};
}
}  // namespace

// The openarm_can limit table must match the Global-Constraints [P,V,T]MAX.
TEST(DmConstants, LimitTableMatchesGlobalConstraints)
{
  auto lim = [](MotorType m) { return MOTOR_LIMIT_PARAMS[static_cast<int>(m)]; };
  EXPECT_DOUBLE_EQ(lim(MotorType::DM4310).pMax, 12.5);
  EXPECT_DOUBLE_EQ(lim(MotorType::DM4310).vMax, 30.0);
  EXPECT_DOUBLE_EQ(lim(MotorType::DM4310).tMax, 10.0);

  EXPECT_DOUBLE_EQ(lim(MotorType::DM4340).vMax, 8.0);
  EXPECT_DOUBLE_EQ(lim(MotorType::DM4340).tMax, 28.0);

  EXPECT_DOUBLE_EQ(lim(MotorType::DM8009).vMax, 45.0);
  EXPECT_DOUBLE_EQ(lim(MotorType::DM8009).tMax, 54.0);

  EXPECT_DOUBLE_EQ(lim(MotorType::DM10010).tMax, 200.0);
}

// p=v=kp=kd=tau=0 over symmetric ranges -> q midpoint = 0x7FFF.
TEST(DmMitPacking, ZeroPacking)
{
  auto b = encode_mit(0, 0, 0, 0, 0, MotorType::DM4310);
  EXPECT_EQ(b[0], 0x7F);
  EXPECT_EQ(b[1], 0xFF);
}

// kp uses [0,500], kd [0,5] -> kp=0 -> 0, kp=500 & kd=5 -> 0xFFF.
TEST(DmMitPacking, KpKdRanges)
{
  auto b_lo = encode_mit(0, 0, 0, 0, 0, MotorType::DM4310);
  auto b_hi = encode_mit(0, 0, 500, 5, 0, MotorType::DM4310);
  const uint16_t kp_lo = ((b_lo[3] & 0xF) << 8) | b_lo[4];
  const uint16_t kp_hi = ((b_hi[3] & 0xF) << 8) | b_hi[4];
  EXPECT_EQ(kp_lo, 0u);
  EXPECT_EQ(kp_hi, 0xFFFu);
  const uint16_t kd_hi = (b_hi[5] << 4) | (b_hi[6] >> 4);
  EXPECT_EQ(kd_hi, 0xFFFu);
}

// Position endpoints: -P_MAX -> 0x0000, +P_MAX -> 0xFFFF.
TEST(DmMitPacking, PositionEndpoints)
{
  auto b_min = encode_mit(-12.5, 0, 0, 0, 0, MotorType::DM4310);
  auto b_max = encode_mit(12.5, 0, 0, 0, 0, MotorType::DM4310);
  const uint16_t q_min = (b_min[0] << 8) | b_min[1];
  const uint16_t q_max = (b_max[0] << 8) | b_max[1];
  EXPECT_EQ(q_min, 0x0000u);
  EXPECT_EQ(q_max, 0xFFFFu);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
