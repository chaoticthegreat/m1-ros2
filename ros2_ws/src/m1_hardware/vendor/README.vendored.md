# Vendored third-party source

## `openarm_can/`

This directory is a **vendored copy** of the OpenArm CAN library
(`openarm_can`, https://github.com/enactic/openarm_can), version 1.2.9,
Copyright 2025 Enactic, Inc., licensed under the **Apache License 2.0**.

The upstream `LICENSE.txt` is retained unmodified at
`openarm_can/LICENSE.txt`. No source files were modified.

`m1_hardware`'s own `CMakeLists.txt` compiles the library's C++ sources
(`openarm_can/src/openarm/**/*.cpp`) directly into a static `openarm_can`
target — the vendored `openarm_can/CMakeLists.txt` is **not** used (it pulls
in `CLI11`, builds CLI/demo executables and Python bindings we don't need).
Only the SocketCAN Damiao motor codec + transport is consumed.

Why vendored rather than a system/rosdep dependency: `openarm_can` is not in
the ROS 2 Jazzy / rosdep index, and we want a reproducible offline build on
this machine (DGX Spark, aarch64) without an extra apt/PPA source.
