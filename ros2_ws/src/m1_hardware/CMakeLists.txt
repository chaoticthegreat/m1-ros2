# Copyright 2026 M1 Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(m1_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# ---------------------------------------------------------------------------
# Vendored openarm_can (Apache-2.0) -- built directly from sources, NOT via its
# own CMakeLists.txt (which pulls in CLI11 + builds CLI/demo/python targets we
# don't need). See vendor/README.vendored.md. License kept at
# vendor/openarm_can/LICENSE.txt.
# ---------------------------------------------------------------------------
set(OPENARM_CAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/openarm_can)
file(GLOB OPENARM_CAN_SOURCES
  ${OPENARM_CAN_DIR}/src/openarm/can/socket/*.cpp
  ${OPENARM_CAN_DIR}/src/openarm/canbus/*.cpp
  ${OPENARM_CAN_DIR}/src/openarm/damiao_motor/*.cpp
)

add_library(openarm_can STATIC ${OPENARM_CAN_SOURCES})
target_include_directories(openarm_can PUBLIC
  $<BUILD_INTERFACE:${OPENARM_CAN_DIR}/include>
)
set_target_properties(openarm_can PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
# Vendored third-party code: don't fail our build on its warnings.
target_compile_options(openarm_can PRIVATE -w)

# ---------------------------------------------------------------------------
# M1 hardware plugin
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED
  src/m1_system_interface.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${PROJECT_NAME} openarm_can)

ament_target_dependencies(${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# Export the pluginlib description so controller_manager can discover the class.
pluginlib_export_plugin_description_file(hardware_interface m1_hardware.xml)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS openarm_can
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)
install(FILES m1_hardware.xml
  DESTINATION share/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_dm_mit_packing test/test_dm_mit_packing.cpp)
  target_include_directories(test_dm_mit_packing PRIVATE
    ${OPENARM_CAN_DIR}/include
  )
  target_link_libraries(test_dm_mit_packing openarm_can)
endif()

ament_export_include_directories(include)
ament_export_dependencies(
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

ament_package()
