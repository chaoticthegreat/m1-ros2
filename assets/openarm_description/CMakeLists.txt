cmake_minimum_required(VERSION 3.10)
project(openarm_description)

find_package(ament_cmake REQUIRED)

# Install the mesh/asset tree so package:// references resolve at runtime.
install(
  DIRECTORY assets
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
