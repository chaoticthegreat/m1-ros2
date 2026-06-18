cmake_minimum_required(VERSION 3.10)
project(ranger_air_description)

find_package(ament_cmake REQUIRED)

# Install the URDF and meshes so package:// references resolve at runtime.
install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
