# Jazzy-in-Docker for the M1 control box on a JetPack 6 (Ubuntu 22.04) Orin.
# Use this ONLY when the host is NOT Ubuntu 24.04 Noble (see DEPLOY_AGX_ORIN.md
# "Path B"). On a native-Noble host, install with setup_agx_orin.sh instead.
#
# Build (from the repo root):
#   docker build -t m1-jazzy -f deploy/agx-orin/Dockerfile .
# Run with host networking + device passthrough so SocketCAN/serial reach in
# (bring up can0 on the HOST, not the container):
#   docker run -it --rm --network host --cap-add NET_ADMIN -v /dev:/dev \
#     -v "$PWD":/work -w /work m1-jazzy
# Inside: rosdep install + colcon build + ros2 launch, exactly as native.
FROM osrf/ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-controller-manager \
        python3-pip python3-rosdep python3-colcon-common-extensions \
        can-utils libyaml-cpp-dev \
    && pip3 install --break-system-packages drake python-can pyserial \
    && rm -rf /var/lib/apt/lists/*

# Drake's geometry import pulls GL libs even headless:
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libopengl0 libglib2.0-0 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
