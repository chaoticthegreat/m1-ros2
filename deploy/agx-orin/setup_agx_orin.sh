#!/usr/bin/env bash
#
# setup_agx_orin.sh -- idempotent software bootstrap for the M1 control box on an
# NVIDIA Jetson AGX Orin (or any aarch64 Ubuntu 24.04 "Noble" host).
#
# It does the DETERMINISTIC, hardware-free part of deployment:
#   * verifies the host is a supported target (aarch64 + Noble + py3.12)
#   * installs ROS 2 Jazzy (ros-base) + the ros2_control runtime the app needs
#   * installs Drake (the IK backend) + python-can/pyserial + can-utils
#   * resolves rosdep deps and builds the colcon workspace
#   * runs a fast preflight (imports + a couple of quick gated suites)
#
# It deliberately does NOT touch hardware: no CAN bring-up, no motor config, no
# Jetson power/clock tuning, no systemd install. Those are Phases 6-11 of
# DEPLOY_AGX_ORIN.md and need real hardware / root policy decisions.
#
# Safe to re-run. See: ../../DEPLOY_AGX_ORIN.md
#
# Usage:
#   ./deploy/agx-orin/setup_agx_orin.sh            # detect + install + build + preflight
#   ./deploy/agx-orin/setup_agx_orin.sh --check    # detect only; change nothing
#   ./deploy/agx-orin/setup_agx_orin.sh --with-moveit   # also install ros-jazzy-moveit
#   ./deploy/agx-orin/setup_agx_orin.sh --full-tests    # run the full (slow) gated suite at the end
#   ./deploy/agx-orin/setup_agx_orin.sh --no-build      # install deps only, skip colcon build
#
set -euo pipefail

# --- locate the repo root (this script lives at deploy/agx-orin/) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WS="${REPO_ROOT}/ros2_ws"

PY=/usr/bin/python3          # the Jazzy interpreter -- NEVER the conda python3 on PATH
ROS_DISTRO=jazzy
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

CHECK_ONLY=0
WITH_MOVEIT=0
FULL_TESTS=0
DO_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --check)       CHECK_ONLY=1 ;;
    --with-moveit) WITH_MOVEIT=1 ;;
    --full-tests)  FULL_TESTS=1 ;;
    --no-build)    DO_BUILD=0 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# --- pretty logging ----------------------------------------------------------
c_bold=$'\033[1m'; c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_off=$'\033[0m'
say()  { printf '%s\n' "${c_bold}==>${c_off} $*"; }
ok()   { printf '%s\n' "${c_grn}[ ok ]${c_off} $*"; }
warn() { printf '%s\n' "${c_yel}[warn]${c_off} $*"; }
die()  { printf '%s\n' "${c_red}[FAIL]${c_off} $*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi

# ============================================================================
# 1. DETECT
# ============================================================================
say "Fingerprinting the host"
ARCH="$(uname -m)"
. /etc/os-release
CODENAME="${VERSION_CODENAME:-unknown}"
DPKG_ARCH="$(dpkg --print-architecture 2>/dev/null || echo unknown)"
KERNEL="$(uname -r)"
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo '?')"
PYVER="$("$PY" -V 2>&1 | awk '{print $2}')"
NPROC="$(nproc)"
MODEL=""
if [[ -r /sys/firmware/devicetree/base/model ]]; then
  MODEL="$(tr -d '\0' < /sys/firmware/devicetree/base/model 2>/dev/null || true)"
fi
if [[ -r /etc/nv_tegra_release ]]; then
  L4T="$(head -1 /etc/nv_tegra_release)"
else
  L4T="(no /etc/nv_tegra_release -- not an L4T/Jetson image)"
fi

cat <<EOF
    model      : ${MODEL:-unknown}
    arch       : ${ARCH} (dpkg: ${DPKG_ARCH})
    os         : ${PRETTY_NAME} [codename: ${CODENAME}]
    kernel     : ${KERNEL}
    glibc      : ${GLIBC}
    /usr/bin/python3 : ${PYVER}
    cores      : ${NPROC}
    L4T        : ${L4T}
EOF

# --- gate the host -----------------------------------------------------------
problems=0
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  warn "arch is '${ARCH}', expected aarch64/arm64."; problems=1
fi
if [[ "$CODENAME" != "noble" ]]; then
  warn "Ubuntu codename is '${CODENAME}', not 'noble' (24.04)."
  problems=1
fi
case "$PYVER" in
  3.12.*) : ;;
  *) warn "/usr/bin/python3 is ${PYVER}, expected 3.12.x (the Jazzy interpreter)."; problems=1 ;;
esac
# glibc >= 2.34 needed by the Drake wheel
if [[ "$GLIBC" != "?" ]]; then
  if ! awk -v g="$GLIBC" 'BEGIN{split(g,a,"."); exit !((a[1]>2)||(a[1]==2&&a[2]>=34))}'; then
    warn "glibc ${GLIBC} < 2.34 -- the Drake aarch64 wheel needs >= 2.34."; problems=1
  fi
fi

if [[ $problems -ne 0 ]]; then
  cat <<EOF

${c_yel}This host is not a native-Jazzy target.${c_off} ROS 2 Jazzy + the Drake wheel
require aarch64 + Ubuntu 24.04 (Noble) + Python 3.12.

If this is a Jetson on JetPack 6 (Ubuntu 22.04 'jammy'), use the DOCKER path
instead -- see DEPLOY_AGX_ORIN.md "Path B", or re-flash to JetPack 7.2 (Noble).
EOF
  if [[ $CHECK_ONLY -eq 1 ]]; then exit 0; fi
  die "Refusing to install on an unsupported host. Use the Docker path, or pass --check to inspect only."
fi
ok "Host is a supported native-Jazzy target."

if [[ $CHECK_ONLY -eq 1 ]]; then
  say "Plan (--check, nothing changed):"
  cat <<EOF
    - install ROS 2 ${ROS_DISTRO} (ros-base) + ros2_control + controllers + tools via apt
    - install Drake + python-can + pyserial via ${PY} (--user --break-system-packages)
    - rosdep install over ${WS}/src
    - colcon build --symlink-install in ${WS}$( [[ $DO_BUILD -eq 0 ]] && echo ' (SKIPPED: --no-build)')
    - fast preflight$( [[ $FULL_TESTS -eq 1 ]] && echo ' + full gated suite (--full-tests)')
EOF
  exit 0
fi

# ============================================================================
# 2. ROS 2 Jazzy + ros2_control (apt)
# ============================================================================
if [[ -f "$ROS_SETUP" ]]; then
  ok "ROS 2 ${ROS_DISTRO} already present at /opt/ros/${ROS_DISTRO}."
else
  say "Installing the ROS 2 ${ROS_DISTRO} apt source"
  $SUDO apt-get update
  $SUDO apt-get install -y locales software-properties-common curl
  $SUDO locale-gen en_US en_US.UTF-8
  $SUDO update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  $SUDO add-apt-repository -y universe
  RAS="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
  [[ -n "$RAS" ]] || die "could not resolve the latest ros-apt-source release tag (network?)."
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${RAS}/ros2-apt-source_${RAS}.${CODENAME}_all.deb"
  $SUDO dpkg -i /tmp/ros2-apt-source.deb
  $SUDO apt-get update
fi

say "Installing ROS packages (ros-base + ros2_control + tooling)"
PKGS=(
  ros-${ROS_DISTRO}-ros-base
  ros-dev-tools
  ros-${ROS_DISTRO}-ros2-control
  ros-${ROS_DISTRO}-ros2-controllers
  ros-${ROS_DISTRO}-controller-manager
  python3-colcon-common-extensions
  python3-rosdep
  can-utils
  libyaml-cpp-dev
)
if [[ $WITH_MOVEIT -eq 1 ]]; then PKGS+=( ros-${ROS_DISTRO}-moveit ); fi
$SUDO apt-get install -y "${PKGS[@]}"
ok "ROS + ros2_control installed."

# ============================================================================
# 3. Drake + python-can + pyserial (pip, the Jazzy interpreter)
# ============================================================================
say "Installing Drake (IK backend) + CAN python deps for ${PY}"
"$PY" -m pip install --user --break-system-packages -U pip
"$PY" -m pip install --user --break-system-packages drake python-can pyserial

say "Verifying Drake imports under ${PY}"
if "$PY" - <<'PYEOF'
import sys
import pydrake
from pydrake.solvers import SnoptSolver, IpoptSolver
import pydrake.all  # pulls GL/geometry libs; surfaces missing .so here
print(f"    drake {pydrake.__version__} on python {sys.version.split()[0]}")
print(f"    snopt available={SnoptSolver().available()}  ipopt available={IpoptSolver().available()}")
PYEOF
then
  ok "Drake import + solvers OK."
else
  warn "Drake import failed. If it was a missing libGL/libOpenGL, run:"
  warn "  ${SUDO} apt-get install -y libgl1 libopengl0 libglib2.0-0 libx11-6"
  warn "then re-run this script."
  die "Drake is a hard runtime dependency of the brain -- cannot proceed without it."
fi

# ============================================================================
# 4. rosdep + colcon build
# ============================================================================
# ROS / ament / colcon prefix scripts reference unbound vars; relax -u while sourcing.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

say "Resolving rosdep dependencies for the workspace"
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  $SUDO rosdep init || true
fi
rosdep update || warn "rosdep update failed (network?) -- continuing; deps may already be present."
rosdep install --from-paths "${WS}/src" --ignore-src -y --rosdistro "${ROS_DISTRO}" \
  || warn "rosdep install reported issues -- check output above (often harmless if deps already installed)."

if [[ $DO_BUILD -eq 1 ]]; then
  say "Building the colcon workspace (--symlink-install)"
  ( cd "$WS" && colcon build --symlink-install )
  ok "colcon build complete."
else
  warn "Skipping colcon build (--no-build)."
fi

# ============================================================================
# 5. Preflight
# ============================================================================
say "Preflight: importing the kinematics module under ${PY}"
PYTHONPATH="${WS}/src/m1_control" "$PY" -c 'import m1_control.kinematics; print("    m1_control.kinematics import OK")' \
  && ok "Brain kinematics imports." || warn "kinematics import failed -- check Drake/numpy."

say "Preflight: fast gated suites (swerve + collision + trajectory)"
( cd "$REPO_ROOT" && "$PY" _swerve_test.py ) && ok "swerve suite passed." || warn "swerve suite did not pass."
( cd "$REPO_ROOT" && PYTHONPATH=ros2_ws/src/m1_control "$PY" -m m1_control.collision )  && ok "collision self-test passed." || warn "collision self-test failed."
( cd "$REPO_ROOT" && PYTHONPATH=ros2_ws/src/m1_control "$PY" -m m1_control.trajectory ) && ok "trajectory smoke passed."  || warn "trajectory smoke failed."

if [[ $FULL_TESTS -eq 1 ]]; then
  say "Full gated suite (slow -- the cold multi-seed IK suites take a while)"
  ( cd "$REPO_ROOT" && for t in _solver_test.py _solver_test_positions.py _solver_test_tracking.py _solver_test_pathing.py _accuracy_bench.py; do
      echo "--- $t ---"; "$PY" "$t" || warn "$t did not pass"; done )
  ( cd "$REPO_ROOT" && PYTHONPATH=ros2_ws/src/m1_control "$PY" ros2_ws/src/m1_control/_bridge_test.py ) || warn "bridge test did not pass"
  ( cd "$REPO_ROOT" && PYTHONPATH=ros2_ws/src/m1_can_tools "$PY" -m pytest ros2_ws/src/m1_can_tools/test -q ) || warn "can_tools tests did not pass"
fi

# ============================================================================
# Done
# ============================================================================
cat <<EOF

${c_grn}${c_bold}Software bootstrap complete.${c_off}

Next steps (see DEPLOY_AGX_ORIN.md):
  source ${ROS_SETUP}
  source ${WS}/install/setup.bash

  # Phase 5b -- mock bring-up (no hardware):
  ros2 launch m1_bringup hardware.launch.py use_mock:=true

  # Phase 6+  -- CAN bus, motor config, real bring-up, tuning, autostart:
  sudo ${SCRIPT_DIR}/can_up.sh            # bring up can0 (CAN-FD 1M/5M)
  ros2 run m1_can_tools m1_hwconfig --ros-args -p transport:=socketcan -p can_channel:=can0
  ros2 launch m1_bringup hardware.launch.py use_mock:=false can_interface:=can0 \\
       can_fd:=true motor_map:=\$HOME/.config/m1/motor_map.yaml

Run the full offline regression any time with:
  ${SCRIPT_DIR}/setup_agx_orin.sh --full-tests --no-build
EOF
