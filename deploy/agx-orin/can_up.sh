#!/usr/bin/env bash
#
# can_up.sh -- idempotently bring up a SocketCAN interface for the M1 Damiao
# motors (CAN-FD 1 Mbps nominal / 5 Mbps data by default), with detection of the
# common Jetson failure modes (missing gs_usb module, unmuxed native pins).
#
# Run as root (or via sudo). See DEPLOY_AGX_ORIN.md "Phase 6".
#
# Usage:
#   sudo ./can_up.sh                       # can0, CAN-FD 1M/5M
#   sudo ./can_up.sh can1                  # a different interface
#   sudo IFACE=can0 BITRATE=1000000 DBITRATE=5000000 FD=yes ./can_up.sh
#   sudo FD=no ./can_up.sh                 # classic CAN @ 1 Mbps (no data phase)
#
set -euo pipefail

IFACE="${1:-${IFACE:-can0}}"
BITRATE="${BITRATE:-1000000}"     # arbitration / nominal bitrate
DBITRATE="${DBITRATE:-5000000}"   # CAN-FD data bitrate
FD="${FD:-yes}"                   # yes -> CAN-FD; no -> classic CAN
TXQLEN="${TXQLEN:-1000}"

if [[ $EUID -ne 0 ]]; then echo "must run as root (use sudo)" >&2; exit 1; fi

c_yel=$'\033[33m'; c_grn=$'\033[32m'; c_off=$'\033[0m'
warn() { printf '%s\n' "${c_yel}[warn]${c_off} $*" >&2; }
ok()   { printf '%s\n' "${c_grn}[ ok ]${c_off} $*"; }

# --- 1. make sure the interface exists, loading modules if needed ------------
if ! ip link show "$IFACE" >/dev/null 2>&1; then
  echo "==> $IFACE not present; attempting to load CAN modules"
  modprobe can     2>/dev/null || true
  modprobe can_raw 2>/dev/null || true

  # Native Tegra controller?
  if modinfo mttcan >/dev/null 2>&1; then
    modprobe mttcan 2>/dev/null || true
  fi
  # USB adapters (only if a module is actually built)
  if lsusb 2>/dev/null | grep -qiE 'peak|pcan'; then
    modinfo peak_usb >/dev/null 2>&1 && modprobe peak_usb 2>/dev/null || \
      warn "PEAK USB seen but peak_usb module not available -- build it (see guide)."
  fi
  if lsusb 2>/dev/null | grep -qiE 'canable|candlelight|gs_usb|openmoko|1d50:'; then
    if modinfo gs_usb >/dev/null 2>&1; then
      modprobe gs_usb 2>/dev/null || true
    else
      warn "A gs_usb-class USB-CAN adapter is plugged in but gs_usb is NOT in this kernel."
      warn "On stock JetPack 6 you must build gs_usb out-of-tree, or use native mttcan / a PEAK dongle."
    fi
  fi
  sleep 1
fi

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  warn "$IFACE still does not exist."
  warn "Diagnose: 'dmesg | grep -iE \"mttcan|gs_usb|peak|can\"', 'lsusb', 'modinfo gs_usb mttcan peak_usb'."
  warn "Native mttcan often needs a pinmux enable: sudo /opt/nvidia/jetson-io/jetson-io.py  (then reboot)."
  exit 1
fi

# --- 2. (re)configure the link -----------------------------------------------
echo "==> configuring $IFACE  (bitrate=$BITRATE fd=$FD dbitrate=$DBITRATE)"
ip link set "$IFACE" down 2>/dev/null || true

if [[ "$FD" == "yes" || "$FD" == "true" || "$FD" == "1" ]]; then
  if ! ip link set "$IFACE" up type can bitrate "$BITRATE" dbitrate "$DBITRATE" fd on restart-ms 100; then
    warn "CAN-FD bring-up failed; retrying in classic CAN mode (no 5 Mbps data phase)."
    ip link set "$IFACE" up type can bitrate "$BITRATE" restart-ms 100
  fi
else
  ip link set "$IFACE" up type can bitrate "$BITRATE" restart-ms 100
fi

ip link set "$IFACE" txqueuelen "$TXQLEN"

# --- 3. report ---------------------------------------------------------------
echo "==> $IFACE state:"
ip -details link show "$IFACE" | sed 's/^/    /'
ok "$IFACE is up. Smoke-test with:  candump $IFACE   (frames appear when motors are powered)"
