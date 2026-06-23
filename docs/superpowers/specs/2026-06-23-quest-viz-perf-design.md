# Quest WebXR viz performance — design

**Date:** 2026-06-23
**Scope:** `ros2_ws/src/m1_control/m1_control/quest_node.py` (the HTTP server +
the embedded three.js client). Plus a new headless server load-test harness.

## Problem

In the Quest in-headset hologram, **the robot model trails and stutters** — it
updates in jerky low-FPS steps and the arms lag behind the operator's hand —
while head-tracked passthrough stays smooth. Isaac Sim and RViz2 tracking are
fine, so the controller/solver is not at fault. Hardware: Quest 3S (capable
GPU), same-WiFi-AP LAN to the DGX Spark.

## Diagnosis (evidence)

The headset re-renders every frame (`renderer.setAnimationLoop(onFrame)` →
always `renderer.render`), so smooth passthrough + a stuttering robot means the
**robot pose data is stale between updates**, not that the GPU is slow.

Measured server cost (headless, `object.__new__` harness, no DDS):

| Measurement | Result |
|---|---|
| Response payload | 4.1 KB (36 links) |
| `on_xr_frame`, FK cache hit (common) | 0.009 ms |
| `on_xr_frame`, FK recomputed every frame | 0.327 ms |
| `on_xr_frame` + `json.dumps` | 0.056 ms |

So the server is not the bottleneck. Root causes, ranked by impact:

1. **No interpolation + round-trip-gated refresh.** The robot pose `lastViz` is
   only assigned in the `/api/xr` fetch `.then`, and the loop is gated to one
   request in flight (`inFlight`). `lastViz` is therefore a step function; the
   render loop poses the model from stale data between POSTs. Over WiFi with
   variable RTT this is the jerky, low-FPS feel.
2. **Nagle's algorithm is ON.** `socketserver.StreamRequestHandler.disable_nagle_algorithm`
   defaults to `False` (verified on this Python 3.12.3) and the Quest `Handler`
   never overrides it; `_send` also writes the response in 5 small chunks. On a
   real network this causes ~40 ms delayed-ACK stalls per request → caps refresh
   near ~25 Hz. (Does not reproduce on loopback, so offline tests never caught it.)
3. **`fetch(..., keepalive:true)`** routes the high-rate POST through the
   browser's constrained beacon pool rather than the optimized HTTP/1.1
   keep-alive connection.

## Chosen approach (A)

### Server (`quest_node.py`)
- Set `disable_nagle_algorithm = True` on the `Handler` (→ `TCP_NODELAY`).
- Coalesce the HTTP response into a **single `wfile.write`** (status line +
  headers + body), so the reply leaves as one segment.

### Client (embedded JS)
- Drop `keepalive:true` from the `/api/xr` fetch.
- **Snapshot interpolation** (the primary fix): retain the two most recent
  server frames with client receive-timestamps; each render frame, interpolate
  link positions (lerp) + orientations (slerp), plus the per-arm markers/lines
  and the base odom pose, at a render clock delayed ~one inter-arrival interval.
  **Hold (clamp, no extrapolation) on a stall** so packet loss can't overshoot.
  This makes a 25–40 Hz data stream render smooth at 72/90 Hz regardless of WiFi
  jitter. Scratch `Vector3`/`Quaternion` objects hoisted out of the loop (no
  per-frame allocation; also fixes the `new` allocations in `updateHud`).
- Inter-arrival interval estimated by EMA of POST round-trips; render delay =
  clamp(interval, ~15..60 ms).

### Instrumentation (metrics)
- In-VR perf HUD gated by **`/?perf`** (hidden on the normal URL): render FPS,
  data-update Hz, POST RTT (ms, EMA), payload KB. A second billboarded canvas
  panel, drawn only when values change (like the existing REACH ERROR HUD).
- Headless server load-test harness (`_quest_perf_bench.py`): drives the real
  `on_xr_frame` + a localhost TLS keep-alive POST loop; reports per-frame cost,
  payload size, and sustained req/s + latency distribution. Honest caveat: the
  full RTT win from the Nagle fix only shows on a real network, not loopback.

## Trade-offs
- Interpolation adds ~one data-interval (~20–40 ms) of latency to the **visual
  preview only**. Arm/base commands publish at 60 Hz server-side, independent of
  the viz, so **control latency is unchanged**. Smooth ≫ 30 ms for a preview.
- Render-delay buffering means the hologram shows state ~one interval old; on a
  hard stall it freezes at the last frame rather than guessing.

## Out of scope (considered, deferred)
- **B: SSE/WebSocket push** of state at 60 Hz decoupled from the input POST —
  the better long-term architecture, but stdlib has no WebSocket, it is a larger
  change, and it still benefits from interpolation. Revisit if A is insufficient.
- GPU-side work (frustum culling, mesh decimation) — symptom says render is not
  the bottleneck; leave the visual fidelity alone.

## Validation
- `_quest_position_test.py` stays 10/10 (the refactor must not disturb the
  clutch / precision / reseed / base / error-window data path).
- `python3 -m py_compile` clean; `_quest_perf_bench.py` reports before/after.
- Interpolation math kept as small pure functions, audited by inspection (no JS
  runtime available in this environment).
- Operator reads the `/?perf` HUD in-headset for before/after FPS / Hz / RTT.

## Implementation checklist
1. Server: `disable_nagle_algorithm = True` + single-write `_send`.
2. Client: drop `keepalive:true`.
3. Client: two-frame buffer + receive timestamps + RTT/interval EMA.
4. Client: interpolate links + markers + base in `updateRobot`; hoist scratch.
5. Client: `/?perf` in-VR HUD panel (FPS / Hz / RTT / KB).
6. Tests: keep `_quest_position_test.py` green; add `_quest_perf_bench.py`.
7. Report metrics + on-device read-off instructions.
