# Quest teleop latency — optimizations + cloud roadmap (2026-07-01)

Goal: an operator wearing a Quest, eventually **anywhere in the world (cloud, RTT
50–300 ms)**, controls the robot with low perceived lag. This is **Phase 1** —
the local-lag relief + the control/viz decoupling seam that the eventual cloud
transport plugs into. No cloud infra is built yet.

## Ground truth (measured, `_quest_perf_bench.py` + live)

- **Server is NOT the bottleneck:** `on_xr_frame` ~0.06 ms (cache hit) / ~0.32 ms
  (recompute); ~1300 req/s sustained over one keep-alive TLS connection.
- **Dominant cause of felt lag = the one-in-flight POST RTT coupling.** Control
  (hand pose → arm target) rode the *same* one-in-flight POST as the ~4 KB viz, so
  the control-update *rate* = `1/RTT`: ~25–65 Hz on WiFi, **~10 Hz @100 ms, ~5 Hz
  @200 ms** at cloud RTTs. Between fresh targets the arm goal just *stepped* (stair-
  cased) while the hand moved continuously.
- **Interpolation** added a fixed ~one-interval (~33 ms) visual delay on the preview.
- The control tick is **Drake-solve-bound** (~96 % CPU on one core, ~30 ms/solve on
  this box while continuously solving an active target), so the *actual* publish
  rate is ~30–48 Hz regardless of nominal rate/executor — the executor/rate is **not**
  the lever (the dwell-freeze from the oscillation fix is, for held targets).

## Phase 1 — shipped now (all local wins + the decoupling seam)

| # | Change | File | Effect |
|---|--------|------|--------|
| 1 | **Control/viz split.** New `/api/ctrl` route (tiny ack, no viz) applies controls; `/api/xr` is viz-only. Client POSTs `/api/ctrl` **every frame** (pipelined ≤3 in flight, monotonic `seq`, server drops stale/reordered); viz fetched on its own ~30 Hz gate. | `quest_node.py` (`on_ctrl_frame`/`_apply_controls`, `do_POST`, client loop) | Control rate **decoupled from RTT** → streams at the 72–90 Hz headset frame rate at *all* RTTs (was ~1/RTT). **The headline win** and the cloud-ready seam. |
| 2 | **BEST_EFFORT / KEEP_LAST depth-1 QoS** for `target_pose` / `cmd_vel` / `gripper` (quest pubs + controller subs). | `quest_node.py`, `controller_node.py` | Removes RELIABLE retransmit head-of-line blocking on a lossy/cloud link (a self-superseding stream doesn't want retransmit). |
| 3 | **Adaptive interpolation delay** — sized off measured *jitter* (`1.5·jitterEMA`, floor 4 ms) instead of a whole inter-arrival interval. | `quest_node.py` (`ingestViz`/`interpState`) | Cuts ~20–28 ms of visual preview latency on a steady LAN; grows to absorb jitter on a bad link. |
| 4 | **Markers off the hot control tick** → a separate 15 Hz timer (skips when no RViz subscriber). | `controller_node.py` | Removes 2 FK + an 8-marker serialize from the control-critical path each tick. |
| 5 | `control_rate` 60→120 nominal (harmless; tick is solve-bound). Kept single-threaded — the **MultiThreadedExecutor was reverted** (measured live it *throttled* the timer to ~25 Hz via rclpy per-callback/GIL overhead). | `controller_node.py` | No regression; callback groups + lock retained so a future MTE is a one-liner. |
| 6 | **`?perf` HUD** now shows **`ctrl Hz`** (the decoupled control rate) separately from `viz Hz`, so the win is visible on-device. | `quest_node.py` | Measurement. |

**Verify (offline, all green):** `_quest_position_test.py` 23/23, `_quest_perf_bench.py`
3/3 (1300 req/s, TCP_NODELAY), `on_ctrl_frame` split unit-checked (control applied,
no viz, stale-seq dropped), solver/tracking/accuracy/swerve/pathing gates unchanged,
`_ros_reach_check.py` 3/3. **Live-only (validate in headset):** the client
`/api/ctrl` loop, the interp delay, and the smooth model under `tc netem` — open
`/?perf` and watch `ctrl Hz` hold near the headset frame rate while you throttle the
link (`sudo tc qdisc add dev <if> root netem delay 150ms 20ms`).

> RLock note: `on_xr_frame`/`on_ctrl_frame` hold `_lock` and call the shared
> `_apply_place`/`_apply_controls`, so `_lock` is now a `threading.RLock`. The three
> offline harnesses (`_quest_position_test.py`, `_quest_perf_bench.py`) were updated
> to match; any new harness building the node via `__new__` must use `RLock`.

## Cloud roadmap (Phases 2–4)

Target transport: **a persistent bidirectional channel** with three decoupled
concerns — (i) an unreliable/latest-wins **control** stream (overwrite, never queue),
(ii) a separate server-push **viz** stream at ~30 Hz (client jitter-buffers +
interpolates — reuse `ingestViz`/`interpState` verbatim), (iii) client-side
**prediction** of the operator's own known state. This removes RTT from the control
*rate*; the irreducible RTT/2 one-way latency is physics. The ROS side is reused
unchanged (control-first ordering is already safe).

Transport ranking: **WSS (WebSocket) now** (full-duplex, HTTPS/NAT-friendly, reuses
the existing cert) **> WebTransport/QUIC or WebRTC datachannel later** (loss-tolerant,
no HOL, but needs UDP + TURN/relay + signaling) **≫ pipelined HTTP** (browser in-order
HOL) **> the old one-in-flight POST** (RTT-rate-coupled, unusable at cloud RTT).

- **Phase 2 (~3–5 d, no infra): threaded WSS `/ws/xr`** alongside the HTTPS server.
  Control sent every frame without awaiting; viz pushed at its own rate; latest-wins
  per connection. Keep `rclpy.spin` on the main thread, run WS in a daemon thread,
  cross into the node only via the existing `_lock`-guarded methods, reuse the exact
  `ssl.SSLContext`, add client reconnect/backoff + a per-conn idle timeout. Because
  Phase 1 already isolated the control channel, this is a **one-channel drop-in**, not
  a rewrite. This is the cloud-enabling change AGENTS.md flags.
- **Phase 3 (~2–3 d): client-side prediction.** Mirror `SwerveOdometry` in JS to
  predict the base pose from the operator's *own* `cmd_vel` (hides ~all base downlink
  latency for free), reconcile on viz arrival with a complementary filter.
- **Phase 4 (LATER, needs infra): WebTransport/QUIC or WebRTC datachannel** for
  packet-loss immunity + the cloud infra it requires (TURN/relay, UDP reachability,
  geo edge, auth/session, WebRTC signaling). Binary int16 + delta-encoded viz payload
  is bandwidth polish that lands here (largely redundant once control is off the viz
  path).

**Do NOT** optimize server compute or touch the Drake solver numerics — server work
is 0.3 ms and every high-value change is transport/scheduling.
