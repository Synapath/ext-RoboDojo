# Strict observation delivery and simulation-clock recording

Launch Kit with `env.camera_manager.capture.render_sync.add_zero_delay_kit_args`
before AppLauncher. A bare render does not prove that annotators have received
the new physics state. Strict capture requires both zero-delay settings and an
advancing SDG dispatcher reference; it fails closed on missing/stalled delivery.
Renders used by the fence do not advance physics. The legacy render-count fallback
remains for callers that have not opted into strict capture.

`ObsManager.require_fresh_frames=True` enables this contract. Each observation's
`additional_info.capture` contains a monotonically increasing capture sequence,
the common native physics counter/dt, before/after dispatcher reference, render
passes and readout wall epoch/monotonic clocks. The dispatcher reference is a
global delivery clock, **not a per-sensor exposure timestamp**. All configured
RGB cameras must be returned. The physics counter must stay fixed during capture.

For complete video, call `env.start_continuous_recording(metadata_path)` after
reset and before any legacy observation/video frame. This also enables strict
capture. The native physics step checks a simulation-clock schedule and captures
at `obs_manager.collect_freq`; at dt0.004/fps25 this is every10 physics steps.
There is an initial frame at time0. Policy/snapshot reads never append video.
Waiting for a model does not create frames or advance the simulation. Missed
sampling deadlines raise instead of inserting duplicate/interpolated frames.
Do not alter control decimation or advance physics just to record video.

The append-only metadata JSONL records every video index, actual physics time,
native action count, all cameras, robot state and capture receipt. Arm joints
and EEF are measured. The legacy native `*_ee_joint_state` is a normalized
gripper command; actual finger joints are recorded separately as
`measured_finger_joints` in native joint units (metres for ARX X5). Consumers
must not confuse those command/readout fields. Call
`stop_continuous_recording()` before reset, then `save_video` to finalize native
per-camera MP4s. The adjacent `.summary.json` records start/last/stop counters,
stride and sample count. A stop between scheduled frames is explicit; its final
unsampled tail is less than one video interval. Interrupted close retains partial
writer files instead of discarding them. This mode is opt-in and does not change
legacy policy observation/recording behavior or native scoring.

Render-product teardown always releases every annotator and the product. An
already-invalid shared SDG NodeObj is recorded as a cleanup warning; other SDK
errors remain errors. Ownership is cleared before release so repeated close or
destructor calls do not try detaching the same product twice.

Unit tests cover absent/stalled delivery and clock misses. They do not prove
render freshness or MP4 coverage on a GPU. Native acceptance must execute known
rotations/translations, compare first delivered images against stationary
confirmation frames and measured state, and independently decode videos/count
frames against the physics schedule. Keep that evidence outside the source tree.
