// Pulsar — Phase 1: faithful env layer.
//
// Exactly mirrors the sprint5 Python prototype's env behaviour:
//   - 47-dim symmetric team-relative observation  (get_symmetric_obs, sprint5:69-148)
//   - Continuous 8-dim actions with Orange steer/yaw/roll mirroring (sprint5:176-202)
//   - Done: goal-scored OR OOB OR max-steps                          (sprint5:206-227)
//   - Full-state injection on reset                                  (sprint5:41-67)
//
// No Torch dependency. Pure C++20, self-contained.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsar/core/types.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

// ============================================================================
// Helpers
// ============================================================================

namespace detail {

// Computes RocketSim's "right" vector from the two stored axes.
// In RocketSim's right-handed convention: right = up × forward.
// Verified: forward=(0,1,0), up=(0,0,1) → right=(-1,0,0)  (matches sprint5 rot dicts).
inline Vec3 right_from_fwd_up(Vec3 fwd, Vec3 up) {
  return {up.y * fwd.z - up.z * fwd.y,
          up.z * fwd.x - up.x * fwd.z,
          up.x * fwd.y - up.y * fwd.x};
}

// Write 19 floats for one car into `out`. Advances `out` by 19.
// `invert` = true for Orange (negates X and Y components, sprint5:99-122).
inline void write_car_obs(float*& out, const CarState& car, bool invert) {
  const float sx = invert ? -1.0f : 1.0f;  // sign for x/y dims
  const Vec3 right = right_from_fwd_up(car.forward, car.up);

  // position
  *out++ = sx * car.position.x / 4000.0f;
  *out++ = sx * car.position.y / 5120.0f;
  *out++ =      car.position.z / 2000.0f;
  // velocity
  *out++ = sx * car.velocity.x / 2300.0f;
  *out++ = sx * car.velocity.y / 2300.0f;
  *out++ =      car.velocity.z / 2300.0f;
  // forward
  *out++ = sx * car.forward.x;
  *out++ = sx * car.forward.y;
  *out++ =      car.forward.z;
  // right
  *out++ = sx * right.x;
  *out++ = sx * right.y;
  *out++ =      right.z;
  // up
  *out++ = sx * car.up.x;
  *out++ = sx * car.up.y;
  *out++ =      car.up.z;
  // angular velocity
  *out++ = sx * car.angular_velocity.x / 5.5f;
  *out++ = sx * car.angular_velocity.y / 5.5f;
  *out++ =      car.angular_velocity.z / 5.5f;
  // boost
  *out++ = car.boost / 100.0f;
}

// Write 9 floats for the ball into `out`. Advances `out` by 9.
// `invert` = true for Orange.
inline void write_ball_obs(float*& out, const BallState& ball, bool invert) {
  const float sx = invert ? -1.0f : 1.0f;

  // position
  *out++ = sx * ball.position.x / 4000.0f;
  *out++ = sx * ball.position.y / 5120.0f;
  *out++ =      ball.position.z / 2000.0f;
  // velocity (note: ball uses /3000, not /2300)
  *out++ = sx * ball.velocity.x / 3000.0f;
  *out++ = sx * ball.velocity.y / 3000.0f;
  *out++ =      ball.velocity.z / 3000.0f;
  // angular velocity (ball uses /6.0, not /5.5)
  *out++ = sx * ball.angular_velocity.x / 6.0f;
  *out++ = sx * ball.angular_velocity.y / 6.0f;
  *out++ =      ball.angular_velocity.z / 6.0f;
}

}  // namespace detail

// ============================================================================
// PulsarObsBuilder — 47-dim symmetric team-relative obs
// ============================================================================
// Layout (per agent): ego_car(19) | ball(9) | opponent_car(19) = 47
//
// For agent 0 (Blue):  ego=cars[0], opp=cars[1], invert=false
// For agent 1 (Orange): ego=cars[1], opp=cars[0], invert=true
//
// The ball obs starting at index 19 encodes (pos_x,y,z, vel_x,y,z) which is
// used as the 6-dim hindsight goal (sprint5:569, 656).

class PulsarObsBuilder final : public ObsBuilder {
 public:
  static constexpr std::size_t kObsDim = 47;

  std::size_t obs_dim() const override { return kObsDim; }

  std::vector<float> build_obs(const EnvState& state, AgentId agent_id) const override {
    std::vector<float> out(kObsDim);
    write_agent_obs(state, agent_id, out.data());
    return out;
  }

  void build_obs_batch(const EnvState& state, std::span<float> out) const override {
    assert(state.cars.size() == 2 && "PulsarObsBuilder expects exactly 2 cars (1v1)");
    assert(out.size() == state.cars.size() * kObsDim);
    for (std::size_t i = 0; i < state.cars.size(); ++i) {
      write_agent_obs(state, i, out.data() + i * kObsDim);
    }
  }

 private:
  static void write_agent_obs(const EnvState& state, AgentId agent_id, float* out) {
    assert(state.cars.size() == 2 && "PulsarObsBuilder expects exactly 2 cars (1v1)");
    const bool invert = (agent_id == 1);  // Orange flips X/Y
    const CarState& ego  = state.cars[agent_id];
    const CarState& opp  = state.cars[1 - agent_id];

    detail::write_car_obs(out, ego,  invert);  // [0..18]
    detail::write_ball_obs(out, state.ball, invert);  // [19..27]
    detail::write_car_obs(out, opp,  invert);  // [28..46]
  }
};

// ============================================================================
// PulsarContinuousActionParser — 8-dim continuous actions, Orange mirrored
// ============================================================================
// Input: float[num_agents * 8], output: ControllerState per agent.
//
// Blue  (agent 0): throttle,steer,pitch,yaw,roll = clip(values); jump/boost/handbrake = >0.5
// Orange(agent 1): steer,yaw,roll are negated before clip (sprint5:195-201).

class PulsarContinuousActionParser {
 public:
  static constexpr int kActionDim = 8;

  // Parse a single agent's 8-dim float action into a ControllerState.
  static ControllerState parse(const float* values, int agent_id) {
    const float mirror = (agent_id == 1) ? -1.0f : 1.0f;

    ControllerState ctrl;
    ctrl.throttle  = std::clamp(values[0], -1.0f, 1.0f);
    ctrl.steer     = std::clamp(mirror * values[1], -1.0f, 1.0f);
    ctrl.pitch     = std::clamp(values[2], -1.0f, 1.0f);
    ctrl.yaw       = std::clamp(mirror * values[3], -1.0f, 1.0f);
    ctrl.roll      = std::clamp(mirror * values[4], -1.0f, 1.0f);
    ctrl.jump      = values[5] > 0.5f;
    ctrl.boost     = values[6] > 0.5f;
    ctrl.handbrake = values[7] > 0.5f;
    return ctrl;
  }

  // Parse a flat float buffer [num_agents * 8] into ControllerStates.
  static void parse_batch(std::span<const float> actions, std::span<ControllerState> out) {
    assert(actions.size() == out.size() * kActionDim);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = parse(actions.data() + i * kActionDim, static_cast<int>(i));
    }
  }
};

// ============================================================================
// PulsarDoneCondition — goal, OOB, or max-steps
// ============================================================================
// terminated: goal_scored OR |ball.x|>4096 OR |ball.y|>5120
// truncated:  episode_ticks >= max_steps * tick_skip
//             (default: 200 steps * 8 ticks = 1600 ticks)

class PulsarDoneCondition final : public DoneCondition {
 public:
  static constexpr float kMaxBallX = 4096.0f;
  static constexpr float kMaxBallY = 5120.0f;

  explicit PulsarDoneCondition(int max_episode_ticks = 1600 /*200 steps * tick_skip 8*/)
      : max_episode_ticks_(max_episode_ticks) {}

  // Returns {terminated, truncated} for all agents (both share the same flag in 1v1).
  std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> is_done(
      const EnvState& state, int episode_ticks) const override {
    const std::size_t n = state.cars.size();
    std::uint8_t term = check_terminated(state);
    std::uint8_t trunc = (!term) && (episode_ticks >= max_episode_ticks_) ? 1u : 0u;
    return {std::vector<std::uint8_t>(n, term), std::vector<std::uint8_t>(n, trunc)};
  }

  void is_done_into(const EnvState& state, int episode_ticks,
                    std::span<std::uint8_t> terminated,
                    std::span<std::uint8_t> truncated) const override {
    const std::uint8_t term  = check_terminated(state);
    const std::uint8_t trunc = (!term) && (episode_ticks >= max_episode_ticks_) ? 1u : 0u;
    std::fill(terminated.begin(), terminated.end(), term);
    std::fill(truncated.begin(), truncated.end(), trunc);
  }

  bool is_episode_over(const EnvState& state, int episode_ticks) const {
    return check_terminated(state) || episode_ticks >= max_episode_ticks_;
  }

 private:
  int max_episode_ticks_;

  static std::uint8_t check_terminated(const EnvState& state) {
    if (state.goal_scored) return 1u;
    const float bx = std::abs(state.ball.position.x);
    const float by = std::abs(state.ball.position.y);
    if (bx > kMaxBallX || by > kMaxBallY) return 1u;
    return 0u;
  }
};

// ============================================================================
// PulsarEnvSnapshot — the full state dict from sprint5 (sprint5:378-394)
// ============================================================================
struct PulsarEnvSnapshot {
  // Blue / scorer
  Vec3 blue_pos{};
  Vec3 blue_vel{};
  Vec3 blue_fwd{0.0f, 1.0f, 0.0f};
  Vec3 blue_right{-1.0f, 0.0f, 0.0f};
  Vec3 blue_up{0.0f, 0.0f, 1.0f};
  Vec3 blue_ang_vel{};
  float blue_boost = 33.0f;

  // Orange / defender
  Vec3 orange_pos{};
  Vec3 orange_vel{};
  Vec3 orange_fwd{0.0f, -1.0f, 0.0f};
  Vec3 orange_right{1.0f, 0.0f, 0.0f};
  Vec3 orange_up{0.0f, 0.0f, 1.0f};
  Vec3 orange_ang_vel{};
  float orange_boost = 33.0f;

  // Ball
  Vec3 ball_pos{0.0f, 0.0f, 93.0f};
  Vec3 ball_vel{};
  Vec3 ball_ang_vel{};
};

// Apply a snapshot to an EnvState (for use by the StateMutator and parity tests).
// Only `forward` and `up` are set in CarState (no `right` field); right is
// reconstructed from them inside PulsarObsBuilder via up × forward.
inline void apply_snapshot(const PulsarEnvSnapshot& snap, EnvState& state) {
  assert(state.cars.size() >= 2);
  CarState& blue   = state.cars[0];
  CarState& orange = state.cars[1];

  blue.position         = snap.blue_pos;
  blue.velocity         = snap.blue_vel;
  blue.forward          = snap.blue_fwd;
  blue.up               = snap.blue_up;
  blue.angular_velocity = snap.blue_ang_vel;
  blue.boost            = snap.blue_boost;

  orange.position         = snap.orange_pos;
  orange.velocity         = snap.orange_vel;
  orange.forward          = snap.orange_fwd;
  orange.up               = snap.orange_up;
  orange.angular_velocity = snap.orange_ang_vel;
  orange.boost            = snap.orange_boost;

  state.ball.position         = snap.ball_pos;
  state.ball.velocity         = snap.ball_vel;
  state.ball.angular_velocity = snap.ball_ang_vel;

  state.goal_scored = false;
}

// ============================================================================
// PulsarStateMutator — inject a fixed PulsarEnvSnapshot on reset
// ============================================================================
// The reachability-grid mutator (Phase 5) wraps this with sampled snapshots.

class PulsarStateMutator final : public StateMutator {
 public:
  // Mutate to a fixed snapshot every reset (useful for parity tests and per-env curriculum).
  explicit PulsarStateMutator(PulsarEnvSnapshot snap) : snap_(std::move(snap)) {}

  // Mutate using a callback that produces a snapshot.
  explicit PulsarStateMutator(std::function<PulsarEnvSnapshot(std::uint64_t)> sample_fn)
      : sample_fn_(std::move(sample_fn)) {}

  // Update the fixed snapshot (used by the collector to set next episode's state).
  void set_snapshot(PulsarEnvSnapshot snap) {
    sample_fn_ = {};
    snap_ = std::move(snap);
  }

  void apply(EnvState& state, std::uint64_t seed) const override {
    const PulsarEnvSnapshot snap = sample_fn_ ? sample_fn_(seed) : snap_;
    apply_snapshot(snap, state);
  }

 private:
  PulsarEnvSnapshot snap_{};
  std::function<PulsarEnvSnapshot(std::uint64_t)> sample_fn_{};
};

// ============================================================================
// Convenience: build EnvState directly from Python-dict-style values.
// Used by parity tests to avoid going through RocketSim.
// ============================================================================
inline EnvState make_env_state_from_snapshot(const PulsarEnvSnapshot& snap) {
  EnvState state;
  state.cars.resize(2);
  state.cars[0].team = Team::Blue;
  state.cars[1].team = Team::Orange;
  apply_snapshot(snap, state);
  return state;
}

// Build the two fixed parity snapshots (matching parity/dump_reference.py).
inline PulsarEnvSnapshot midfield_snapshot() {
  PulsarEnvSnapshot s;
  s.blue_pos    = {100.0f, -1200.0f, 17.0f};
  s.blue_vel    = {0.0f,    400.0f,   0.0f};
  s.blue_fwd    = {0.0f,    1.0f,     0.0f};
  // right = up × forward = (0,0,1)×(0,1,0) = (-1,0,0)
  s.blue_up     = {0.0f,    0.0f,     1.0f};
  s.blue_ang_vel = {0.0f, 0.0f, 0.0f};
  s.blue_boost  = 33.0f;

  s.orange_pos  = {300.0f,  4600.0f, 17.0f};
  s.orange_vel  = {0.0f,    0.0f,    0.0f};
  s.orange_fwd  = {0.0f,   -1.0f,    0.0f};
  // right = up × forward = (0,0,1)×(0,-1,0) = (1,0,0)
  s.orange_up   = {0.0f,    0.0f,    1.0f};
  s.orange_ang_vel = {0.0f, 0.0f, 0.0f};
  s.orange_boost = 33.0f;

  s.ball_pos    = {150.0f, -400.0f,  93.0f};
  s.ball_vel    = {0.0f,    0.0f,    0.0f};
  s.ball_ang_vel = {0.0f,   0.0f,    0.0f};
  return s;
}

inline PulsarEnvSnapshot dynamic_snapshot() {
  PulsarEnvSnapshot s;
  s.blue_pos    = {-1500.0f,  800.0f,  50.0f};
  s.blue_vel    = { -300.0f, 1200.0f,  80.0f};
  s.blue_fwd    = {    0.6f,    0.8f,   0.0f};
  // right = up × forward = (0,0,1)×(0.6,0.8,0) = (-0.8,0.6,0)
  s.blue_up     = {    0.0f,    0.0f,   1.0f};
  s.blue_ang_vel = {1.0f, -2.0f, 0.5f};
  s.blue_boost  = 87.0f;

  s.orange_pos  = { 1200.0f,  3000.0f, 120.0f};
  s.orange_vel  = {  500.0f,  -700.0f, -30.0f};
  s.orange_fwd  = {    0.3f,   -0.95f,   0.0f};
  // right = up × forward = (0,0,1)×(0.3,-0.95,0) = (0.95,0.3,0)
  s.orange_up   = {    0.0f,     0.0f,   1.0f};
  s.orange_ang_vel = {-0.7f, 1.5f, -1.2f};
  s.orange_boost = 12.0f;

  s.ball_pos    = { -600.0f,  1500.0f, 300.0f};
  s.ball_vel    = { 1000.0f,  -200.0f, 400.0f};
  s.ball_ang_vel = {2.0f, -3.0f, 1.0f};
  return s;
}

}  // namespace pulsar
