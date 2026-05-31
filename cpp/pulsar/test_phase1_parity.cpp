// Phase 1 parity gate: obs builder, action parser, done condition.
//
// Loads the reference numpy dumps from parity/ref/ (written by
// parity/dump_reference.py) and compares the C++ Pulsar env layer against them.
// Exits non-zero on any failure. The ref dir can be passed as argv[1]
// (defaults to "parity/ref").
//
// Gate criteria: max abs diff < 1e-4 for all obs vectors.
// Action parsing is deterministic (no tolerance needed).
// Done/reward is discrete (exact).
#include <cstdio>
#include <cstring>
#include <string>

#include "pulsar/parity.hpp"
#include "pulsar/pulsar_env.hpp"

namespace par = pulsar::parity;
using namespace pulsar;
using namespace pulsar;

namespace {

// Compare a C++ float vector against a named reference record.
int check_obs(const std::string& ref_dir, const std::string& name,
              const std::vector<float>& got, double tol = 1e-4) {
  const par::NpyArray ref = par::load_npy(ref_dir + "/" + name + ".npy");
  if (static_cast<std::int64_t>(got.size()) != ref.numel()) {
    std::printf("  FAIL %-38s size mismatch: got=%zu ref=%lld\n",
                name.c_str(), got.size(), static_cast<long long>(ref.numel()));
    return 1;
  }
  std::vector<double> got_d(got.begin(), got.end());
  const double diff = par::max_abs_diff(ref, got_d.data(), ref.numel());
  if (diff > tol) {
    std::printf("  FAIL %-38s max_abs_diff=%.2e (tol=%.2e)\n",
                name.c_str(), diff, tol);
    // Print first few mismatches for debugging
    for (std::int64_t i = 0; i < ref.numel(); ++i) {
      const double d = std::abs(ref.data[i] - got_d[i]);
      if (d > tol)
        std::printf("         [%lld] ref=%.6f got=%.6f diff=%.2e\n",
                    static_cast<long long>(i), ref.data[i], got_d[i], d);
    }
    return 1;
  }
  std::printf("  ok   %-38s max_abs_diff=%.2e\n", name.c_str(), diff);
  return 0;
}

// Check a single float scalar against a named record.
int check_scalar(const std::string& ref_dir, const std::string& name,
                 float got_val, double tol = 0.0) {
  const par::NpyArray ref = par::load_npy(ref_dir + "/" + name + ".npy");
  const double diff = std::abs(ref.data[0] - static_cast<double>(got_val));
  if (diff > tol) {
    std::printf("  FAIL %-38s got=%.6f ref=%.6f diff=%.2e\n",
                name.c_str(), got_val, ref.data[0], diff);
    return 1;
  }
  std::printf("  ok   %-38s got=%.6f\n", name.c_str(), got_val);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ref_dir = argc > 1 ? argv[1] : "parity/ref";
  std::printf("[pulsar_phase1_parity] ref dir: %s\n\n", ref_dir.c_str());

  int fails = 0;
  PulsarObsBuilder obs_builder;

  // ---- obs parity for both fixed states, both agents ----
  std::printf("--- obs parity ---\n");
  for (const auto& [name, snap] : std::initializer_list<std::pair<const char*, PulsarEnvSnapshot>>{
           {"midfield", midfield_snapshot()},
           {"dynamic",  dynamic_snapshot()},
       }) {
    const EnvState state = make_env_state_from_snapshot(snap);
    fails += check_obs(ref_dir, std::string("obs/") + name + "/blue",
                       obs_builder.build_obs(state, 0));
    fails += check_obs(ref_dir, std::string("obs/") + name + "/orange",
                       obs_builder.build_obs(state, 1));
  }

  // ---- action parity: C++ parsed action -> check throttle/steer/etc manually ----
  // (No reference for ControllerState fields; verify structural constraints instead.)
  std::printf("\n--- action parser structural checks ---\n");
  {
    // Blue: steer/yaw/roll should NOT be mirrored
    float act_b[8] = {1.0f, -0.5f, 0.2f, 0.7f, -0.3f, 1.0f, 1.0f, 0.0f};
    const ControllerState ctrl_b = PulsarContinuousActionParser::parse(act_b, 0);
    const bool blue_ok =
        ctrl_b.throttle == std::clamp(act_b[0], -1.0f, 1.0f) &&
        ctrl_b.steer    == std::clamp(act_b[1], -1.0f, 1.0f) &&  // no mirror
        ctrl_b.pitch    == std::clamp(act_b[2], -1.0f, 1.0f) &&
        ctrl_b.yaw      == std::clamp(act_b[3], -1.0f, 1.0f) &&  // no mirror
        ctrl_b.roll     == std::clamp(act_b[4], -1.0f, 1.0f) &&  // no mirror
        ctrl_b.jump      == true &&
        ctrl_b.boost     == true &&
        ctrl_b.handbrake == false;
    std::printf("  %s blue action (no mirror)\n", blue_ok ? "ok  " : "FAIL");
    if (!blue_ok) ++fails;

    // Orange: steer/yaw/roll should be mirrored (negated)
    float act_o[8] = {0.3f, 0.9f, -0.4f, -0.8f, 0.6f, 0.0f, 1.0f, 1.0f};
    const ControllerState ctrl_o = PulsarContinuousActionParser::parse(act_o, 1);
    const bool orange_ok =
        ctrl_o.throttle  == std::clamp( act_o[0], -1.0f, 1.0f) &&
        ctrl_o.steer     == std::clamp(-act_o[1], -1.0f, 1.0f) &&  // NEGATED
        ctrl_o.pitch     == std::clamp( act_o[2], -1.0f, 1.0f) &&
        ctrl_o.yaw       == std::clamp(-act_o[3], -1.0f, 1.0f) &&  // NEGATED
        ctrl_o.roll      == std::clamp(-act_o[4], -1.0f, 1.0f) &&  // NEGATED
        ctrl_o.jump      == false &&
        ctrl_o.boost     == true &&
        ctrl_o.handbrake == true;
    std::printf("  %s orange action (steer/yaw/roll mirrored)\n",
                orange_ok ? "ok  " : "FAIL");
    if (!orange_ok) ++fails;
  }

  // ---- done condition ----
  std::printf("\n--- done condition ---\n");
  {
    PulsarDoneCondition done(1600);
    // goal scored
    {
      EnvState s = make_env_state_from_snapshot(midfield_snapshot());
      s.goal_scored = true;
      auto [term, trunc] = done.is_done(s, 0);
      const bool ok = term[0] == 1u && trunc[0] == 0u;
      std::printf("  %s goal_scored -> terminated\n", ok ? "ok  " : "FAIL");
      if (!ok) ++fails;
    }
    // OOB x
    {
      EnvState s = make_env_state_from_snapshot(midfield_snapshot());
      s.ball.position.x = 4097.0f;
      auto [term, trunc] = done.is_done(s, 0);
      const bool ok = term[0] == 1u;
      std::printf("  %s |ball.x|>4096 -> terminated\n", ok ? "ok  " : "FAIL");
      if (!ok) ++fails;
    }
    // OOB y
    {
      EnvState s = make_env_state_from_snapshot(midfield_snapshot());
      s.ball.position.y = -5121.0f;
      auto [term, trunc] = done.is_done(s, 0);
      const bool ok = term[0] == 1u;
      std::printf("  %s |ball.y|>5120 -> terminated\n", ok ? "ok  " : "FAIL");
      if (!ok) ++fails;
    }
    // timeout
    {
      EnvState s = make_env_state_from_snapshot(midfield_snapshot());
      auto [term, trunc] = done.is_done(s, 1600);
      const bool ok = term[0] == 0u && trunc[0] == 1u;
      std::printf("  %s ticks>=1600 -> truncated\n", ok ? "ok  " : "FAIL");
      if (!ok) ++fails;
    }
    // no done within bounds
    {
      EnvState s = make_env_state_from_snapshot(midfield_snapshot());
      auto [term, trunc] = done.is_done(s, 800);
      const bool ok = term[0] == 0u && trunc[0] == 0u;
      std::printf("  %s mid-episode -> no done\n", ok ? "ok  " : "FAIL");
      if (!ok) ++fails;
    }
  }

  std::printf("\n[pulsar_phase1_parity] %s (%d failure(s))\n",
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
