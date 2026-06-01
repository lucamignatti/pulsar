// Phase 4 parity gate: update loop components.
//
// Loads the reference tensors from parity/ref/update/ and verifies:
//   - GAE advantages and returns
//   - Quasimetric critic advantages
//   - Gaussian log_prob and entropy
//   - PPO minibatch loss components (pg, vl, cl, total)
//
// All checks use the reference weights (seed=0 init, same as Phase 2).
// Tolerances follow the parity manifest (1e-5 for all update records).
#include <cstdio>
#include <string>
#include <torch/torch.h>

#include "pulsar/parity.hpp"
#include "pulsar/pulsar_models.hpp"
#include "pulsar/pulsar_update.hpp"

namespace par = pulsar::parity;
using namespace pulsar;

// ---- helpers ---------------------------------------------------------------
static torch::Tensor load_tensor(const std::string& ref_dir, const std::string& name) {
  const par::NpyArray a = par::load_npy(ref_dir + "/" + name + ".npy");
  std::vector<float> buf(a.data.begin(), a.data.end());
  std::vector<int64_t> shape(a.shape.begin(), a.shape.end());
  if (shape.empty()) shape = {1};
  return torch::from_blob(buf.data(), shape, torch::kFloat32).clone();
}

static void load_weights(torch::nn::Module& mod, const std::string& prefix,
                          const std::string& ref_dir) {
  for (const auto& item : mod.named_parameters()) {
    const std::string path = ref_dir + "/weights/" + prefix + "/" + item.key() + ".npy";
    const par::NpyArray a = par::load_npy(path);
    std::vector<float> buf(a.data.begin(), a.data.end());
    std::vector<int64_t> shape(a.shape.begin(), a.shape.end());
    if (shape.empty()) shape = {1};
    torch::Tensor loaded = torch::from_blob(buf.data(), shape, torch::kFloat32).clone();
    torch::NoGradGuard ng;
    item.value().set_data(loaded);
  }
}

static int check(const std::string& ref_dir, const std::string& name,
                  const torch::Tensor& got, double tol = 1e-5) {
  const par::NpyArray ref = par::load_npy(ref_dir + "/" + name + ".npy");
  std::vector<float> buf(ref.data.begin(), ref.data.end());
  std::vector<int64_t> shape(ref.shape.begin(), ref.shape.end());
  if (shape.empty()) shape = {1};
  const auto ref_t = torch::from_blob(buf.data(), shape, torch::kFloat32).clone();
  const auto got_c = got.detach().cpu().contiguous();
  const double diff = (got_c - ref_t).abs().max().item<double>();
  if (diff > tol) {
    std::printf("  FAIL %-38s max_abs_diff=%.2e (tol=%.2e)\n", name.c_str(), diff, tol);
    // Print first few mismatches
    const auto diff_t = (got_c.reshape(-1) - ref_t.reshape(-1)).abs();
    for (int i = 0; i < std::min(static_cast<int64_t>(3), diff_t.numel()); ++i) {
      if (diff_t[i].item<float>() > tol)
        std::printf("    [%d] ref=%.6f got=%.6f\n", i,
                    ref_t.reshape(-1)[i].item<float>(),
                    got_c.reshape(-1)[i].item<float>());
    }
    return 1;
  }
  std::printf("  ok   %-38s max_abs_diff=%.2e\n", name.c_str(), diff);
  return 0;
}

int main(int argc, char** argv) {
  const std::string ref_dir = argc > 1 ? argv[1] : "parity/ref";
  std::printf("[pulsar_phase4_parity] ref dir: %s\n\n", ref_dir.c_str());

  const torch::Device dev = torch::kCPU;  // parity always on CPU
  torch::manual_seed(0);

  int fails = 0;

  // Load reference rollout
  const auto obs_buf   = load_tensor(ref_dir, "update/obs_buf");    // [8,4,47]
  const auto act_buf   = load_tensor(ref_dir, "update/act_buf");    // [8,4,8]
  const auto logp_buf  = load_tensor(ref_dir, "update/logp_buf");   // [8,4]
  const auto rew_buf   = load_tensor(ref_dir, "update/rew_buf");    // [8,4]
  const auto done_buf  = load_tensor(ref_dir, "update/dones_buf");  // [8,4]
  const auto val_buf   = load_tensor(ref_dir, "update/val_buf");    // [8,4]

  const int T = static_cast<int>(obs_buf.size(0));
  const int N = static_cast<int>(obs_buf.size(1));

  // Build and load reference models
  PulsarActor actor;   actor->to(dev);  load_weights(*actor,  "actor",  ref_dir);  actor->eval();
  PulsarValue value;   value->to(dev);  load_weights(*value,  "value",  ref_dir);  value->eval();
  PulsarQuasimetricCritic critic; critic->to(dev); load_weights(*critic, "critic", ref_dir); critic->eval();

  // ---- GAE -------------------------------------------------------------------
  std::printf("--- GAE ---\n");
  {
    const auto agent_goals = PulsarUpdateLoop::build_agent_goals(N, dev);
    torch::NoGradGuard ng;
    const auto nv   = value->forward(obs_buf[-1], agent_goals).flatten(); // last obs as next
    const auto nd   = done_buf[-1];
    const auto adv  = compute_gae_pulsar(rew_buf, val_buf, done_buf, nv, nd);
    const auto ret  = adv + val_buf;

    fails += check(ref_dir, "update/gae_advantages", adv);
    fails += check(ref_dir, "update/returns", ret);
  }

  // ---- Quasimetric advantage -------------------------------------------------
  std::printf("\n--- Quasimetric advantage ---\n");
  {
    const auto b_obs  = obs_buf.reshape({T * N, kObsDim});
    const auto b_act  = act_buf.reshape({T * N, kActionDim});
    const auto fg     = load_tensor(ref_dir, "update/future_goals");
    const auto b_fg   = fg.reshape({T * N, kGoalDim});

    torch::NoGradGuard ng;
    const auto q    = critic->forward(b_obs, b_act, b_fg);
    const auto adv  = compute_quasimetric_advantage(q);

    fails += check(ref_dir, "update/q_values",   q);
    fails += check(ref_dir, "update/q_adv_norm", adv);
  }

  // ---- Gaussian log_prob + entropy -------------------------------------------
  std::printf("\n--- Gaussian log_prob / entropy ---\n");
  {
    // Use the first minibatch (indices 0..7) like the reference
    const auto b_obs = obs_buf.reshape({T * N, kObsDim});
    const auto b_act = act_buf.reshape({T * N, kActionDim});
    const auto mb    = torch::arange(8);
    const auto gs = PulsarUpdateLoop::goal_score_tensor(dev);
    const auto gd = PulsarUpdateLoop::goal_defend_tensor(dev);
    auto mb_goals = torch::zeros({8, kGoalDim});
    for (int i = 0; i < 8; ++i) {
      mb_goals[i] = (i % 2 == 0) ? gs : gd;
    }

    torch::NoGradGuard ng;
    auto [am, astd, _] = actor->forward(b_obs.index_select(0, mb), mb_goals);
    const auto nlp = gaussian_log_prob(b_act.index_select(0, mb), am, astd);
    const auto ent = gaussian_entropy(astd);
    const auto b_logp_old = logp_buf.reshape({T * N});
    const auto ratio = torch::exp(nlp - b_logp_old.index_select(0, mb));

    fails += check(ref_dir, "update/mb_logprobs", nlp);
    fails += check(ref_dir, "update/mb_entropy",  ent);
    fails += check(ref_dir, "update/mb_ratio",    ratio);
  }

  // ---- Full minibatch loss components ----------------------------------------
  std::printf("\n--- Minibatch loss components ---\n");
  {
    const auto b_obs  = obs_buf.reshape({T * N, kObsDim});
    const auto b_act  = act_buf.reshape({T * N, kActionDim});
    const auto b_logp_flat = logp_buf.reshape({T * N});
    const auto b_ret  = [&] {
      const auto fg = load_tensor(ref_dir, "update/future_goals");
      const auto agent_goals = PulsarUpdateLoop::build_agent_goals(N, dev);
      torch::NoGradGuard ng;
      const auto nv = value->forward(obs_buf[-1], agent_goals).flatten();
      const auto nd = done_buf[-1];
      const auto adv = compute_gae_pulsar(rew_buf, val_buf, done_buf, nv, nd);
      return (adv + val_buf).reshape({T * N});
    }();
    const auto fg     = load_tensor(ref_dir, "update/future_goals");
    const auto b_fg   = fg.reshape({T * N, kGoalDim});

    // q-advantage
    torch::Tensor b_adv;
    {
      torch::NoGradGuard ng;
      const auto q = critic->forward(b_obs, b_act, b_fg);
      b_adv = compute_quasimetric_advantage(q);
    }

    const auto mb = torch::arange(8);
    const auto gs = PulsarUpdateLoop::goal_score_tensor(dev);
    const auto gd = PulsarUpdateLoop::goal_defend_tensor(dev);
    auto mb_goals = torch::zeros({8, kGoalDim});
    for (int i = 0; i < 8; ++i) mb_goals[i] = (i % 2 == 0) ? gs : gd;

    torch::NoGradGuard ng;
    auto [am, astd, _feat] = actor->forward(b_obs.index_select(0, mb), mb_goals);
    const auto nlp   = gaussian_log_prob(b_act.index_select(0, mb), am, astd);
    const auto ent   = gaussian_entropy(astd);
    const auto ratio = torch::exp(nlp - b_logp_flat.index_select(0, mb));

    // Advantage typed: sign-flip for Orange (odd)
    auto mb_adv_typed = b_adv.index_select(0, mb).clone();
    const auto mb_parity = mb % 2;
    mb_adv_typed.index_put_({mb_parity == 1}, -mb_adv_typed.index({mb_parity == 1}));
    const auto ma = (mb_adv_typed - mb_adv_typed.mean()) / (mb_adv_typed.std() + 1e-8f);

    const auto pg_ind = torch::max(-ma * ratio, -ma * torch::clamp(ratio, 0.8f, 1.2f));
    const auto nv_mb  = value->forward(b_obs.index_select(0, mb), mb_goals).flatten();
    const auto vl_ind = 0.5f * (nv_mb - b_ret.index_select(0, mb)).square();
    const auto cl     = compute_pairwise_infonce_loss(critic, b_obs.index_select(0, mb),
                                                       b_act.index_select(0, mb),
                                                       b_fg.index_select(0, mb));

    const auto mb_w = torch::where(mb_parity == 0, torch::ones({8}), torch::full({8}, 0.85f));
    const auto pg   = (pg_ind * mb_w).mean();
    const auto vl   = (vl_ind * mb_w).mean();
    const auto en   = ent.mean();
    const auto tot  = pg - 0.01f * en + 0.5f * vl + cl;

    fails += check(ref_dir, "update/mb_adv_typed",     mb_adv_typed);
    fails += check(ref_dir, "update/mb_adv_normed",    ma);
    fails += check(ref_dir, "update/mb_pg_individual", pg_ind);
    fails += check(ref_dir, "update/mb_vl_individual", vl_ind);
    fails += check(ref_dir, "update/mb_pg",            pg.unsqueeze(0));
    fails += check(ref_dir, "update/mb_vl",            vl.unsqueeze(0));
    fails += check(ref_dir, "update/mb_cl",            cl.unsqueeze(0));
    fails += check(ref_dir, "update/mb_total_loss",    tot.unsqueeze(0));
  }

  std::printf("\n[pulsar_phase4_parity] %s (%d failure(s))\n",
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
