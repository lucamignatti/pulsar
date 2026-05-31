// Pulsar — Phase 4: update loop implementation.
#include "pulsar/pulsar_update.hpp"

#ifdef PULSAR_HAS_TORCH
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>

namespace pulsar {

// ============================================================================
// GAE — exactly matches sprint5:660-678
// ============================================================================
torch::Tensor compute_gae_pulsar(
    const torch::Tensor& rewards,    // [T, N]
    const torch::Tensor& values,     // [T, N]
    const torch::Tensor& dones,      // [T, N]
    const torch::Tensor& next_values, // [N]
    const torch::Tensor& next_done,   // [N]
    float gamma, float gae_lambda) {

  const int T = static_cast<int>(rewards.size(0));
  auto adv  = torch::zeros_like(rewards);
  auto lastgae = torch::zeros_like(next_values);

  // Reverse temporal accumulation.  sprint5:669-677
  for (int t = T - 1; t >= 0; --t) {
    torch::Tensor nnt, nval;
    if (t == T - 1) {
      nnt  = 1.0f - next_done;           // sprint5:671
      nval = next_values;                // sprint5:672
    } else {
      nnt  = 1.0f - dones[t + 1];       // sprint5:674
      nval = values[t + 1];             // sprint5:675
    }
    const auto delta = rewards[t] + gamma * nval * nnt - values[t]; // sprint5:676
    lastgae = delta + gamma * gae_lambda * nnt * lastgae;           // sprint5:677
    adv[t]  = lastgae;
  }
  return adv;
}

// ============================================================================
// Quasimetric advantage — sprint5:686-691
// ============================================================================
torch::Tensor compute_quasimetric_advantage(const torch::Tensor& q) {
  const auto v = q.mean();
  auto adv = (q - v);
  return (adv - adv.mean()) / (adv.std() + 1e-8f);
}

// ============================================================================
// Gaussian log-prob and entropy (diagonal, independent dimensions)
// ============================================================================

// log p(x) = -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
// summed over action dims.  sprint5: pr.log_prob(act).sum(-1)
torch::Tensor gaussian_log_prob(const torch::Tensor& actions,
                                const torch::Tensor& mean,
                                const torch::Tensor& std) {
  static constexpr float kHalfLog2Pi = 0.9189385332f;  // 0.5 * log(2π)
  const auto z   = (actions - mean) / std;
  const auto lp  = -0.5f * z.square() - std.log() - kHalfLog2Pi;
  return lp.sum(-1);  // [B]
}

// H[N(μ, σ)] = 0.5 + 0.5*log(2π) + log(σ) per dim, summed.
// = (0.5*(1 + log(2π)) + log(σ)).sum(-1)
// sprint5: pr.entropy().sum(-1)
torch::Tensor gaussian_entropy(const torch::Tensor& std) {
  static constexpr float kConst = 1.4189385332f;  // 0.5 * (1 + log(2π))
  return (kConst + std.log()).sum(-1);  // [B]
}

// ============================================================================
// Hindsight goal sampler — sprint5:649-658
// ============================================================================
// goal = obs_buf[t + geom_offset, e, 19:25] within the episode.
// Uses CPU torch RNG; for reproducibility the caller should seed torch.
torch::Tensor sample_hindsight_goals(const torch::Tensor& obs_buf,
                                     const torch::Tensor& dones_buf,
                                     float geom_p) {
  const int T = static_cast<int>(obs_buf.size(0));
  const int N = static_cast<int>(obs_buf.size(1));

  auto future_goals = torch::zeros({T, N, kGoalDim});
  // Work on CPU regardless of obs device
  const auto obs_cpu  = obs_buf.cpu().contiguous();
  const auto done_cpu = dones_buf.cpu().contiguous();

  const float log1mp = std::log(1.0f - geom_p);

  for (int e = 0; e < N; ++e) {
    int ep_end = T - 1;
    for (int t = T - 1; t >= 0; --t) {
      const int max_off = ep_end - t;
      int off = 0;
      if (max_off > 0) {
        // Geometric sample via inverse CDF: k = ceil(log(U) / log(1-p)) - 1
        const float u = torch::rand({1}).item<float>();
        const int k = static_cast<int>(std::ceil(std::log(std::max(u, 1e-7f)) / log1mp)) - 1;
        off = std::min(k, max_off);
      }
      // Goal = ball obs slice from obs at t+off: indices [19..24] → pos_x,y,z + vel_x,y,z
      future_goals[t][e] = obs_cpu[t + off][e].slice(0, 19, 25);
      // On episode boundary, reset ep_end to track within-episode horizon
      if (done_cpu[t][e].item<float>() == 1.0f && t > 0) {
        ep_end = t - 1;
      }
    }
  }
  return future_goals;
}

// ============================================================================
// PulsarUpdateLoop
// ============================================================================

torch::Tensor PulsarUpdateLoop::goal_score_tensor(torch::Device dev) {
  // sprint5:569: [0, 1, 321/2000, 0, 0, 0]
  return torch::tensor({0.0f, 1.0f, 321.0f / 2000.0f, 0.0f, 0.0f, 0.0f}).to(dev);
}

torch::Tensor PulsarUpdateLoop::goal_defend_tensor(torch::Device dev) {
  // sprint5:570: [0, -1, 321/2000, 0, 0, 0]
  return torch::tensor({0.0f, -1.0f, 321.0f / 2000.0f, 0.0f, 0.0f, 0.0f}).to(dev);
}

torch::Tensor PulsarUpdateLoop::build_agent_goals(int num_agents, torch::Device dev) {
  auto gs = goal_score_tensor(dev);
  auto gd = goal_defend_tensor(dev);
  auto goals = torch::zeros({num_agents, kGoalDim}, dev);
  goals.index_put_({torch::arange(0, num_agents, 2, dev)}, gs.unsqueeze(0));
  goals.index_put_({torch::arange(1, num_agents, 2, dev)}, gd.unsqueeze(0));
  return goals;
}

PulsarUpdateLoop::PulsarUpdateLoop(PulsarActor actor, PulsarValue value,
                               PulsarQuasimetricCritic critic, float lr)
    : actor_(std::move(actor)),
      value_(std::move(value)),
      critic_(std::move(critic)),
      actor_opt_(actor_->parameters(),  torch::optim::AdamOptions(lr).eps(1e-5)),
      value_opt_(value_->parameters(),  torch::optim::AdamOptions(lr).eps(1e-5)),
      critic_opt_(critic_->parameters(), torch::optim::AdamOptions(lr).eps(1e-5)) {}

void PulsarUpdateLoop::zero_grad() {
  actor_opt_.zero_grad();
  value_opt_.zero_grad();
  critic_opt_.zero_grad();
}

PulsarUpdateResult PulsarUpdateLoop::update(
    PulsarRolloutStorage& rollout,
    const torch::Tensor& next_obs,
    const torch::Tensor& next_done,
    int num_epochs, int minibatch_size,
    torch::Device device) {

  const int T       = rollout.steps();
  const int N       = rollout.num_agents();
  const int B       = T * N;

  // Move rollout data to device
  const auto obs_d    = rollout.obs.to(device);      // [T, N, 47]
  const auto act_d    = rollout.actions.to(device);  // [T, N, 8]
  const auto logp_d   = rollout.logprobs.to(device); // [T, N]
  const auto rew_d    = rollout.rewards.to(device);  // [T, N]
  const auto done_d   = rollout.dones.to(device);    // [T, N]
  const auto val_d    = rollout.values.to(device);   // [T, N]
  const auto next_obs_d  = next_obs.to(device);      // [N, 47]
  const auto next_done_d = next_done.to(device);     // [N]

  // ---- Hindsight goal sampling  sprint5:649-658 ----
  const auto future_goals = sample_hindsight_goals(obs_d, done_d).to(device); // [T,N,6]

  // ---- GAE  sprint5:660-678 ----
  const auto agent_goals = build_agent_goals(N, device);   // [N, 6]
  torch::Tensor nv;
  {
    torch::NoGradGuard no_grad;
    nv = value_->forward(next_obs_d, agent_goals).flatten(); // [N]
  }
  auto advantages = compute_gae_pulsar(rew_d, val_d, done_d, nv, next_done_d).detach();
  const auto returns = (advantages + val_d).detach();

  // Flatten for minibatch processing
  const auto b_obs    = obs_d.reshape({B, kObsDim});
  const auto b_act    = act_d.reshape({B, kActionDim});
  const auto b_logp   = logp_d.reshape({B});
  const auto b_ret    = returns.reshape({B});
  const auto b_fgoals = future_goals.reshape({B, kGoalDim});

  // ---- Quasimetric advantage  sprint5:686-691 ----
  torch::Tensor b_adv;
  {
    torch::NoGradGuard no_grad;
    const auto q = critic_->forward(b_obs, b_act, b_fgoals);
    b_adv = compute_quasimetric_advantage(q).detach();
  }

  // ---- Optimization  sprint5:693-754 ----
  // Precompute per-agent parity mask (True = odd = Orange/defender)
  const auto is_orange_mask = torch::arange(B, device) % 2 == 1; // [B] bool
  // offense_weight = 1.0 for blue, 0.85 for orange
  const auto agent_weights = torch::where(
      is_orange_mask, torch::full({B}, 0.85f, device), torch::ones({B}, device));

  auto indices = torch::randperm(B, device);  // shuffled each epoch

  float sum_pg = 0.0f, sum_vl = 0.0f, sum_ent = 0.0f;
  float sum_cl = 0.0f, sum_total = 0.0f, sum_kl = 0.0f;
  int n_updates = 0;

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    indices = torch::randperm(B, device);

    for (int start = 0; start < B; start += minibatch_size) {
      const auto mb_idx = indices.slice(0, start, std::min(start + minibatch_size, B));
      const int  mb_n   = static_cast<int>(mb_idx.size(0));

      // Build per-minibatch goals (based on parity of flat agent index)
      // sprint5:702-708
      const auto gs = goal_score_tensor(device);
      const auto gd = goal_defend_tensor(device);
      auto mb_goals = torch::zeros({mb_n, kGoalDim}, device);
      const auto mb_parity = mb_idx % 2;  // 0=blue, 1=orange
      mb_goals.index_put_({mb_parity == 0}, gs.unsqueeze(0));
      mb_goals.index_put_({mb_parity == 1}, gd.unsqueeze(0));

      const auto mb_obs  = b_obs.index_select(0, mb_idx);
      const auto mb_act  = b_act.index_select(0, mb_idx);
      const auto mb_logp_old = b_logp.index_select(0, mb_idx);
      const auto mb_ret  = b_ret.index_select(0, mb_idx);
      const auto mb_fg   = b_fgoals.index_select(0, mb_idx);
      const auto mb_w    = agent_weights.index_select(0, mb_idx);

      // Actor forward
      auto [am, astd, _feat] = actor_->forward(mb_obs, mb_goals);
      const auto nlp = gaussian_log_prob(mb_act, am, astd);   // [mb_n]
      const auto ent = gaussian_entropy(astd);                  // [mb_n]
      const auto ratio = torch::exp(nlp - mb_logp_old);        // [mb_n]

      // Advantage sign flip for Orange  sprint5:718-722
      auto mb_adv_typed = b_adv.index_select(0, mb_idx).clone();
      mb_adv_typed.index_put_({mb_parity == 1}, -mb_adv_typed.index({mb_parity == 1}));

      // Per-minibatch normalise  sprint5:724
      const auto ma = (mb_adv_typed - mb_adv_typed.mean()) / (mb_adv_typed.std() + 1e-8f);

      // Clipped PPO surrogate  sprint5:725  (positive = loss to minimise)
      const auto pg_ind = torch::max(-ma * ratio,
                                      -ma * torch::clamp(ratio, 0.8f, 1.2f));

      // Value loss  sprint5:727-728
      const auto nv_mb = value_->forward(mb_obs, mb_goals).flatten();
      const auto vl_ind = 0.5f * (nv_mb - mb_ret).square();

      // InfoNCE contrastive loss on the critic  sprint5:730
      const auto cl = compute_pairwise_infonce_loss(critic_, mb_obs, mb_act, mb_fg);

      // Weighted mean  sprint5:740-743
      const auto pg    = (pg_ind * mb_w).mean();
      const auto vl    = (vl_ind * mb_w).mean();
      const auto en    = ent.mean();
      const auto total = pg - 0.01f * en + 0.5f * vl + cl;

      zero_grad();
      total.backward();
      torch::nn::utils::clip_grad_norm_(actor_->parameters(), 0.5);
      torch::nn::utils::clip_grad_norm_(value_->parameters(), 0.5);
      torch::nn::utils::clip_grad_norm_(critic_->parameters(), 0.5);
      actor_opt_.step();
      value_opt_.step();
      critic_opt_.step();

      // Accumulate diagnostics
      {
        torch::NoGradGuard ng;
        const auto kl = (-(nlp - mb_logp_old)).mean();
        sum_kl    += kl.item<float>();
        sum_pg    += pg.item<float>();
        sum_vl    += vl.item<float>();
        sum_ent   += en.item<float>();
        sum_cl    += cl.item<float>();
        sum_total += total.item<float>();
        ++n_updates;
      }
    }
  }

  const float inv = 1.0f / static_cast<float>(std::max(n_updates, 1));
  return {sum_pg * inv, sum_vl * inv, sum_ent * inv,
          sum_cl * inv, sum_total * inv, sum_kl * inv};
}

}  // namespace pulsar
#endif  // PULSAR_HAS_TORCH
