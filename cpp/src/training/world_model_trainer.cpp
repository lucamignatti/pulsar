#include "pulsar/training/world_model_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <cmath>
#include <torch/torch.h>
#include "pulsar/training/ppo_math.hpp"

namespace pulsar {

WorldModelTrainer::WorldModelTrainer(const WorldModelConfig& config, const torch::Device& device)
    : config_(config), device_(device) {}

torch::Tensor WorldModelTrainer::kl_divergence_free_bits(
    const torch::Tensor& post_mean,
    const torch::Tensor& post_logvar,
    const torch::Tensor& prior_mean,
    const torch::Tensor& prior_logvar,
    float free_bits) {
  const torch::Tensor logvar_diff = post_logvar - prior_logvar;
  const torch::Tensor kl = 0.5F * (torch::exp(logvar_diff)
      + (post_mean - prior_mean).square() * torch::exp(-prior_logvar)
      - 1.0F - logvar_diff);
  return torch::clamp_min(kl, free_bits).mean();
}

// Process one TBPTT chunk: forward + compute all losses + backward.
// Returns per-step intrinsic rewards [chunk_len, agents] on CPU.
// Writes loss scalars into the WorldModelLosses accumulators.
// Updates `state` in-place with detached final state (for next chunk).
static torch::Tensor process_rssm_chunk(
    RSSMWorldModel& rssm,
    const torch::Tensor& obs_chunk,          // [L, agents, obs_dim]   on device
    const torch::Tensor& action_chunk,       // [L, agents]             on device
    const torch::Tensor& ep_starts_chunk,    // [L, agents] float       on device
    const torch::Tensor& goal_chunk,         // [L, agents, 4]          on device
    const WorldModelConfig& config,
    RSSMState& state,                        // in: initial; out: detached final
    WorldModelLosses& accum) {

  const int L = static_cast<int>(obs_chunk.size(0));
  const int agents = static_cast<int>(obs_chunk.size(1));
  const torch::Device dev = obs_chunk.device();

  // Forward through this chunk with gradient tracking
  std::vector<RSSMStepOutput> steps = rssm->forward_sequence(
      obs_chunk, action_chunk, ep_starts_chunk, goal_chunk, &state);

  // Stack posterior/prior params: [L, agents, stochastic_dim]
  std::vector<torch::Tensor> pm, plv, qm, qlv;
  pm.reserve(static_cast<std::size_t>(L));
  plv.reserve(static_cast<std::size_t>(L));
  qm.reserve(static_cast<std::size_t>(L));
  qlv.reserve(static_cast<std::size_t>(L));
  for (const auto& s : steps) {
    pm.push_back(s.posterior_mean);
    plv.push_back(s.posterior_logvar);
    qm.push_back(s.prior_mean);
    qlv.push_back(s.prior_logvar);
  }
  const torch::Tensor kl_loss = WorldModelTrainer::kl_divergence_free_bits(
      torch::stack(pm,  0),
      torch::stack(plv, 0),
      torch::stack(qm,  0),
      torch::stack(qlv, 0),
      config.kl_free_bits);

  // ICM forward and inverse losses
  torch::Tensor icm_fwd_loss  = torch::zeros({}, torch::TensorOptions().device(dev));
  torch::Tensor icm_inv_loss  = torch::zeros({}, torch::TensorOptions().device(dev));
  torch::Tensor intrinsic_cpu = torch::zeros({L, agents});

  if (L > 1) {
    std::vector<torch::Tensor> icm_preds_v, latents_v;
    icm_preds_v.reserve(static_cast<std::size_t>(L));
    latents_v.reserve(static_cast<std::size_t>(L));
    for (const auto& s : steps) {
      icm_preds_v.push_back(s.icm_forward_pred);
      latents_v.push_back(s.latent);
    }
    const torch::Tensor latents = torch::stack(latents_v, 0);           // [L, agents, ld]
    const torch::Tensor icm_preds = torch::stack(icm_preds_v, 0);       // [L, agents, ld]
    const torch::Tensor future = latents.narrow(0, 1, L - 1).detach();  // [L-1, agents, ld]
    const torch::Tensor cur_pred = icm_preds.narrow(0, 0, L - 1);       // [L-1, agents, ld]
    const torch::Tensor ep_mask = (1.0F - ep_starts_chunk.narrow(0, 1, L - 1)).unsqueeze(-1);

    icm_fwd_loss = (cur_pred - future).square().sum(-1, true).mul(ep_mask).mean();

    // Intrinsic rewards: detached ICM forward error per step (no grad)
    {
      torch::NoGradGuard ng;
      const torch::Tensor r_int = (cur_pred.detach() - future).square().sum(-1) * ep_mask.squeeze(-1);
      intrinsic_cpu.narrow(0, 0, L - 1).copy_(r_int.to(torch::kCPU));
    }

    // ICM inverse loss
    const int flat_size = (L - 1) * agents;
    const torch::Tensor lat_t   = latents.narrow(0, 0, L - 1).reshape({flat_size, -1});
    const torch::Tensor lat_tp1 = future.reshape({flat_size, -1});
    const torch::Tensor inv_logits = rssm->inverse_model_forward(lat_t.detach(), lat_tp1);
    const torch::Tensor inv_targets = action_chunk.narrow(0, 0, L - 1).reshape({flat_size});
    const torch::Tensor inv_ep_mask = ep_mask.squeeze(-1).reshape({flat_size});
    const torch::Tensor inv_loss_per = torch::nn::functional::cross_entropy(
        inv_logits, inv_targets,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
    icm_inv_loss = (inv_loss_per * inv_ep_mask).mean();
  }

  // Goal head loss
  std::vector<torch::Tensor> latents_for_goal;
  latents_for_goal.reserve(static_cast<std::size_t>(L));
  for (const auto& s : steps) latents_for_goal.push_back(s.latent);
  const torch::Tensor latents_stacked = torch::stack(latents_for_goal, 0);  // [L, agents, ld]
  const torch::Tensor goal_preds = rssm->predict_goal(latents_stacked.reshape({L * agents, -1}));
  const torch::Tensor goal_loss = torch::nn::functional::mse_loss(
      goal_preds, goal_chunk.reshape({L * agents, 4}).detach());

  // Multi-step consistency (capped within chunk to avoid cross-boundary issues)
  torch::Tensor consist_loss = torch::zeros({}, torch::TensorOptions().device(dev));
  const int k = std::min(config.num_consistency_steps, L - 1);
  if (k > 0) {
    std::vector<torch::Tensor> consist_terms;
    for (int t = 0; t < L - k; ++t) {
      // Check for any reset in [t+1, t+k]
      bool any_reset = false;
      for (int j = t + 1; j <= t + k; ++j) {
        if (ep_starts_chunk[j].any().item<bool>()) { any_reset = true; break; }
      }
      if (any_reset) continue;
      RSSMState cs = rssm->split_latent(latents_stacked[t].detach());
      torch::Tensor imag_latent;
      for (int j = 0; j < k; ++j) {
        RSSMStepOutput io = rssm->imagine_step(action_chunk[t + j], cs);
        cs.h = io.h; cs.z = io.z;
        imag_latent = io.latent;
      }
      consist_terms.push_back((imag_latent - latents_stacked[t + k].detach()).square().mean());
    }
    if (!consist_terms.empty()) {
      consist_loss = torch::stack(consist_terms).mean();
    }
  }

  // Combined loss for this chunk
  torch::Tensor chunk_loss = config.kl_weight       * kl_loss
                           + config.icm_weight       * icm_fwd_loss
                           + config.icm_inverse_weight * icm_inv_loss
                           + 1.0F                    * goal_loss
                           + config.consistency_weight * consist_loss;

  if (chunk_loss.defined() && torch::isfinite(chunk_loss).item<bool>()) {
    chunk_loss.backward();  // accumulates into params; caller zeros_grad before first chunk
  }

  // Accumulate logging metrics
  accum.kl_loss_val          += kl_loss.item<double>();
  accum.icm_forward_loss_val += icm_fwd_loss.item<double>();
  accum.icm_inverse_loss_val += icm_inv_loss.item<double>();
  accum.goal_head_loss_val   += goal_loss.item<double>();
  accum.consistency_loss_val += consist_loss.item<double>();

  // Detach final state for next chunk (breaks autograd graph across chunks)
  state.h = steps.back().h.detach();
  state.z = steps.back().z.detach();

  return intrinsic_cpu;
}

torch::Tensor WorldModelTrainer::compute_losses(
    RSSMWorldModel& rssm,
    const RolloutStorage& rollout,
    const ExperimentConfig& config,
    WorldModelLosses& losses_out) {

  const int T = rollout.rollout_length();
  const int agents = rollout.num_agents();
  if (T <= 0 || agents <= 0 || !rssm) {
    losses_out.total = torch::zeros({}, torch::TensorOptions().device(device_));
    return torch::zeros({T, agents});
  }

  // TBPTT chunk size: match Mamba2's sequence_length so the chunk fits comfortably on GPU.
  // This bounds the autograd graph depth to chunk_size steps instead of T.
  const int chunk_size = std::max(4, config.model.sequence_length);

  torch::Tensor intrinsic_rewards = torch::zeros({T, agents});
  RSSMState state = rssm->zero_state(static_cast<int64_t>(agents), device_);

  // Process rollout in TBPTT chunks — backward() called per chunk to cap graph depth
  for (int t0 = 0; t0 < T; t0 += chunk_size) {
    const int chunk_len = std::min(chunk_size, T - t0);

    // Transfer only this chunk to the device
    const torch::Tensor obs_c    = rollout.obs.narrow(0, t0, chunk_len).to(device_);
    const torch::Tensor act_c    = rollout.actions.narrow(0, t0, chunk_len).to(device_);
    const torch::Tensor ep_c     = rollout.episode_starts.narrow(0, t0, chunk_len)
                                    .to(device_).to(torch::kFloat32);
    const torch::Tensor goal_c   = rollout.goal_positions.narrow(0, t0, chunk_len).to(device_);

    const torch::Tensor chunk_int = process_rssm_chunk(
        rssm, obs_c, act_c, ep_c, goal_c, config_, state, losses_out);

    intrinsic_rewards.narrow(0, t0, chunk_len).copy_(chunk_int);
  }

  // losses_out.total is left undefined — backward() was already called inside chunks.
  // Set a scalar for the caller's finitude check / logging.
  losses_out.total = torch::tensor(
      losses_out.kl_loss_val + losses_out.icm_forward_loss_val + losses_out.goal_head_loss_val,
      torch::TensorOptions().device(device_));

  return intrinsic_rewards;
}

torch::Tensor WorldModelTrainer::imagined_her_goals(
    RSSMWorldModel& rssm,
    const Actor& actor,
    const torch::Tensor& terminal_latents,
    const torch::Tensor& terminal_obs,
    const torch::Tensor& action_masks,
    int horizon,
    float icm_threshold) {

  if (!rssm || !actor || terminal_latents.size(0) == 0) return torch::Tensor{};

  torch::NoGradGuard no_grad;
  RSSMState state = rssm->split_latent(terminal_latents);
  std::vector<torch::Tensor> reliable_goals;

  for (int step = 0; step < horizon; ++step) {
    ActorStepOutput policy_out = const_cast<ActorImpl&>(*actor)
        .forward_step(terminal_obs, {}, /*compute_value=*/false);
    const torch::Tensor actions = sample_masked_actions(
        policy_out.policy_logits, action_masks, false, nullptr, 1.0F);

    RSSMStepOutput imag = rssm->imagine_step(actions, state);
    state.h = imag.h;
    state.z = imag.z;

    // Quality gate: prior logvar entropy (high entropy = uncertain = discard)
    const torch::Tensor prior_entropy = 0.5F * (1.0F + imag.prior_logvar);  // [batch, sdim]
    const torch::Tensor mean_entropy  = prior_entropy.mean(-1);              // [batch]
    const torch::Tensor reliable      = mean_entropy < icm_threshold;

    if (reliable.any().item<bool>()) {
      reliable_goals.push_back(rssm->predict_goal(imag.latent).index({reliable}));
    }
  }

  if (reliable_goals.empty()) return torch::Tensor{};
  return torch::cat(reliable_goals, 0);
}

}  // namespace pulsar

#endif
