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
  // KL(q||p) = 0.5 * sum[ exp(logvar_q - logvar_p) + (mu_q-mu_p)^2/exp(logvar_p)
  //                       - 1 - (logvar_q - logvar_p) ]
  const torch::Tensor logvar_diff = post_logvar - prior_logvar;
  const torch::Tensor kl = 0.5F * (torch::exp(logvar_diff)
      + (post_mean - prior_mean).square() * torch::exp(-prior_logvar)
      - 1.0F - logvar_diff);
  // Free-bits clipping: per-latent-dimension, not summed
  const torch::Tensor kl_free = torch::clamp_min(kl, free_bits).mean();
  return kl_free;
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
    return torch::zeros({T, agents}, torch::TensorOptions().device(device_));
  }

  const torch::Tensor obs_seq = rollout.obs.narrow(0, 0, T).to(device_);        // [T, agents, obs_dim]
  const torch::Tensor action_seq = rollout.actions.narrow(0, 0, T).to(device_);  // [T, agents]
  const torch::Tensor ep_starts = rollout.episode_starts.narrow(0, 0, T).to(device_).to(torch::kFloat32);
  const torch::Tensor goal_seq = rollout.goal_positions.narrow(0, 0, T).to(device_); // [T, agents, 4]

  // Forward sequence through RSSM
  std::vector<RSSMStepOutput> steps = rssm->forward_sequence(obs_seq, action_seq, ep_starts, goal_seq);

  // Accumulate losses
  torch::Tensor kl_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor icm_fwd_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor icm_inv_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor goal_total = torch::zeros({}, torch::TensorOptions().device(device_));
  torch::Tensor consistency_total = torch::zeros({}, torch::TensorOptions().device(device_));

  // Per-step intrinsic rewards (detached from graph)
  torch::Tensor intrinsic_rewards = torch::zeros({T, agents}, torch::TensorOptions().device(device_));

  // Stack all posterior and prior params for batch KL computation
  std::vector<torch::Tensor> post_means, post_logvars, prior_means, prior_logvars;
  std::vector<torch::Tensor> all_latents, all_icm_preds;

  post_means.reserve(static_cast<std::size_t>(T));
  post_logvars.reserve(static_cast<std::size_t>(T));
  prior_means.reserve(static_cast<std::size_t>(T));
  prior_logvars.reserve(static_cast<std::size_t>(T));
  all_latents.reserve(static_cast<std::size_t>(T));
  all_icm_preds.reserve(static_cast<std::size_t>(T));

  for (int t = 0; t < T; ++t) {
    post_means.push_back(steps[static_cast<std::size_t>(t)].posterior_mean);
    post_logvars.push_back(steps[static_cast<std::size_t>(t)].posterior_logvar);
    prior_means.push_back(steps[static_cast<std::size_t>(t)].prior_mean);
    prior_logvars.push_back(steps[static_cast<std::size_t>(t)].prior_logvar);
    all_latents.push_back(steps[static_cast<std::size_t>(t)].latent);
    all_icm_preds.push_back(steps[static_cast<std::size_t>(t)].icm_forward_pred);
  }

  // KL loss (batched over all timesteps)
  const torch::Tensor pm = torch::stack(post_means, 0);    // [T, agents, stochastic_dim]
  const torch::Tensor plv = torch::stack(post_logvars, 0);
  const torch::Tensor qm = torch::stack(prior_means, 0);
  const torch::Tensor qlv = torch::stack(prior_logvars, 0);
  kl_total = kl_divergence_free_bits(pm, plv, qm, qlv, config_.kl_free_bits);

  // ICM losses: predict next latent from current latent+action, compare to actual next latent
  // Also compute per-step intrinsic rewards from ICM forward error (detached)
  const torch::Tensor latents_stacked = torch::stack(all_latents, 0);          // [T, agents, ld]
  const torch::Tensor icm_preds_stacked = torch::stack(all_icm_preds, 0);      // [T, agents, ld]

  if (T > 1) {
    // ICM forward loss: icm_pred[t] should match latent[t+1]
    // Both masked by episode boundaries (don't compute across resets)
    const torch::Tensor future_latents = latents_stacked.narrow(0, 1, T - 1).detach();  // [T-1, agents, ld]
    const torch::Tensor cur_preds = icm_preds_stacked.narrow(0, 0, T - 1);              // [T-1, agents, ld]
    const torch::Tensor future_ep_starts = ep_starts.narrow(0, 1, T - 1).unsqueeze(-1); // [T-1, agents, 1]
    // Mask: don't compute ICM across episode boundaries (next step is a reset)
    const torch::Tensor icm_mask = (1.0F - future_ep_starts);
    const torch::Tensor icm_fwd_err = (cur_preds - future_latents).square().sum(-1, /*keepdim=*/true);
    icm_fwd_total = (icm_fwd_err * icm_mask).mean();

    // ICM forward error becomes intrinsic reward (detached, per step)
    {
      torch::NoGradGuard no_grad;
      const torch::Tensor r_int = (cur_preds.detach() - future_latents).square().sum(-1);  // [T-1, agents]
      intrinsic_rewards.narrow(0, 0, T - 1).copy_(r_int * icm_mask.squeeze(-1));
    }

    // ICM inverse loss: predict action from consecutive latents
    const torch::Tensor cur_latents = latents_stacked.narrow(0, 0, T - 1).reshape({(T - 1) * agents, -1});
    const torch::Tensor nxt_latents = future_latents.reshape({(T - 1) * agents, -1});
    const torch::Tensor inv_logits = rssm->inverse_model_forward(cur_latents.detach(), nxt_latents);
    const torch::Tensor inv_targets = action_seq.narrow(0, 0, T - 1).reshape({(T - 1) * agents});
    const torch::Tensor icm_mask_flat = icm_mask.squeeze(-1).reshape({(T - 1) * agents});
    const torch::Tensor inv_loss_per = torch::nn::functional::cross_entropy(
        inv_logits,
        inv_targets,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
    icm_inv_total = (inv_loss_per * icm_mask_flat).mean();
  }

  // Goal head loss: latent[t] should predict goal[t]
  const torch::Tensor latents_flat = latents_stacked.reshape({T * agents, -1});
  const torch::Tensor goal_flat = goal_seq.reshape({T * agents, 4});
  const torch::Tensor goal_preds = rssm->predict_goal(latents_flat);
  goal_total = torch::nn::functional::mse_loss(goal_preds, goal_flat.detach());

  // Multi-step consistency loss: imagined latent from t should match posterior at t+k
  if (config_.num_consistency_steps > 1 && T > config_.num_consistency_steps) {
    const int k = std::min(config_.num_consistency_steps, T - 1);
    // For each t in [0, T-k), roll forward k steps in imagination and compare to posterior
    std::vector<torch::Tensor> consist_terms;
    for (int t = 0; t < T - k; ++t) {
      RSSMState state = rssm->split_latent(latents_stacked[t].detach());
      // Check no episode boundary in [t+1, t+k]
      bool any_reset = false;
      for (int j = 1; j <= k; ++j) {
        if (ep_starts[t + j].any().item<bool>()) {
          any_reset = true;
          break;
        }
      }
      if (any_reset) continue;

      // Roll forward k steps in imagination
      torch::Tensor imag_latent;
      for (int j = 0; j < k; ++j) {
        RSSMStepOutput imag_out = rssm->imagine_step(action_seq[t + j], state);
        state.h = imag_out.h;
        state.z = imag_out.z;
        imag_latent = imag_out.latent;
      }
      // Compare to actual posterior at t+k
      const torch::Tensor target = latents_stacked[t + k].detach();
      consist_terms.push_back((imag_latent - target).square().mean());
    }
    if (!consist_terms.empty()) {
      consistency_total = torch::stack(consist_terms).mean();
    }
  }

  // Total RSSM loss
  losses_out.kl_loss = kl_total;
  losses_out.icm_forward_loss = icm_fwd_total;
  losses_out.icm_inverse_loss = icm_inv_total;
  losses_out.goal_head_loss = goal_total;
  losses_out.consistency_loss = consistency_total;

  losses_out.kl_loss_val = kl_total.item<double>();
  losses_out.icm_forward_loss_val = icm_fwd_total.defined() ? icm_fwd_total.item<double>() : 0.0;
  losses_out.icm_inverse_loss_val = icm_inv_total.defined() ? icm_inv_total.item<double>() : 0.0;
  losses_out.goal_head_loss_val = goal_total.item<double>();
  losses_out.consistency_loss_val = consistency_total.defined() ? consistency_total.item<double>() : 0.0;

  losses_out.total = config_.kl_weight * kl_total
      + config_.icm_weight * icm_fwd_total
      + config_.icm_inverse_weight * icm_inv_total
      + 1.0F * goal_total
      + config_.consistency_weight * consistency_total;

  return intrinsic_rewards.to(torch::kCPU);
}

torch::Tensor WorldModelTrainer::imagined_her_goals(
    RSSMWorldModel& rssm,
    const Actor& actor,
    const torch::Tensor& terminal_latents,
    const torch::Tensor& terminal_obs,
    const torch::Tensor& action_masks,
    int horizon,
    float icm_threshold) {

  if (!rssm || !actor || terminal_latents.size(0) == 0) {
    return torch::Tensor{};
  }

  torch::NoGradGuard no_grad;
  const int batch = static_cast<int>(terminal_latents.size(0));

  RSSMState state = rssm->split_latent(terminal_latents);
  std::vector<torch::Tensor> reliable_goals;

  for (int step = 0; step < horizon; ++step) {
    // Use prior entropy as quality gate: H(prior) = 0.5 * sum(1 + log_logvar) (element-wise)
    // Get actions from actor policy (stochastic)
    const torch::Tensor norm_obs = terminal_obs;  // already normalized by caller
    ActorStepOutput policy_out = const_cast<ActorImpl&>(*actor).forward_step(norm_obs, {}, /*compute_value=*/false);
    const torch::Tensor actions = sample_masked_actions(
        policy_out.policy_logits, action_masks, /*deterministic=*/false, nullptr, 1.0F);

    RSSMStepOutput imag_out = rssm->imagine_step(actions, state);
    state.h = imag_out.h;
    state.z = imag_out.z;

    // Quality gate: use prior logvar entropy as uncertainty proxy
    // High entropy prior → model uncertain → unreliable imagined goal
    // H(prior_i) = 0.5*(1 + logvar_i): if mean exceeds threshold, discard
    const torch::Tensor prior_entropy = 0.5F * (1.0F + imag_out.prior_logvar);  // [batch, stochastic_dim]
    const torch::Tensor mean_entropy = prior_entropy.mean(-1);  // [batch]

    // Adaptive gate: compare to icm_threshold (set from ~75th pct of training distribution)
    const torch::Tensor reliable_mask = mean_entropy < icm_threshold;  // [batch]

    if (reliable_mask.any().item<bool>()) {
      const torch::Tensor imag_goals = rssm->predict_goal(imag_out.latent);  // [batch, 4]
      reliable_goals.push_back(imag_goals.index({reliable_mask}));
    }
  }

  if (reliable_goals.empty()) {
    return torch::Tensor{};
  }
  return torch::cat(reliable_goals, 0);
}

}  // namespace pulsar

#endif
