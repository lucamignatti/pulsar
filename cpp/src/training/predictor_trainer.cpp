#include "pulsar/training/predictor_trainer.hpp"

#include <iostream>
#include <limits>
#include <cmath>

#include "pulsar/training/gradient_surgery.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

PredictorTrainer::PredictorTrainer(
    const ExperimentConfig& config,
    const torch::Device& device)
    : config_(config), device_(device) {
  ema_losses_.assign(kSparseEventChannelCount, -1.0F);
  deltas_.assign(kSparseEventChannelCount, std::numeric_limits<float>::infinity());
  consecutive_low_delta_.assign(kSparseEventChannelCount, 0);
}

void PredictorTrainer::initialize_optimizers(VRPOActor& actor) {
  optimizers_.clear();
  optimizers_.resize(kSparseEventChannelCount);
  for (std::size_t i = 0; i < kSparseEventChannelCount; ++i) {
    const std::string channel_name(kSparseEventChannels[i].name);
    const auto& channel = config_.predictor.channels.at(channel_name);
    if (channel.enabled) {
      auto& head = actor->predictor_head(i);
      optimizers_[i] = std::make_unique<torch::optim::Adam>(
          head->parameters(),
          torch::optim::AdamOptions(channel.learning_rate).weight_decay(1.0e-4));
    }
  }
}

void PredictorTrainer::save_optimizers(torch::serialize::OutputArchive& archive) const {
  for (std::size_t i = 0; i < optimizers_.size() && i < kSparseEventChannelCount; ++i) {
    if (optimizers_[i]) {
      torch::serialize::OutputArchive sub_archive;
      optimizers_[i]->save(sub_archive);
      archive.write("predictor_optimizer_" + std::to_string(i), sub_archive);
    }
  }
}

void PredictorTrainer::load_optimizers(torch::serialize::InputArchive& archive) {
  for (std::size_t i = 0; i < optimizers_.size() && i < kSparseEventChannelCount; ++i) {
    if (optimizers_[i]) {
      torch::serialize::InputArchive sub_archive;
      if (archive.try_read("predictor_optimizer_" + std::to_string(i), sub_archive)) {
        optimizers_[i]->load(sub_archive);
      }
    }
  }
}

void PredictorTrainer::update_predictors(
    VRPOActor& actor,
    const torch::Tensor& flat_features,
    const torch::Tensor& targets_all,
    int64_t rollout_steps,
    int64_t count,
    int64_t agent_offset,
    int update_index,
    std::vector<torch::Tensor>& channel_loss_sums,
    std::vector<double>& channel_sample_counts,
    std::vector<torch::Tensor>& channel_positive_sums,
    std::vector<double>& channel_positive_sample_counts) {
  PULSAR_TRACE_SCOPE_CAT("predictor_trainer", "update_predictors");

  const double samples = static_cast<double>(rollout_steps * count);

  // 1. Loss of Plasticity NaN sanitizer and active optimizer resetting
  for (std::size_t channel_index = 0; channel_index < kSparseEventChannelCount; ++channel_index) {
    const std::string channel_name(kSparseEventChannels[channel_index].name);
    const auto& channel = config_.predictor.channels.at(channel_name);
    if (!channel.enabled || channel_index >= optimizers_.size() || !optimizers_[channel_index]) {
      continue;
    }
    SparseRewardPredictor& head = actor->predictor_head(channel_index);
    const GradientSanitizeResult sanitized = zero_nonfinite_parameters(*head);
    if (sanitized.changed) {
      std::cerr << "Warning: repaired non-finite predictor_" << channel_name
                << " parameter " << sanitized.first_parameter
                << "; resetting its optimizer state.\n";
      optimizers_[channel_index] = std::make_unique<torch::optim::Adam>(
          head->parameters(),
          torch::optim::AdamOptions(channel.learning_rate).weight_decay(1.0e-4));
    }
    optimizers_[channel_index]->zero_grad();
  }

  // 2. Forward pass all predictors
  torch::Tensor logits_all = finite_or_zero(actor->forward_all_predictors(flat_features));

  std::vector<torch::Tensor> losses;
  losses.reserve(kSparseEventChannelCount);
  std::vector<std::size_t> loss_channels;
  loss_channels.reserve(kSparseEventChannelCount);

  // 3. Compute loss per enabled channel
  for (std::size_t channel_index = 0; channel_index < kSparseEventChannelCount; ++channel_index) {
    const std::string channel_name(kSparseEventChannels[channel_index].name);
    const auto& channel = config_.predictor.channels.at(channel_name);
    if (!channel.enabled || channel_index >= optimizers_.size() || !optimizers_[channel_index]) {
      continue;
    }

    torch::Tensor targets = targets_all
        .narrow(1, agent_offset, count)
        .select(2, static_cast<int64_t>(channel_index))
        .reshape({rollout_steps * count});

    // Compute dynamic class weighting for target imbalance
    const double pos_sum = (targets > 0).to(torch::kFloat64).sum().item<double>();
    channel_positive_sums[channel_index].add_(pos_sum);
    channel_positive_sample_counts[channel_index] += samples;

    const double batch_pos_rate = pos_sum / static_cast<double>(targets.numel());
    float w1 = 1.0F;
    if (batch_pos_rate > 1e-6) {
      w1 = std::max(1.0F, std::min(50.0F, static_cast<float>((1.0 - batch_pos_rate) / (batch_pos_rate + 1e-6))));
    }

    torch::Tensor logits = logits_all.select(1, static_cast<int64_t>(channel_index));
    torch::Tensor weight = torch::tensor({1.0F, w1}, logits.options());
    torch::Tensor loss = torch::nn::functional::cross_entropy(
        logits,
        targets,
        torch::nn::functional::CrossEntropyFuncOptions().weight(weight));

    losses.push_back(loss);
    loss_channels.push_back(channel_index);

    // Accumulate on GPU — no sync stalls
    channel_loss_sums[channel_index].add_(loss.to(torch::kFloat64) * samples);
    channel_sample_counts[channel_index] += samples;
  }

  // 4. Backward pass & step optimizers
  if (!losses.empty()) {
    const torch::Tensor stacked_losses = torch::stack(losses);
    if (!torch::isfinite(stacked_losses).all().item<bool>()) {
      std::cerr << "Warning: skipping predictor optimizer step after non-finite loss"
                << " at update=" << update_index
                << " agent_offset=" << agent_offset
                << '\n';
      for (std::size_t channel_index : loss_channels) {
        optimizers_[channel_index]->zero_grad();
      }
      return;
    }
    torch::Tensor total_loss = stacked_losses.sum();
    total_loss.backward();

    for (std::size_t channel_index = 0; channel_index < kSparseEventChannelCount; ++channel_index) {
      const std::string channel_name(kSparseEventChannels[channel_index].name);
      const auto& channel = config_.predictor.channels.at(channel_name);
      if (!channel.enabled || channel_index >= optimizers_.size() || !optimizers_[channel_index]) {
        continue;
      }
      auto& optimizer = optimizers_[channel_index];
      SparseRewardPredictor& head = actor->predictor_head(channel_index);
      GradientSanitizeResult sanitized = zero_nonfinite_gradients(*head);
      if (sanitized.changed && !gradients_are_finite(*head)) {
        std::cerr << "Warning: skipping predictor_" << channel_name
                  << " optimizer step after non-finite gradient in "
                  << sanitized.first_parameter << '\n';
        optimizer->zero_grad();
        continue;
      }
      const double grad_norm = torch::nn::utils::clip_grad_norm_(head->parameters(), 0.1);
      if (!std::isfinite(grad_norm)) {
        std::cerr << "Warning: skipping predictor_" << channel_name
                  << " optimizer step with non-finite grad_norm=" << grad_norm << '\n';
        optimizer->zero_grad();
        continue;
      }
      optimizer->step();
      optimizer->zero_grad();
    }
  }
}

void PredictorTrainer::apply_reward_shaping(
    const VRPOActor& actor,
    const torch::Tensor& flat_features,
    const torch::Tensor& sparse_events,
    torch::Tensor& shaped_rewards,
    int64_t rollout_steps,
    int64_t count,
    int64_t agent_offset) {
  PULSAR_TRACE_SCOPE_CAT("predictor_trainer", "apply_reward_shaping");
  torch::NoGradGuard no_grad;

  // Initialize hindsight EMAs once
  if (!hindsight_initialized_) {
    ema_p_goal_given_event_.assign(kSparseEventChannelCount, -1.0F);
    ema_p_goal_given_no_event_.assign(kSparseEventChannelCount, -1.0F);
    hindsight_initialized_ = true;
  }

  // Get all predictor probabilities
  torch::Tensor shaping_logits_all = finite_or_zero(actor->forward_all_predictors(flat_features)).detach();

  // Goal predictor P_goal
  const torch::Tensor goal_logits = shaping_logits_all.select(1, kGoalChannelIndex);
  const torch::Tensor goal_probs = torch::softmax(goal_logits, -1);
  const torch::Tensor p_goal = goal_probs.select(1, 1).reshape({rollout_steps, count});

  constexpr float kAlpha = 0.95F;
  const torch::Tensor shaped_view = shaped_rewards.narrow(1, agent_offset, count);

  for (std::size_t channel_index = 1; channel_index < kSparseEventChannelCount; ++channel_index) {
    const std::string channel_name(kSparseEventChannels[channel_index].name);
    const auto& channel = config_.predictor.channels.at(channel_name);
    const bool active = channel_index < actor->predictor_active_.size() &&
        actor->predictor_active_[channel_index];
    if (!channel.enabled || !active) {
      continue;
    }

    // Get event indicators from sparse_events buffer
    const torch::Tensor events_flat = sparse_events
        .narrow(1, agent_offset, count)
        .select(2, static_cast<int64_t>(channel_index))
        .reshape({rollout_steps * count});

    const torch::Tensor p_goal_flat = p_goal.reshape({rollout_steps * count});

    // Update EMAs: P_goal when event fires vs when it doesn't
    const torch::Tensor event_mask = events_flat > 0.5F;
    const bool has_events = event_mask.any().item<bool>();
    const bool has_no_events = (~event_mask).any().item<bool>();

    if (has_events) {
      const float pg_event = p_goal_flat.masked_select(event_mask).mean().item<float>();
      if (ema_p_goal_given_event_[channel_index] < 0.0F) {
        ema_p_goal_given_event_[channel_index] = pg_event;
      } else {
        ema_p_goal_given_event_[channel_index] =
            kAlpha * ema_p_goal_given_event_[channel_index] + (1.0F - kAlpha) * pg_event;
      }
    }
    if (has_no_events) {
      const float pg_no_event = p_goal_flat.masked_select(~event_mask).mean().item<float>();
      if (ema_p_goal_given_no_event_[channel_index] < 0.0F) {
        ema_p_goal_given_no_event_[channel_index] = pg_no_event;
      } else {
        ema_p_goal_given_no_event_[channel_index] =
            kAlpha * ema_p_goal_given_no_event_[channel_index] + (1.0F - kAlpha) * pg_no_event;
      }
    }

    // Compute advantage: how much does this event type predict goals?
    float advantage = 0.0F;
    if (ema_p_goal_given_event_[channel_index] >= 0.0F &&
        ema_p_goal_given_no_event_[channel_index] >= 0.0F) {
      advantage = ema_p_goal_given_event_[channel_index] - ema_p_goal_given_no_event_[channel_index];
      if (advantage < 0.0F) advantage = 0.0F;
    }

    // Dynamic weight = P_goal * (1 + advantage)
    // Events that happen when P_goal is high and that historically correlate with goals get more reward
    const torch::Tensor dynamic_weight = p_goal_flat * (1.0F + advantage);
    const torch::Tensor hindsight_reward = dynamic_weight.reshape({rollout_steps, count}) *
        events_flat.reshape({rollout_steps, count});
    shaped_view.add_(hindsight_reward);
  }
}

void PredictorTrainer::process_convergence(
    VRPOActor& actor,
    const std::vector<PredictorUpdateStats>& stats,
    int update_index) {
  const auto finish_channel = [&](double mean_loss,
                                  double positive_rate,
                                  float& ema_loss,
                                  float& delta,
                                  int& low_delta_streak,
                                  std::uint8_t& active,
                                  const PredictorChannelConfig& channel,
                                  const char* channel_name) {
    if (mean_loss <= 0.0 || !std::isfinite(mean_loss)) {
      return;
    }
    if (ema_loss < 0.0F) {
      ema_loss = static_cast<float>(mean_loss);
      delta = std::numeric_limits<float>::infinity();
    } else {
      const float next_ema = 0.9F * ema_loss + 0.1F * static_cast<float>(mean_loss);
      delta = std::abs(next_ema - ema_loss);
      ema_loss = next_ema;
    }

    // Track sustained stability: require N consecutive updates whose delta is
    // below threshold. A single low-delta update can come from transient noise
    // or from a still-learning loss whose per-step change happens to be small.
    if (std::isfinite(delta) && delta <= channel.convergence_threshold) {
      ++low_delta_streak;
    } else {
      low_delta_streak = 0;
    }

    if (active == 0 &&
        update_index >= channel.warmup_updates &&
        positive_rate >= 0.0005 &&
        low_delta_streak >= kRequiredConsecutiveStableUpdates) {
      active = 1;
      std::cout << "predictor_enabled channel=" << channel_name
                << " update=" << update_index
                << " ema_loss=" << ema_loss
                << " delta=" << delta
                << " streak=" << low_delta_streak
                << " positive_rate=" << positive_rate
                << '\n';
    }
  };

  if (consecutive_low_delta_.size() < kSparseEventChannelCount) {
    consecutive_low_delta_.resize(kSparseEventChannelCount, 0);
  }
  for (std::size_t channel_index = 0; channel_index < kSparseEventChannelCount; ++channel_index) {
    const std::string channel_name(kSparseEventChannels[channel_index].name);
    const auto& channel = config_.predictor.channels.at(channel_name);
    if (channel_index >= actor->predictor_active_.size()) {
      actor->predictor_active_.resize(kSparseEventChannelCount, 0);
    }
    const double mean_loss = stats[channel_index].mean_loss();
    const double positive_rate = stats[channel_index].positive_rate();
    finish_channel(
        mean_loss,
        positive_rate,
        ema_losses_[channel_index],
        deltas_[channel_index],
        consecutive_low_delta_[channel_index],
        actor->predictor_active_[channel_index],
        channel,
        channel_name.c_str());
  }
}

void PredictorTrainer::log_metrics(
    WandbLogger& logger,
    const std::map<std::string, double>& losses,
    const std::map<std::string, double>& deltas,
    const std::map<std::string, double>& positive_rates,
    const std::map<std::string, double>& active_states) {
  // First-class visual logging for predictor channels
  for (const auto& [channel, val] : losses) {
    logger.add_metric("Predictor", "predictor_loss_" + channel, val);
  }
  for (const auto& [channel, val] : deltas) {
    logger.add_metric("Predictor", "predictor_delta_" + channel, val);
  }
  for (const auto& [channel, val] : positive_rates) {
    logger.add_metric("Predictor", "predictor_positive_" + channel, val);
  }
  for (const auto& [channel, val] : active_states) {
    logger.add_metric("Predictor", "predictor_active_" + channel, val);
  }
  
  // Tabular summary W&B Table logging
  std::vector<std::string> columns = {"Channel", "Active", "BCE Loss (EMA)", "Convergence Delta", "Positive Rate"};
  std::vector<std::vector<nlohmann::json>> row_data;
  
  for (std::size_t i = 0; i < kSparseEventChannelCount; ++i) {
    const std::string name(kSparseEventChannels[i].name);
    if (!config_.predictor.channels.at(name).enabled) {
      continue;
    }
    bool active = active_states.count(name) ? (active_states.at(name) > 0.5) : false;
    double loss = losses.count(name) ? losses.at(name) : 0.0;
    double delta = deltas.count(name) ? deltas.at(name) : 0.0;
    double pos_rate = positive_rates.count(name) ? positive_rates.at(name) : 0.0;
    
    std::vector<nlohmann::json> row = {
      name,
      active ? "ACTIVE" : "INACTIVE",
      loss,
      delta,
      pos_rate
    };
    row_data.push_back(row);
  }
  
  logger.add_table("predictor_status_table", columns, row_data);
}

}  // namespace pulsar

#endif
