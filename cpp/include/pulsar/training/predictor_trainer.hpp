#pragma once

#ifdef PULSAR_HAS_TORCH

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/vrpo_actor.hpp"
#include "pulsar/training/sparse_event_channels.hpp"
#include "pulsar/logging/wandb_logger.hpp"

namespace pulsar {

struct PredictorUpdateStats {
  double loss_sum = 0.0;
  double samples = 0.0;
  double positive_sum = 0.0;
  double positive_samples = 0.0;

  [[nodiscard]] double mean_loss() const {
    return samples > 0.0 ? loss_sum / samples : 0.0;
  }

  [[nodiscard]] double positive_rate() const {
    return positive_samples > 0.0 ? positive_sum / positive_samples : 0.0;
  }
};

class PredictorTrainer {
 public:
  PredictorTrainer(
      const ExperimentConfig& config,
      const torch::Device& device);

  void initialize_optimizers(VRPOActor& actor);
  void save_optimizers(torch::serialize::OutputArchive& archive) const;
  void load_optimizers(torch::serialize::InputArchive& archive);

  void update_predictors(
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
      std::vector<double>& channel_positive_sample_counts);

  void apply_reward_shaping(
      const VRPOActor& actor,
      const torch::Tensor& flat_features,
      const torch::Tensor& sparse_events,
      torch::Tensor& shaped_rewards,
      int64_t rollout_steps,
      int64_t count,
      int64_t agent_offset);

  void process_convergence(
      VRPOActor& actor,
      const std::vector<PredictorUpdateStats>& stats,
      int update_index);

  void log_metrics(
      WandbLogger& logger,
      const std::map<std::string, double>& losses,
      const std::map<std::string, double>& deltas,
      const std::map<std::string, double>& positive_rates,
      const std::map<std::string, double>& active_states);

  [[nodiscard]] const std::vector<float>& ema_losses() const { return ema_losses_; }
  [[nodiscard]] const std::vector<float>& deltas() const { return deltas_; }
  
  std::vector<float>& mutable_ema_losses() { return ema_losses_; }
  std::vector<float>& mutable_deltas() { return deltas_; }

 private:
  ExperimentConfig config_;
  torch::Device device_;
  std::vector<std::unique_ptr<torch::optim::Adam>> optimizers_{};
  std::vector<float> ema_losses_{};
  std::vector<float> deltas_{};
  // Count of consecutive updates with delta <= convergence_threshold. A
  // predictor activates only once this exceeds kRequiredConsecutiveStableUpdates,
  // which filters out both transient noise and slow-but-still-learning losses.
  std::vector<int> consecutive_low_delta_{};
  static constexpr int kRequiredConsecutiveStableUpdates = 10;

  // Hindsight weighting state
  static constexpr int kGoalChannelIndex = 0;
  std::vector<float> ema_p_goal_given_event_{};
  std::vector<float> ema_p_goal_given_no_event_{};
  bool hindsight_initialized_ = false;

  // Smoothed positive rate for stable class weighting (fix 2)
  std::vector<float> ema_pos_rate_{};
};

}  // namespace pulsar

#endif
