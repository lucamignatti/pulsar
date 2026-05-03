#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace pulsar {

struct TrainerMetrics {
  double collection_agent_steps_per_second = 0.0;
  double update_agent_steps_per_second = 0.0;
  double overall_agent_steps_per_second = 0.0;
  double update_seconds = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
  double forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  double adaptive_epsilon = 0.0;
  double critic_variance = 0.0;
  double mean_confidence_weight = 0.0;
  double extrinsic_reward_mean = 0.0;
  double curiosity_reward_mean = 0.0;
  double learning_progress_reward_mean = 0.0;
  double bc_regularization_beta = 0.0;
  double novelty_ema = 0.0;
  double learning_progress_ema = 0.0;
  std::map<std::string, double> elo_ratings{};
  std::map<std::string, double> value_losses{};
};

class APPOTrainer {
 public:
  APPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");

 private:
  [[nodiscard]] torch::Tensor map_outcome_labels_to_rewards(const torch::Tensor& labels) const;
  void maybe_initialize_from_checkpoint();
  void save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const;
  void prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const;
  TrainerMetrics run_update(std::int64_t* global_step, int update_index);
  TrainerMetrics update_actor();
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, int update_index) const;

  void update_weight_schedule();
  void decay_bc_beta();

  ExperimentConfig config_{};
  std::unique_ptr<BatchedRocketSimCollector> collector_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  ControllerActionTable action_table_{};
  PPOActor actor_{nullptr};
  ObservationNormalizer actor_normalizer_;
  torch::optim::Adam actor_optimizer_;
  RolloutStorage rollout_;
  torch::Device device_{torch::kCPU};
  std::filesystem::path run_output_root_{};
  bool log_initialization_ = true;
  std::int64_t resumed_global_step_ = 0;
  std::int64_t resumed_update_index_ = 0;
  std::size_t total_agents_ = 0;
  ContinuumState collection_state_{};
  ContinuumState opponent_collection_state_{};
  bool use_pinned_host_buffers_ = false;

  std::unordered_map<std::string, float> head_weights_{};
  float current_beta_ = 0.0F;
  double novelty_ema_ = 0.0;
  double learning_progress_ema_ = 0.0;

};

}  // namespace pulsar

#endif