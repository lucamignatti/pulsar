#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/core/parallel_executor.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/rl/interfaces.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace pulsar {

struct TrainerMetrics {
  double collection_agent_steps_per_second = 0.0;
  double update_agent_steps_per_second = 0.0;
  double overall_agent_steps_per_second = 0.0;
  double update_seconds = 0.0;
  double reward_mean = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double value_entropy = 0.0;
  double value_variance = 0.0;
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double reward_done_seconds = 0.0;
  double ngp_reward_seconds = 0.0;
  double rollout_append_seconds = 0.0;
  double gae_seconds = 0.0;
  double ppo_forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  std::map<std::string, double> elo_ratings{};
};

class PPOTrainer {
 public:
  PPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      ActionParserPtr action_parser,
      std::unique_ptr<SelfPlayManager> self_play_manager);

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");

 private:
  ContinuumState replay_state_until(std::int64_t start_step, const torch::Tensor& agent_indices);
  torch::Tensor sample_actions(
      const torch::Tensor& logits,
      const torch::Tensor& action_masks,
      bool deterministic,
      torch::Tensor* log_probs) const;
  std::vector<std::int64_t> actions_to_indices(const torch::Tensor& actions) const;
  torch::Tensor categorical_projection(const torch::Tensor& returns) const;
  torch::Tensor confidence_weights(const torch::Tensor& value_logits) const;
  torch::Tensor adaptive_epsilon(const torch::Tensor& value_logits) const;
  void maybe_initialize_from_checkpoint();
  void maybe_initialize_ngp_reward();
  torch::Tensor ngp_scalar(const torch::Tensor& logits) const;
  void save_checkpoint_to_directory(
      const std::filesystem::path& directory,
      std::int64_t global_step,
      std::int64_t update_index);
  TrainerMetrics update_policy();
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, std::int64_t update_index) const;
  void save_checkpoint(const std::string& checkpoint_dir, std::int64_t global_step, std::int64_t update_index);

  ExperimentConfig config_{};
  std::unique_ptr<BatchedRocketSimCollector> collector_{};
  ActionParserPtr action_parser_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  ControllerActionTable action_table_{};
  SharedActorCritic model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::optim::Adam optimizer_;
  RolloutStorage rollout_;
  torch::Device device_{torch::kCPU};
  SharedActorCritic ngp_model_{nullptr};
  ObservationNormalizer ngp_normalizer_{1};
  std::size_t total_agents_ = 0;
  ContinuumState collection_state_{};
  ContinuumState opponent_collection_state_{};
  ContinuumState ngp_collection_state_{};
  std::vector<ControllerState> host_actions_{};
  bool use_pinned_host_buffers_ = false;
  bool use_ngp_reward_ = false;
  bool use_shaped_reward_ = true;
  double best_reward_mean_ = -1.0e30;
};

}  // namespace pulsar

#endif
