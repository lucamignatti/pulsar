#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
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
#include "pulsar/training/rollout_storage.hpp"

namespace pulsar {

struct TrainerMetrics {
  double collection_fps = 0.0;
  double update_seconds = 0.0;
  double reward_mean = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double value_entropy = 0.0;
  double value_variance = 0.0;
};

class PPOTrainer {
 public:
  PPOTrainer(
      ExperimentConfig config,
      std::vector<TransitionEnginePtr> engines,
      ObsBuilderPtr obs_builder,
      ActionParserPtr action_parser,
      RewardFunctionPtr reward_fn,
      DoneConditionPtr done_condition);

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");

 private:
  torch::Tensor collect_observations();
  void step_envs(std::span<const std::int64_t> action_indices, std::int64_t global_step);
  ContinuumState replay_state_until(std::int64_t start_step, const torch::Tensor& agent_indices);
  torch::Tensor sample_actions(const torch::Tensor& logits, torch::Tensor* log_probs) const;
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
  std::vector<TransitionEnginePtr> engines_{};
  ObsBuilderPtr obs_builder_{};
  ActionParserPtr action_parser_{};
  RewardFunctionPtr reward_fn_{};
  DoneConditionPtr done_condition_{};
  ControllerActionTable action_table_{};
  SharedActorCritic model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::optim::Adam optimizer_;
  RolloutStorage rollout_;
  torch::Device device_{torch::kCPU};
  ParallelExecutor collection_executor_;
  SharedActorCritic ngp_model_{nullptr};
  ObservationNormalizer ngp_normalizer_{1};
  std::vector<std::size_t> agent_offsets_{};
  std::size_t total_agents_ = 0;
  ContinuumState collection_state_{};
  ContinuumState ngp_collection_state_{};
  std::vector<ControllerState> host_actions_{};
  std::vector<std::uint8_t> host_terminated_{};
  std::vector<std::uint8_t> host_truncated_{};
  torch::Tensor host_obs_;
  torch::Tensor host_post_step_obs_;
  torch::Tensor host_episode_starts_;
  torch::Tensor host_rewards_;
  torch::Tensor host_dones_;
  bool use_pinned_host_buffers_ = false;
  bool use_ngp_reward_ = false;
  bool use_shaped_reward_ = true;
  double best_reward_mean_ = -1.0e30;
};

}  // namespace pulsar

#endif
