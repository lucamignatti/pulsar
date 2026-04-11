#pragma once

#ifdef PULSAR_HAS_TORCH

#include <memory>
#include <string>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/core/parallel_executor.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/rl/interfaces.hpp"
#include "pulsar/training/rollout_storage.hpp"

namespace pulsar {

struct TrainerMetrics {
  double collection_fps = 0.0;
  double update_seconds = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
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

  void train(int updates, const std::string& checkpoint_dir);

 private:
  torch::Tensor collect_observations();
  void step_envs(std::span<const std::int64_t> action_indices, std::int64_t global_step);
  torch::Tensor sample_actions(const torch::Tensor& logits, torch::Tensor* log_probs) const;
  std::vector<std::int64_t> actions_to_indices(const torch::Tensor& actions) const;
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
  std::vector<std::size_t> agent_offsets_{};
  std::size_t total_agents_ = 0;
  std::vector<ControllerState> host_actions_{};
  std::vector<std::uint8_t> host_terminated_{};
  std::vector<std::uint8_t> host_truncated_{};
  torch::Tensor host_obs_;
  torch::Tensor host_rewards_;
  torch::Tensor host_dones_;
  bool use_pinned_host_buffers_ = false;
};

}  // namespace pulsar

#endif
