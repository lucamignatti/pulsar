#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/future_evaluator.hpp"
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/offline_dataset.hpp"
#include "pulsar/training/online_outcome_replay_buffer.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "pulsar/training/self_play_manager.hpp"

namespace pulsar {

struct TrainerMetrics {
  double collection_agent_steps_per_second = 0.0;
  double update_agent_steps_per_second = 0.0;
  double overall_agent_steps_per_second = 0.0;
  double update_seconds = 0.0;
  double policy_loss = 0.0;
  double latent_loss = 0.0;
  double evaluator_loss = 0.0;
  double evaluator_online_loss = 0.0;
  double evaluator_anchor_loss = 0.0;
  double evaluator_outcome_loss = 0.0;
  double evaluator_delta_loss = 0.0;
  double entropy = 0.0;
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
  double lfpo_forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  std::int64_t online_outcome_samples = 0;
  std::int64_t online_outcome_trajectories = 0;
  std::int64_t evaluator_online_samples = 0;
  std::int64_t evaluator_anchor_samples = 0;
  std::int64_t evaluator_outcome_samples = 0;
  std::int64_t evaluator_delta_samples = 0;
  std::int64_t evaluator_target_update_index = 0;
  std::map<std::string, double> elo_ratings{};
};

class LFPOTrainer {
 public:
  LFPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");

 private:
  torch::Tensor sample_actions(
      const torch::Tensor& logits,
      const torch::Tensor& action_masks,
      bool deterministic,
      torch::Tensor* log_probs) const;
  torch::Tensor sample_candidate_actions(
      const torch::Tensor& logits,
      const torch::Tensor& action_masks,
      torch::Tensor* log_probs) const;
  void maybe_initialize_from_checkpoint();
  void load_future_evaluator_checkpoint(const std::filesystem::path& base);
  void save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index) const;
  void prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const;
  void update_evaluator_from_self_play(int update_index, TrainerMetrics* metrics);
  void update_target_evaluator(int update_index);
  TrainerMetrics run_update(std::int64_t* global_step, int update_index);
  TrainerMetrics update_actor();
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, int update_index) const;

  ExperimentConfig config_{};
  std::unique_ptr<BatchedRocketSimCollector> collector_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  ControllerActionTable action_table_{};
  LatentFutureActor actor_{nullptr};
  FutureEvaluator evaluator_{nullptr};
  FutureEvaluator target_evaluator_{nullptr};
  ObservationNormalizer actor_normalizer_;
  ObservationNormalizer evaluator_normalizer_;
  torch::optim::Adam actor_optimizer_;
  torch::optim::AdamW evaluator_optimizer_;
  RolloutStorage rollout_;
  std::unique_ptr<OnlineOutcomeReplayBuffer> outcome_buffer_{};
  std::unique_ptr<OfflineTensorDataset> evaluator_anchor_dataset_{};
  torch::Device device_{torch::kCPU};
  std::filesystem::path run_output_root_{};
  bool log_initialization_ = true;
  std::int64_t resumed_global_step_ = 0;
  std::int64_t resumed_update_index_ = 0;
  std::int64_t evaluator_target_update_index_ = 0;
  std::size_t total_agents_ = 0;
  ContinuumState collection_state_{};
  ContinuumState opponent_collection_state_{};
  torch::Tensor collection_trajectory_ids_{};
  std::int64_t next_trajectory_id_ = 0;
  bool use_pinned_host_buffers_ = false;
};

}  // namespace pulsar

#endif
