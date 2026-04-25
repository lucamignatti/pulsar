#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <map>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <optional>
#include <string>
#include <thread>
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
#include "pulsar/training/online_ngp_dataset_writer.hpp"
#include "pulsar/training/online_ngp_replay_buffer.hpp"
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
  double event_reward_mean = 0.0;
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
  double done_reset_seconds = 0.0;
  double reward_model_seconds = 0.0;
  double rollout_append_seconds = 0.0;
  double gae_seconds = 0.0;
  double ppo_forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  std::int64_t ngp_promotion_index = 0;
  std::int64_t ngp_promoted_global_step = 0;
  std::int64_t ngp_source_global_step = 0;
  std::int64_t ngp_source_update_index = 0;
  std::int64_t ngp_online_samples_written = 0;
  std::int64_t ngp_online_trajectories_written = 0;
  std::string ngp_label{};
  std::string ngp_checkpoint{};
  std::string ngp_config_hash{};
  std::map<std::string, double> elo_ratings{};
};

class PPOTrainer {
 public:
  PPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);
  ~PPOTrainer();

  TrainerMetrics benchmark(int warmup_updates, int measured_updates);
  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");

 private:
  struct NGPRefreshTask {
    int update_index = 0;
    std::int64_t global_step = 0;
    std::vector<NGPTrajectory> online_train{};
    std::vector<NGPTrajectory> online_val{};
    std::int64_t target_anchor_samples = 0;
    std::uint64_t anchor_seed = 0;
  };

  struct NGPRefreshResult {
    int update_index = 0;
    std::int64_t global_step = 0;
    std::int64_t online_train_samples = 0;
    std::int64_t online_val_samples = 0;
    double active_anchor_loss = 0.0;
    double active_recent_loss = 0.0;
    double candidate_anchor_loss = 0.0;
    double candidate_recent_loss = 0.0;
    double recent_loss_improvement = 0.0;
    double anchor_loss_regression = 0.0;
    bool promote = false;
  };

  struct ModelSnapshot {
    SharedActorCritic model{nullptr};
    ObservationNormalizer normalizer{1};
  };

  struct RefreshStateSnapshot {
    std::optional<ModelSnapshot> candidate_ngp{};
    std::optional<std::string> candidate_trunk_optimizer_bytes{};
    std::optional<std::string> candidate_ngp_head_optimizer_bytes{};
    std::shared_ptr<OnlineNGPReplayBuffer> replay_buffer{};
    std::int64_t last_ngp_promotion_update = 0;
  };

  struct CheckpointSnapshot {
    CheckpointMetadata metadata{};
    ModelSnapshot policy{};
    std::optional<std::string> optimizer_bytes{};
    std::optional<ModelSnapshot> active_ngp{};
    std::optional<RefreshStateSnapshot> refresh_state{};
  };

  struct CandidateCheckpointSnapshot {
    ModelSnapshot candidate{};
    std::optional<std::string> trunk_optimizer_bytes{};
    std::optional<std::string> ngp_head_optimizer_bytes{};
    std::int64_t global_step = 0;
    std::int64_t update_index = 0;
  };

  enum class PersistenceKind {
    RollingCheckpoint,
    BestCheckpoint,
    FinalCheckpoint,
    CandidateCheckpoint,
  };

  struct PersistenceRequest {
    PersistenceKind kind = PersistenceKind::RollingCheckpoint;
    std::filesystem::path directory{};
    std::shared_ptr<CheckpointSnapshot> checkpoint{};
    std::shared_ptr<CandidateCheckpointSnapshot> candidate{};
  };

  torch::Tensor sample_actions(
      const torch::Tensor& logits,
      const torch::Tensor& action_masks,
      bool deterministic,
      torch::Tensor* log_probs) const;
  torch::Tensor actions_to_cpu(const torch::Tensor& actions) const;
  torch::Tensor categorical_projection(const torch::Tensor& returns) const;
  torch::Tensor confidence_weights(const torch::Tensor& value_logits) const;
  torch::Tensor adaptive_epsilon(const torch::Tensor& value_logits) const;
  void maybe_initialize_from_checkpoint();
  void load_ngp_reward_checkpoint(
      const std::string& checkpoint_path,
      const std::string& configured_label,
      std::int64_t promotion_index,
      std::int64_t promoted_global_step);
  void maybe_initialize_ngp_reward();
  void maybe_promote_ngp_reward(std::int64_t global_step, int update_index, const std::string& checkpoint_dir);
  void maybe_refresh_ngp_candidate_in_process(
      std::int64_t global_step,
      int update_index,
      const std::string& checkpoint_dir);
  void maybe_initialize_in_process_ngp_refresh(const std::filesystem::path& init_checkpoint_dir);
  void ensure_candidate_ngp_initialized();
  void save_model_snapshot(
      SharedActorCritic model,
      const ObservationNormalizer& normalizer,
      const std::filesystem::path& directory,
      std::int64_t global_step,
      std::int64_t update_index) const;
  [[nodiscard]] std::shared_ptr<CheckpointSnapshot> capture_checkpoint_snapshot(
      std::int64_t global_step,
      std::int64_t update_index);
  [[nodiscard]] std::shared_ptr<CandidateCheckpointSnapshot> capture_candidate_checkpoint_snapshot(
      std::int64_t global_step,
      std::int64_t update_index);
  void write_refresh_state_snapshot(
      const RefreshStateSnapshot& snapshot,
      const std::filesystem::path& directory) const;
  void write_checkpoint_snapshot(
      const CheckpointSnapshot& snapshot,
      const std::filesystem::path& directory) const;
  void write_candidate_checkpoint_snapshot(
      const CandidateCheckpointSnapshot& snapshot,
      const std::filesystem::path& directory) const;
  void load_in_process_ngp_refresh_state(const std::filesystem::path& directory);
  void start_ngp_refresh_worker();
  void stop_ngp_refresh_worker();
  void start_persistence_worker();
  void flush_persistence_worker();
  void stop_persistence_worker();
  void enqueue_persistence_request(PersistenceRequest request);
  void persistence_worker_loop();
  void rethrow_persistence_error_if_any();
  void maybe_schedule_ngp_refresh_task(std::int64_t global_step, int update_index);
  void maybe_collect_ngp_refresh_result(const std::string& checkpoint_dir);
  void ngp_refresh_worker_loop();
  void train_candidate_on_trajectories(const std::vector<NGPTrajectory>& trajectories, int epochs);
  NGPRefreshResult evaluate_candidate_refresh(
      SharedActorCritic active_model,
      const ObservationNormalizer& active_normalizer,
      const std::vector<NGPTrajectory>& recent_val,
      const std::vector<NGPTrajectory>& anchor_val) const;
  std::pair<double, double> evaluate_ngp_trajectories(
      SharedActorCritic model,
      const ObservationNormalizer& normalizer,
      const std::vector<NGPTrajectory>& trajectories) const;
  torch::Tensor ngp_scalar(const torch::Tensor& logits) const;
  torch::Tensor compute_rollout_ngp_rewards(
      const torch::Tensor& obs_seq,
      const torch::Tensor& episode_starts_seq);
  TrainerMetrics run_update(std::int64_t* global_step, int current_update_index);
  TrainerMetrics update_policy();
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, std::int64_t update_index) const;

  ExperimentConfig config_{};
  std::unique_ptr<BatchedRocketSimCollector> collector_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  ControllerActionTable action_table_{};
  SharedActorCritic model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::optim::Adam optimizer_;
  RolloutStorage rollout_;
  torch::Device device_{torch::kCPU};
  torch::Device ngp_refresh_device_{torch::kCPU};
  std::filesystem::path run_output_root_{};
  bool log_initialization_ = true;
  SharedActorCritic ngp_model_{nullptr};
  ObservationNormalizer ngp_normalizer_{1};
  std::unique_ptr<OnlineNGPDatasetWriter> online_ngp_dataset_writer_{};
  std::unique_ptr<OnlineNGPReplayBuffer> ngp_replay_buffer_{};
  AnchorManifest anchor_train_manifest_{};
  AnchorManifest anchor_val_manifest_{};
  SharedActorCritic candidate_ngp_model_{nullptr};
  ObservationNormalizer candidate_ngp_normalizer_{1};
  std::vector<torch::Tensor> candidate_trunk_parameters_{};
  std::vector<torch::Tensor> candidate_ngp_head_parameters_{};
  std::unique_ptr<torch::optim::AdamW> candidate_trunk_optimizer_{};
  std::unique_ptr<torch::optim::AdamW> candidate_ngp_head_optimizer_{};
  std::int64_t last_ngp_promotion_update_ = 0;
  std::mutex candidate_mutex_{};
  std::thread ngp_refresh_thread_{};
  std::mutex ngp_refresh_mutex_{};
  std::condition_variable ngp_refresh_cv_{};
  bool ngp_refresh_worker_stop_ = false;
  bool ngp_refresh_task_pending_ = false;
  bool ngp_refresh_task_in_progress_ = false;
  bool ngp_refresh_result_ready_ = false;
  NGPRefreshTask pending_ngp_refresh_task_{};
  NGPRefreshResult latest_ngp_refresh_result_{};
  std::string active_ngp_checkpoint_{};
  std::string active_ngp_label_{};
  std::string active_ngp_config_hash_{};
  std::thread persistence_thread_{};
  std::mutex persistence_mutex_{};
  std::condition_variable persistence_cv_{};
  std::condition_variable persistence_idle_cv_{};
  std::deque<PersistenceRequest> persistence_requests_{};
  std::exception_ptr persistence_error_{};
  bool persistence_worker_stop_ = false;
  bool persistence_request_in_progress_ = false;
  std::int64_t active_ngp_global_step_ = 0;
  std::int64_t active_ngp_update_index_ = 0;
  std::int64_t active_ngp_promotion_index_ = 0;
  std::int64_t active_ngp_promoted_global_step_ = 0;
  std::int64_t resumed_global_step_ = 0;
  std::int64_t resumed_update_index_ = 0;
  std::size_t total_agents_ = 0;
  ContinuumState collection_state_{};
  ContinuumState opponent_collection_state_{};
  ContinuumState ngp_collection_state_{};
  bool use_pinned_host_buffers_ = false;
  double best_reward_mean_ = -1.0e30;
};

}  // namespace pulsar

#endif
