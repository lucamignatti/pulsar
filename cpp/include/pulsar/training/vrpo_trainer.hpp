#pragma once

#ifdef PULSAR_HAS_TORCH

#include <atomic>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/core/work_stealing_thread_pool.hpp"
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/vrpo_actor.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "pulsar/training/self_play_manager.hpp"
#include "pulsar/training/gradient_surgery.hpp"
#include "pulsar/training/predictor_trainer.hpp"
#include "pulsar/training/gcrl_trainer.hpp"
#include "pulsar/training/optimizers.hpp"

#ifdef PULSAR_HAS_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/autocast_mode.h>
#endif

namespace pulsar {

struct TrainerMetrics {
  double collection_agent_steps_per_second = 0.0;
  double update_agent_steps_per_second = 0.0;
  double overall_agent_steps_per_second = 0.0;
  double update_seconds = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double grad_norm = 0.0;
  double policy_approx_kl = 0.0;
  double policy_clip_fraction = 0.0;
  double policy_log_ratio_abs_max = 0.0;
  int64_t nonfinite_loss_skips = 0;
  int64_t nonfinite_grad_norm_skips = 0;
  int64_t kl_guard_skips = 0;
  int64_t grad_norm_guard_skips = 0;
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
  double forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  double process_rss_mb = 0.0;
  double process_peak_rss_mb = 0.0;
  double cgroup_memory_current_mb = 0.0;
  double cgroup_memory_limit_mb = 0.0;
  double cuda_memory_allocated_mb = 0.0;
  double cuda_memory_reserved_mb = 0.0;
  double cuda_max_memory_allocated_mb = 0.0;
  double cuda_max_memory_reserved_mb = 0.0;
  int64_t cuda_alloc_retries = 0;
  int64_t cuda_ooms = 0;
  double total_reward_mean = 0.0;
  double gameplay_reward_mean = 0.0;
  double mechanic_reward_mean = 0.0;
  double sampled_value_win_mean = 0.0;
  int64_t rollout_steps = 0;
  int64_t completed_episodes = 0;
  int64_t scored_episodes = 0;
  int64_t conceded_episodes = 0;
  int64_t neutral_episodes = 0;
  int64_t no_touch_episodes = 0;
  int64_t truncated_episodes = 0;

  double goal_critic_loss = 0.0;
  double mean_goal_score = 0.0;
  double mean_sampled_goal_distance = 0.0;
  double mean_goal_distance = 0.0;
  double min_goal_distance = 0.0;
  double ball_proximity_rate = 0.0;
  int64_t goals_scored = 0;
  int64_t goals_conceded = 0;

  double es_fitness_mean = 0.0;
  double es_fitness_std = 0.0;
  double es_fitness_best = 0.0;
  double es_reward_mean = 0.0;
  double es_winrate_mean = 0.0;
  double es_kl_mean = 0.0;
  double es_update_norm = 0.0;
  double es_lora_a_norm = 0.0;
  double es_lora_b_norm = 0.0;
  double es_seconds = 0.0;
  double es_eval_seconds = 0.0;
  double es_agent_steps_per_second = 0.0;
  double es_effective_population = 0.0;
  double es_virtual_population_waves = 0.0;
  double es_eval_shards = 0.0;
  double es_policy_update_norm = 0.0;
  double scored_episode_rate = 0.0;
  double conceded_episode_rate = 0.0;
  double neutral_episode_rate = 0.0;
  double no_touch_episode_rate = 0.0;
  double truncated_episode_rate = 0.0;
  double touch_episode_rate = 0.0;
  double multi_touch_episode_rate = 0.0;
  double effective_entropy_coef = 0.0;
  int self_play_snapshot_count = 0;

  double q_critic_loss = 0.0;
  double advantage_std = 0.0;

  std::map<std::string, double> predictor_loss{};
  std::map<std::string, double> predictor_delta{};
  std::map<std::string, double> predictor_positive{};
  std::map<std::string, double> predictor_active{};

  std::map<std::string, double> elo_ratings{};
  std::map<std::string, double> mode_touch_rates{};
  std::map<std::string, double> mode_multi_touch_rates{};
  std::map<std::string, double> mode_scored_rates{};
  std::map<std::string, int> mode_completed_episodes{};
};

struct TrainerBenchmarkMetrics {
  int updates = 0;
  std::int64_t agent_steps = 0;
  double total_seconds = 0.0;
  double collection_seconds = 0.0;
  double update_seconds = 0.0;
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
  double forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double grad_norm = 0.0;
  int es_updates = 0;
  double es_seconds = 0.0;
  double es_eval_seconds = 0.0;
  double es_agent_steps_per_second = 0.0;
  double es_effective_population = 0.0;
  double es_virtual_population_waves = 0.0;
  double es_eval_shards = 0.0;
  double es_policy_update_norm = 0.0;
};

class VRPOTrainer {
 public:
  VRPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);
  VRPOTrainer(
      ExperimentConfig config,
      std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);
  ~VRPOTrainer();

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");
  TrainerBenchmarkMetrics benchmark(int updates);
  [[nodiscard]] std::int64_t model_parameter_count() const;

 private:
  void maybe_initialize_from_checkpoint();
  void save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index, const std::string& wandb_run_id) const;
  void save_checkpoint_with_metadata(const std::filesystem::path& directory, const CheckpointMetadata& metadata) const;
  void save_training_state(const std::filesystem::path& path) const;
  void load_training_state(const std::filesystem::path& path);
  void prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const;
  void collect_rollout(
      RolloutStorage& dest,
      TrainerMetrics& metrics,
      std::int64_t* collected_agent_steps,
      VRPOActor rollout_actor,
      ObservationNormalizer& normalizer);
  TrainerMetrics update_actor(RolloutStorage& rollout, int update_index);
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, int update_index, const std::string& wandb_run_id) const;

  void run_es_lora_update(int update_index, TrainerMetrics& metrics);
  std::pair<torch::Tensor, torch::Tensor> compute_es_deltas();

  void rebuild_collectors();
  void sync_self_play_assignments_to_collectors();

  struct ESPopulationFitness {
    std::vector<float> fitness{};
    std::vector<float> reward{};
    std::vector<float> winrate{};
    std::vector<float> kl{};
    std::int64_t agent_steps = 0;
  };

  ESPopulationFitness evaluate_es_population(
      const torch::Tensor& A_stack,
      const torch::Tensor& B_stack,
      int update_index,
      int wave_index);

  ExperimentConfig config_{};
  std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  ControllerActionTable action_table_{};
  VRPOActor actor_{nullptr};
  VRPOActor actor_snapshot_{nullptr};
  std::vector<VRPOActor> compute_actors_;
  ObservationNormalizer actor_normalizer_;
  MagSGD actor_optimizer_;
  std::unique_ptr<WorkStealingThreadPool> task_pool_;
  PopArtNormalizer q_normalizer_{};
  torch::Device device_{torch::kCPU};
  std::vector<torch::Device> compute_devices_{};
  std::vector<torch::Device> shard_devices_{};
  std::vector<VRPOActor> collection_actors_;
  RolloutStorage rollout_;
  RolloutStorage rollout_B_;
  std::filesystem::path run_output_root_{};
  bool log_initialization_ = true;
  std::int64_t resumed_global_step_ = 0;
  std::int64_t resumed_update_index_ = 0;
  std::size_t total_agents_ = 0;
  std::vector<std::int64_t> shard_agent_offsets_{};
  std::vector<torch::Tensor> shard_action_buffers_cpu_{};
  bool use_pinned_host_buffers_ = false;
  bool benchmark_progress_ = false;
  std::atomic<bool> es_deltas_ready_{false};
  torch::Tensor es_delta_A_;
  torch::Tensor es_delta_B_;

  // Modular trainers
  std::unique_ptr<PredictorTrainer> predictor_trainer_{nullptr};
  std::unique_ptr<GCRLTrainer> gcrl_trainer_{nullptr};

  std::vector<torch::Tensor> predictor_ema_loss_{};
  std::vector<torch::Tensor> predictor_delta_{};
  torch::Tensor predictor_horizons_device_;

#ifdef PULSAR_HAS_CUDA
  std::vector<c10::cuda::CUDAStream> shard_collection_streams_;
  std::optional<c10::cuda::CUDAStream> training_stream_;
#endif
};

}  // namespace pulsar

#endif
