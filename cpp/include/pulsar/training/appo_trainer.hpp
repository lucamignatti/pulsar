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
#include "pulsar/logging/wandb_logger.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/curriculum.hpp"
#include "pulsar/training/rollout_storage.hpp"
#include "pulsar/training/self_play_manager.hpp"

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
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
  double forward_backward_seconds = 0.0;
  double optimizer_step_seconds = 0.0;
  double self_play_eval_seconds = 0.0;
  double total_reward_mean = 0.0;
  double gameplay_reward_mean = 0.0;
  double mechanic_reward_mean = 0.0;
  double sampled_value_win_mean = 0.0;
  int64_t rollout_steps = 0;
  int64_t completed_episodes = 0;
  int64_t scored_episodes = 0;

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
  double es_winrate_mean = 0.0;
  double es_kl_mean = 0.0;
  double es_update_norm = 0.0;
  double es_lora_a_norm = 0.0;
  double es_lora_b_norm = 0.0;
  double es_seconds = 0.0;
  double scored_episode_rate = 0.0;
  double touch_episode_rate = 0.0;
  double effective_entropy_coef = 0.0;

  std::map<std::string, double> elo_ratings{};
  std::map<std::string, double> mode_touch_rates{};
  std::map<std::string, double> mode_scored_rates{};
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
};

class APPOTrainer {
 public:
  APPOTrainer(
      ExperimentConfig config,
      std::unique_ptr<BatchedRocketSimCollector> collector,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);
  APPOTrainer(
      ExperimentConfig config,
      std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors,
      std::unique_ptr<SelfPlayManager> self_play_manager,
      std::filesystem::path run_output_root = {},
      bool log_initialization = true);
  ~APPOTrainer();

  void train(int updates, const std::string& checkpoint_dir, const std::string& config_path = "");
  TrainerBenchmarkMetrics benchmark(int updates);
  [[nodiscard]] std::int64_t model_parameter_count() const;

 private:
  void maybe_initialize_from_checkpoint();
  void save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index, const std::string& wandb_run_id) const;
  void save_training_state(const std::filesystem::path& path) const;
  void load_training_state(const std::filesystem::path& path);
  void prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const;
  void collect_rollout(
      RolloutStorage& dest,
      TrainerMetrics& metrics,
      std::int64_t* collected_agent_steps,
      PPOActor rollout_actor,
      ObservationNormalizer& normalizer);
  TrainerMetrics update_actor(RolloutStorage& rollout);
  CheckpointMetadata make_checkpoint_metadata(std::int64_t global_step, int update_index, const std::string& wandb_run_id) const;

  void run_es_lora_update(int update_index, TrainerMetrics& metrics);
  std::pair<torch::Tensor, torch::Tensor> compute_es_deltas();

  void apply_curriculum_to_collectors();
  void apply_curriculum_lr();
  void rebuild_collectors();

  struct ESPopulationFitness {
    std::vector<float> fitness{};
    std::vector<float> winrate{};
    std::vector<float> kl{};
  };

  ESPopulationFitness evaluate_es_population(
      const torch::Tensor& A_stack,
      const torch::Tensor& B_stack,
      int update_index);

  ExperimentConfig config_{};
  std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors_{};
  std::unique_ptr<SelfPlayManager> self_play_manager_{};
  Curriculum curriculum_{config_.curriculum};
  ControllerActionTable action_table_{};
  PPOActor actor_{nullptr};
  PPOActor actor_snapshot_{nullptr};
  ObservationNormalizer actor_normalizer_;
  torch::optim::Adam actor_optimizer_;
  torch::Device device_{torch::kCPU};
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
  std::deque<double> recent_scored_rates_{};
  static constexpr int kRecentScoredRateWindow = 20;
#ifdef PULSAR_HAS_CUDA
  std::vector<c10::cuda::CUDAStream> shard_collection_streams_;
  std::optional<c10::cuda::CUDAStream> training_stream_;
#endif
};

}  // namespace pulsar

#endif
