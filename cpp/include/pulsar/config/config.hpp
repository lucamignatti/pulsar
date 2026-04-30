#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "pulsar/core/types.hpp"

namespace pulsar {

struct OutcomeConfig {
  float score = 1.0F;
  float concede = -1.0F;
  float neutral = 0.0F;
};

struct ActionTableConfig {
  std::string builtin = "rlgym_lookup_v1";
  std::vector<ControllerState> actions{};
};

struct EnvConfig {
  std::string mode = "soccar";
  std::string collision_meshes_path = "collision_meshes";
  int team_size = 2;
  int tick_skip = 8;
  int tick_rate = 120;
  int max_episode_ticks = 2250;
  float no_touch_timeout_seconds = 10.0F;
  bool spawn_opponents = true;
  bool randomize_kickoffs = true;
  std::uint64_t seed = 0;
};

struct ModelConfig {
  int observation_dim = 132;
  int action_dim = 90;
  bool use_layer_norm = true;
  int encoder_dim = 512;
  int workspace_dim = 512;
  int stm_slots = 48;
  int stm_key_dim = 128;
  int stm_value_dim = 128;
  int ltm_slots = 32;
  int ltm_dim = 128;
  int controller_dim = 512;
  int consolidation_stride = 8;
  float retired_decay = 0.96F;
  int action_embedding_dim = 64;
  int future_latent_dim = 128;
  int future_horizon_count = 3;
};

struct LFPOConfig {
  int num_envs = 64;
  int collection_workers = 0;
  std::string init_checkpoint{};
  int rollout_length = 256;
  int minibatch_size = 32768;
  int update_epochs = 3;
  float clip_range = 0.2F;
  float entropy_coef = 0.01F;
  float latent_loss_coef = 1.0F;
  float behavior_prior_coef = 0.0F;
  int behavior_prior_decay_updates = 0;
  float learning_rate = 3.0e-4F;
  float max_grad_norm = 1.0F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
  int max_rolling_checkpoints = 5;
  int sequence_length = 16;
  int burn_in = 0;
  int candidate_count = 8;
  int evaluator_update_interval = 4;
  int evaluator_target_update_interval = 1;
  float evaluator_target_ema_tau = 0.01F;
  int online_window_capacity = 64;
  float evaluator_anchor_ratio = 0.25F;
};

struct FutureEvaluatorConfig {
  std::vector<int> horizons{8, 32, 96};
  int latent_dim = 128;
  int model_dim = 256;
  int layers = 4;
  int heads = 8;
  int feedforward_dim = 1024;
  float dropout = 0.0F;
  int outcome_classes = 3;
  float learning_rate = 3.0e-4F;
  float weight_decay = 1.0e-6F;
  float max_grad_norm = 1.0F;
  std::vector<float> class_weights{1.0F, 1.0F, 0.25F};
  float future_delta_loss_coef = 0.1F;
};

struct OfflineDatasetConfig {
  std::string train_manifest = "";
  std::string val_manifest = "";
  int batch_size = 4096;
  bool shuffle = true;
  std::uint64_t seed = 0;
};

struct OfflinePretrainingConfig {
  int evaluator_epochs = 2;
  int actor_epochs = 2;
  int sequence_length = 32;
  float behavior_cloning_learning_rate = 3.0e-4F;
  float actor_learning_rate = 3.0e-4F;
  float evaluator_learning_rate = 3.0e-4F;
  float weight_decay = 1.0e-6F;
  float label_smoothing = 0.0F;
  float behavior_cloning_loss_coef = 1.0F;
  float latent_loss_coef = 1.0F;
  float max_grad_norm = 1.0F;
};

struct SelfPlayLeagueConfig {
  bool enabled = false;
  float opponent_probability = 0.0F;
  int snapshot_interval_updates = 10;
  int max_snapshots = 8;
  std::string training_opponent_policy = "stochastic";
  int eval_interval_updates = 10;
  int eval_num_envs = 8;
  int eval_matches_per_snapshot = 4;
  std::string eval_policy = "deterministic";
  float elo_initial = 1000.0F;
  float elo_k = 32.0F;
};

struct WandbConfig {
  bool enabled = false;
  std::string project = "pulsar";
  std::string entity{};
  std::string run_name{};
  std::string group{};
  std::string job_type{};
  std::string dir{};
  std::string mode = "online";
  std::string python_executable = "python3";
  std::string script_path = "scripts/wandb_stream.py";
  double log_interval_seconds = 30.0;
  std::vector<std::string> tags{};
};

struct ExperimentConfig {
  int schema_version = 4;
  int obs_schema_version = 1;
  EnvConfig env{};
  OutcomeConfig outcome{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  LFPOConfig lfpo{};
  FutureEvaluatorConfig future_evaluator{};
  OfflineDatasetConfig offline_dataset{};
  OfflinePretrainingConfig offline_pretraining{};
  SelfPlayLeagueConfig self_play_league{};
  WandbConfig wandb{};
};

struct CheckpointMetadata {
  int schema_version = 4;
  int obs_schema_version = 1;
  std::string config_hash{};
  std::string action_table_hash{};
  std::string architecture_name = "lfpo_continuum";
  std::string device = "cpu";
  std::int64_t global_step = 0;
  std::int64_t update_index = 0;
  std::string future_evaluator_checkpoint{};
  std::string future_evaluator_config_hash{};
  std::int64_t future_evaluator_global_step = 0;
  std::int64_t future_evaluator_update_index = 0;
  std::int64_t future_evaluator_target_update_index = 0;
};

void to_json(nlohmann::json& j, const ControllerState& value);
void from_json(const nlohmann::json& j, ControllerState& value);

void to_json(nlohmann::json& j, const OutcomeConfig& value);
void from_json(const nlohmann::json& j, OutcomeConfig& value);
void to_json(nlohmann::json& j, const ActionTableConfig& value);
void from_json(const nlohmann::json& j, ActionTableConfig& value);
void to_json(nlohmann::json& j, const EnvConfig& value);
void from_json(const nlohmann::json& j, EnvConfig& value);
void to_json(nlohmann::json& j, const ModelConfig& value);
void from_json(const nlohmann::json& j, ModelConfig& value);
void to_json(nlohmann::json& j, const LFPOConfig& value);
void from_json(const nlohmann::json& j, LFPOConfig& value);
void to_json(nlohmann::json& j, const FutureEvaluatorConfig& value);
void from_json(const nlohmann::json& j, FutureEvaluatorConfig& value);
void to_json(nlohmann::json& j, const OfflineDatasetConfig& value);
void from_json(const nlohmann::json& j, OfflineDatasetConfig& value);
void to_json(nlohmann::json& j, const OfflinePretrainingConfig& value);
void from_json(const nlohmann::json& j, OfflinePretrainingConfig& value);
void to_json(nlohmann::json& j, const SelfPlayLeagueConfig& value);
void from_json(const nlohmann::json& j, SelfPlayLeagueConfig& value);
void to_json(nlohmann::json& j, const WandbConfig& value);
void from_json(const nlohmann::json& j, WandbConfig& value);
void to_json(nlohmann::json& j, const ExperimentConfig& value);
void from_json(const nlohmann::json& j, ExperimentConfig& value);
void to_json(nlohmann::json& j, const CheckpointMetadata& value);
void from_json(const nlohmann::json& j, CheckpointMetadata& value);

ExperimentConfig load_experiment_config(const std::string& path);
void save_experiment_config(const ExperimentConfig& config, const std::string& path);
std::string stable_json(const ExperimentConfig& config);
std::string stable_json(const CheckpointMetadata& metadata);
std::string hash_string(const std::string& value);
std::string config_hash(const ExperimentConfig& config);
std::string action_table_hash(const ActionTableConfig& config);

}  // namespace pulsar
