#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "pulsar/core/types.hpp"

namespace pulsar {

struct RewardConfig {
  std::string ngp_checkpoint{};
  float ngp_scale = 1.0F;
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
};

struct PPOConfig {
  struct SelfPlayConfig {
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

  int num_envs = 64;
  int collection_workers = 0;
  std::string init_checkpoint{};
  int rollout_length = 256;
  int minibatch_size = 2048;
  int epochs = 4;
  float gamma = 0.99F;
  float gae_lambda = 0.95F;
  float clip_range = 0.2F;
  float entropy_coef = 0.01F;
  float value_coef = 1.0F;
  float learning_rate = 3.0e-4F;
  float max_grad_norm = 1.0F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
  int sequence_length = 16;
  int burn_in = 4;
  float value_v_min = -10.0F;
  float value_v_max = 10.0F;
  int value_num_atoms = 51;
  bool use_adaptive_epsilon = true;
  float adaptive_epsilon_beta = 1.0F;
  float epsilon_min = 0.05F;
  float epsilon_max = 0.3F;
  bool use_confidence_weighting = true;
  std::string confidence_weight_type = "entropy";
  float confidence_weight_delta = 1.0e-6F;
  bool normalize_confidence_weights = false;
  SelfPlayConfig self_play{};
};

struct OfflineDatasetConfig {
  std::string train_manifest = "";
  std::string val_manifest = "";
  int batch_size = 4096;
  bool shuffle = true;
  std::uint64_t seed = 0;
};

struct BehaviorCloningConfig {
  bool enabled = true;
  int epochs = 10;
  float learning_rate = 3.0e-4F;
  float weight_decay = 1.0e-6F;
  float label_smoothing = 0.0F;
  float max_grad_norm = 1.0F;
  int sequence_length = 32;
};

struct NextGoalPredictorConfig {
  bool enabled = true;
  int epochs = 10;
  float learning_rate = 3.0e-4F;
  float weight_decay = 1.0e-6F;
  float label_smoothing = 0.0F;
  float max_grad_norm = 1.0F;
  std::vector<float> class_weights{1.0F, 1.0F, 0.25F};
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
  std::vector<std::string> tags{};
};

struct ExperimentConfig {
  int schema_version = 1;
  int obs_schema_version = 1;
  EnvConfig env{};
  RewardConfig reward{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  PPOConfig ppo{};
  OfflineDatasetConfig offline_dataset{};
  BehaviorCloningConfig behavior_cloning{};
  NextGoalPredictorConfig next_goal_predictor{};
  WandbConfig wandb{};
};

struct CheckpointMetadata {
  int schema_version = 1;
  int obs_schema_version = 1;
  std::string config_hash{};
  std::string action_table_hash{};
  std::string architecture_name = "shared_actor_critic";
  std::string device = "cpu";
  std::int64_t global_step = 0;
  std::int64_t update_index = 0;
};

void to_json(nlohmann::json& j, const ControllerState& value);
void from_json(const nlohmann::json& j, ControllerState& value);

void to_json(nlohmann::json& j, const RewardConfig& value);
void from_json(const nlohmann::json& j, RewardConfig& value);
void to_json(nlohmann::json& j, const ActionTableConfig& value);
void from_json(const nlohmann::json& j, ActionTableConfig& value);
void to_json(nlohmann::json& j, const EnvConfig& value);
void from_json(const nlohmann::json& j, EnvConfig& value);
void to_json(nlohmann::json& j, const ModelConfig& value);
void from_json(const nlohmann::json& j, ModelConfig& value);
void to_json(nlohmann::json& j, const PPOConfig::SelfPlayConfig& value);
void from_json(const nlohmann::json& j, PPOConfig::SelfPlayConfig& value);
void to_json(nlohmann::json& j, const PPOConfig& value);
void from_json(const nlohmann::json& j, PPOConfig& value);
void to_json(nlohmann::json& j, const OfflineDatasetConfig& value);
void from_json(const nlohmann::json& j, OfflineDatasetConfig& value);
void to_json(nlohmann::json& j, const BehaviorCloningConfig& value);
void from_json(const nlohmann::json& j, BehaviorCloningConfig& value);
void to_json(nlohmann::json& j, const NextGoalPredictorConfig& value);
void from_json(const nlohmann::json& j, NextGoalPredictorConfig& value);
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
