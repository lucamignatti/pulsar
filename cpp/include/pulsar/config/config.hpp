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
  int value_hidden_dim = 256;
  int value_num_atoms = 51;
  float value_v_min = -10.0F;
  float value_v_max = 10.0F;
  int policy_hidden_dim = 0;
};

struct GoalMappingConfig {
  float goal = 0.0F;
  float kernel_sigma = 0.08F;
  float arena_max_distance = 8192.0F;
};

struct GoalCriticConfig {
  int horizon_H = 256;
  float gamma_g = 0.99F;
  int num_atoms = 51;
  float v_min = 0.0F;
  float v_max = 0.0F;
  float lambda_Zg = 1.0F;
};

struct ActorGoalConfig {
  float lambda_g = 0.1F;
};

struct ESLoraConfig {
  int rank = 4;
  float lora_alpha = 4.0F;
  int population_size = 16;
  float sigma_ES = 0.01F;
  float eta_ES = 0.003F;
  int es_interval = 100;
  int eval_episodes_per_member = 8;
  float alpha_g = 0.05F;
  float beta_KL = 0.01F;
  bool antithetic_sampling = true;
  bool update_norm_clip = true;
};

struct PPOConfig {
  int num_envs = 64;
  int collection_workers = 0;
  std::string init_checkpoint{};
  int rollout_length = 256;
  int minibatch_size = 32768;
  int update_epochs = 3;
  float clip_range = 0.2F;
  float entropy_coef = 0.01F;
  float value_coef = 1.0F;
  float gamma = 0.99F;
  float gae_lambda = 0.95F;
  float learning_rate = 3.0e-4F;
  float max_grad_norm = 1.0F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
  int max_rolling_checkpoints = 5;
  int sequence_length = 16;
  int burn_in = 0;
  bool use_adaptive_epsilon = true;
  float adaptive_epsilon_beta = 1.0F;
  float epsilon_min = 0.05F;
  float epsilon_max = 0.3F;
  bool use_confidence_weighting = true;
  std::string confidence_weight_type = "entropy";
  float confidence_weight_delta = 1.0e-6F;
  bool normalize_confidence_weights = false;
  bool synchronize_cuda_timing = false;
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
  int schema_version = 5;
  int obs_schema_version = 2;
  EnvConfig env{};
  OutcomeConfig outcome{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  PPOConfig ppo{};
  GoalMappingConfig goal_mapping{};
  GoalCriticConfig goal_critic{};
  ActorGoalConfig actor_goal{};
  ESLoraConfig es_lora{};
  SelfPlayLeagueConfig self_play_league{};
  WandbConfig wandb{};
};

struct CheckpointMetadata {
  int schema_version = 5;
  int obs_schema_version = 2;
  std::string config_hash{};
  std::string action_table_hash{};
  std::string architecture_name = "continuum_goal_conditioned";
  std::string device = "cpu";
  std::int64_t global_step = 0;
  std::int64_t update_index = 0;
  std::vector<std::string> critic_heads{};
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
void to_json(nlohmann::json& j, const GoalMappingConfig& value);
void from_json(const nlohmann::json& j, GoalMappingConfig& value);
void to_json(nlohmann::json& j, const GoalCriticConfig& value);
void from_json(const nlohmann::json& j, GoalCriticConfig& value);
void to_json(nlohmann::json& j, const ActorGoalConfig& value);
void from_json(const nlohmann::json& j, ActorGoalConfig& value);
void to_json(nlohmann::json& j, const ESLoraConfig& value);
void from_json(const nlohmann::json& j, ESLoraConfig& value);
void to_json(nlohmann::json& j, const PPOConfig& value);
void from_json(const nlohmann::json& j, PPOConfig& value);
void to_json(nlohmann::json& j, const SelfPlayLeagueConfig& value);
void from_json(const nlohmann::json& j, SelfPlayLeagueConfig& value);
void to_json(nlohmann::json& j, const WandbConfig& value);
void from_json(const nlohmann::json& j, WandbConfig& value);
void to_json(nlohmann::json& j, const ExperimentConfig& value);
void from_json(const nlohmann::json& j, ExperimentConfig& value);
void to_json(nlohmann::json& j, const CheckpointMetadata& value);
void from_json(const nlohmann::json& j, CheckpointMetadata& value);

void validate_experiment_config(const ExperimentConfig& config);

ExperimentConfig load_experiment_config(const std::string& path);
void save_experiment_config(const ExperimentConfig& config, const std::string& path);
std::string stable_json(const ExperimentConfig& config);
std::string stable_json(const CheckpointMetadata& metadata);
std::string hash_string(const std::string& value);
std::string config_hash(const ExperimentConfig& config);
std::string action_table_hash(const ActionTableConfig& config);

float compute_goal_critic_v_max(const GoalCriticConfig& cfg);

}  // namespace pulsar
