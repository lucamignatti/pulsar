#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "pulsar/core/types.hpp"

namespace pulsar {

struct RewardTermConfig {
  std::string name{};
  float weight = 0.0F;
};

struct RewardConfig {
  std::vector<RewardTermConfig> terms{};
  float team_spirit = 0.0F;
  float opponent_scale = 0.0F;
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
  std::vector<int> hidden_sizes{1024, 1024, 512, 512};
  int action_dim = 90;
  bool use_layer_norm = true;
};

struct PPOConfig {
  int num_envs = 64;
  int rollout_length = 256;
  int minibatch_size = 2048;
  int epochs = 4;
  float gamma = 0.99F;
  float gae_lambda = 0.95F;
  float clip_range = 0.2F;
  float value_clip_range = 0.2F;
  float entropy_coef = 0.01F;
  float value_coef = 1.0F;
  float learning_rate = 3.0e-4F;
  float max_grad_norm = 1.0F;
  float target_kl = 0.03F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
};

struct ExperimentConfig {
  int schema_version = 1;
  int obs_schema_version = 1;
  EnvConfig env{};
  RewardConfig reward{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  PPOConfig ppo{};
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

void to_json(nlohmann::json& j, const RewardTermConfig& value);
void from_json(const nlohmann::json& j, RewardTermConfig& value);
void to_json(nlohmann::json& j, const RewardConfig& value);
void from_json(const nlohmann::json& j, RewardConfig& value);
void to_json(nlohmann::json& j, const ActionTableConfig& value);
void from_json(const nlohmann::json& j, ActionTableConfig& value);
void to_json(nlohmann::json& j, const EnvConfig& value);
void from_json(const nlohmann::json& j, EnvConfig& value);
void to_json(nlohmann::json& j, const ModelConfig& value);
void from_json(const nlohmann::json& j, ModelConfig& value);
void to_json(nlohmann::json& j, const PPOConfig& value);
void from_json(const nlohmann::json& j, PPOConfig& value);
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
