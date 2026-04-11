#include "pulsar/config/config.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pulsar/rl/action_table.hpp"

namespace pulsar {
namespace {

using nlohmann::json;

template <typename T>
std::string dump_stable(const T& value) {
  json j = value;
  return j.dump(-1, ' ', false, json::error_handler_t::strict);
}

}  // namespace

using nlohmann::json;

void to_json(json& j, const ControllerState& value) {
  j = json{
      {"throttle", value.throttle},
      {"steer", value.steer},
      {"yaw", value.yaw},
      {"pitch", value.pitch},
      {"roll", value.roll},
      {"jump", value.jump},
      {"boost", value.boost},
      {"handbrake", value.handbrake},
  };
}

void from_json(const json& j, ControllerState& value) {
  value.throttle = j.at("throttle").get<float>();
  value.steer = j.at("steer").get<float>();
  value.yaw = j.at("yaw").get<float>();
  value.pitch = j.at("pitch").get<float>();
  value.roll = j.at("roll").get<float>();
  value.jump = j.at("jump").get<bool>();
  value.boost = j.at("boost").get<bool>();
  value.handbrake = j.at("handbrake").get<bool>();
}

void to_json(json& j, const RewardTermConfig& value) {
  j = json{{"name", value.name}, {"weight", value.weight}};
}

void from_json(const json& j, RewardTermConfig& value) {
  value.name = j.at("name").get<std::string>();
  value.weight = j.at("weight").get<float>();
}

void to_json(json& j, const RewardConfig& value) {
  j = json{
      {"terms", value.terms},
      {"team_spirit", value.team_spirit},
      {"opponent_scale", value.opponent_scale},
  };
}

void from_json(const json& j, RewardConfig& value) {
  value.terms = j.at("terms").get<std::vector<RewardTermConfig>>();
  value.team_spirit = j.value("team_spirit", 0.0F);
  value.opponent_scale = j.value("opponent_scale", 0.0F);
}

void to_json(json& j, const ActionTableConfig& value) {
  j = json{
      {"builtin", value.builtin},
      {"actions", value.actions},
  };
}

void from_json(const json& j, ActionTableConfig& value) {
  value.builtin = j.value("builtin", std::string{});
  value.actions = j.value("actions", std::vector<ControllerState>{});
}

void to_json(json& j, const EnvConfig& value) {
  j = json{
      {"mode", value.mode},
      {"collision_meshes_path", value.collision_meshes_path},
      {"team_size", value.team_size},
      {"tick_skip", value.tick_skip},
      {"tick_rate", value.tick_rate},
      {"max_episode_ticks", value.max_episode_ticks},
      {"no_touch_timeout_seconds", value.no_touch_timeout_seconds},
      {"spawn_opponents", value.spawn_opponents},
      {"randomize_kickoffs", value.randomize_kickoffs},
      {"seed", value.seed},
  };
}

void from_json(const json& j, EnvConfig& value) {
  value.mode = j.at("mode").get<std::string>();
  value.collision_meshes_path = j.value("collision_meshes_path", std::string{"collision_meshes"});
  value.team_size = j.at("team_size").get<int>();
  value.tick_skip = j.at("tick_skip").get<int>();
  value.tick_rate = j.at("tick_rate").get<int>();
  value.max_episode_ticks = j.at("max_episode_ticks").get<int>();
  value.no_touch_timeout_seconds = j.value("no_touch_timeout_seconds", 10.0F);
  value.spawn_opponents = j.at("spawn_opponents").get<bool>();
  value.randomize_kickoffs = j.at("randomize_kickoffs").get<bool>();
  value.seed = j.at("seed").get<std::uint64_t>();
}

void to_json(json& j, const ModelConfig& value) {
  j = json{
      {"observation_dim", value.observation_dim},
      {"hidden_sizes", value.hidden_sizes},
      {"action_dim", value.action_dim},
      {"use_layer_norm", value.use_layer_norm},
  };
}

void from_json(const json& j, ModelConfig& value) {
  value.observation_dim = j.at("observation_dim").get<int>();
  value.hidden_sizes = j.at("hidden_sizes").get<std::vector<int>>();
  value.action_dim = j.at("action_dim").get<int>();
  value.use_layer_norm = j.at("use_layer_norm").get<bool>();
}

void to_json(json& j, const PPOConfig& value) {
  j = json{
      {"num_envs", value.num_envs},
      {"rollout_length", value.rollout_length},
      {"minibatch_size", value.minibatch_size},
      {"epochs", value.epochs},
      {"gamma", value.gamma},
      {"gae_lambda", value.gae_lambda},
      {"clip_range", value.clip_range},
      {"value_clip_range", value.value_clip_range},
      {"entropy_coef", value.entropy_coef},
      {"value_coef", value.value_coef},
      {"learning_rate", value.learning_rate},
      {"max_grad_norm", value.max_grad_norm},
      {"target_kl", value.target_kl},
      {"device", value.device},
      {"checkpoint_interval", value.checkpoint_interval},
  };
}

void from_json(const json& j, PPOConfig& value) {
  value.num_envs = j.at("num_envs").get<int>();
  value.rollout_length = j.at("rollout_length").get<int>();
  value.minibatch_size = j.at("minibatch_size").get<int>();
  value.epochs = j.at("epochs").get<int>();
  value.gamma = j.at("gamma").get<float>();
  value.gae_lambda = j.at("gae_lambda").get<float>();
  value.clip_range = j.at("clip_range").get<float>();
  value.value_clip_range = j.at("value_clip_range").get<float>();
  value.entropy_coef = j.at("entropy_coef").get<float>();
  value.value_coef = j.at("value_coef").get<float>();
  value.learning_rate = j.at("learning_rate").get<float>();
  value.max_grad_norm = j.at("max_grad_norm").get<float>();
  value.target_kl = j.at("target_kl").get<float>();
  value.device = j.at("device").get<std::string>();
  value.checkpoint_interval = j.at("checkpoint_interval").get<int>();
}

void to_json(json& j, const ExperimentConfig& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"env", value.env},
      {"reward", value.reward},
      {"action_table", value.action_table},
      {"model", value.model},
      {"ppo", value.ppo},
  };
}

void from_json(const json& j, ExperimentConfig& value) {
  value.schema_version = j.at("schema_version").get<int>();
  value.obs_schema_version = j.at("obs_schema_version").get<int>();
  value.env = j.at("env").get<EnvConfig>();
  value.reward = j.at("reward").get<RewardConfig>();
  value.action_table = j.at("action_table").get<ActionTableConfig>();
  value.model = j.at("model").get<ModelConfig>();
  value.ppo = j.at("ppo").get<PPOConfig>();
}

void to_json(json& j, const CheckpointMetadata& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"config_hash", value.config_hash},
      {"action_table_hash", value.action_table_hash},
      {"architecture_name", value.architecture_name},
      {"device", value.device},
      {"global_step", value.global_step},
      {"update_index", value.update_index},
  };
}

void from_json(const json& j, CheckpointMetadata& value) {
  value.schema_version = j.at("schema_version").get<int>();
  value.obs_schema_version = j.at("obs_schema_version").get<int>();
  value.config_hash = j.at("config_hash").get<std::string>();
  value.action_table_hash = j.at("action_table_hash").get<std::string>();
  value.architecture_name = j.at("architecture_name").get<std::string>();
  value.device = j.at("device").get<std::string>();
  value.global_step = j.at("global_step").get<std::int64_t>();
  value.update_index = j.at("update_index").get<std::int64_t>();
}

ExperimentConfig load_experiment_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  nlohmann::json j;
  input >> j;
  return j.get<ExperimentConfig>();
}

void save_experiment_config(const ExperimentConfig& config, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write config file: " + path);
  }

  nlohmann::json j = config;
  output << std::setw(2) << j << '\n';
}

std::string stable_json(const ExperimentConfig& config) {
  return dump_stable(config);
}

std::string stable_json(const CheckpointMetadata& metadata) {
  return dump_stable(metadata);
}

std::string hash_string(const std::string& value) {
  std::uint64_t hashed = 1469598103934665603ULL;
  for (const unsigned char ch : value) {
    hashed ^= ch;
    hashed *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << hashed;
  return out.str();
}

std::string config_hash(const ExperimentConfig& config) {
  return hash_string(stable_json(config));
}

std::string action_table_hash(const ActionTableConfig& config) {
  ActionTableConfig materialized = config;
  if (materialized.actions.empty() && !materialized.builtin.empty()) {
    materialized = ControllerActionTable::make_builtin(materialized.builtin);
  }
  nlohmann::json j = materialized;
  return hash_string(j.dump());
}

}  // namespace pulsar
