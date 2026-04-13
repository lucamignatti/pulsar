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

void to_json(json& j, const RewardConfig& value) {
  j = json{
      {"ngp_checkpoint", value.ngp_checkpoint},
      {"ngp_scale", value.ngp_scale},
  };
}

void from_json(const json& j, RewardConfig& value) {
  value.ngp_checkpoint = j.value("ngp_checkpoint", std::string{});
  value.ngp_scale = j.value("ngp_scale", 1.0F);
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
      {"action_dim", value.action_dim},
      {"use_layer_norm", value.use_layer_norm},
      {"encoder_dim", value.encoder_dim},
      {"workspace_dim", value.workspace_dim},
      {"stm_slots", value.stm_slots},
      {"stm_key_dim", value.stm_key_dim},
      {"stm_value_dim", value.stm_value_dim},
      {"ltm_slots", value.ltm_slots},
      {"ltm_dim", value.ltm_dim},
      {"controller_dim", value.controller_dim},
      {"consolidation_stride", value.consolidation_stride},
      {"retired_decay", value.retired_decay},
  };
}

void from_json(const json& j, ModelConfig& value) {
  value.observation_dim = j.at("observation_dim").get<int>();
  value.action_dim = j.at("action_dim").get<int>();
  value.use_layer_norm = j.at("use_layer_norm").get<bool>();
  value.encoder_dim = j.value("encoder_dim", 512);
  value.workspace_dim = j.value("workspace_dim", 512);
  value.stm_slots = j.value("stm_slots", 48);
  value.stm_key_dim = j.value("stm_key_dim", 128);
  value.stm_value_dim = j.value("stm_value_dim", 128);
  value.ltm_slots = j.value("ltm_slots", 32);
  value.ltm_dim = j.value("ltm_dim", 128);
  value.controller_dim = j.value("controller_dim", 512);
  value.consolidation_stride = j.value("consolidation_stride", 8);
  value.retired_decay = j.value("retired_decay", 0.96F);
}

void to_json(json& j, const PPOConfig::SelfPlayConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"opponent_probability", value.opponent_probability},
      {"snapshot_interval_updates", value.snapshot_interval_updates},
      {"max_snapshots", value.max_snapshots},
      {"training_opponent_policy", value.training_opponent_policy},
      {"eval_interval_updates", value.eval_interval_updates},
      {"eval_num_envs", value.eval_num_envs},
      {"eval_matches_per_snapshot", value.eval_matches_per_snapshot},
      {"eval_policy", value.eval_policy},
      {"elo_initial", value.elo_initial},
      {"elo_k", value.elo_k},
  };
}

void from_json(const json& j, PPOConfig::SelfPlayConfig& value) {
  value.enabled = j.value("enabled", false);
  value.opponent_probability = j.value("opponent_probability", 0.0F);
  value.snapshot_interval_updates = j.value("snapshot_interval_updates", 10);
  value.max_snapshots = j.value("max_snapshots", 8);
  value.training_opponent_policy = j.value("training_opponent_policy", std::string{"stochastic"});
  value.eval_interval_updates = j.value("eval_interval_updates", 10);
  value.eval_num_envs = j.value("eval_num_envs", 8);
  value.eval_matches_per_snapshot = j.value("eval_matches_per_snapshot", 4);
  value.eval_policy = j.value("eval_policy", std::string{"deterministic"});
  value.elo_initial = j.value("elo_initial", 1000.0F);
  value.elo_k = j.value("elo_k", 32.0F);
}

void to_json(json& j, const PPOConfig& value) {
  j = json{
      {"num_envs", value.num_envs},
      {"collection_workers", value.collection_workers},
      {"init_checkpoint", value.init_checkpoint},
      {"rollout_length", value.rollout_length},
      {"minibatch_size", value.minibatch_size},
      {"epochs", value.epochs},
      {"gamma", value.gamma},
      {"gae_lambda", value.gae_lambda},
      {"clip_range", value.clip_range},
      {"entropy_coef", value.entropy_coef},
      {"value_coef", value.value_coef},
      {"learning_rate", value.learning_rate},
      {"max_grad_norm", value.max_grad_norm},
      {"device", value.device},
      {"checkpoint_interval", value.checkpoint_interval},
      {"sequence_length", value.sequence_length},
      {"burn_in", value.burn_in},
      {"value_v_min", value.value_v_min},
      {"value_v_max", value.value_v_max},
      {"value_num_atoms", value.value_num_atoms},
      {"use_adaptive_epsilon", value.use_adaptive_epsilon},
      {"adaptive_epsilon_beta", value.adaptive_epsilon_beta},
      {"epsilon_min", value.epsilon_min},
      {"epsilon_max", value.epsilon_max},
      {"use_confidence_weighting", value.use_confidence_weighting},
      {"confidence_weight_type", value.confidence_weight_type},
      {"confidence_weight_delta", value.confidence_weight_delta},
      {"normalize_confidence_weights", value.normalize_confidence_weights},
      {"self_play", value.self_play},
  };
}

void from_json(const json& j, PPOConfig& value) {
  value.num_envs = j.at("num_envs").get<int>();
  value.collection_workers = j.value("collection_workers", 0);
  value.init_checkpoint = j.value("init_checkpoint", std::string{});
  value.rollout_length = j.at("rollout_length").get<int>();
  value.minibatch_size = j.at("minibatch_size").get<int>();
  value.epochs = j.at("epochs").get<int>();
  value.gamma = j.at("gamma").get<float>();
  value.gae_lambda = j.at("gae_lambda").get<float>();
  value.clip_range = j.at("clip_range").get<float>();
  value.entropy_coef = j.at("entropy_coef").get<float>();
  value.value_coef = j.at("value_coef").get<float>();
  value.learning_rate = j.at("learning_rate").get<float>();
  value.max_grad_norm = j.at("max_grad_norm").get<float>();
  value.device = j.at("device").get<std::string>();
  value.checkpoint_interval = j.at("checkpoint_interval").get<int>();
  value.sequence_length = j.value("sequence_length", 16);
  value.burn_in = j.value("burn_in", 4);
  value.value_v_min = j.value("value_v_min", -10.0F);
  value.value_v_max = j.value("value_v_max", 10.0F);
  value.value_num_atoms = j.value("value_num_atoms", 51);
  value.use_adaptive_epsilon = j.value("use_adaptive_epsilon", true);
  value.adaptive_epsilon_beta = j.value("adaptive_epsilon_beta", 1.0F);
  value.epsilon_min = j.value("epsilon_min", 0.05F);
  value.epsilon_max = j.value("epsilon_max", 0.3F);
  value.use_confidence_weighting = j.value("use_confidence_weighting", true);
  value.confidence_weight_type = j.value("confidence_weight_type", std::string{"entropy"});
  value.confidence_weight_delta = j.value("confidence_weight_delta", 1.0e-6F);
  value.normalize_confidence_weights = j.value("normalize_confidence_weights", false);
  value.self_play = j.value("self_play", PPOConfig::SelfPlayConfig{});
}

void to_json(json& j, const OfflineDatasetConfig& value) {
  j = json{
      {"train_manifest", value.train_manifest},
      {"val_manifest", value.val_manifest},
      {"batch_size", value.batch_size},
      {"shuffle", value.shuffle},
      {"seed", value.seed},
  };
}

void from_json(const json& j, OfflineDatasetConfig& value) {
  value.train_manifest = j.value("train_manifest", std::string{});
  value.val_manifest = j.value("val_manifest", std::string{});
  value.batch_size = j.value("batch_size", 4096);
  value.shuffle = j.value("shuffle", true);
  value.seed = j.value("seed", static_cast<std::uint64_t>(0));
}

void to_json(json& j, const BehaviorCloningConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"epochs", value.epochs},
      {"learning_rate", value.learning_rate},
      {"weight_decay", value.weight_decay},
      {"label_smoothing", value.label_smoothing},
      {"max_grad_norm", value.max_grad_norm},
      {"sequence_length", value.sequence_length},
  };
}

void from_json(const json& j, BehaviorCloningConfig& value) {
  value.enabled = j.value("enabled", true);
  value.epochs = j.value("epochs", 10);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.weight_decay = j.value("weight_decay", 1.0e-6F);
  value.label_smoothing = j.value("label_smoothing", 0.0F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.sequence_length = j.value("sequence_length", 32);
}

void to_json(json& j, const NextGoalPredictorConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"epochs", value.epochs},
      {"learning_rate", value.learning_rate},
      {"weight_decay", value.weight_decay},
      {"label_smoothing", value.label_smoothing},
      {"max_grad_norm", value.max_grad_norm},
      {"class_weights", value.class_weights},
  };
}

void from_json(const json& j, NextGoalPredictorConfig& value) {
  value.enabled = j.value("enabled", true);
  value.epochs = j.value("epochs", 10);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.weight_decay = j.value("weight_decay", 1.0e-6F);
  value.label_smoothing = j.value("label_smoothing", 0.0F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.class_weights = j.value("class_weights", std::vector<float>{1.0F, 1.0F, 0.25F});
}

void to_json(json& j, const WandbConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"project", value.project},
      {"entity", value.entity},
      {"run_name", value.run_name},
      {"group", value.group},
      {"job_type", value.job_type},
      {"dir", value.dir},
      {"mode", value.mode},
      {"python_executable", value.python_executable},
      {"script_path", value.script_path},
      {"tags", value.tags},
  };
}

void from_json(const json& j, WandbConfig& value) {
  value.enabled = j.value("enabled", false);
  value.project = j.value("project", std::string{"pulsar"});
  value.entity = j.value("entity", std::string{});
  value.run_name = j.value("run_name", std::string{});
  value.group = j.value("group", std::string{});
  value.job_type = j.value("job_type", std::string{});
  value.dir = j.value("dir", std::string{});
  value.mode = j.value("mode", std::string{"online"});
  value.python_executable = j.value("python_executable", std::string{"python3"});
  value.script_path = j.value("script_path", std::string{"scripts/wandb_stream.py"});
  value.tags = j.value("tags", std::vector<std::string>{});
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
      {"offline_dataset", value.offline_dataset},
      {"behavior_cloning", value.behavior_cloning},
      {"next_goal_predictor", value.next_goal_predictor},
      {"wandb", value.wandb},
  };
}

void from_json(const json& j, ExperimentConfig& value) {
  value.schema_version = j.at("schema_version").get<int>();
  value.obs_schema_version = j.at("obs_schema_version").get<int>();
  value.env = j.at("env").get<EnvConfig>();
  if (j.contains("reward")) {
    value.reward = j.at("reward").get<RewardConfig>();
  }
  value.action_table = j.at("action_table").get<ActionTableConfig>();
  value.model = j.at("model").get<ModelConfig>();
  value.ppo = j.at("ppo").get<PPOConfig>();
  if (j.contains("offline_dataset")) {
    value.offline_dataset = j.at("offline_dataset").get<OfflineDatasetConfig>();
  }
  if (j.contains("behavior_cloning")) {
    value.behavior_cloning = j.at("behavior_cloning").get<BehaviorCloningConfig>();
  }
  if (j.contains("next_goal_predictor")) {
    value.next_goal_predictor = j.at("next_goal_predictor").get<NextGoalPredictorConfig>();
  }
  if (j.contains("wandb")) {
    value.wandb = j.at("wandb").get<WandbConfig>();
  }
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
