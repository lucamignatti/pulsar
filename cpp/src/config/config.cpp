#include "pulsar/config/config.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

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

void reject_removed_section(const json& j, const std::string& name) {
  if (j.contains(name)) {
    throw std::runtime_error("Removed training config section present: " + name);
  }
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

void to_json(json& j, const OutcomeConfig& value) {
  j = json{{"score", value.score}, {"concede", value.concede}, {"neutral", value.neutral}};
}

void from_json(const json& j, OutcomeConfig& value) {
  value.score = j.value("score", 1.0F);
  value.concede = j.value("concede", -1.0F);
  value.neutral = j.value("neutral", 0.0F);
}

void to_json(json& j, const ActionTableConfig& value) {
  j = json{{"builtin", value.builtin}, {"actions", value.actions}};
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
  value.mode = j.value("mode", std::string{"soccar"});
  value.collision_meshes_path = j.value("collision_meshes_path", std::string{"collision_meshes"});
  value.team_size = j.value("team_size", 2);
  value.tick_skip = j.value("tick_skip", 8);
  value.tick_rate = j.value("tick_rate", 120);
  value.max_episode_ticks = j.value("max_episode_ticks", 2250);
  value.no_touch_timeout_seconds = j.value("no_touch_timeout_seconds", 10.0F);
  value.spawn_opponents = j.value("spawn_opponents", true);
  value.randomize_kickoffs = j.value("randomize_kickoffs", true);
  value.seed = j.value("seed", static_cast<std::uint64_t>(0));
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
      {"action_embedding_dim", value.action_embedding_dim},
      {"future_latent_dim", value.future_latent_dim},
      {"future_horizon_count", value.future_horizon_count},
  };
}

void from_json(const json& j, ModelConfig& value) {
  value.observation_dim = j.value("observation_dim", 132);
  value.action_dim = j.value("action_dim", 90);
  value.use_layer_norm = j.value("use_layer_norm", true);
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
  value.action_embedding_dim = j.value("action_embedding_dim", 64);
  value.future_latent_dim = j.value("future_latent_dim", 128);
  value.future_horizon_count = j.value("future_horizon_count", 3);
}

void to_json(json& j, const LFPOConfig& value) {
  j = json{
      {"num_envs", value.num_envs},
      {"collection_workers", value.collection_workers},
      {"init_checkpoint", value.init_checkpoint},
      {"rollout_length", value.rollout_length},
      {"minibatch_size", value.minibatch_size},
      {"update_epochs", value.update_epochs},
      {"clip_range", value.clip_range},
      {"entropy_coef", value.entropy_coef},
      {"latent_loss_coef", value.latent_loss_coef},
      {"behavior_prior_coef", value.behavior_prior_coef},
      {"behavior_prior_decay_updates", value.behavior_prior_decay_updates},
      {"learning_rate", value.learning_rate},
      {"max_grad_norm", value.max_grad_norm},
      {"device", value.device},
      {"checkpoint_interval", value.checkpoint_interval},
      {"max_rolling_checkpoints", value.max_rolling_checkpoints},
      {"sequence_length", value.sequence_length},
      {"burn_in", value.burn_in},
      {"candidate_count", value.candidate_count},
      {"evaluator_update_interval", value.evaluator_update_interval},
      {"evaluator_target_update_interval", value.evaluator_target_update_interval},
      {"evaluator_target_ema_tau", value.evaluator_target_ema_tau},
      {"online_window_capacity", value.online_window_capacity},
  };
}

void from_json(const json& j, LFPOConfig& value) {
  value.num_envs = j.value("num_envs", 64);
  value.collection_workers = j.value("collection_workers", 0);
  value.init_checkpoint = j.value("init_checkpoint", std::string{});
  value.rollout_length = j.value("rollout_length", 256);
  value.minibatch_size = j.value("minibatch_size", 32768);
  value.update_epochs = j.value("update_epochs", 3);
  value.clip_range = j.value("clip_range", 0.2F);
  value.entropy_coef = j.value("entropy_coef", 0.01F);
  value.latent_loss_coef = j.value("latent_loss_coef", 1.0F);
  value.behavior_prior_coef = j.value("behavior_prior_coef", 0.0F);
  value.behavior_prior_decay_updates = j.value("behavior_prior_decay_updates", 0);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.device = j.value("device", std::string{"cpu"});
  value.checkpoint_interval = j.value("checkpoint_interval", 10);
  value.max_rolling_checkpoints = j.value("max_rolling_checkpoints", 5);
  value.sequence_length = j.value("sequence_length", 16);
  value.burn_in = j.value("burn_in", 0);
  value.candidate_count = j.value("candidate_count", 8);
  value.evaluator_update_interval = j.value("evaluator_update_interval", 4);
  value.evaluator_target_update_interval = j.value("evaluator_target_update_interval", 500);
  value.evaluator_target_ema_tau = j.value("evaluator_target_ema_tau", 0.01F);
  value.online_window_capacity = j.value("online_window_capacity", 64);
}

void to_json(json& j, const FutureEvaluatorConfig& value) {
  j = json{
      {"horizons", value.horizons},
      {"latent_dim", value.latent_dim},
      {"model_dim", value.model_dim},
      {"layers", value.layers},
      {"heads", value.heads},
      {"feedforward_dim", value.feedforward_dim},
      {"dropout", value.dropout},
      {"outcome_classes", value.outcome_classes},
      {"learning_rate", value.learning_rate},
      {"weight_decay", value.weight_decay},
      {"max_grad_norm", value.max_grad_norm},
      {"class_weights", value.class_weights},
  };
}

void from_json(const json& j, FutureEvaluatorConfig& value) {
  value.horizons = j.value("horizons", std::vector<int>{8, 32, 96});
  value.latent_dim = j.value("latent_dim", 128);
  value.model_dim = j.value("model_dim", 256);
  value.layers = j.value("layers", 4);
  value.heads = j.value("heads", 8);
  value.feedforward_dim = j.value("feedforward_dim", 1024);
  value.dropout = j.value("dropout", 0.0F);
  value.outcome_classes = j.value("outcome_classes", 3);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.weight_decay = j.value("weight_decay", 1.0e-6F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.class_weights = j.value("class_weights", std::vector<float>{1.0F, 1.0F, 0.25F});
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

void to_json(json& j, const OfflinePretrainingConfig& value) {
  j = json{
      {"evaluator_epochs", value.evaluator_epochs},
      {"actor_epochs", value.actor_epochs},
      {"sequence_length", value.sequence_length},
      {"behavior_cloning_learning_rate", value.behavior_cloning_learning_rate},
      {"actor_learning_rate", value.actor_learning_rate},
      {"evaluator_learning_rate", value.evaluator_learning_rate},
      {"weight_decay", value.weight_decay},
      {"label_smoothing", value.label_smoothing},
      {"behavior_cloning_loss_coef", value.behavior_cloning_loss_coef},
      {"latent_loss_coef", value.latent_loss_coef},
      {"max_grad_norm", value.max_grad_norm},
  };
}

void from_json(const json& j, OfflinePretrainingConfig& value) {
  value.evaluator_epochs = j.value("evaluator_epochs", 2);
  value.actor_epochs = j.value("actor_epochs", 2);
  value.sequence_length = j.value("sequence_length", 32);
  value.behavior_cloning_learning_rate = j.value("behavior_cloning_learning_rate", 3.0e-4F);
  value.actor_learning_rate = j.value("actor_learning_rate", 3.0e-4F);
  value.evaluator_learning_rate = j.value("evaluator_learning_rate", 3.0e-4F);
  value.weight_decay = j.value("weight_decay", 1.0e-6F);
  value.label_smoothing = j.value("label_smoothing", 0.0F);
  value.behavior_cloning_loss_coef = j.value("behavior_cloning_loss_coef", 1.0F);
  value.latent_loss_coef = j.value("latent_loss_coef", 1.0F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
}

void to_json(json& j, const SelfPlayLeagueConfig& value) {
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

void from_json(const json& j, SelfPlayLeagueConfig& value) {
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
      {"log_interval_seconds", value.log_interval_seconds},
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
  value.log_interval_seconds = j.value("log_interval_seconds", 30.0);
  value.tags = j.value("tags", std::vector<std::string>{});
}

void to_json(json& j, const ExperimentConfig& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"env", value.env},
      {"outcome", value.outcome},
      {"action_table", value.action_table},
      {"model", value.model},
      {"lfpo", value.lfpo},
      {"future_evaluator", value.future_evaluator},
      {"offline_dataset", value.offline_dataset},
      {"offline_pretraining", value.offline_pretraining},
      {"self_play_league", value.self_play_league},
      {"wandb", value.wandb},
  };
}

void from_json(const json& j, ExperimentConfig& value) {
  reject_removed_section(j, "reward");
  reject_removed_section(j, std::string{"p"} + "po");
  reject_removed_section(j, std::string{"next_"} + "goal_predictor");
  reject_removed_section(j, std::string{"value_"} + "pretraining");
  value.schema_version = j.value("schema_version", 4);
  value.obs_schema_version = j.value("obs_schema_version", 1);
  value.env = j.value("env", EnvConfig{});
  value.outcome = j.value("outcome", OutcomeConfig{});
  value.action_table = j.value("action_table", ActionTableConfig{});
  value.model = j.value("model", ModelConfig{});
  value.lfpo = j.value("lfpo", LFPOConfig{});
  value.future_evaluator = j.value("future_evaluator", FutureEvaluatorConfig{});
  value.offline_dataset = j.value("offline_dataset", OfflineDatasetConfig{});
  value.offline_pretraining = j.value("offline_pretraining", OfflinePretrainingConfig{});
  value.self_play_league = j.value("self_play_league", SelfPlayLeagueConfig{});
  value.wandb = j.value("wandb", WandbConfig{});
  value.model.future_latent_dim = value.future_evaluator.latent_dim;
  value.model.future_horizon_count = static_cast<int>(value.future_evaluator.horizons.size());
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
      {"future_evaluator_checkpoint", value.future_evaluator_checkpoint},
      {"future_evaluator_config_hash", value.future_evaluator_config_hash},
      {"future_evaluator_global_step", value.future_evaluator_global_step},
      {"future_evaluator_update_index", value.future_evaluator_update_index},
      {"future_evaluator_target_update_index", value.future_evaluator_target_update_index},
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
  value.future_evaluator_checkpoint = j.value("future_evaluator_checkpoint", std::string{});
  value.future_evaluator_config_hash = j.value("future_evaluator_config_hash", std::string{});
  value.future_evaluator_global_step = j.value("future_evaluator_global_step", static_cast<std::int64_t>(0));
  value.future_evaluator_update_index = j.value("future_evaluator_update_index", static_cast<std::int64_t>(0));
  value.future_evaluator_target_update_index =
      j.value("future_evaluator_target_update_index", static_cast<std::int64_t>(0));
}

ExperimentConfig load_experiment_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + path);
  }
  json j;
  input >> j;
  return j.get<ExperimentConfig>();
}

void save_experiment_config(const ExperimentConfig& config, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write config file: " + path);
  }
  json j = config;
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
  json j = materialized;
  return hash_string(j.dump());
}

}  // namespace pulsar
