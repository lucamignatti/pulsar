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
      {"value_hidden_dim", value.value_hidden_dim},
      {"value_num_atoms", value.value_num_atoms},
      {"value_v_min", value.value_v_min},
      {"value_v_max", value.value_v_max},
      {"policy_hidden_dim", value.policy_hidden_dim},
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
  value.value_hidden_dim = j.value("value_hidden_dim", 256);
  value.value_num_atoms = j.value("value_num_atoms", 51);
  value.value_v_min = j.value("value_v_min", -10.0F);
  value.value_v_max = j.value("value_v_max", 10.0F);
  value.policy_hidden_dim = j.value("policy_hidden_dim", 0);
}

void to_json(json& j, const PPOConfig& value) {
  j = json{
      {"num_envs", value.num_envs},
      {"collection_workers", value.collection_workers},
      {"init_checkpoint", value.init_checkpoint},
      {"rollout_length", value.rollout_length},
      {"minibatch_size", value.minibatch_size},
      {"update_epochs", value.update_epochs},
      {"clip_range", value.clip_range},
      {"entropy_coef", value.entropy_coef},
      {"value_coef", value.value_coef},
      {"gamma", value.gamma},
      {"gae_lambda", value.gae_lambda},
      {"learning_rate", value.learning_rate},
      {"max_grad_norm", value.max_grad_norm},
      {"device", value.device},
      {"checkpoint_interval", value.checkpoint_interval},
      {"max_rolling_checkpoints", value.max_rolling_checkpoints},
      {"sequence_length", value.sequence_length},
      {"burn_in", value.burn_in},
      {"use_adaptive_epsilon", value.use_adaptive_epsilon},
      {"adaptive_epsilon_beta", value.adaptive_epsilon_beta},
      {"epsilon_min", value.epsilon_min},
      {"epsilon_max", value.epsilon_max},
      {"use_confidence_weighting", value.use_confidence_weighting},
      {"confidence_weight_type", value.confidence_weight_type},
      {"confidence_weight_delta", value.confidence_weight_delta},
      {"normalize_confidence_weights", value.normalize_confidence_weights},
      {"synchronize_cuda_timing", value.synchronize_cuda_timing},
  };
}

void from_json(const json& j, PPOConfig& value) {
  value.num_envs = j.value("num_envs", 64);
  value.collection_workers = j.value("collection_workers", 0);
  value.init_checkpoint = j.value("init_checkpoint", std::string{});
  value.rollout_length = j.value("rollout_length", 256);
  value.minibatch_size = j.value("minibatch_size", 32768);
  value.update_epochs = j.value("update_epochs", 3);
  value.clip_range = j.value("clip_range", 0.2F);
  value.entropy_coef = j.value("entropy_coef", 0.01F);
  value.value_coef = j.value("value_coef", 1.0F);
  value.gamma = j.value("gamma", 0.99F);
  value.gae_lambda = j.value("gae_lambda", 0.95F);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.device = j.value("device", std::string{"cpu"});
  value.checkpoint_interval = j.value("checkpoint_interval", 10);
  value.max_rolling_checkpoints = j.value("max_rolling_checkpoints", 5);
  value.sequence_length = j.value("sequence_length", 16);
  value.burn_in = j.value("burn_in", 0);
  value.use_adaptive_epsilon = j.value("use_adaptive_epsilon", true);
  value.adaptive_epsilon_beta = j.value("adaptive_epsilon_beta", 1.0F);
  value.epsilon_min = j.value("epsilon_min", 0.05F);
  value.epsilon_max = j.value("epsilon_max", 0.3F);
  value.use_confidence_weighting = j.value("use_confidence_weighting", true);
  value.confidence_weight_type = j.value("confidence_weight_type", std::string{"entropy"});
  value.confidence_weight_delta = j.value("confidence_weight_delta", 1.0e-6F);
  value.normalize_confidence_weights = j.value("normalize_confidence_weights", false);
  value.synchronize_cuda_timing = j.value("synchronize_cuda_timing", false);
}

void to_json(json& j, const OfflineDatasetConfig& value) {
  j = json{
      {"train_manifest", value.train_manifest},
      {"val_manifest", value.val_manifest},
      {"batch_size", value.batch_size},
      {"shuffle", value.shuffle},
      {"seed", value.seed},
      {"allow_pickle", value.allow_pickle},
  };
}

void from_json(const json& j, OfflineDatasetConfig& value) {
  value.train_manifest = j.value("train_manifest", std::string{});
  value.val_manifest = j.value("val_manifest", std::string{});
  value.batch_size = j.value("batch_size", 4096);
  value.shuffle = j.value("shuffle", true);
  value.seed = j.value("seed", static_cast<std::uint64_t>(0));
  value.allow_pickle = j.value("allow_pickle", false);
}

void to_json(json& j, const BehaviorCloningConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"epochs", value.epochs},
      {"sequence_length", value.sequence_length},
      {"learning_rate", value.learning_rate},
      {"weight_decay", value.weight_decay},
      {"label_smoothing", value.label_smoothing},
      {"max_grad_norm", value.max_grad_norm},
  };
}

void from_json(const json& j, BehaviorCloningConfig& value) {
  value.enabled = j.value("enabled", true);
  value.epochs = j.value("epochs", 10);
  value.sequence_length = j.value("sequence_length", 32);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.weight_decay = j.value("weight_decay", 1.0e-6F);
  value.label_smoothing = j.value("label_smoothing", 0.0F);
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

void to_json(json& j, const CriticHeadConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"value_hidden_dim", value.value_hidden_dim},
      {"value_num_atoms", value.value_num_atoms},
      {"value_v_min", value.value_v_min},
      {"value_v_max", value.value_v_max},
  };
}

void from_json(const json& j, CriticHeadConfig& value) {
  value.enabled = j.value("enabled", true);
  value.value_hidden_dim = j.value("value_hidden_dim", 0);
  value.value_num_atoms = j.value("value_num_atoms", 0);
  value.value_v_min = j.value("value_v_min", 0.0F);
  value.value_v_max = j.value("value_v_max", 0.0F);
}

void to_json(json& j, const CriticConfig& value) {
  j = json{
      {"extrinsic", value.extrinsic},
      {"curiosity", value.curiosity},
      {"learning_progress", value.learning_progress},
      {"controllability", value.controllability},
  };
}

void from_json(const json& j, CriticConfig& value) {
  value.extrinsic = j.value("extrinsic", CriticHeadConfig{});
  value.extrinsic.enabled = true;
  value.curiosity = j.value("curiosity", CriticHeadConfig{});
  value.learning_progress = j.value("learning_progress", CriticHeadConfig{});
  value.controllability = j.value("controllability", CriticHeadConfig{false});
}

void to_json(json& j, const ForwardModelConfig& value) {
  j = json{
      {"hidden_dim", value.hidden_dim},
      {"num_layers", value.num_layers},
      {"learning_rate", value.learning_rate},
  };
}

void from_json(const json& j, ForwardModelConfig& value) {
  value.hidden_dim = j.value("hidden_dim", 256);
  value.num_layers = j.value("num_layers", 2);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
}

void to_json(json& j, const InverseModelConfig& value) {
  j = json{
      {"hidden_dim", value.hidden_dim},
      {"num_layers", value.num_layers},
      {"learning_rate", value.learning_rate},
  };
}

void from_json(const json& j, InverseModelConfig& value) {
  value.hidden_dim = j.value("hidden_dim", 256);
  value.num_layers = j.value("num_layers", 2);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
}

void to_json(json& j, const IntrinsicRewardConfig& value) {
  j = json{
      {"curiosity_weight", value.curiosity_weight},
      {"learning_progress_weight", value.learning_progress_weight},
      {"controllability_weight", value.controllability_weight},
      {"novelty_ema_decay", value.novelty_ema_decay},
      {"learning_progress_ema_decay", value.learning_progress_ema_decay},
      {"use_controllability_gate", value.use_controllability_gate},
  };
}

void from_json(const json& j, IntrinsicRewardConfig& value) {
  value.curiosity_weight = j.value("curiosity_weight", 1.0F);
  value.learning_progress_weight = j.value("learning_progress_weight", 1.0F);
  value.controllability_weight = j.value("controllability_weight", 0.5F);
  value.novelty_ema_decay = j.value("novelty_ema_decay", 0.99F);
  value.learning_progress_ema_decay = j.value("learning_progress_ema_decay", 0.95F);
  value.use_controllability_gate = j.value("use_controllability_gate", true);
}

void to_json(json& j, const IntrinsicModelConfig& value) {
  j = json{
      {"forward_loss_coef", value.forward_loss_coef},
      {"inverse_loss_coef", value.inverse_loss_coef},
  };
}

void from_json(const json& j, IntrinsicModelConfig& value) {
  value.forward_loss_coef = j.value("forward_loss_coef", 1.0F);
  value.inverse_loss_coef = j.value("inverse_loss_coef", 1.0F);
}

void to_json(json& j, const BCRegularizationConfig& value) {
  j = json{
      {"initial_beta", value.initial_beta},
      {"beta_decay", value.beta_decay},
      {"min_beta", value.min_beta},
  };
}

void from_json(const json& j, BCRegularizationConfig& value) {
  value.initial_beta = j.value("initial_beta", 0.1F);
  value.beta_decay = j.value("beta_decay", 0.999F);
  value.min_beta = j.value("min_beta", 0.0F);
}

void to_json(json& j, const WeightScheduleConfig& value) {
  j = json{
      {"initial_extrinsic_weight", value.initial_extrinsic_weight},
      {"initial_curiosity_weight", value.initial_curiosity_weight},
      {"initial_learning_progress_weight", value.initial_learning_progress_weight},
      {"initial_controllability_weight", value.initial_controllability_weight},
      {"extrinsic_weight_growth_rate", value.extrinsic_weight_growth_rate},
      {"intrinsic_weight_decay_rate", value.intrinsic_weight_decay_rate},
      {"max_extrinsic_weight", value.max_extrinsic_weight},
      {"min_intrinsic_weight", value.min_intrinsic_weight},
  };
}

void from_json(const json& j, WeightScheduleConfig& value) {
  value.initial_extrinsic_weight = j.value("initial_extrinsic_weight", 0.1F);
  value.initial_curiosity_weight = j.value("initial_curiosity_weight", 1.0F);
  value.initial_learning_progress_weight = j.value("initial_learning_progress_weight", 1.0F);
  value.initial_controllability_weight = j.value("initial_controllability_weight", 0.0F);
  value.extrinsic_weight_growth_rate = j.value("extrinsic_weight_growth_rate", 1.001F);
  value.intrinsic_weight_decay_rate = j.value("intrinsic_weight_decay_rate", 0.999F);
  value.max_extrinsic_weight = j.value("max_extrinsic_weight", 1.0F);
  value.min_intrinsic_weight = j.value("min_intrinsic_weight", 0.01F);
}

void to_json(json& j, const SuccessBufferConfig& value) {
  j = json{
      {"capacity", value.capacity},
      {"oversample_ratio", value.oversample_ratio},
  };
}

void from_json(const json& j, SuccessBufferConfig& value) {
  value.capacity = j.value("capacity", 10000);
  value.oversample_ratio = j.value("oversample_ratio", 0.1F);
}

void to_json(json& j, const ExperimentConfig& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"env", value.env},
      {"outcome", value.outcome},
      {"action_table", value.action_table},
      {"model", value.model},
      {"ppo", value.ppo},
      {"offline_dataset", value.offline_dataset},
      {"behavior_cloning", value.behavior_cloning},
      {"self_play_league", value.self_play_league},
      {"wandb", value.wandb},
      {"critic", value.critic},
      {"forward_model", value.forward_model},
      {"inverse_model", value.inverse_model},
      {"intrinsic_rewards", value.intrinsic_rewards},
      {"intrinsic_model", value.intrinsic_model},
      {"bc_regularization", value.bc_regularization},
      {"weight_schedule", value.weight_schedule},
      {"success_buffer", value.success_buffer},
  };
}

void from_json(const json& j, ExperimentConfig& value) {
  reject_removed_section(j, "lfpo");
  reject_removed_section(j, "future_evaluator");
  reject_removed_section(j, "offline_pretraining");
  value.schema_version = j.value("schema_version", 4);
  value.obs_schema_version = j.value("obs_schema_version", 1);
  value.env = j.value("env", EnvConfig{});
  value.outcome = j.value("outcome", OutcomeConfig{});
  value.action_table = j.value("action_table", ActionTableConfig{});
  value.model = j.value("model", ModelConfig{});
  value.ppo = j.value("ppo", PPOConfig{});
  value.offline_dataset = j.value("offline_dataset", OfflineDatasetConfig{});
  value.behavior_cloning = j.value("behavior_cloning", BehaviorCloningConfig{});
  value.self_play_league = j.value("self_play_league", SelfPlayLeagueConfig{});
  value.wandb = j.value("wandb", WandbConfig{});
  value.critic = j.value("critic", CriticConfig{});
  value.forward_model = j.value("forward_model", ForwardModelConfig{});
  value.inverse_model = j.value("inverse_model", InverseModelConfig{});
  value.model.forward_model = value.forward_model;
  value.model.inverse_model = value.inverse_model;
  value.intrinsic_rewards = j.value("intrinsic_rewards", IntrinsicRewardConfig{});
  value.intrinsic_model = j.value("intrinsic_model", IntrinsicModelConfig{});
  value.bc_regularization = j.value("bc_regularization", BCRegularizationConfig{});
  value.weight_schedule = j.value("weight_schedule", WeightScheduleConfig{});
  value.success_buffer = j.value("success_buffer", SuccessBufferConfig{});
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
      {"critic_heads", value.critic_heads},
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
  value.critic_heads = j.value("critic_heads", std::vector<std::string>{});
}

CriticHeadConfig materialize_critic_head_config(
    const CriticHeadConfig& cfg,
    const ModelConfig& model,
    bool enabled) {
  CriticHeadConfig out = cfg;
  out.enabled = out.enabled && enabled;

  if (out.value_hidden_dim <= 0) {
    out.value_hidden_dim = model.value_hidden_dim;
  }
  if (out.value_num_atoms <= 0) {
    out.value_num_atoms = model.value_num_atoms;
  }
  if (!(out.value_v_max > out.value_v_min)) {
    out.value_v_min = model.value_v_min;
    out.value_v_max = model.value_v_max;
  }

  return out;
}

void validate_experiment_config(const ExperimentConfig& config) {
  if (config.ppo.rollout_length <= 1) {
    throw std::invalid_argument("ppo.rollout_length must be > 1.");
  }
  if (config.ppo.sequence_length <= 0) {
    throw std::invalid_argument("ppo.sequence_length must be positive.");
  }
  if (config.ppo.minibatch_size < config.ppo.sequence_length) {
    throw std::invalid_argument("ppo.minibatch_size must be >= ppo.sequence_length.");
  }
  if (config.ppo.burn_in < 0 || config.ppo.burn_in >= config.ppo.sequence_length) {
    throw std::invalid_argument("ppo.burn_in must satisfy 0 <= burn_in < sequence_length.");
  }
  if (config.behavior_cloning.sequence_length <= 0) {
    throw std::invalid_argument("behavior_cloning.sequence_length must be positive.");
  }
  if (config.model.encoder_dim <= 0) {
    throw std::invalid_argument("model.encoder_dim must be positive.");
  }

  const bool uses_intrinsic =
      config.intrinsic_rewards.curiosity_weight > 0.0F ||
      config.intrinsic_rewards.learning_progress_weight > 0.0F ||
      config.intrinsic_rewards.controllability_weight > 0.0F ||
      config.intrinsic_rewards.use_controllability_gate;

  if (uses_intrinsic) {
    if (config.intrinsic_model.forward_loss_coef <= 0.0F) {
      throw std::invalid_argument("Intrinsic rewards require forward_loss_coef > 0.");
    }
    if (config.intrinsic_model.inverse_loss_coef <= 0.0F) {
      throw std::invalid_argument("Intrinsic rewards require inverse_loss_coef > 0.");
    }
  }

  if (!config.critic.curiosity.enabled &&
      config.weight_schedule.initial_curiosity_weight != 0.0F) {
    throw std::invalid_argument("curiosity weight is nonzero but curiosity critic is disabled.");
  }
  if (!config.critic.learning_progress.enabled &&
      config.weight_schedule.initial_learning_progress_weight != 0.0F) {
    throw std::invalid_argument("learning_progress weight is nonzero but learning_progress critic is disabled.");
  }
  if (!config.critic.controllability.enabled &&
      config.weight_schedule.initial_controllability_weight != 0.0F) {
    throw std::invalid_argument("controllability weight is nonzero but controllability critic is disabled.");
  }
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
