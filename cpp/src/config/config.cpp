#include "pulsar/config/config.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/sparse_event_channels.hpp"

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
  j = json{
      {"score", value.score},
      {"concede", value.concede},
      {"neutral", value.neutral},
      {"neutral_no_touch", value.neutral_no_touch},
      {"team_spirit", value.team_spirit},
      {"step_penalty", value.step_penalty},
  };
}

void from_json(const json& j, OutcomeConfig& value) {
  value.score = j.value("score", 1.0F);
  value.concede = j.value("concede", -1.0F);
  value.neutral = j.value("neutral", 0.0F);
  value.neutral_no_touch = j.value("neutral_no_touch", -1.0F);
  value.team_spirit = j.value("team_spirit", 0.0F);
  value.step_penalty = j.value("step_penalty", 0.0F);
}

void to_json(json& j, const PredictorChannelConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"horizon", value.horizon},
      {"learning_rate", value.learning_rate},
      {"convergence_threshold", value.convergence_threshold},
      {"warmup_updates", value.warmup_updates},
  };
}

void from_json(const json& j, PredictorChannelConfig& value) {
  value.enabled = j.value("enabled", true);
  value.horizon = j.value("horizon", 40);
  value.learning_rate = j.value("learning_rate", 0.005F);
  value.convergence_threshold = j.value("convergence_threshold", 0.002F);
  value.warmup_updates = j.value("warmup_updates", 15);
}

void to_json(json& j, const PredictorConfig& value) {
  j = json{{"enabled", value.enabled}, {"channels", value.channels}};
}

void from_json(const json& j, PredictorConfig& value) {
  value.enabled = j.value("enabled", true);
  value.channels.clear();
  for (const auto& spec : kSparseEventChannels) {
    PredictorChannelConfig channel{};
    channel.horizon = spec.default_horizon;
    value.channels.emplace(std::string(spec.name), channel);
  }

  if (j.contains("channels")) {
    for (const auto& item : j.at("channels").items()) {
      if (value.channels.find(item.key()) == value.channels.end()) {
        throw std::runtime_error("Unknown predictor channel: " + item.key());
      }
      value.channels[item.key()] = item.value().get<PredictorChannelConfig>();
    }
  } else {
    for (const auto& spec : kSparseEventChannels) {
      const std::string name(spec.name);
      if (j.contains(name)) {
        value.channels[name] = j.at(name).get<PredictorChannelConfig>();
      }
    }
  }
}

void to_json(json& j, const ActionTableConfig& value) {
  j = json{{"builtin", value.builtin}, {"actions", value.actions}, {"refined_action_masking", value.refined_action_masking}};
}

void from_json(const json& j, ActionTableConfig& value) {
  value.builtin = j.value("builtin", std::string{});
  value.actions = j.value("actions", std::vector<ControllerState>{});
  value.refined_action_masking = j.value("refined_action_masking", false);
}

void to_json(json& j, const EnvConfig& value) {
  j = json{
      {"mode", value.mode},
      {"collision_meshes_path", value.collision_meshes_path},
      {"team_size", value.team_size},
      {"tick_skip", value.tick_skip},
      {"tick_rate", value.tick_rate},
      {"max_episode_ticks", value.max_episode_ticks},
      {"disable_truncation", value.disable_truncation},
      {"no_touch_timeout_seconds", value.no_touch_timeout_seconds},
      {"no_touch_timeout_only_before_first_touch", value.no_touch_timeout_only_before_first_touch},
      {"spawn_opponents", value.spawn_opponents},
      {"randomize_kickoffs", value.randomize_kickoffs},
      {"seed", value.seed},
      {"obs_x_mirror", value.obs_x_mirror},
      {"obs_local_frame", value.obs_local_frame},
      {"obs_relative_goals", value.obs_relative_goals},
      {"obs_proximity_boosts", value.obs_proximity_boosts},
      {"obs_explicit_kinematics", value.obs_explicit_kinematics},
      {"obs_flip_decay", value.obs_flip_decay},
      {"obs_action_history", value.obs_action_history},
      {"obs_ball_prediction", value.obs_ball_prediction},
  };
}

void from_json(const json& j, EnvConfig& value) {
  value.mode = j.value("mode", std::string{"soccar"});
  value.collision_meshes_path = j.value("collision_meshes_path", std::string{"collision_meshes"});
  value.team_size = j.value("team_size", 2);
  value.tick_skip = j.value("tick_skip", 8);
  value.tick_rate = j.value("tick_rate", 120);
  value.max_episode_ticks = j.value("max_episode_ticks", 2250);
  value.disable_truncation = j.value("disable_truncation", false);
  value.no_touch_timeout_seconds = j.value("no_touch_timeout_seconds", 10.0F);
  value.no_touch_timeout_only_before_first_touch = j.value("no_touch_timeout_only_before_first_touch", false);
  value.spawn_opponents = j.value("spawn_opponents", true);
  value.randomize_kickoffs = j.value("randomize_kickoffs", true);
  value.seed = j.value("seed", static_cast<std::uint64_t>(0));
  value.obs_x_mirror = j.value("obs_x_mirror", false);
  value.obs_local_frame = j.value("obs_local_frame", false);
  value.obs_relative_goals = j.value("obs_relative_goals", false);
  value.obs_proximity_boosts = j.value("obs_proximity_boosts", false);
  value.obs_explicit_kinematics = j.value("obs_explicit_kinematics", false);
  value.obs_flip_decay = j.value("obs_flip_decay", false);
  value.obs_action_history = j.value("obs_action_history", false);
  value.obs_ball_prediction = j.value("obs_ball_prediction", false);
}

void to_json(json& j, const ModelConfig& value) {
  j = json{
      {"observation_dim", value.observation_dim},
      {"action_dim", value.action_dim},
      {"use_layer_norm", value.use_layer_norm},
      {"encoder_dim", value.encoder_dim},
      {"num_encoder_blocks", value.num_encoder_blocks},
      {"sequence_length", value.sequence_length},
      {"max_forward_samples", value.max_forward_samples},
      {"value_hidden_dim", value.value_hidden_dim},
      {"policy_hidden_dim", value.policy_hidden_dim},
  };
}

void from_json(const json& j, ModelConfig& value) {
  value.observation_dim = j.value("observation_dim", 132);
  value.action_dim = j.value("action_dim", 90);
  value.use_layer_norm = j.value("use_layer_norm", true);
  value.encoder_dim = j.value("encoder_dim", 640);
  value.num_encoder_blocks = j.value("num_encoder_blocks", 5);
  value.sequence_length = j.value("sequence_length", j.value("transformer_window_size", 16));
  value.max_forward_samples = j.value("max_forward_samples", j.value("transformer_max_batch_size", 0));
  value.value_hidden_dim = j.value("value_hidden_dim", 256);
  value.policy_hidden_dim = j.value("policy_hidden_dim", 0);
}

void to_json(json& j, const GoalMappingConfig& value) {
  j = json{
      {"arena_max_distance", value.arena_max_distance},
  };
}

void from_json(const json& j, GoalMappingConfig& value) {
  value.arena_max_distance = j.value("arena_max_distance", 8192.0F);
}

void to_json(json& j, const GoalCriticConfig& value) {
  j = json{
      {"goal_dim", value.goal_dim},
      {"hidden_dim", value.hidden_dim},
      {"embedding_dim", value.embedding_dim},
      {"logsumexp_penalty_coeff", value.logsumexp_penalty_coeff},
      {"lambda_Zg", value.lambda_Zg},
      {"lambda_goal_actor", value.lambda_goal_actor},
      {"contrastive_batch_size", value.contrastive_batch_size},
      {"max_future_horizon", value.max_future_horizon},
      {"temperature", value.temperature},
      {"gcrl_actor_alpha_gain", value.gcrl_actor_alpha_gain},
  };
}

void from_json(const json& j, GoalCriticConfig& value) {
  value.goal_dim = j.value("goal_dim", 3);
  value.hidden_dim = j.value("hidden_dim", 256);
  value.embedding_dim = j.value("embedding_dim", 64);
  value.logsumexp_penalty_coeff = j.value("logsumexp_penalty_coeff", 0.01F);
  value.lambda_Zg = j.value("lambda_Zg", 1.0F);
  value.lambda_goal_actor = j.value("lambda_goal_actor", 0.1F);
  value.contrastive_batch_size = j.value("contrastive_batch_size", 2048);
  value.max_future_horizon = j.value("max_future_horizon", 256);
  value.temperature = j.value("temperature", 0.1F);
  value.gcrl_actor_alpha_gain = j.value("gcrl_actor_alpha_gain", 2.0F);
}

void to_json(json& j, const VRPOConfig& value) {
  j = json{
      {"q_critic_hidden_dim", value.q_critic_hidden_dim},
      {"q_critic_coef", value.q_critic_coef},
      {"q_value_abs_cap", value.q_value_abs_cap},
      {"q_target_abs_cap", value.q_target_abs_cap},
      {"value_coef_sparse", value.value_coef_sparse},
  };
}

void from_json(const json& j, VRPOConfig& value) {
  value.q_critic_hidden_dim = j.value("q_critic_hidden_dim", 384);
  value.q_critic_coef = j.value("q_critic_coef", 0.5F);
  value.q_value_abs_cap = j.value("q_value_abs_cap", 50.0F);
  value.q_target_abs_cap = j.value("q_target_abs_cap", 50.0F);
  value.value_coef_sparse = j.value("value_coef_sparse", 0.5F);
}

void to_json(json& j, const ESLoraConfig& value) {
  j = json{
      {"rank", value.rank},
      {"lora_alpha", value.lora_alpha},
      {"population_size", value.population_size},
      {"virtual_population_waves", value.virtual_population_waves},
      {"sigma_ES", value.sigma_ES},
      {"eta_ES", value.eta_ES},
      {"es_interval", value.es_interval},
      {"eval_shards", value.eval_shards},
      {"eval_workers", value.eval_workers},
      {"eval_episodes_per_member", value.eval_episodes_per_member},
      {"eval_num_envs", value.eval_num_envs},
      {"eval_rollout_length", value.eval_rollout_length},
      {"kl_eval_stride", value.kl_eval_stride},
      {"beta_KL", value.beta_KL},
      {"rank_transform", value.rank_transform},
      {"antithetic_sampling", value.antithetic_sampling},
      {"update_norm_clip", value.update_norm_clip},
      {"max_update_norm", value.max_update_norm},
      {"max_kl_mean", value.max_kl_mean},
      {"require_fitness_signal", value.require_fitness_signal},
      {"min_fitness_std", value.min_fitness_std},
      {"sigma_alpha_gain", value.sigma_alpha_gain},
      {"es_interval_min", value.es_interval_min},
  };
}

void from_json(const json& j, ESLoraConfig& value) {
  if (j.contains("alpha_g")) {
    throw std::invalid_argument("es_lora.alpha_g was removed; ES fitness is reward minus KL.");
  }
  value.rank = j.value("rank", 4);
  value.lora_alpha = j.value("lora_alpha", 4.0F);
  value.population_size = j.value("population_size", 8);
  value.virtual_population_waves = j.value("virtual_population_waves", 1);
  value.sigma_ES = j.value("sigma_ES", 0.05F);
  value.eta_ES = j.value("eta_ES", 0.003F);
  value.es_interval = j.value("es_interval", 25);
  value.eval_shards = j.value("eval_shards", 0);
  value.eval_workers = j.value("eval_workers", 0);
  value.eval_episodes_per_member = j.value("eval_episodes_per_member", 2);
  value.eval_num_envs = j.value("eval_num_envs", 8);
  value.eval_rollout_length = j.value("eval_rollout_length", 450);
  value.kl_eval_stride = j.value("kl_eval_stride", 1);
  value.beta_KL = j.value("beta_KL", 0.01F);
  value.rank_transform = j.value("rank_transform", false);
  value.antithetic_sampling = j.value("antithetic_sampling", true);
  value.update_norm_clip = j.value("update_norm_clip", true);
  value.max_update_norm = j.value("max_update_norm", 0.002F);
  value.max_kl_mean = j.value("max_kl_mean", 0.01F);
  value.require_fitness_signal = j.value(
      "require_fitness_signal",
      j.value("require_winrate_signal", true));
  value.min_fitness_std = j.value(
      "min_fitness_std",
      j.value("min_winrate_std", 1.0e-6F));
  value.sigma_alpha_gain = j.value("sigma_alpha_gain", 1.0F);
  value.es_interval_min = j.value("es_interval_min", 20);
}

void to_json(json& j, const PPOConfig& value) {
  j = json{
      {"num_envs", value.num_envs},
      {"collection_workers", value.collection_workers},
      {"collection_shards", value.collection_shards},
      {"init_checkpoint", value.init_checkpoint},
      {"rollout_length", value.rollout_length},
      {"minibatch_size", value.minibatch_size},
      {"update_epochs", value.update_epochs},
      {"optimizer_accumulation_steps", value.optimizer_accumulation_steps},
      {"clip_range", value.clip_range},
      {"entropy_coef", value.entropy_coef},
      {"entropy_floor", value.entropy_floor},
      {"entropy_floor_coef", value.entropy_floor_coef},
      {"value_coef", value.value_coef},
      {"value_loss_delta", value.value_loss_delta},
      {"gamma", value.gamma},
      {"gae_lambda", value.gae_lambda},
      {"learning_rate", value.learning_rate},
      {"overbatching", value.overbatching},
      {"policy_temperature", value.policy_temperature},
      {"max_grad_norm", value.max_grad_norm},
      {"device", value.device},
      {"checkpoint_interval", value.checkpoint_interval},
      {"max_rolling_checkpoints", value.max_rolling_checkpoints},

      {"synchronize_cuda_timing", value.synchronize_cuda_timing},
      {"cuda_amp", value.cuda_amp},
      {"adaptive_entropy", value.adaptive_entropy},
      {"entropy_decay_score", value.entropy_decay_score},
      {"entropy_low_coef", value.entropy_low_coef},
      {"max_policy_log_ratio", value.max_policy_log_ratio},
      {"target_kl", value.target_kl},
      {"max_preclip_grad_norm", value.max_preclip_grad_norm},
      {"pcgrad", value.pcgrad},
      {"overlap_collection_update", value.overlap_collection_update},
      {"value_clipping", value.value_clipping},
      {"value_clip_range", value.value_clip_range},
      {"weight_decay", value.weight_decay},
      {"popart_enabled", value.popart_enabled},
      {"popart_beta", value.popart_beta},
      {"popart_epsilon", value.popart_epsilon},
      {"ent_coef_min", value.ent_coef_min},
      {"entropy_window", value.entropy_window},
  };
}

void from_json(const json& j, PPOConfig& value) {
  value.num_envs = j.value("num_envs", 64);
  value.collection_workers = j.value("collection_workers", 0);
  value.collection_shards = j.value("collection_shards", 1);
  value.init_checkpoint = j.value("init_checkpoint", std::string{});
  value.rollout_length = j.value("rollout_length", 256);
  value.minibatch_size = j.value("minibatch_size", 32768);
  value.update_epochs = j.value("update_epochs", 3);
  value.optimizer_accumulation_steps = j.value("optimizer_accumulation_steps", 1);
  value.clip_range = j.value("clip_range", 0.2F);
  value.entropy_coef = j.value("entropy_coef", 0.01F);
  value.entropy_floor = j.value("entropy_floor", 0.0F);
  value.entropy_floor_coef = j.value("entropy_floor_coef", 0.0F);
  value.value_coef = j.value("value_coef", 1.0F);
  value.value_loss_delta = j.value("value_loss_delta", 10.0F);
  value.gamma = j.value("gamma", 0.99F);
  value.gae_lambda = j.value("gae_lambda", 0.95F);
  value.learning_rate = j.value("learning_rate", 3.0e-4F);
  value.overbatching = j.value("overbatching", true);
  value.policy_temperature = j.value("policy_temperature", 1.0F);
  value.max_grad_norm = j.value("max_grad_norm", 1.0F);
  value.device = j.value("device", std::string{"cpu"});
  value.checkpoint_interval = j.value("checkpoint_interval", 10);
  value.max_rolling_checkpoints = j.value("max_rolling_checkpoints", 5);

  value.synchronize_cuda_timing = j.value("synchronize_cuda_timing", false);
  value.cuda_amp = j.value("cuda_amp", false);
  value.adaptive_entropy = j.value("adaptive_entropy", false);
  value.entropy_decay_score = j.value("entropy_decay_score", 0.60F);
  value.entropy_low_coef = j.value("entropy_low_coef", 0.005F);
  value.max_policy_log_ratio = j.value("max_policy_log_ratio", 5.0F);
  value.target_kl = j.value("target_kl", 0.0F);
  value.max_preclip_grad_norm = j.value("max_preclip_grad_norm", 0.0F);
  value.pcgrad = j.value("pcgrad", false);
  value.overlap_collection_update = j.value("overlap_collection_update", false);
  value.value_clipping = j.value("value_clipping", false);
  value.value_clip_range = j.value("value_clip_range", 0.2F);
  value.weight_decay = j.value("weight_decay", 0.0F);
  value.popart_enabled = j.value("popart_enabled", false);
  value.popart_beta = j.value("popart_beta", 0.0001);
  value.popart_epsilon = j.value("popart_epsilon", 1e-4);
  value.ent_coef_min = j.value("ent_coef_min", 0.025F);
  value.entropy_window = j.value("entropy_window", 10);
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
      {"pfsp_enabled", value.pfsp_enabled},
      {"pfsp_sigma", value.pfsp_sigma},
  };
}

void from_json(const json& j, SelfPlayLeagueConfig& value) {
  value.enabled = j.value("enabled", false);
  value.opponent_probability = j.value("opponent_probability", 0.0F);
  value.snapshot_interval_updates = j.value("snapshot_interval_updates", 10);
  value.max_snapshots = j.value("max_snapshots", 8);
  value.training_opponent_policy = j.value("training_opponent_policy", std::string{"current_policy"});
  value.eval_interval_updates = j.value("eval_interval_updates", 10);
  value.eval_num_envs = j.value("eval_num_envs", 8);
  value.eval_matches_per_snapshot = j.value("eval_matches_per_snapshot", 4);
  value.eval_policy = j.value("eval_policy", std::string{"deterministic"});
  value.elo_initial = j.value("elo_initial", 1000.0F);
  value.elo_k = j.value("elo_k", 32.0F);
  value.pfsp_enabled = j.value("pfsp_enabled", false);
  value.pfsp_sigma = j.value("pfsp_sigma", 200.0F);
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
      {"run_id", value.run_id},
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
  value.run_id = j.value("run_id", std::string{});
}

void to_json(json& j, const CurriculumModeConfig& value) {
  j = json{
      {"mode", value.mode},
      {"shards", value.shards},
      {"num_envs", value.num_envs},
      {"collection_workers", value.collection_workers},
  };
}

void from_json(const json& j, CurriculumModeConfig& value) {
  value.mode = j.at("mode").get<std::string>();
  value.shards = j.value("shards", 1);
  value.num_envs = j.value("num_envs", 0);
  value.collection_workers = j.value("collection_workers", 0);
}

void to_json(json& j, const CurriculumConfig& value) {
  j = json{{"modes", value.modes}};
}

void from_json(const json& j, CurriculumConfig& value) {
  value.modes = j.value("modes", std::vector<CurriculumModeConfig>{});
}

void to_json(json& j, const ExperimentConfig& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"env", value.env},
      {"outcome", value.outcome},
      {"predictor", value.predictor},
      {"action_table", value.action_table},
      {"model", value.model},
      {"ppo", value.ppo},
      {"goal_mapping", value.goal_mapping},
      {"goal_critic", value.goal_critic},
      {"vrpo", value.vrpo},
      {"es_lora", value.es_lora},
      {"self_play_league", value.self_play_league},
      {"wandb", value.wandb},
      {"curriculum", value.curriculum},
  };
}

void from_json(const json& j, ExperimentConfig& value) {
  reject_removed_section(j, "lfpo");
  reject_removed_section(j, "future_evaluator");
  reject_removed_section(j, "offline_pretraining");
  reject_removed_section(j, "offline_dataset");
  reject_removed_section(j, "behavior_cloning");
  reject_removed_section(j, "critic");
  reject_removed_section(j, "forward_model");
  reject_removed_section(j, "inverse_model");
  reject_removed_section(j, "intrinsic_rewards");
  reject_removed_section(j, "intrinsic_model");
  reject_removed_section(j, "bc_regularization");
  reject_removed_section(j, "weight_schedule");
  reject_removed_section(j, "success_buffer");
  reject_removed_section(j, "next_goal_predictor");
  reject_removed_section(j, "offline_optimization");
  reject_removed_section(j, "reward");
  reject_removed_section(j, "value_pretraining");
  reject_removed_section(j, "actor_goal");
  value.schema_version = j.value("schema_version", 6);
  value.obs_schema_version = j.value("obs_schema_version", 2);
  value.env = j.value("env", EnvConfig{});
  value.outcome = j.value("outcome", OutcomeConfig{});
  value.predictor = j.contains("predictor") ? j.at("predictor").get<PredictorConfig>() : json::object().get<PredictorConfig>();
  value.action_table = j.value("action_table", ActionTableConfig{});
  value.model = j.value("model", ModelConfig{});
  value.ppo = j.value("ppo", PPOConfig{});
  value.goal_mapping = j.value("goal_mapping", GoalMappingConfig{});
  value.goal_critic = j.value("goal_critic", GoalCriticConfig{});
  value.vrpo = j.value("vrpo", VRPOConfig{});
  value.es_lora = j.value("es_lora", ESLoraConfig{});
  value.self_play_league = j.value("self_play_league", SelfPlayLeagueConfig{});
  value.wandb = j.value("wandb", WandbConfig{});
  value.curriculum = j.value("curriculum", CurriculumConfig{});
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
      {"extra", value.extra},
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
  value.extra = j.value("extra", json::object());
}

void validate_experiment_config(const ExperimentConfig& config) {
  if (config.ppo.rollout_length <= 1) {
    throw std::invalid_argument("ppo.rollout_length must be > 1.");
  }
  if (config.ppo.num_envs <= 0) {
    throw std::invalid_argument("ppo.num_envs must be positive.");
  }
  if (config.ppo.collection_shards <= 0) {
    throw std::invalid_argument("ppo.collection_shards must be positive.");
  }
  if (config.ppo.collection_shards > config.ppo.num_envs) {
    throw std::invalid_argument("ppo.collection_shards must be <= ppo.num_envs.");
  }

  if (config.model.encoder_dim <= 0) {
    throw std::invalid_argument("model.encoder_dim must be positive.");
  }
  if (config.model.num_encoder_blocks <= 0) {
    throw std::invalid_argument("model.num_encoder_blocks must be positive.");
  }
  if (config.model.sequence_length <= 0) {
    throw std::invalid_argument("model.sequence_length must be positive.");
  }
  if (config.model.max_forward_samples < 0) {
    throw std::invalid_argument("model.max_forward_samples must be non-negative (0 = unlimited).");
  }
  if (config.goal_mapping.arena_max_distance <= 0.0F) {
    throw std::invalid_argument("goal_mapping.arena_max_distance must be positive.");
  }
  if (config.goal_critic.goal_dim <= 0) {
    throw std::invalid_argument("goal_critic.goal_dim must be positive.");
  }
  if (config.goal_critic.hidden_dim <= 0) {
    throw std::invalid_argument("goal_critic.hidden_dim must be positive.");
  }
  if (config.goal_critic.embedding_dim <= 0) {
    throw std::invalid_argument("goal_critic.embedding_dim must be positive.");
  }
  if (config.goal_critic.logsumexp_penalty_coeff < 0.0F) {
    throw std::invalid_argument("goal_critic.logsumexp_penalty_coeff must be non-negative.");
  }
  if (config.goal_critic.lambda_Zg < 0.0F) {
    throw std::invalid_argument("goal_critic.lambda_Zg must be non-negative.");
  }
  if (config.goal_critic.contrastive_batch_size <= 0) {
    throw std::invalid_argument("goal_critic.contrastive_batch_size must be positive.");
  }
  if (config.es_lora.rank <= 0) {
    throw std::invalid_argument("es_lora.rank must be positive.");
  }
  if (config.es_lora.lora_alpha <= 0.0F) {
    throw std::invalid_argument("es_lora.lora_alpha must be positive.");
  }
  if (config.es_lora.sigma_ES <= 0.0F) {
    throw std::invalid_argument("es_lora.sigma_ES must be positive.");
  }
  if (config.es_lora.population_size <= 0) {
    throw std::invalid_argument("es_lora.population_size must be positive.");
  }
  if (config.es_lora.virtual_population_waves <= 0) {
    throw std::invalid_argument("es_lora.virtual_population_waves must be positive.");
  }
  if (config.es_lora.antithetic_sampling && (config.es_lora.population_size % 2) != 0) {
    throw std::invalid_argument("es_lora.population_size must be even when antithetic_sampling is enabled.");
  }
  if (config.es_lora.eta_ES <= 0.0F) {
    throw std::invalid_argument("es_lora.eta_ES must be positive.");
  }
  if (config.es_lora.beta_KL < 0.0F) {
    throw std::invalid_argument("es_lora.beta_KL must be non-negative.");
  }
  if (config.es_lora.max_update_norm < 0.0F) {
    throw std::invalid_argument("es_lora.max_update_norm must be non-negative.");
  }
  if (config.es_lora.max_kl_mean < 0.0F) {
    throw std::invalid_argument("es_lora.max_kl_mean must be non-negative.");
  }
  if (config.es_lora.min_fitness_std < 0.0F) {
    throw std::invalid_argument("es_lora.min_fitness_std must be non-negative.");
  }
  if (config.es_lora.eval_shards < 0) {
    throw std::invalid_argument("es_lora.eval_shards must be non-negative.");
  }
  if (config.es_lora.eval_workers < 0) {
    throw std::invalid_argument("es_lora.eval_workers must be non-negative.");
  }
  if (config.es_lora.eval_episodes_per_member <= 0) {
    throw std::invalid_argument("es_lora.eval_episodes_per_member must be positive.");
  }
  if (config.es_lora.eval_num_envs <= 0) {
    throw std::invalid_argument("es_lora.eval_num_envs must be positive.");
  }
  if (config.es_lora.eval_rollout_length <= 0) {
    throw std::invalid_argument("es_lora.eval_rollout_length must be positive.");
  }
  if (config.es_lora.kl_eval_stride <= 0) {
    throw std::invalid_argument("es_lora.kl_eval_stride must be positive.");
  }
  if (config.goal_critic.max_future_horizon <= 0) {
    throw std::invalid_argument("goal_critic.max_future_horizon must be positive.");
  }
  if (config.ppo.minibatch_size <= 0) {
    throw std::invalid_argument("ppo.minibatch_size must be positive.");
  }
  if (config.ppo.optimizer_accumulation_steps <= 0) {
    throw std::invalid_argument("ppo.optimizer_accumulation_steps must be positive.");
  }
  if (config.ppo.update_epochs <= 0) {
    throw std::invalid_argument("ppo.update_epochs must be positive.");
  }
  if (config.ppo.clip_range < 0.0F) {
    throw std::invalid_argument("ppo.clip_range must be non-negative.");
  }
  if (config.ppo.learning_rate <= 0.0F) {
    throw std::invalid_argument("ppo.learning_rate must be positive.");
  }
  if (config.ppo.max_grad_norm <= 0.0F) {
    throw std::invalid_argument("ppo.max_grad_norm must be positive.");
  }
  if (config.ppo.entropy_floor < 0.0F) {
    throw std::invalid_argument("ppo.entropy_floor must be non-negative.");
  }
  if (config.ppo.entropy_floor_coef < 0.0F) {
    throw std::invalid_argument("ppo.entropy_floor_coef must be non-negative.");
  }
  if (config.ppo.max_policy_log_ratio < 0.0F) {
    throw std::invalid_argument("ppo.max_policy_log_ratio must be non-negative.");
  }
  if (config.ppo.target_kl < 0.0F) {
    throw std::invalid_argument("ppo.target_kl must be non-negative.");
  }
  if (config.ppo.max_preclip_grad_norm < 0.0F) {
    throw std::invalid_argument("ppo.max_preclip_grad_norm must be non-negative.");
  }
  if (config.ppo.value_loss_delta < 0.0F) {
    throw std::invalid_argument("ppo.value_loss_delta must be non-negative.");
  }
  if (config.ppo.value_clip_range <= 0.0F) {
    throw std::invalid_argument("ppo.value_clip_range must be positive.");
  }
  if (config.ppo.weight_decay < 0.0F) {
    throw std::invalid_argument("ppo.weight_decay must be non-negative.");
  }
  if (config.ppo.popart_beta <= 0.0) {
    throw std::invalid_argument("ppo.popart_beta must be positive.");
  }
  if (config.ppo.popart_epsilon <= 0.0) {
    throw std::invalid_argument("ppo.popart_epsilon must be positive.");
  }
  if (config.goal_critic.temperature <= 0.0F) {
    throw std::invalid_argument("goal_critic.temperature must be positive.");
  }
  if (config.self_play_league.pfsp_sigma <= 0.0F) {
    throw std::invalid_argument("self_play_league.pfsp_sigma must be positive.");
  }
  if (config.vrpo.q_critic_hidden_dim <= 0) {
    throw std::invalid_argument("vrpo.q_critic_hidden_dim must be positive.");
  }
  if (config.vrpo.q_critic_coef < 0.0F) {
    throw std::invalid_argument("vrpo.q_critic_coef must be non-negative.");
  }
  if (config.vrpo.q_value_abs_cap <= 0.0F) {
    throw std::invalid_argument("vrpo.q_value_abs_cap must be positive.");
  }
  if (config.vrpo.q_target_abs_cap <= 0.0F) {
    throw std::invalid_argument("vrpo.q_target_abs_cap must be positive.");
  }
  if (config.vrpo.value_coef_sparse < 0.0F) {
    throw std::invalid_argument("vrpo.value_coef_sparse must be non-negative.");
  }
  if (config.env.team_size <= 0 || config.env.team_size > 4) {
    throw std::invalid_argument("env.team_size must be between 1 and 4.");
  }
  if (config.env.tick_rate != 120) {
    std::cerr << "Warning: env.tick_rate is currently ignored. Simulation hardcodes 120 Hz.\n";
  }

  if (config.predictor.enabled) {
    auto validate_channel = [](const PredictorChannelConfig& chan, const std::string& name) {
      if (chan.enabled) {
        if (chan.horizon <= 0) {
          throw std::invalid_argument("predictor." + name + ".horizon must be positive.");
        }
        if (chan.learning_rate <= 0.0F) {
          throw std::invalid_argument("predictor." + name + ".learning_rate must be positive.");
        }
        if (chan.convergence_threshold <= 0.0F) {
          throw std::invalid_argument("predictor." + name + ".convergence_threshold must be positive.");
        }
        if (chan.warmup_updates < 0) {
          throw std::invalid_argument("predictor." + name + ".warmup_updates must be non-negative.");
        }
      }
    };
    for (const auto& spec : kSparseEventChannels) {
      const std::string name(spec.name);
      auto it = config.predictor.channels.find(name);
      if (it == config.predictor.channels.end()) {
        throw std::invalid_argument("predictor.channels missing required channel: " + name);
      }
      validate_channel(it->second, name);
    }
    for (const auto& [name, channel] : config.predictor.channels) {
      if (sparse_event_channel_index(name) < 0) {
        throw std::invalid_argument("predictor.channels contains unknown channel: " + name);
      }
      validate_channel(channel, name);
    }
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
  ExperimentConfig copy = config;
  if (copy.action_table.actions.empty() && !copy.action_table.builtin.empty()) {
    copy.action_table = ControllerActionTable::make_builtin(copy.action_table.builtin);
  }
  return hash_string(stable_json(copy));
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
