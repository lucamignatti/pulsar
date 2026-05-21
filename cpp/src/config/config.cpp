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
  };
}

void from_json(const json& j, OutcomeConfig& value) {
  value.score = j.value("score", 1.0F);
  value.concede = j.value("concede", -1.0F);
  value.neutral = j.value("neutral", 0.0F);
  value.neutral_no_touch = j.value("neutral_no_touch", -1.0F);
}

void to_json(json& j, const MechanicRewardConfig& value) {
  j = json{
      {"kickoff_first_touch", value.kickoff_first_touch},
      {"kickoff_50_50_touch", value.kickoff_50_50_touch},
      {"speed_flip", value.speed_flip},
      {"wavedash", value.wavedash},
      {"chain_dash_bonus", value.chain_dash_bonus},
      {"half_flip", value.half_flip},
      {"wall_dash", value.wall_dash},
      {"air_dribble_base", value.air_dribble_base},
      {"air_dribble_scale", value.air_dribble_scale},
      {"flip_reset", value.flip_reset},
      {"ceiling_shot", value.ceiling_shot},
      {"double_tap", value.double_tap},
      {"preflip", value.preflip},
      {"redirect", value.redirect},
      {"pogo", value.pogo},
      {"pinch", value.pinch},
      {"team_pinch", value.team_pinch},
  };
}

void from_json(const json& j, MechanicRewardConfig& value) {
  value.kickoff_first_touch = j.value("kickoff_first_touch", 0.0F);
  value.kickoff_50_50_touch = j.value("kickoff_50_50_touch", 0.0F);
  value.speed_flip = j.value("speed_flip", 0.0F);
  value.wavedash = j.value("wavedash", 0.0F);
  value.chain_dash_bonus = j.value("chain_dash_bonus", 0.0F);
  value.half_flip = j.value("half_flip", 0.0F);
  value.wall_dash = j.value("wall_dash", 0.0F);
  value.air_dribble_base = j.value("air_dribble_base", 0.0F);
  value.air_dribble_scale = j.value("air_dribble_scale", 0.0F);
  value.flip_reset = j.value("flip_reset", 0.0F);
  value.ceiling_shot = j.value("ceiling_shot", 0.0F);
  value.double_tap = j.value("double_tap", 0.0F);
  value.preflip = j.value("preflip", 0.0F);
  value.redirect = j.value("redirect", 0.0F);
  value.pogo = j.value("pogo", 0.0F);
  value.pinch = j.value("pinch", 0.0F);
  value.team_pinch = j.value("team_pinch", 0.0F);
}

void to_json(json& j, const DenseRewardConfig& value) {
  j = json{
      {"ball_touch_vel_weight", value.ball_touch_vel_weight},
      {"touch_direction_weight", value.touch_direction_weight},
      {"speed_toward_ball_weight", value.speed_toward_ball_weight},
      {"speed_toward_ball_decay", value.speed_toward_ball_decay},
      {"face_ball_weight", value.face_ball_weight},
      {"air_reward_weight", value.air_reward_weight},
      {"air_reward_ball_z_min", value.air_reward_ball_z_min},
      {"velocity_ball_to_goal_weight", value.velocity_ball_to_goal_weight},
      {"max_ball_speed", value.max_ball_speed},
      {"air_touch_weight", value.air_touch_weight},
      {"air_touch_max_air_time", value.air_touch_max_air_time},
      {"save_boost_weight", value.save_boost_weight},
      {"boost_efficiency_weight", value.boost_efficiency_weight},
      {"boost_used_weight", value.boost_used_weight},
      {"defensive_positioning_weight", value.defensive_positioning_weight},
      {"defensive_positioning_decay", value.defensive_positioning_decay},
      {"shot_accuracy_weight", value.shot_accuracy_weight},
      {"boost_pickup_big_weight", value.boost_pickup_big_weight},
      {"boost_pickup_small_weight", value.boost_pickup_small_weight},
      {"boost_pickup_big_threshold", value.boost_pickup_big_threshold},
      {"boost_pickup_cap_per_episode", value.boost_pickup_cap_per_episode},
      {"air_touch_cap_per_episode", value.air_touch_cap_per_episode},
      {"possession_chain_weight", value.possession_chain_weight},
      {"possession_chain_scale", value.possession_chain_scale},
      {"possession_chain_timeout_ticks", value.possession_chain_timeout_ticks},
      {"possession_proximity_weight", value.possession_proximity_weight},
      {"possession_speed_toward_ball_weight", value.possession_speed_toward_ball_weight},
      {"possession_face_ball_weight", value.possession_face_ball_weight},
      {"possession_window_ticks", value.possession_window_ticks},
      {"possession_distance_decay", value.possession_distance_decay},
      {"flat_touch_weight", value.flat_touch_weight},
      {"team_spirit", value.team_spirit},
  };
}

void from_json(const json& j, DenseRewardConfig& value) {
  value.ball_touch_vel_weight = j.value("ball_touch_vel_weight", 0.0F);
  value.touch_direction_weight = j.value("touch_direction_weight", 0.0F);
  value.speed_toward_ball_weight = j.value("speed_toward_ball_weight", 0.0F);
  value.speed_toward_ball_decay = j.value("speed_toward_ball_decay", 300.0F);
  value.face_ball_weight = j.value("face_ball_weight", 0.0F);
  value.air_reward_weight = j.value("air_reward_weight", 0.0F);
  value.air_reward_ball_z_min = j.value("air_reward_ball_z_min", 200.0F);
  value.velocity_ball_to_goal_weight = j.value("velocity_ball_to_goal_weight", 0.0F);
  value.max_ball_speed = j.value("max_ball_speed", 6000.0F);
  value.air_touch_weight = j.value("air_touch_weight", 0.0F);
  value.air_touch_max_air_time = j.value("air_touch_max_air_time", 1.75F);
  value.save_boost_weight = j.value("save_boost_weight", 0.0F);
  value.boost_efficiency_weight = j.value("boost_efficiency_weight", 0.0F);
  value.boost_used_weight = j.value("boost_used_weight", 0.0F);
  value.defensive_positioning_weight = j.value("defensive_positioning_weight", 0.0F);
  value.defensive_positioning_decay = j.value("defensive_positioning_decay", 300.0F);
  value.shot_accuracy_weight = j.value("shot_accuracy_weight", 0.0F);
  value.boost_pickup_big_weight = j.value("boost_pickup_big_weight", 0.0F);
  value.boost_pickup_small_weight = j.value("boost_pickup_small_weight", 0.0F);
  value.boost_pickup_big_threshold = j.value("boost_pickup_big_threshold", 0.5F);
  value.boost_pickup_cap_per_episode = j.value("boost_pickup_cap_per_episode", 0.0F);
  value.air_touch_cap_per_episode = j.value("air_touch_cap_per_episode", 0.0F);
  value.possession_chain_weight = j.value("possession_chain_weight", 0.0F);
  value.possession_chain_scale = j.value("possession_chain_scale", 0.0F);
  value.possession_chain_timeout_ticks = j.value("possession_chain_timeout_ticks", 360);
  value.possession_proximity_weight = j.value("possession_proximity_weight", 0.0F);
  value.possession_speed_toward_ball_weight = j.value("possession_speed_toward_ball_weight", 0.0F);
  value.possession_face_ball_weight = j.value("possession_face_ball_weight", 0.0F);
  value.possession_window_ticks = j.value("possession_window_ticks", 180);
  value.possession_distance_decay = j.value("possession_distance_decay", 1000.0F);
  value.flat_touch_weight = j.value("flat_touch_weight", 0.0F);
  value.team_spirit = j.value("team_spirit", 0.0F);
}

void to_json(json& j, const CurriculumStageConfig& value) {
  j = json{
      {"name", value.name},
      {"mode", value.mode},
      {"mode_allocation", value.mode_allocation},
      {"outcome_override", value.outcome_override},
      {"mechanic_rewards_override", value.mechanic_rewards_override},
      {"dense_rewards_override", value.dense_rewards_override},
      {"unlocked_mechanics", value.unlocked_mechanics},
      {"learning_rate", value.learning_rate},
      {"min_agent_steps", value.min_agent_steps},
      {"rolling_window_size", value.rolling_window_size},
      {"consecutive_success_threshold", value.consecutive_success_threshold},
      {"min_completed_episodes_per_mode", value.min_completed_episodes_per_mode},
      {"required_touch_episode_rate", value.required_touch_episode_rate},
      {"required_multi_touch_episode_rate", value.required_multi_touch_episode_rate},
      {"required_scored_episode_rate", value.required_scored_episode_rate},
  };
  if (!value.mode_scored_thresholds.empty()) {
    j["mode_scored_thresholds"] = value.mode_scored_thresholds;
  }
  if (!value.mode_touch_thresholds.empty()) {
    j["mode_touch_thresholds"] = value.mode_touch_thresholds;
  }
  if (!value.mode_multi_touch_thresholds.empty()) {
    j["mode_multi_touch_thresholds"] = value.mode_multi_touch_thresholds;
  }
}

void from_json(const json& j, CurriculumStageConfig& value) {
  value.name = j.value("name", std::string{});
  value.mode = j.value("mode", std::string{"1v1"});
  value.outcome_override = j.value("outcome_override", OutcomeConfig{});
  value.mechanic_rewards_override = j.value("mechanic_rewards_override", MechanicRewardConfig{});
  value.dense_rewards_override = j.value("dense_rewards_override", DenseRewardConfig{});
  value.unlocked_mechanics = j.value("unlocked_mechanics", std::vector<std::string>{});
  value.learning_rate = j.value("learning_rate", 0.0001F);
  value.min_agent_steps = j.value("min_agent_steps", 20'000'000LL);
  const int window_size = j.value("promotion_window_updates", j.value("rolling_window_size", 10));
  value.rolling_window_size = j.value("rolling_window_size", window_size);
  value.consecutive_success_threshold = j.value("consecutive_success_threshold", 5);
  value.min_completed_episodes_per_mode = j.value("min_completed_episodes_per_mode", 0);
  value.required_touch_episode_rate = j.value("required_touch_episode_rate", 0.0F);
  value.required_multi_touch_episode_rate = j.value("required_multi_touch_episode_rate", 0.0F);
  value.required_scored_episode_rate = j.value("required_scored_episode_rate", 0.0F);
  if (j.contains("mode_scored_thresholds")) {
    value.mode_scored_thresholds = j["mode_scored_thresholds"].get<std::map<std::string, float>>();
  }
  if (j.contains("mode_touch_thresholds")) {
    value.mode_touch_thresholds = j["mode_touch_thresholds"].get<std::map<std::string, float>>();
  }
  if (j.contains("mode_multi_touch_thresholds")) {
    value.mode_multi_touch_thresholds = j["mode_multi_touch_thresholds"].get<std::map<std::string, float>>();
  }

  // mode_allocation: if present in JSON, use it; otherwise derive from single "mode" field
  if (j.contains("mode_allocation")) {
    value.mode_allocation = j["mode_allocation"].get<std::map<std::string, float>>();
  }
  if (value.mode_allocation.empty()) {
    // backward compat: use the single mode field with 100% allocation
    value.mode_allocation[value.mode] = 1.0F;
  }
}

void validate_mode_allocation(const std::map<std::string, float>& alloc, const std::string& stage_name) {
  if (alloc.empty()) {
    throw std::invalid_argument("curriculum stage " + stage_name + " has empty mode_allocation");
  }
  float sum = 0.0F;
  for (const auto& [mode, frac] : alloc) {
    if (frac < 0.0F) {
      throw std::invalid_argument("negative allocation for mode " + mode + " in stage " + stage_name);
    }
    if (frac > 1.0F) {
      throw std::invalid_argument("allocation > 1.0 for mode " + mode + " in stage " + stage_name);
    }
    if (mode != "1v1" && mode != "2v2" && mode != "3v3") {
      throw std::invalid_argument("unknown mode \"" + mode + "\" in stage " + stage_name);
    }
    sum += frac;
  }
  if (std::abs(sum - 1.0F) > 0.01F) {
    throw std::invalid_argument(
        "mode allocations sum to " + std::to_string(sum) + " (expected 1.0) in stage " + stage_name);
  }
}

void to_json(json& j, const CurriculumConfig& value) {
  j = json{
      {"enabled", value.enabled},
      {"stages", value.stages},
  };
}

void from_json(const json& j, CurriculumConfig& value) {
  value.enabled = j.value("enabled", false);
  value.stages = j.value("stages", std::vector<CurriculumStageConfig>{});
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
      {"disable_truncation", value.disable_truncation},
      {"no_touch_timeout_seconds", value.no_touch_timeout_seconds},
      {"no_touch_timeout_only_before_first_touch", value.no_touch_timeout_only_before_first_touch},
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
  value.disable_truncation = j.value("disable_truncation", false);
  value.no_touch_timeout_seconds = j.value("no_touch_timeout_seconds", 10.0F);
  value.no_touch_timeout_only_before_first_touch = j.value("no_touch_timeout_only_before_first_touch", false);
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
      {"plasticity", value.plasticity},
      {"plasticity_interval", value.plasticity_interval},
      {"plasticity_shrink", value.plasticity_shrink},
      {"plasticity_noise", value.plasticity_noise},
      {"pcgrad", value.pcgrad},
      {"overlap_collection_update", value.overlap_collection_update},
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
  value.plasticity = j.value("plasticity", false);
  value.plasticity_interval = j.value("plasticity_interval", 40);
  value.plasticity_shrink = j.value("plasticity_shrink", 0.999F);
  value.plasticity_noise = j.value("plasticity_noise", 1.0e-4F);
  value.pcgrad = j.value("pcgrad", false);
  value.overlap_collection_update = j.value("overlap_collection_update", false);
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

void to_json(json& j, const ExperimentConfig& value) {
  j = json{
      {"schema_version", value.schema_version},
      {"obs_schema_version", value.obs_schema_version},
      {"env", value.env},
      {"outcome", value.outcome},
      {"mechanic_rewards", value.mechanic_rewards},
      {"dense_rewards", value.dense_rewards},
      {"curriculum", value.curriculum},
      {"action_table", value.action_table},
      {"model", value.model},
      {"ppo", value.ppo},
      {"goal_mapping", value.goal_mapping},
      {"goal_critic", value.goal_critic},
      {"es_lora", value.es_lora},
      {"self_play_league", value.self_play_league},
      {"wandb", value.wandb},
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
  value.mechanic_rewards = j.value("mechanic_rewards", MechanicRewardConfig{});
  value.dense_rewards = j.value("dense_rewards", DenseRewardConfig{});
  value.curriculum = j.value("curriculum", CurriculumConfig{});
  value.action_table = j.value("action_table", ActionTableConfig{});
  value.model = j.value("model", ModelConfig{});
  value.ppo = j.value("ppo", PPOConfig{});
  value.goal_mapping = j.value("goal_mapping", GoalMappingConfig{});
  value.goal_critic = j.value("goal_critic", GoalCriticConfig{});
  value.es_lora = j.value("es_lora", ESLoraConfig{});
  value.self_play_league = j.value("self_play_league", SelfPlayLeagueConfig{});
  value.wandb = j.value("wandb", WandbConfig{});
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
  if (config.env.team_size <= 0 || config.env.team_size > 4) {
    throw std::invalid_argument("env.team_size must be between 1 and 4.");
  }
  if (config.env.tick_rate != 120) {
    std::cerr << "Warning: env.tick_rate is currently ignored. Simulation hardcodes 120 Hz.\n";
  }

  if (config.curriculum.enabled) {
    for (const auto& stage : config.curriculum.stages) {
      if (!stage.mode_allocation.empty()) {
        validate_mode_allocation(stage.mode_allocation, stage.name);
      }
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
