#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pulsar/core/types.hpp"

namespace pulsar {

struct OutcomeConfig {
  float score = 1.0F;
  float concede = -1.0F;
  float neutral = 0.0F;
  float neutral_no_touch = -1.0F;
  float team_spirit = 0.0F;
  float step_penalty = 0.0F;
};

struct ActionTableConfig {
  std::string builtin = "rlgym_lookup_v1";
  std::vector<ControllerState> actions{};
  bool refined_action_masking = false;
};

struct EnvConfig {
  std::string mode = "soccar";
  std::string collision_meshes_path = "collision_meshes";
  int team_size = 2;
  int tick_skip = 8;
  int half_tick_skip = 4;
  int tick_rate = 120;
  int max_episode_ticks = 2250;
  bool disable_truncation = false;
  float no_touch_timeout_seconds = 10.0F;
  bool no_touch_timeout_only_before_first_touch = false;
  bool spawn_opponents = true;
  bool randomize_kickoffs = true;
  std::uint64_t seed = 0;
  bool obs_x_mirror = false;
  bool obs_local_frame = false;
  bool obs_relative_goals = false;
  bool obs_proximity_boosts = false;
  bool obs_explicit_kinematics = false;
  bool obs_flip_decay = false;
  bool obs_action_history = false;
  bool obs_ball_prediction = false;
};

struct ModelConfig {
  int observation_dim = 132;
  int action_dim = 90;
  bool use_layer_norm = true;
  int encoder_dim = 256;
  int num_encoder_blocks = 2;
  int sequence_length = 16;
  int max_forward_samples = 0;
  int value_hidden_dim = 256;
  int policy_hidden_dim = 0;
};

struct GoalMappingConfig {
  float arena_max_distance = 8192.0F;
  int goal_dim = 4;  // must equal goal_critic.goal_dim
};

struct GoalCriticConfig {
  int goal_dim = 4;
  int hidden_dim = 256;
  int embedding_dim = 64;
  float logsumexp_penalty_coeff = 0.01F;
  float lambda_Zg = 1.0F;
  float lambda_goal_actor = 0.1F;
  int contrastive_batch_size = 2048;
  int max_future_horizon = 256;
  float temperature = 0.1F;
};

struct ESLoraConfig {
  int rank = 4;
  float lora_alpha = 4.0F;
  int population_size = 8;
  int virtual_population_waves = 1;
  float sigma_ES = 0.05F;
  float eta_ES = 0.003F;
  int es_interval = 25;
  int eval_shards = 0;
  int eval_workers = 0;
  int eval_episodes_per_member = 2;
  int eval_num_envs = 8;
  int eval_rollout_length = 450;
  int kl_eval_stride = 1;
  float beta_KL = 0.01F;
  bool rank_transform = false;
  bool antithetic_sampling = true;
  bool update_norm_clip = true;
  float max_update_norm = 0.002F;
  float max_kl_mean = 0.01F;
  bool require_fitness_signal = true;
  float min_fitness_std = 1.0e-6F;
};

struct PPOConfig {
  int num_envs = 64;
  int collection_workers = 0;
  int collection_shards = 1;
  std::string init_checkpoint{};
  int rollout_length = 256;
  int minibatch_size = 32768;
  int update_epochs = 3;
  int optimizer_accumulation_steps = 1;
  float clip_range = 0.2F;
  float entropy_coef = 0.01F;
  float value_coef = 1.0F;
  float value_loss_delta = 10.0F;
  float gamma = 0.99F;
  float gae_lambda = 0.95F;
  float learning_rate = 3.0e-4F;
  bool overbatching = true;
  float policy_temperature = 1.0F;
  float max_grad_norm = 1.0F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
  int max_rolling_checkpoints = 5;
  bool synchronize_cuda_timing = false;
  bool cuda_amp = false;
  float max_policy_log_ratio = 5.0F;
  float target_kl = 0.0F;
  bool overlap_collection_update = false;
  bool value_clipping = false;
  float value_clip_range = 0.2F;
  float weight_decay = 0.0F;
  int anchor_eval_interval = 50;        // 0 = disabled; run anchor eval every N updates
  int anchor_eval_envs = 8;             // number of envs for anchor eval
  int anchor_eval_steps = 1800;         // steps per anchor eval run
  float anchor_update_threshold = 0.65F; // replace anchor when win_rate >= this (0=always, 1=never)
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
  std::string run_id{};
};

struct CurriculumModeConfig {
  std::string mode;           // "1v1", "2v2", or "3v3"
  int shards = 1;
  int num_envs = 0;           // total envs for this mode, distributed across shards
  int collection_workers = 0; // 0 = derive from ppo.collection_workers proportionally
};

struct CurriculumConfig {
  // Non-empty enables concurrent multi-mode training; each entry gets its own
  // shard group so all modes collect simultaneously. Empty = single-mode from
  // env.team_size (backward-compatible).
  std::vector<CurriculumModeConfig> modes;
};

struct WorldModelConfig {
  int latent_dim = 256;
  int stochastic_dim = 32;
  int obs_encoder_dim = 128;
  int num_consistency_steps = 8;
  float kl_free_bits = 1.0F;
  float kl_weight = 0.1F;
  float consistency_weight = 0.5F;
  float icm_weight = 0.5F;
  float icm_inverse_weight = 0.1F;
  float icm_uncertainty_threshold = 0.1F;
  float intrinsic_reward_scale = 0.01F;
  float intrinsic_anneal_steps = 1000.0F;
};

struct ReplayBufferConfig {
  int capacity = 1048576;
  int min_fill_before_sampling = 8192;
  int segment_length = 64;
  float her_relabel_fraction = 0.8F;
  int her_future_k = 4;
  float goal_reach_epsilon = 0.15F;
  bool epsilon_position_only = true;
};

struct SACCriticConfig {
  int hidden_dim = 512;
  int num_layers = 3;
  float learning_rate = 3.0e-4F;
  float gamma = 0.99F;
  float tau = 0.005F;
  float temperature = 0.2F;
  int update_frequency = 4;
  int batch_size = 2048;
  float lql_lambda_ub = 1.0F;
  float lql_lambda_lb = 1.0F;
};

struct PBRSConfig {
  float initial_weight = 0.1F;
  float final_weight = 1.0F;
  float warmup_updates = 500.0F;
  int recompute_interval = 4;
};

struct SubgoalPlannerConfig {
  int commit_horizon = 12;
  int candidate_buffer_size = 256;
  int imagination_depth = 15;
  float reachability_weight = 0.5F;
  float uncertainty_penalty = 0.1F;
  float min_reachability = 0.15F;
  float shaped_reward_scale = 0.1F;
};

struct ExperimentConfig {
  int schema_version = 6;
  int obs_schema_version = 2;
  EnvConfig env{};
  OutcomeConfig outcome{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  PPOConfig ppo{};
  GoalMappingConfig goal_mapping{};
  GoalCriticConfig goal_critic{};
  ESLoraConfig es_lora{};
  WandbConfig wandb{};
  CurriculumConfig curriculum{};
  WorldModelConfig world_model{};
  ReplayBufferConfig replay_buffer{};
  SACCriticConfig sac_critic{};
  PBRSConfig pbrs{};
  SubgoalPlannerConfig subgoal_planner{};
};

struct CheckpointMetadata {
  int schema_version = 6;
  int obs_schema_version = 2;
  std::string config_hash{};
  std::string action_table_hash{};
  std::string architecture_name = "mamba2_goal";
  std::string device = "cpu";
  std::int64_t global_step = 0;
  std::int64_t update_index = 0;
  std::vector<std::string> critic_heads{};
  nlohmann::json extra{};
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
void to_json(nlohmann::json& j, const ESLoraConfig& value);
void from_json(const nlohmann::json& j, ESLoraConfig& value);
void to_json(nlohmann::json& j, const PPOConfig& value);
void from_json(const nlohmann::json& j, PPOConfig& value);
void to_json(nlohmann::json& j, const WandbConfig& value);
void from_json(const nlohmann::json& j, WandbConfig& value);
void to_json(nlohmann::json& j, const CurriculumModeConfig& value);
void from_json(const nlohmann::json& j, CurriculumModeConfig& value);
void to_json(nlohmann::json& j, const CurriculumConfig& value);
void from_json(const nlohmann::json& j, CurriculumConfig& value);
void to_json(nlohmann::json& j, const WorldModelConfig& value);
void from_json(const nlohmann::json& j, WorldModelConfig& value);
void to_json(nlohmann::json& j, const ReplayBufferConfig& value);
void from_json(const nlohmann::json& j, ReplayBufferConfig& value);
void to_json(nlohmann::json& j, const SACCriticConfig& value);
void from_json(const nlohmann::json& j, SACCriticConfig& value);
void to_json(nlohmann::json& j, const PBRSConfig& value);
void from_json(const nlohmann::json& j, PBRSConfig& value);
void to_json(nlohmann::json& j, const SubgoalPlannerConfig& value);
void from_json(const nlohmann::json& j, SubgoalPlannerConfig& value);
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

}  // namespace pulsar
