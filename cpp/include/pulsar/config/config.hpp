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
};

struct MechanicRewardConfig {
  float kickoff_first_touch = 0.0F;
  float speed_flip = 0.0F;
  float wavedash = 0.0F;
  float chain_dash_bonus = 0.0F;
  float half_flip = 0.0F;
  float wall_dash = 0.0F;
  float air_dribble_base = 0.0F;
  float air_dribble_scale = 0.0F;
  float flip_reset = 0.0F;
  float ceiling_shot = 0.0F;
  float double_tap = 0.0F;
  float preflip = 0.0F;
  float redirect = 0.0F;
  float pogo = 0.0F;
  float pinch = 0.0F;
  float team_pinch = 0.0F;
  float mechanic_reward_cap_per_episode = 0.10F;
};

struct DenseRewardConfig {
  float ball_touch_vel_weight = 0.0F;
  float touch_direction_weight = 0.0F;
  float speed_toward_ball_weight = 0.0F;
  float speed_toward_ball_decay = 300.0F;
  float face_ball_weight = 0.0F;
  float air_reward_weight = 0.0F;
  float air_reward_ball_z_min = 200.0F;
  float velocity_ball_to_goal_weight = 0.0F;
  float max_ball_speed = 6000.0F;
  float air_touch_weight = 0.0F;
  float air_touch_max_air_time = 1.75F;
  float save_boost_weight = 0.0F;
  float boost_efficiency_weight = 0.0F;
  float boost_used_weight = 0.0F;
  float defensive_positioning_weight = 0.0F;
  float defensive_positioning_decay = 300.0F;
  float shot_accuracy_weight = 0.0F;
  float boost_pickup_big_weight = 0.0F;
  float boost_pickup_small_weight = 0.0F;
  float boost_pickup_big_threshold = 0.5F;
  float boost_pickup_cap_per_episode = 0.0F;
  float air_touch_cap_per_episode = 0.0F;
  float possession_chain_weight = 0.0F;
  float possession_chain_scale = 0.0F;
  int possession_chain_timeout_ticks = 360;
  float possession_proximity_weight = 0.0F;
  float possession_speed_toward_ball_weight = 0.0F;
  float possession_face_ball_weight = 0.0F;
  int possession_window_ticks = 180;
  float possession_distance_decay = 1000.0F;
  float dense_reward_cap_per_episode = 0.0F;
  float flat_touch_weight = 0.0F;
  float team_spirit = 0.0F;
};

struct CurriculumStageConfig {
  std::string name;
  std::string mode = "1v1";
  OutcomeConfig outcome_override{};
  MechanicRewardConfig mechanic_rewards_override{};
  DenseRewardConfig dense_rewards_override{};
  std::vector<std::string> unlocked_mechanics;
  std::map<std::string, float> mode_allocation;
  float learning_rate = 0.0001F;
  int64_t min_agent_steps = 20'000'000LL;
  int rolling_window_size = 10;
  int consecutive_success_threshold = 5;
  int min_completed_episodes_per_mode = 0;
  float required_touch_episode_rate = 0.0F;
  float required_multi_touch_episode_rate = 0.0F;
  float required_scored_episode_rate = 0.0F;
  // Per-mode overrides for promotion thresholds.  When non-empty, the value
  // for a given mode takes precedence over the scalar fields above.
  std::map<std::string, float> mode_scored_thresholds{};
  std::map<std::string, float> mode_touch_thresholds{};
  std::map<std::string, float> mode_multi_touch_thresholds{};
};

struct CurriculumConfig {
  bool enabled = false;
  std::vector<CurriculumStageConfig> stages;
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
  bool disable_truncation = false;
  float no_touch_timeout_seconds = 10.0F;
  bool no_touch_timeout_only_before_first_touch = false;
  bool spawn_opponents = true;
  bool randomize_kickoffs = true;
  std::uint64_t seed = 0;
};

struct ModelConfig {
  int observation_dim = 132;
  int action_dim = 90;
  bool use_layer_norm = true;
  int encoder_dim = 640;
  int num_encoder_blocks = 5;
  int sequence_length = 16;
  int max_forward_samples = 0;
  int value_hidden_dim = 256;
  int policy_hidden_dim = 0;
};

struct GoalMappingConfig {
  float arena_max_distance = 8192.0F;
};

struct GoalCriticConfig {
  int goal_dim = 3;
  int hidden_dim = 256;
  int embedding_dim = 64;
  float logsumexp_penalty_coeff = 0.01F;
  float lambda_Zg = 1.0F;
  float lambda_goal_actor = 0.1F;
  int contrastive_batch_size = 2048;
  int max_future_horizon = 256;
};

struct ESLoraConfig {
  int rank = 4;
  float lora_alpha = 4.0F;
  int population_size = 8;
  float sigma_ES = 0.05F;
  float eta_ES = 0.003F;
  int es_interval = 25;
  int eval_episodes_per_member = 2;
  int eval_num_envs = 8;
  int eval_rollout_length = 450;
  float beta_KL = 0.01F;
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
  float entropy_floor = 0.0F;
  float entropy_floor_coef = 0.0F;
  float value_coef = 1.0F;
  float value_loss_delta = 10.0F;
  float gamma = 0.99F;
  float gae_lambda = 0.95F;
  float learning_rate = 3.0e-4F;
  float max_grad_norm = 1.0F;
  std::string device = "cpu";
  int checkpoint_interval = 10;
  int max_rolling_checkpoints = 5;
  bool synchronize_cuda_timing = false;
  bool cuda_amp = false;
  bool adaptive_entropy = false;
  float entropy_decay_score = 0.60F;
  float entropy_low_coef = 0.005F;
  float max_policy_log_ratio = 5.0F;
  float target_kl = 0.0F;
  float max_preclip_grad_norm = 0.0F;
  bool plasticity = false;
  int plasticity_interval = 40;
  float plasticity_shrink = 0.999F;
  float plasticity_noise = 1.0e-4F;
  bool pcgrad = false;
  bool overlap_collection_update = false;
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
  std::string run_id{};
};

struct ExperimentConfig {
  int schema_version = 6;
  int obs_schema_version = 2;
  EnvConfig env{};
  OutcomeConfig outcome{};
  MechanicRewardConfig mechanic_rewards{};
  DenseRewardConfig dense_rewards{};
  CurriculumConfig curriculum{};
  ActionTableConfig action_table{};
  ModelConfig model{};
  PPOConfig ppo{};
  GoalMappingConfig goal_mapping{};
  GoalCriticConfig goal_critic{};
  ESLoraConfig es_lora{};
  SelfPlayLeagueConfig self_play_league{};
  WandbConfig wandb{};
};

struct CheckpointMetadata {
  int schema_version = 6;
  int obs_schema_version = 2;
  std::string config_hash{};
  std::string action_table_hash{};
  std::string architecture_name = "mamba2_goal_appo";
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
void to_json(nlohmann::json& j, const MechanicRewardConfig& value);
void from_json(const nlohmann::json& j, MechanicRewardConfig& value);
void to_json(nlohmann::json& j, const DenseRewardConfig& value);
void from_json(const nlohmann::json& j, DenseRewardConfig& value);
void to_json(nlohmann::json& j, const CurriculumStageConfig& value);
void from_json(const nlohmann::json& j, CurriculumStageConfig& value);
void to_json(nlohmann::json& j, const CurriculumConfig& value);
void from_json(const nlohmann::json& j, CurriculumConfig& value);
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

}  // namespace pulsar
