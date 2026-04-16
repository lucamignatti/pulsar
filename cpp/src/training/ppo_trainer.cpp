#include "pulsar/training/ppo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pulsar {
namespace {

torch::Tensor gather_state_tensor(const torch::Tensor& tensor, const torch::Tensor& agent_indices) {
  if (!tensor.defined()) {
    return tensor;
  }
  return tensor.index_select(0, agent_indices);
}

torch::Device resolve_runtime_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

RolloutStorage make_rollout_storage(
    const ExperimentConfig& config,
    int num_agents,
    int action_dim) {
  const torch::Device device = resolve_runtime_device(config.ppo.device);
  if (device.is_cuda()) {
    c10::DeviceGuard device_guard(device);
    return RolloutStorage(
        config.ppo.rollout_length,
        num_agents,
        config.model.observation_dim,
        action_dim,
        device);
  }
  return RolloutStorage(
      config.ppo.rollout_length,
      num_agents,
      config.model.observation_dim,
      action_dim,
      device);
}

std::string stable_model_signature(const ExperimentConfig& config) {
  nlohmann::json j = {
      {"model", config.model},
      {"value_num_atoms", config.ppo.value_num_atoms},
      {"value_v_min", config.ppo.value_v_min},
      {"value_v_max", config.ppo.value_v_max},
  };
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
}

void validate_init_checkpoint_compatibility(
    const CheckpointMetadata& metadata,
    const ExperimentConfig& checkpoint_config,
    const ExperimentConfig& active_config) {
  if (metadata.schema_version != active_config.schema_version) {
    throw std::runtime_error("Init checkpoint schema_version does not match the active config.");
  }
  if (metadata.obs_schema_version != active_config.obs_schema_version) {
    throw std::runtime_error("Init checkpoint obs_schema_version does not match the active config.");
  }
  if (metadata.action_table_hash != action_table_hash(active_config.action_table)) {
    throw std::runtime_error("Init checkpoint action table hash does not match the active config.");
  }
  if (stable_model_signature(checkpoint_config) != stable_model_signature(active_config)) {
    throw std::runtime_error("Init checkpoint model/value signature does not match the active config.");
  }
}

void validate_aux_checkpoint_compatibility(
    const CheckpointMetadata& metadata,
    const ExperimentConfig& checkpoint_config,
    const ExperimentConfig& active_config,
    const char* checkpoint_name) {
  if (metadata.schema_version != active_config.schema_version) {
    throw std::runtime_error(std::string(checkpoint_name) + " schema_version does not match the active config.");
  }
  if (metadata.obs_schema_version != active_config.obs_schema_version) {
    throw std::runtime_error(std::string(checkpoint_name) + " obs_schema_version does not match the active config.");
  }
  if (metadata.action_table_hash != action_table_hash(active_config.action_table)) {
    throw std::runtime_error(std::string(checkpoint_name) + " action table hash does not match the active config.");
  }
  if (stable_model_signature(checkpoint_config) != stable_model_signature(active_config)) {
    throw std::runtime_error(std::string(checkpoint_name) + " model/value signature does not match the active config.");
  }
}

bool same_checkpoint_directory(const std::string& lhs, const std::string& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  namespace fs = std::filesystem;
  const fs::path lhs_path(lhs);
  const fs::path rhs_path(rhs);
  std::error_code ec;
  if (fs::exists(lhs_path, ec) && fs::exists(rhs_path, ec)) {
    return fs::equivalent(lhs_path, rhs_path, ec) && !ec;
  }
  return lhs_path.lexically_normal() == rhs_path.lexically_normal();
}

void freeze_model_parameters(const SharedActorCritic& model) {
  for (auto& parameter : model->parameters()) {
    parameter.set_requires_grad(false);
  }
}

std::vector<torch::Tensor> collect_head_parameters(
    const SharedActorCritic& model,
    const std::string& prefix) {
  std::vector<torch::Tensor> params;
  for (const auto& item : model->named_parameters(true)) {
    if (item.key().rfind(prefix, 0) == 0) {
      params.push_back(item.value());
    }
  }
  return params;
}

std::vector<torch::Tensor> collect_trunk_parameters(const SharedActorCritic& model) {
  std::vector<torch::Tensor> params;
  for (const auto& item : model->named_parameters(true)) {
    if (item.key().rfind("policy_head.", 0) == 0 ||
        item.key().rfind("value_head.", 0) == 0 ||
        item.key().rfind("next_goal_head.", 0) == 0) {
      continue;
    }
    params.push_back(item.value());
  }
  return params;
}

torch::Tensor weighted_cross_entropy(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& weights,
    double label_smoothing,
    const torch::Tensor& class_weights = {}) {
  auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
  if (label_smoothing > 0.0) {
    options = options.label_smoothing(label_smoothing);
  }
  if (class_weights.defined()) {
    options = options.weight(class_weights);
  }
  const torch::Tensor per_sample = torch::nn::functional::cross_entropy(logits, labels, options);
  const torch::Tensor normalized_weights = weights / weights.mean().clamp_min(1.0e-6);
  return (per_sample * normalized_weights).mean();
}

double weighted_accuracy(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& weights) {
  const torch::Tensor predictions = logits.argmax(-1);
  const torch::Tensor correct = predictions.eq(labels).to(torch::kFloat32);
  return ((correct * weights).sum() / weights.sum().clamp_min(1.0e-6)).item<double>();
}

double relative_improvement(double active_loss, double candidate_loss) {
  const double scale = std::max(std::abs(active_loss), 1.0e-8);
  return (active_loss - candidate_loss) / scale;
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics) {
  nlohmann::json line = {
      {"update", update_index},
      {"global_step", global_step},
      {"collection_agent_steps_per_second", metrics.collection_agent_steps_per_second},
      {"update_agent_steps_per_second", metrics.update_agent_steps_per_second},
      {"overall_agent_steps_per_second", metrics.overall_agent_steps_per_second},
      {"update_seconds", metrics.update_seconds},
      {"reward_mean", metrics.reward_mean},
      {"policy_loss", metrics.policy_loss},
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"value_entropy", metrics.value_entropy},
      {"value_variance", metrics.value_variance},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"done_reset_seconds", metrics.done_reset_seconds},
      {"reward_model_seconds", metrics.reward_model_seconds},
      {"rollout_append_seconds", metrics.rollout_append_seconds},
      {"gae_seconds", metrics.gae_seconds},
      {"ppo_forward_backward_seconds", metrics.ppo_forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
      {"ngp_promotion_index", metrics.ngp_promotion_index},
      {"ngp_promoted_global_step", metrics.ngp_promoted_global_step},
      {"ngp_source_global_step", metrics.ngp_source_global_step},
      {"ngp_source_update_index", metrics.ngp_source_update_index},
      {"ngp_online_samples_written", metrics.ngp_online_samples_written},
      {"ngp_online_trajectories_written", metrics.ngp_online_trajectories_written},
      {"ngp_label", metrics.ngp_label},
      {"ngp_checkpoint", metrics.ngp_checkpoint},
      {"ngp_config_hash", metrics.ngp_config_hash},
  };
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }

  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

void append_ngp_promotion_line(
    const std::filesystem::path& checkpoint_dir,
    std::int64_t global_step,
    int update_index,
    const std::string& old_checkpoint,
    const std::string& new_checkpoint,
    const TrainerMetrics& metrics) {
  nlohmann::json line = {
      {"global_step", global_step},
      {"update", update_index},
      {"old_ngp_checkpoint", old_checkpoint},
      {"new_ngp_checkpoint", new_checkpoint},
      {"ngp_promotion_index", metrics.ngp_promotion_index},
      {"ngp_promoted_global_step", metrics.ngp_promoted_global_step},
      {"ngp_source_global_step", metrics.ngp_source_global_step},
      {"ngp_source_update_index", metrics.ngp_source_update_index},
      {"ngp_label", metrics.ngp_label},
      {"ngp_config_hash", metrics.ngp_config_hash},
  };

  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "ngp_promotions.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

}  // namespace

PPOTrainer::PPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root)
    : config_(std::move(config)),
      collector_(std::move(collector)),
      self_play_manager_(std::move(self_play_manager)),
      action_table_(config_.action_table),
      model_(SharedActorCritic(config_.model, config_.ppo)),
      normalizer_(config_.model.observation_dim),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate)),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(collector_->total_agents()),
          collector_->action_dim())),
      device_(resolve_runtime_device(config_.ppo.device)),
      run_output_root_(std::move(run_output_root)),
      ngp_normalizer_(config_.model.observation_dim),
      candidate_ngp_normalizer_(config_.model.observation_dim) {
  if (!collector_) {
    throw std::invalid_argument("PPOTrainer requires a collector.");
  }

  total_agents_ = collector_->total_agents();
  collection_state_ = model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  opponent_collection_state_ = model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
#ifdef USE_ROCM
  use_pinned_host_buffers_ = false;
#else
  use_pinned_host_buffers_ = device_.is_cuda();
#endif
  model_->to(device_);
  normalizer_.to(device_);
  if (config_.reward.online_dataset.enabled && config_.reward.online_dataset.output_dir.empty()) {
    const std::filesystem::path default_output =
        run_output_root_.empty() ? std::filesystem::path("online_ngp") : run_output_root_ / "online_ngp";
    config_.reward.online_dataset.output_dir = default_output.string();
  }
  if (config_.reward.online_dataset.enabled) {
    if (config_.reward.online_dataset.output_dir.empty()) {
      throw std::invalid_argument("reward.online_dataset.output_dir must be set when online dataset export is enabled.");
    }
  }
  if (config_.reward.refresh.enabled && config_.reward.refresh.train_candidate_in_process) {
    if (!config_.reward.online_dataset.enabled) {
      throw std::invalid_argument("reward.online_dataset.enabled must be true for in-process NGP refresh.");
    }
    if (config_.reward.refresh.online_train_fraction <= 0.0F ||
        config_.reward.refresh.online_train_fraction >= 1.0F) {
      throw std::invalid_argument(
          "reward.refresh.online_train_fraction must be in (0, 1) for in-process NGP refresh.");
    }
    if (config_.reward.refresh.anchor_train_manifest.empty() ||
        config_.reward.refresh.anchor_val_manifest.empty()) {
      throw std::invalid_argument(
          "reward.refresh.anchor_train_manifest and anchor_val_manifest must be set for in-process NGP refresh.");
    }
    if (config_.reward.refresh.candidate_epochs <= 0) {
      throw std::invalid_argument("reward.refresh.candidate_epochs must be positive for in-process NGP refresh.");
    }
  }
  const std::filesystem::path init_checkpoint_dir =
      config_.ppo.init_checkpoint.empty() ? std::filesystem::path{} : std::filesystem::path(config_.ppo.init_checkpoint);
  maybe_initialize_from_checkpoint();
  maybe_initialize_ngp_reward();
  maybe_initialize_in_process_ngp_refresh(init_checkpoint_dir);

  if (self_play_manager_ && self_play_manager_->enabled()) {
    collector_->set_self_play_assignment_fn(
        [this](std::size_t env_idx, std::uint64_t seed) {
          return self_play_manager_->sample_assignment(env_idx, seed);
        });
  }
}

PPOTrainer::~PPOTrainer() {
  stop_ngp_refresh_worker();
}

ContinuumState PPOTrainer::replay_state_until(std::int64_t start_step, const torch::Tensor& agent_indices) {
  ContinuumState state = model_->initial_state(agent_indices.size(0), device_);
  if (start_step <= 0) {
    return state;
  }

  torch::NoGradGuard no_grad;
  for (std::int64_t step = 0; step < start_step; ++step) {
    const torch::Tensor obs = rollout_.obs[step].index_select(0, agent_indices);
    const torch::Tensor starts = rollout_.episode_starts[step].index_select(0, agent_indices);
    SequenceOutput out = model_->forward_sequence(obs.unsqueeze(0), std::move(state), starts.unsqueeze(0));
    state = std::move(out.final_state);
  }
  return state;
}

torch::Tensor PPOTrainer::sample_actions(
    const torch::Tensor& logits,
    const torch::Tensor& action_masks,
    bool deterministic,
    torch::Tensor* log_probs) const {
  return sample_masked_actions(logits, action_masks, deterministic, log_probs);
}

torch::Tensor PPOTrainer::actions_to_cpu(const torch::Tensor& actions) const {
  return actions.contiguous().to(torch::kCPU);
}

torch::Tensor PPOTrainer::categorical_projection(const torch::Tensor& returns) const {
  return categorical_value_projection(
      returns,
      config_.ppo.value_v_min,
      config_.ppo.value_v_max,
      config_.ppo.value_num_atoms);
}

torch::Tensor PPOTrainer::confidence_weights(const torch::Tensor& value_logits) const {
  return compute_confidence_weights(model_, config_.ppo, value_logits);
}

torch::Tensor PPOTrainer::adaptive_epsilon(const torch::Tensor& value_logits) const {
  return compute_adaptive_epsilon(model_, config_.ppo, value_logits);
}

void PPOTrainer::maybe_initialize_from_checkpoint() {
  if (config_.ppo.init_checkpoint.empty()) {
    return;
  }

  namespace fs = std::filesystem;
  const fs::path base(config_.ppo.init_checkpoint);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_init_checkpoint_compatibility(metadata, checkpoint_config, config_);

  torch::serialize::InputArchive archive;
  archive.load_from((base / "model.pt").string());
  model_->load(archive);
  normalizer_.load(archive);
  model_->to(device_);
  normalizer_.to(device_);
  if (fs::exists(base / "optimizer.pt")) {
    torch::serialize::InputArchive optimizer_archive;
    optimizer_archive.load_from((base / "optimizer.pt").string());
    optimizer_.load(optimizer_archive);
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  } else {
    resumed_global_step_ = 0;
    resumed_update_index_ = 0;
  }

  if (fs::exists(base / "active_ngp" / "model.pt")) {
    load_ngp_reward_checkpoint(
        (base / "active_ngp").string(),
        metadata.reward_ngp_label,
        metadata.reward_ngp_promotion_index,
        metadata.reward_ngp_promoted_global_step);
  }

  std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
}

void PPOTrainer::load_ngp_reward_checkpoint(
    const std::string& checkpoint_path,
    const std::string& configured_label,
    std::int64_t promotion_index,
    std::int64_t promoted_global_step) {
  namespace fs = std::filesystem;
  const fs::path base(checkpoint_path);
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  validate_aux_checkpoint_compatibility(metadata, checkpoint_config, config_, "NGP checkpoint");

  if (same_checkpoint_directory(checkpoint_path, config_.ppo.init_checkpoint)) {
    ngp_model_ = clone_shared_model(model_, device_);
    ngp_normalizer_ = normalizer_.clone();
  } else {
    ngp_model_ = SharedActorCritic(config_.model, config_.ppo);
    torch::serialize::InputArchive archive;
    archive.load_from((base / "model.pt").string());
    ngp_model_->load(archive);
    ngp_normalizer_.load(archive);
  }
  ngp_model_->to(device_);
  ngp_normalizer_.to(device_);
  freeze_model_parameters(ngp_model_);
  ngp_model_->eval();
  ngp_collection_state_ = ngp_model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  active_ngp_checkpoint_ = base.string();
  active_ngp_label_ = !configured_label.empty() ? configured_label : base.filename().string();
  active_ngp_config_hash_ = metadata.config_hash;
  active_ngp_global_step_ = metadata.global_step;
  active_ngp_update_index_ = metadata.update_index;
  active_ngp_promotion_index_ = promotion_index;
  active_ngp_promoted_global_step_ = promoted_global_step;

  std::cout << "initialized_ngp_reward_from_checkpoint=" << base.string()
            << " ngp_label=" << active_ngp_label_
            << " ngp_promotion_index=" << active_ngp_promotion_index_
            << '\n';
}

void PPOTrainer::maybe_initialize_ngp_reward() {
  if (ngp_model_) {
    return;
  }
  if (config_.reward.ngp_checkpoint.empty()) {
    throw std::runtime_error("reward.ngp_checkpoint must be set for PPO training.");
  }
  load_ngp_reward_checkpoint(config_.reward.ngp_checkpoint, config_.reward.ngp_label, 0, 0);
}

void PPOTrainer::maybe_initialize_in_process_ngp_refresh(const std::filesystem::path& init_checkpoint_dir) {
  if (!config_.reward.refresh.enabled || !config_.reward.refresh.train_candidate_in_process) {
    if (config_.reward.online_dataset.enabled) {
      const std::size_t num_envs = collector_->num_envs();
      const std::size_t agents_per_env = num_envs == 0 ? 0 : total_agents_ / num_envs;
      online_ngp_dataset_writer_ = std::make_unique<OnlineNGPDatasetWriter>(
          config_.reward.online_dataset,
          config_.reward.online_dataset.output_dir,
          config_.model.observation_dim,
          config_.model.action_dim,
          num_envs,
          agents_per_env);
    }
    return;
  }

  const std::size_t num_envs = collector_->num_envs();
  const std::size_t agents_per_env = num_envs == 0 ? 0 : total_agents_ / num_envs;
  if (!ngp_replay_buffer_) {
    ngp_replay_buffer_ = std::make_unique<OnlineNGPReplayBuffer>(
        config_.reward.online_dataset,
        config_.reward.refresh,
        config_.model.observation_dim,
        num_envs,
        agents_per_env);
  }
  if (anchor_train_trajectories_.empty()) {
    anchor_train_trajectories_ = load_ngp_trajectories_from_manifest(config_.reward.refresh.anchor_train_manifest);
  }
  if (anchor_val_trajectories_.empty()) {
    anchor_val_trajectories_ = load_ngp_trajectories_from_manifest(config_.reward.refresh.anchor_val_manifest);
  }

  const std::filesystem::path refresh_state_dir = init_checkpoint_dir / "ngp_refresh_state";
  if (!init_checkpoint_dir.empty() && std::filesystem::exists(refresh_state_dir / "metadata.json")) {
    load_in_process_ngp_refresh_state(refresh_state_dir);
  }
  ensure_candidate_ngp_initialized();
  start_ngp_refresh_worker();
}

void PPOTrainer::ensure_candidate_ngp_initialized() {
  std::lock_guard<std::mutex> lock(candidate_mutex_);
  if (candidate_ngp_model_) {
    return;
  }

  candidate_ngp_model_ = clone_shared_model(ngp_model_, device_);
  candidate_ngp_model_->train();
  candidate_ngp_normalizer_ = ngp_normalizer_.clone();
  candidate_ngp_normalizer_.to(device_);
  candidate_trunk_parameters_ = collect_trunk_parameters(candidate_ngp_model_);
  candidate_ngp_head_parameters_ = collect_head_parameters(candidate_ngp_model_, "next_goal_head.");

  if (!config_.reward.refresh.train_trunk) {
    for (auto& parameter : candidate_trunk_parameters_) {
      parameter.set_requires_grad(false);
    }
  }

  candidate_trunk_optimizer_.reset();
  if (config_.reward.refresh.train_trunk && !candidate_trunk_parameters_.empty()) {
    candidate_trunk_optimizer_ = std::make_unique<torch::optim::AdamW>(
        candidate_trunk_parameters_,
        torch::optim::AdamWOptions(config_.offline_optimization.trunk_learning_rate)
            .weight_decay(config_.offline_optimization.trunk_weight_decay));
  }
  candidate_ngp_head_optimizer_ = std::make_unique<torch::optim::AdamW>(
      candidate_ngp_head_parameters_,
      torch::optim::AdamWOptions(config_.next_goal_predictor.learning_rate)
          .weight_decay(config_.next_goal_predictor.weight_decay));
}

void PPOTrainer::save_model_snapshot(
    SharedActorCritic model,
    const ObservationNormalizer& normalizer,
    const std::filesystem::path& directory,
    std::int64_t global_step,
    std::int64_t update_index) const {
  namespace fs = std::filesystem;
  fs::remove_all(directory);
  fs::create_directories(directory);
  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "continuum_dppo",
          .device = config_.ppo.device,
          .global_step = global_step,
          .update_index = update_index,
      },
      (directory / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  model->save(archive);
  normalizer.save(archive);
  archive.save_to((directory / "model.pt").string());
}

void PPOTrainer::save_candidate_checkpoint(
    const std::filesystem::path& directory,
    std::int64_t global_step,
    std::int64_t update_index) {
  std::lock_guard<std::mutex> lock(candidate_mutex_);
  if (!candidate_ngp_model_) {
    return;
  }
  save_model_snapshot(candidate_ngp_model_, candidate_ngp_normalizer_, directory, global_step, update_index);
  if (candidate_trunk_optimizer_) {
    torch::serialize::OutputArchive trunk_optimizer_archive;
    candidate_trunk_optimizer_->save(trunk_optimizer_archive);
    trunk_optimizer_archive.save_to((directory / "trunk_optimizer.pt").string());
  }
  if (candidate_ngp_head_optimizer_) {
    torch::serialize::OutputArchive head_optimizer_archive;
    candidate_ngp_head_optimizer_->save(head_optimizer_archive);
    head_optimizer_archive.save_to((directory / "ngp_head_optimizer.pt").string());
  }
}

void PPOTrainer::save_in_process_ngp_refresh_state(const std::filesystem::path& directory) {
  if (!config_.reward.refresh.enabled || !config_.reward.refresh.train_candidate_in_process || !ngp_replay_buffer_) {
    return;
  }
  namespace fs = std::filesystem;
  fs::remove_all(directory);
  fs::create_directories(directory);

  {
    std::lock_guard<std::mutex> candidate_lock(candidate_mutex_);
    if (candidate_ngp_model_) {
      save_model_snapshot(candidate_ngp_model_, candidate_ngp_normalizer_, directory / "candidate_ngp", 0, 0);
      if (candidate_trunk_optimizer_) {
        torch::serialize::OutputArchive trunk_optimizer_archive;
        candidate_trunk_optimizer_->save(trunk_optimizer_archive);
        trunk_optimizer_archive.save_to((directory / "candidate_ngp" / "trunk_optimizer.pt").string());
      }
      if (candidate_ngp_head_optimizer_) {
        torch::serialize::OutputArchive head_optimizer_archive;
        candidate_ngp_head_optimizer_->save(head_optimizer_archive);
        head_optimizer_archive.save_to((directory / "candidate_ngp" / "ngp_head_optimizer.pt").string());
      }
    }
  }

  ngp_replay_buffer_->save(directory / "replay_buffer");
  nlohmann::json metadata = {
      {"last_ngp_promotion_update", last_ngp_promotion_update_},
  };
  std::ofstream output(directory / "metadata.json");
  output << metadata.dump(2) << '\n';
}

void PPOTrainer::load_in_process_ngp_refresh_state(const std::filesystem::path& directory) {
  std::ifstream input(directory / "metadata.json");
  if (!input) {
    return;
  }
  nlohmann::json metadata;
  input >> metadata;
  last_ngp_promotion_update_ = metadata.value("last_ngp_promotion_update", static_cast<std::int64_t>(0));

  if (ngp_replay_buffer_ && std::filesystem::exists(directory / "replay_buffer" / "metadata.json")) {
    ngp_replay_buffer_->load(directory / "replay_buffer");
  }

  const std::filesystem::path candidate_dir = directory / "candidate_ngp";
  if (!std::filesystem::exists(candidate_dir / "model.pt")) {
    return;
  }

  std::lock_guard<std::mutex> lock(candidate_mutex_);
  candidate_ngp_model_ = SharedActorCritic(config_.model, config_.ppo);
  torch::serialize::InputArchive archive;
  archive.load_from((candidate_dir / "model.pt").string());
  candidate_ngp_model_->load(archive);
  candidate_ngp_model_->to(device_);
  candidate_ngp_model_->train();
  candidate_ngp_normalizer_.load(archive);
  candidate_ngp_normalizer_.to(device_);
  candidate_trunk_parameters_ = collect_trunk_parameters(candidate_ngp_model_);
  candidate_ngp_head_parameters_ = collect_head_parameters(candidate_ngp_model_, "next_goal_head.");

  if (!config_.reward.refresh.train_trunk) {
    for (auto& parameter : candidate_trunk_parameters_) {
      parameter.set_requires_grad(false);
    }
  }
  candidate_trunk_optimizer_.reset();
  if (config_.reward.refresh.train_trunk && !candidate_trunk_parameters_.empty()) {
    candidate_trunk_optimizer_ = std::make_unique<torch::optim::AdamW>(
        candidate_trunk_parameters_,
        torch::optim::AdamWOptions(config_.offline_optimization.trunk_learning_rate)
            .weight_decay(config_.offline_optimization.trunk_weight_decay));
    if (std::filesystem::exists(candidate_dir / "trunk_optimizer.pt")) {
      torch::serialize::InputArchive optimizer_archive;
      optimizer_archive.load_from((candidate_dir / "trunk_optimizer.pt").string());
      candidate_trunk_optimizer_->load(optimizer_archive);
    }
  }
  candidate_ngp_head_optimizer_ = std::make_unique<torch::optim::AdamW>(
      candidate_ngp_head_parameters_,
      torch::optim::AdamWOptions(config_.next_goal_predictor.learning_rate)
          .weight_decay(config_.next_goal_predictor.weight_decay));
  if (std::filesystem::exists(candidate_dir / "ngp_head_optimizer.pt")) {
    torch::serialize::InputArchive optimizer_archive;
    optimizer_archive.load_from((candidate_dir / "ngp_head_optimizer.pt").string());
    candidate_ngp_head_optimizer_->load(optimizer_archive);
  }
}

void PPOTrainer::start_ngp_refresh_worker() {
  if (!config_.reward.refresh.enabled || !config_.reward.refresh.train_candidate_in_process ||
      !config_.reward.refresh.async_candidate_updates || ngp_refresh_thread_.joinable()) {
    return;
  }
  ngp_refresh_worker_stop_ = false;
  ngp_refresh_thread_ = std::thread([this]() { ngp_refresh_worker_loop(); });
}

void PPOTrainer::stop_ngp_refresh_worker() {
  {
    std::lock_guard<std::mutex> lock(ngp_refresh_mutex_);
    ngp_refresh_worker_stop_ = true;
  }
  ngp_refresh_cv_.notify_all();
  if (ngp_refresh_thread_.joinable()) {
    ngp_refresh_thread_.join();
  }
  ngp_refresh_worker_stop_ = false;
  ngp_refresh_task_pending_ = false;
  ngp_refresh_task_in_progress_ = false;
}

void PPOTrainer::train_candidate_on_trajectories(const std::vector<NGPTrajectory>& trajectories, int epochs) {
  if (!candidate_ngp_model_ || trajectories.empty()) {
    return;
  }
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);
  const std::int64_t sequence_length = std::max<std::int64_t>(1, config_.behavior_cloning.sequence_length);
  std::vector<std::size_t> ordering(trajectories.size());
  std::iota(ordering.begin(), ordering.end(), 0);
  std::mt19937_64 rng(config_.offline_dataset.seed);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    if (config_.offline_dataset.shuffle) {
      std::shuffle(ordering.begin(), ordering.end(), rng);
    }
    candidate_ngp_model_->train();
    for (const std::size_t index : ordering) {
      const NGPTrajectory& trajectory = trajectories[index];
      const torch::Tensor obs = trajectory.obs_cpu.to(device_);
      const torch::Tensor normalized = candidate_ngp_normalizer_.normalize(obs).contiguous();
      torch::Tensor starts = torch::zeros({normalized.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      if (normalized.size(0) > 0) {
        starts[0] = 1.0F;
      }
      const torch::Tensor weights = torch::ones({normalized.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      const torch::Tensor labels =
          torch::full({normalized.size(0)}, trajectory.label, torch::TensorOptions().dtype(torch::kLong).device(device_));
      ContinuumState state = candidate_ngp_model_->initial_state(1, device_);

      for (std::int64_t offset = 0; offset < normalized.size(0); offset += sequence_length) {
        const std::int64_t length = std::min<std::int64_t>(sequence_length, normalized.size(0) - offset);
        candidate_ngp_model_->zero_grad();
        if (candidate_trunk_optimizer_) {
          candidate_trunk_optimizer_->zero_grad();
        }
        candidate_ngp_head_optimizer_->zero_grad();

        const SequenceOutput output = candidate_ngp_model_->forward_sequence(
            normalized.narrow(0, offset, length).unsqueeze(1),
            std::move(state),
            starts.narrow(0, offset, length).unsqueeze(1));
        const torch::Tensor loss = weighted_cross_entropy(
            output.next_goal_logits.squeeze(1),
            labels.narrow(0, offset, length),
            weights.narrow(0, offset, length),
            config_.next_goal_predictor.label_smoothing,
            ngp_class_weights);
        loss.backward();
        if (candidate_trunk_optimizer_ && !candidate_trunk_parameters_.empty()) {
          torch::nn::utils::clip_grad_norm_(
              candidate_trunk_parameters_,
              config_.offline_optimization.trunk_max_grad_norm);
          candidate_trunk_optimizer_->step();
        }
        if (!candidate_ngp_head_parameters_.empty()) {
          torch::nn::utils::clip_grad_norm_(
              candidate_ngp_head_parameters_,
              config_.next_goal_predictor.max_grad_norm);
        }
        candidate_ngp_head_optimizer_->step();
        state = output.final_state;
      }
    }
  }
}

std::pair<double, double> PPOTrainer::evaluate_ngp_trajectories(
    SharedActorCritic model,
    const ObservationNormalizer& normalizer,
    const std::vector<NGPTrajectory>& trajectories) const {
  if (trajectories.empty()) {
    return {0.0, 0.0};
  }
  const torch::Tensor ngp_class_weights =
      torch::tensor(config_.next_goal_predictor.class_weights, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_);
  const std::int64_t sequence_length = std::max<std::int64_t>(1, config_.behavior_cloning.sequence_length);
  double total_loss = 0.0;
  double total_accuracy = 0.0;
  std::int64_t total_samples = 0;

  model->eval();
  for (const auto& trajectory : trajectories) {
    const torch::Tensor obs = trajectory.obs_cpu.to(device_);
    const torch::Tensor normalized = normalizer.normalize(obs).contiguous();
    torch::Tensor starts = torch::zeros({normalized.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    if (normalized.size(0) > 0) {
      starts[0] = 1.0F;
    }
    const torch::Tensor weights = torch::ones({normalized.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    const torch::Tensor labels =
        torch::full({normalized.size(0)}, trajectory.label, torch::TensorOptions().dtype(torch::kLong).device(device_));
    ContinuumState state = model->initial_state(1, device_);

    for (std::int64_t offset = 0; offset < normalized.size(0); offset += sequence_length) {
      const std::int64_t length = std::min<std::int64_t>(sequence_length, normalized.size(0) - offset);
      torch::NoGradGuard no_grad;
      const SequenceOutput output = model->forward_sequence(
          normalized.narrow(0, offset, length).unsqueeze(1),
          std::move(state),
          starts.narrow(0, offset, length).unsqueeze(1));
      const torch::Tensor logits = output.next_goal_logits.squeeze(1);
      const torch::Tensor chunk_weights = weights.narrow(0, offset, length);
      const torch::Tensor chunk_labels = labels.narrow(0, offset, length);
      const torch::Tensor loss = weighted_cross_entropy(
          logits,
          chunk_labels,
          chunk_weights,
          config_.next_goal_predictor.label_smoothing,
          ngp_class_weights);
      total_loss += loss.item<double>() * static_cast<double>(length);
      total_accuracy += weighted_accuracy(logits, chunk_labels, chunk_weights) * static_cast<double>(length);
      total_samples += length;
      state = output.final_state;
    }
  }

  if (total_samples <= 0) {
    return {0.0, 0.0};
  }
  return {total_loss / static_cast<double>(total_samples), total_accuracy / static_cast<double>(total_samples)};
}

PPOTrainer::NGPRefreshResult PPOTrainer::evaluate_candidate_refresh(
    SharedActorCritic active_model,
    const ObservationNormalizer& active_normalizer,
    const std::vector<NGPTrajectory>& recent_val) const {
  NGPRefreshResult result{};
  const auto [active_anchor_loss, _active_anchor_acc] =
      evaluate_ngp_trajectories(active_model, active_normalizer, anchor_val_trajectories_);
  const auto [active_recent_loss, _active_recent_acc] =
      evaluate_ngp_trajectories(active_model, active_normalizer, recent_val);
  const auto [candidate_anchor_loss, _candidate_anchor_acc] =
      evaluate_ngp_trajectories(candidate_ngp_model_, candidate_ngp_normalizer_, anchor_val_trajectories_);
  const auto [candidate_recent_loss, _candidate_recent_acc] =
      evaluate_ngp_trajectories(candidate_ngp_model_, candidate_ngp_normalizer_, recent_val);

  result.active_anchor_loss = active_anchor_loss;
  result.active_recent_loss = active_recent_loss;
  result.candidate_anchor_loss = candidate_anchor_loss;
  result.candidate_recent_loss = candidate_recent_loss;
  result.recent_loss_improvement = relative_improvement(active_recent_loss, candidate_recent_loss);
  result.anchor_loss_regression = -relative_improvement(active_anchor_loss, candidate_anchor_loss);
  result.promote =
      result.recent_loss_improvement >= config_.reward.refresh.min_recent_loss_improvement &&
      result.anchor_loss_regression <= config_.reward.refresh.max_anchor_loss_regression;
  return result;
}

void PPOTrainer::ngp_refresh_worker_loop() {
  while (true) {
    NGPRefreshTask task;
    {
      std::unique_lock<std::mutex> lock(ngp_refresh_mutex_);
      ngp_refresh_cv_.wait(lock, [this]() { return ngp_refresh_worker_stop_ || ngp_refresh_task_pending_; });
      if (ngp_refresh_worker_stop_ && !ngp_refresh_task_pending_) {
        break;
      }
      if (!ngp_refresh_task_pending_) {
        continue;
      }
      task = std::move(pending_ngp_refresh_task_);
      ngp_refresh_task_pending_ = false;
      ngp_refresh_task_in_progress_ = true;
    }

    NGPRefreshResult result{};
    result.update_index = task.update_index;
    result.global_step = task.global_step;
    result.online_train_samples = ngp_trajectory_sample_count(task.online_train);
    result.online_val_samples = ngp_trajectory_sample_count(task.online_val);

    {
      std::lock_guard<std::mutex> candidate_lock(candidate_mutex_);
      std::vector<NGPTrajectory> training_trajectories = task.online_train;
      training_trajectories.insert(training_trajectories.end(), task.anchor_train.begin(), task.anchor_train.end());
      train_candidate_on_trajectories(training_trajectories, config_.reward.refresh.candidate_epochs);
      NGPRefreshResult eval_result = evaluate_candidate_refresh(task.active_model, task.active_normalizer, task.online_val);
      result.active_anchor_loss = eval_result.active_anchor_loss;
      result.active_recent_loss = eval_result.active_recent_loss;
      result.candidate_anchor_loss = eval_result.candidate_anchor_loss;
      result.candidate_recent_loss = eval_result.candidate_recent_loss;
      result.recent_loss_improvement = eval_result.recent_loss_improvement;
      result.anchor_loss_regression = eval_result.anchor_loss_regression;
      result.promote = eval_result.promote;
    }

    {
      std::lock_guard<std::mutex> lock(ngp_refresh_mutex_);
      latest_ngp_refresh_result_ = std::move(result);
      ngp_refresh_task_in_progress_ = false;
      ngp_refresh_result_ready_ = true;
    }
  }
}

void PPOTrainer::maybe_schedule_ngp_refresh_task(std::int64_t global_step, int update_index) {
  const auto& refresh = config_.reward.refresh;
  if (!ngp_replay_buffer_) {
    return;
  }
  const std::vector<NGPTrajectory> online_train = ngp_replay_buffer_->train_trajectories();
  const std::vector<NGPTrajectory> online_val = ngp_replay_buffer_->val_trajectories();
  const std::int64_t online_train_samples = ngp_trajectory_sample_count(online_train);
  const std::int64_t online_val_samples = ngp_trajectory_sample_count(online_val);
  if (online_train_samples < refresh.min_online_train_samples || online_val_samples <= 0) {
    return;
  }

  std::vector<NGPTrajectory> anchor_subset;
  if (refresh.old_data_fraction > 0.0F) {
    const double old_fraction = static_cast<double>(refresh.old_data_fraction);
    if (old_fraction >= 1.0) {
      throw std::runtime_error("reward.refresh.old_data_fraction must be in [0, 1).");
    }
    const auto target_anchor_samples = static_cast<std::int64_t>(
        std::ceil(static_cast<double>(online_train_samples) * old_fraction / (1.0 - old_fraction)));
    anchor_subset =
        select_ngp_trajectory_subset(anchor_train_trajectories_, target_anchor_samples, config_.offline_dataset.seed + update_index);
  }

  NGPRefreshTask task;
  task.update_index = update_index;
  task.global_step = global_step;
  task.online_train = online_train;
  task.online_val = online_val;
  task.anchor_train = std::move(anchor_subset);
  task.active_model = clone_shared_model(ngp_model_, device_);
  task.active_model->eval();
  task.active_normalizer = ngp_normalizer_.clone();
  task.active_normalizer.to(device_);

  if (refresh.async_candidate_updates) {
    std::lock_guard<std::mutex> lock(ngp_refresh_mutex_);
    if (ngp_refresh_task_pending_ || ngp_refresh_task_in_progress_ || ngp_refresh_result_ready_) {
      return;
    }
    pending_ngp_refresh_task_ = std::move(task);
    ngp_refresh_task_pending_ = true;
    ngp_refresh_cv_.notify_one();
    return;
  }

  {
    std::lock_guard<std::mutex> candidate_lock(candidate_mutex_);
    std::vector<NGPTrajectory> training_trajectories = task.online_train;
    training_trajectories.insert(training_trajectories.end(), task.anchor_train.begin(), task.anchor_train.end());
    train_candidate_on_trajectories(training_trajectories, config_.reward.refresh.candidate_epochs);
    latest_ngp_refresh_result_ = evaluate_candidate_refresh(task.active_model, task.active_normalizer, task.online_val);
    latest_ngp_refresh_result_.update_index = update_index;
    latest_ngp_refresh_result_.global_step = global_step;
    latest_ngp_refresh_result_.online_train_samples = online_train_samples;
    latest_ngp_refresh_result_.online_val_samples = online_val_samples;
  }
  ngp_refresh_result_ready_ = true;
}

void PPOTrainer::maybe_collect_ngp_refresh_result(const std::string& checkpoint_dir) {
  NGPRefreshResult result;
  {
    std::lock_guard<std::mutex> lock(ngp_refresh_mutex_);
    if (!ngp_refresh_result_ready_) {
      return;
    }
    result = latest_ngp_refresh_result_;
    ngp_refresh_result_ready_ = false;
  }

  const std::filesystem::path refresh_dir =
      std::filesystem::path(checkpoint_dir) / "ngp_runtime" / ("update_" + std::to_string(result.update_index));
  std::filesystem::create_directories(refresh_dir);
  nlohmann::json summary = {
      {"update", result.update_index},
      {"global_step", result.global_step},
      {"online_train_samples", result.online_train_samples},
      {"online_val_samples", result.online_val_samples},
      {"active_anchor_loss", result.active_anchor_loss},
      {"active_recent_loss", result.active_recent_loss},
      {"candidate_anchor_loss", result.candidate_anchor_loss},
      {"candidate_recent_loss", result.candidate_recent_loss},
      {"recent_loss_improvement", result.recent_loss_improvement},
      {"anchor_loss_regression", result.anchor_loss_regression},
      {"promoted", result.promote},
  };
  std::ofstream summary_output(refresh_dir / "summary.json");
  summary_output << summary.dump(2) << '\n';

  if (!result.promote ||
      result.update_index - last_ngp_promotion_update_ < std::max(1, config_.reward.refresh.promotion_cooldown_updates)) {
    return;
  }

  const std::string old_checkpoint = active_ngp_checkpoint_;
  const std::filesystem::path promoted_dir =
      std::filesystem::path(checkpoint_dir) / "ngp_versions" /
      ("promotion_" + std::to_string(static_cast<long long>(active_ngp_promotion_index_ + 1)));
  save_candidate_checkpoint(promoted_dir, result.global_step, result.update_index);

  {
    std::lock_guard<std::mutex> candidate_lock(candidate_mutex_);
    ngp_model_ = clone_shared_model(candidate_ngp_model_, device_);
    ngp_normalizer_ = candidate_ngp_normalizer_.clone();
    ngp_normalizer_.to(device_);
  }
  freeze_model_parameters(ngp_model_);
  ngp_model_->eval();
  ngp_collection_state_ = ngp_model_->initial_state(static_cast<std::int64_t>(total_agents_), device_);
  active_ngp_checkpoint_ = promoted_dir.string();
  active_ngp_label_ = promoted_dir.filename().string();
  active_ngp_config_hash_ = config_hash(config_);
  active_ngp_global_step_ = result.global_step;
  active_ngp_update_index_ = result.update_index;
  active_ngp_promotion_index_ += 1;
  active_ngp_promoted_global_step_ = result.global_step;
  last_ngp_promotion_update_ = result.update_index;

  TrainerMetrics promotion_metrics{};
  promotion_metrics.ngp_promotion_index = active_ngp_promotion_index_;
  promotion_metrics.ngp_promoted_global_step = active_ngp_promoted_global_step_;
  promotion_metrics.ngp_source_global_step = active_ngp_global_step_;
  promotion_metrics.ngp_source_update_index = active_ngp_update_index_;
  promotion_metrics.ngp_label = active_ngp_label_;
  promotion_metrics.ngp_checkpoint = active_ngp_checkpoint_;
  promotion_metrics.ngp_config_hash = active_ngp_config_hash_;
  append_ngp_promotion_line(checkpoint_dir, result.global_step, result.update_index, old_checkpoint, active_ngp_checkpoint_, promotion_metrics);
}

void PPOTrainer::maybe_refresh_ngp_candidate_in_process(
    std::int64_t global_step,
    int update_index,
    const std::string& checkpoint_dir) {
  const auto& refresh = config_.reward.refresh;
  if (!refresh.enabled || !refresh.train_candidate_in_process || !ngp_replay_buffer_) {
    return;
  }
  const int interval = std::max(1, refresh.check_interval_updates);
  if (update_index % interval != 0) {
    return;
  }

  ngp_replay_buffer_->close_window();
  maybe_collect_ngp_refresh_result(checkpoint_dir);
  maybe_schedule_ngp_refresh_task(global_step, update_index);
}

void PPOTrainer::maybe_promote_ngp_reward(
    std::int64_t global_step,
    int update_index,
    const std::string& checkpoint_dir) {
  if (!config_.reward.refresh.enabled || config_.reward.refresh.candidate_checkpoint.empty()) {
    return;
  }
  const int interval = std::max(1, config_.reward.refresh.check_interval_updates);
  if (update_index % interval != 0) {
    return;
  }

  namespace fs = std::filesystem;
  const fs::path candidate = fs::path(config_.reward.refresh.candidate_checkpoint);
  if (!fs::exists(candidate / "model.pt") || !fs::exists(candidate / "config.json") || !fs::exists(candidate / "metadata.json")) {
    return;
  }
  if (candidate.string() == active_ngp_checkpoint_) {
    return;
  }

  const std::string old_checkpoint = active_ngp_checkpoint_;
  try {
    load_ngp_reward_checkpoint(candidate.string(), candidate.filename().string(), active_ngp_promotion_index_ + 1, global_step);
  } catch (const std::exception& exc) {
    std::cerr << "skipping_ngp_promotion checkpoint=" << candidate.string() << " reason=" << exc.what() << '\n';
    return;
  }

  TrainerMetrics promotion_metrics{};
  promotion_metrics.ngp_promotion_index = active_ngp_promotion_index_;
  promotion_metrics.ngp_promoted_global_step = active_ngp_promoted_global_step_;
  promotion_metrics.ngp_source_global_step = active_ngp_global_step_;
  promotion_metrics.ngp_source_update_index = active_ngp_update_index_;
  promotion_metrics.ngp_label = active_ngp_label_;
  promotion_metrics.ngp_checkpoint = active_ngp_checkpoint_;
  promotion_metrics.ngp_config_hash = active_ngp_config_hash_;
  append_ngp_promotion_line(checkpoint_dir, global_step, update_index, old_checkpoint, active_ngp_checkpoint_, promotion_metrics);
}

torch::Tensor PPOTrainer::ngp_scalar(const torch::Tensor& logits) const {
  const torch::Tensor probs = torch::softmax(logits, -1);
  return probs.select(-1, 0) - probs.select(-1, 1);
}

TrainerMetrics PPOTrainer::update_policy() {
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};
  const int seq_len = std::max(1, config_.ppo.sequence_length);
  const int burn_in = std::max(0, config_.ppo.burn_in);
  const int agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const auto total_agents = static_cast<int>(rollout_.num_agents());
  std::int64_t metric_steps = 0;

  for (int epoch = 0; epoch < config_.ppo.epochs; ++epoch) {
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(device_));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += agents_per_batch) {
      const int count = std::min(agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);

      for (int seq_start = 0; seq_start < rollout_.rollout_length(); seq_start += seq_len) {
        const int effective_start = seq_start;
        const int context_start = std::max(0, effective_start - burn_in);
        const int effective_end = std::min(rollout_.rollout_length(), effective_start + seq_len);
        const int total_steps = effective_end - context_start;
        const int burn = effective_start - context_start;

        ContinuumState state = replay_state_until(context_start, agent_indices);
        const torch::Tensor obs = rollout_.obs.narrow(0, context_start, total_steps).index_select(1, agent_indices);
        const torch::Tensor episode_starts =
            rollout_.episode_starts.narrow(0, context_start, total_steps).index_select(1, agent_indices);
        const torch::Tensor action_masks =
            rollout_.action_masks.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices).to(torch::kBool);
        const torch::Tensor learner_active =
            rollout_.learner_active.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor actions =
            rollout_.actions.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor old_log_probs =
            rollout_.log_probs.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        const torch::Tensor returns =
            rollout_.returns.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);
        torch::Tensor advantages =
            rollout_.advantages.narrow(0, effective_start, effective_end - effective_start).index_select(1, agent_indices);

        const auto ppo_start = std::chrono::steady_clock::now();
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-6);
        SequenceOutput output;
        output = model_->forward_sequence(obs, std::move(state), episode_starts);
        torch::Tensor policy_logits = output.policy_logits.narrow(0, burn, effective_end - effective_start);
        torch::Tensor value_logits = output.value_logits.narrow(0, burn, effective_end - effective_start);

        const torch::Tensor flat_policy_logits = policy_logits.reshape({-1, config_.model.action_dim});
        const torch::Tensor flat_action_masks = action_masks.reshape({-1, config_.model.action_dim});
        const torch::Tensor flat_value_logits = value_logits.reshape({-1, config_.ppo.value_num_atoms});
        const torch::Tensor flat_actions = actions.reshape({-1});
        const torch::Tensor flat_old_log_probs = old_log_probs.reshape({-1});
        const torch::Tensor flat_advantages = advantages.reshape({-1});
        const torch::Tensor flat_returns = returns.reshape({-1});
        const torch::Tensor flat_active = learner_active.reshape({-1}) > 0.5;

        if (flat_active.sum().item<std::int64_t>() == 0) {
          continue;
        }

        const torch::Tensor active_policy_logits = flat_policy_logits.index({flat_active});
        const torch::Tensor active_action_masks = flat_action_masks.index({flat_active});
        const torch::Tensor active_value_logits = flat_value_logits.index({flat_active});
        const torch::Tensor active_actions = flat_actions.index({flat_active});
        const torch::Tensor active_old_log_probs = flat_old_log_probs.index({flat_active});
        const torch::Tensor active_advantages = flat_advantages.index({flat_active});
        const torch::Tensor active_returns = flat_returns.index({flat_active});

        const torch::Tensor active_masked_logits = apply_action_mask_to_logits(active_policy_logits, active_action_masks);
        const torch::Tensor current_log_probs =
            torch::log_softmax(active_masked_logits, -1).gather(-1, active_actions.unsqueeze(-1)).squeeze(-1);
        const torch::Tensor ratio = torch::exp(current_log_probs - active_old_log_probs);
        const torch::Tensor epsilon = adaptive_epsilon(active_value_logits);
        const torch::Tensor clipped_ratio = torch::clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
        const torch::Tensor weights = confidence_weights(active_value_logits);
        const torch::Tensor unclipped = ratio * active_advantages;
        const torch::Tensor clipped = clipped_ratio * active_advantages;
        const torch::Tensor policy_loss = -(torch::min(unclipped, clipped) * weights).mean();

        const torch::Tensor action_entropy = masked_action_entropy(active_policy_logits, active_action_masks).mean();
        const torch::Tensor target_dist = categorical_projection(active_returns);
        const torch::Tensor critic_loss =
            -(target_dist * torch::log_softmax(active_value_logits, -1)).sum(-1).mean();
        const torch::Tensor loss =
            policy_loss + config_.ppo.value_coef * critic_loss - config_.ppo.entropy_coef * action_entropy;
        metrics.ppo_forward_backward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ppo_start).count();

        const auto optim_start = std::chrono::steady_clock::now();
        optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.ppo.max_grad_norm);
        optimizer_.step();
        metrics.optimizer_step_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();

        metrics.policy_loss += policy_loss.item<double>() * active_actions.size(0);
        metrics.value_loss += critic_loss.item<double>() * active_actions.size(0);
        metrics.entropy += action_entropy.item<double>() * active_actions.size(0);
        metrics.value_entropy += model_->value_entropy(active_value_logits).mean().item<double>() * active_actions.size(0);
        metrics.value_variance += model_->value_variance(active_value_logits).mean().item<double>() * active_actions.size(0);
        metric_steps += active_actions.size(0);
      }
    }
  }

  if (metric_steps > 0) {
    metrics.policy_loss /= static_cast<double>(metric_steps);
    metrics.value_loss /= static_cast<double>(metric_steps);
    metrics.entropy /= static_cast<double>(metric_steps);
    metrics.value_entropy /= static_cast<double>(metric_steps);
    metrics.value_variance /= static_cast<double>(metric_steps);
  }
  metrics.update_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  return metrics;
}

CheckpointMetadata PPOTrainer::make_checkpoint_metadata(
    std::int64_t global_step,
    std::int64_t update_index) const {
  return {
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_.hash(),
      .architecture_name = "continuum_dppo",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .reward_ngp_label = active_ngp_label_,
      .reward_ngp_checkpoint = active_ngp_checkpoint_,
      .reward_ngp_config_hash = active_ngp_config_hash_,
      .reward_ngp_global_step = active_ngp_global_step_,
      .reward_ngp_update_index = active_ngp_update_index_,
      .reward_ngp_promotion_index = active_ngp_promotion_index_,
      .reward_ngp_promoted_global_step = active_ngp_promoted_global_step_,
  };
}

void PPOTrainer::save_checkpoint(
    const std::string& checkpoint_dir,
    std::int64_t global_step,
    std::int64_t update_index) {
  namespace fs = std::filesystem;
  const fs::path directory = fs::path(checkpoint_dir) / ("update_" + std::to_string(update_index));
  save_checkpoint_to_directory(directory, global_step, update_index);
}

void PPOTrainer::save_checkpoint_to_directory(
    const std::filesystem::path& directory,
    std::int64_t global_step,
    std::int64_t update_index) {
  namespace fs = std::filesystem;
  fs::remove_all(directory);
  fs::create_directories(directory);

  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index), (directory / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  model_->save(archive);
  normalizer_.save(archive);
  archive.save_to((directory / "model.pt").string());

  torch::serialize::OutputArchive optimizer_archive;
  optimizer_.save(optimizer_archive);
  optimizer_archive.save_to((directory / "optimizer.pt").string());

  if (ngp_model_) {
    save_model_snapshot(ngp_model_, ngp_normalizer_, directory / "active_ngp", global_step, update_index);
  }
  save_in_process_ngp_refresh_state(directory / "ngp_refresh_state");
}

void PPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  std::int64_t global_step = resumed_global_step_;
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "ppo_train");
  int last_completed_update_index = static_cast<int>(resumed_update_index_);

  for (int update_index = 0; update_index < updates; ++update_index) {
    const int current_update_index = static_cast<int>(resumed_update_index_) + update_index + 1;
    last_completed_update_index = current_update_index;
    const auto update_start = std::chrono::steady_clock::now();
    TrainerMetrics metrics{};
    CollectorTimings collector_timings{};
    std::int64_t collected_agent_steps = 0;

    const auto collection_start = std::chrono::steady_clock::now();
    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      torch::Tensor raw_obs_host = collector_->host_observations();
      const torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
      const torch::Tensor episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
      const torch::Tensor action_masks = collector_->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
      const torch::Tensor learner_active = collector_->host_learner_active().to(device_, use_pinned_host_buffers_);
      const torch::Tensor snapshot_ids = collector_->host_snapshot_ids().to(device_, use_pinned_host_buffers_);

      torch::Tensor obs = raw_obs;
      torch::Tensor actions;
      torch::Tensor current_actions;
      torch::Tensor log_probs;
      torch::Tensor sampled_values;
      torch::Tensor ngp_prev_scalar;
      ContinuumState next_ngp_state;

      const auto policy_start = std::chrono::steady_clock::now();
      {
        torch::NoGradGuard no_grad;
        const torch::Tensor normalized_ngp_obs = ngp_normalizer_.normalize(raw_obs);
        PolicyOutput ngp_output =
            ngp_model_->forward_step(normalized_ngp_obs, std::move(ngp_collection_state_), episode_starts);
        ngp_prev_scalar = ngp_scalar(ngp_output.next_goal_logits);
        next_ngp_state = std::move(ngp_output.state);

        normalizer_.update(obs);
        obs = normalizer_.normalize(obs);
        PolicyOutput output = model_->forward_step(obs, std::move(collection_state_), episode_starts);
        collection_state_ = std::move(output.state);
        current_actions = sample_actions(output.policy_logits, action_masks, false, &log_probs);
        sampled_values = output.sampled_values;
      }
      actions = current_actions;
      metrics.policy_forward_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

      if (self_play_manager_ && self_play_manager_->has_snapshots()) {
        torch::Tensor opponent_actions;
        self_play_manager_->infer_opponent_actions(
            model_,
            raw_obs,
            action_masks,
            episode_starts,
            snapshot_ids,
            opponent_collection_state_,
            &opponent_actions,
            &metrics.policy_forward_seconds);
        actions = torch::where(snapshot_ids >= 0, opponent_actions, current_actions);
      }

      const auto decode_start = std::chrono::steady_clock::now();
      const torch::Tensor action_indices_cpu = actions_to_cpu(actions);
      collector_->step(
          std::span<const std::int64_t>(
              action_indices_cpu.data_ptr<std::int64_t>(),
              static_cast<std::size_t>(action_indices_cpu.numel())),
          &collector_timings);
      metrics.action_decode_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
      const torch::Tensor dones = collector_->host_dones().to(device_, use_pinned_host_buffers_);
      if (online_ngp_dataset_writer_) {
        online_ngp_dataset_writer_->record_step(
            raw_obs_host,
            collector_->host_dones(),
            collector_->host_terminated(),
            collector_->host_truncated(),
            collector_->host_terminal_next_goal_labels());
      }
      if (ngp_replay_buffer_) {
        ngp_replay_buffer_->record_step(
            raw_obs_host,
            collector_->host_dones(),
            collector_->host_terminated(),
            collector_->host_truncated(),
            collector_->host_terminal_next_goal_labels());
      }
      const auto ngp_start = std::chrono::steady_clock::now();
      torch::Tensor rewards;
      {
        torch::NoGradGuard no_grad;
        const torch::Tensor post_step_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
        const torch::Tensor zero_starts = torch::zeros_like(dones);
        const torch::Tensor normalized_post_step_obs = ngp_normalizer_.normalize(post_step_obs);
        PolicyOutput ngp_output =
            ngp_model_->forward_step(normalized_post_step_obs, std::move(next_ngp_state), zero_starts);
        const torch::Tensor ngp_current_scalar = ngp_scalar(ngp_output.next_goal_logits);
        rewards = config_.reward.ngp_scale * (ngp_current_scalar - ngp_prev_scalar);
        ngp_collection_state_ = std::move(ngp_output.state);
      }
      metrics.reward_model_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - ngp_start).count();

      const auto append_start = std::chrono::steady_clock::now();
      rollout_.append(
          step,
          obs,
          episode_starts,
          action_masks.to(torch::kUInt8),
          learner_active,
          actions,
          log_probs,
          rewards,
          dones,
          sampled_values);
      metrics.rollout_append_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - append_start).count();
      collected_agent_steps += learner_active.sum().item<std::int64_t>();
    }
    const double collection_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

    const auto gae_start = std::chrono::steady_clock::now();
    const torch::Tensor last_obs_host = collector_->host_observations();
    const torch::Tensor last_obs = normalizer_.normalize(last_obs_host.to(device_, use_pinned_host_buffers_));
    const torch::Tensor last_episode_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    torch::Tensor last_sampled_value;
    const auto bootstrap_start = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      PolicyOutput output = model_->forward_step(last_obs, collection_state_, last_episode_starts);
      last_sampled_value = output.sampled_values;
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - bootstrap_start).count();
    rollout_.compute_returns_and_advantages(last_sampled_value, config_.ppo.gamma, config_.ppo.gae_lambda);
    metrics.gae_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - gae_start).count();

    TrainerMetrics update_metrics = update_policy();
    metrics.policy_loss = update_metrics.policy_loss;
    metrics.value_loss = update_metrics.value_loss;
    metrics.entropy = update_metrics.entropy;
    metrics.value_entropy = update_metrics.value_entropy;
    metrics.value_variance = update_metrics.value_variance;
    metrics.update_seconds = update_metrics.update_seconds;
    metrics.ppo_forward_backward_seconds = update_metrics.ppo_forward_backward_seconds;
    metrics.optimizer_step_seconds = update_metrics.optimizer_step_seconds;
    metrics.obs_build_seconds = collector_timings.obs_build_seconds;
    metrics.mask_build_seconds = collector_timings.mask_build_seconds;
    metrics.env_step_seconds = collector_timings.env_step_seconds;
    metrics.done_reset_seconds = collector_timings.done_reset_seconds;

    metrics.collection_agent_steps_per_second =
        collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / collection_seconds : 0.0;
    metrics.update_agent_steps_per_second =
        collected_agent_steps > 0 ? static_cast<double>(collected_agent_steps) / std::max(metrics.update_seconds, 1.0e-9) : 0.0;

    global_step += collected_agent_steps;

    if (self_play_manager_) {
      SelfPlayMetrics self_play_metrics =
          self_play_manager_->on_update(model_, normalizer_, global_step, current_update_index);
      metrics.self_play_eval_seconds = self_play_metrics.eval_seconds;
      metrics.elo_ratings = std::move(self_play_metrics.ratings);
    }

    metrics.overall_agent_steps_per_second =
        collected_agent_steps > 0
            ? static_cast<double>(collected_agent_steps) /
                  std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count(), 1.0e-9)
            : 0.0;
    const torch::Tensor active_rewards = rollout_.rewards * rollout_.learner_active;
    const double active_count = rollout_.learner_active.sum().item<double>();
    metrics.reward_mean = active_count > 0.0 ? active_rewards.sum().item<double>() / active_count : 0.0;
    metrics.ngp_promotion_index = active_ngp_promotion_index_;
    metrics.ngp_promoted_global_step = active_ngp_promoted_global_step_;
    metrics.ngp_source_global_step = active_ngp_global_step_;
    metrics.ngp_source_update_index = active_ngp_update_index_;
    metrics.ngp_label = active_ngp_label_;
    metrics.ngp_checkpoint = active_ngp_checkpoint_;
    metrics.ngp_config_hash = active_ngp_config_hash_;
    if (online_ngp_dataset_writer_) {
      metrics.ngp_online_samples_written = online_ngp_dataset_writer_->samples_written();
      metrics.ngp_online_trajectories_written = online_ngp_dataset_writer_->trajectories_written();
    } else if (ngp_replay_buffer_) {
      metrics.ngp_online_samples_written =
          ngp_replay_buffer_->train_sample_count() + ngp_replay_buffer_->val_sample_count();
      metrics.ngp_online_trajectories_written = ngp_replay_buffer_->trajectories_written();
    }

    std::cout << "update=" << current_update_index
              << " collection_sps=" << metrics.collection_agent_steps_per_second
              << " update_sps=" << metrics.update_agent_steps_per_second
              << " overall_sps=" << metrics.overall_agent_steps_per_second
              << " reward_mean=" << metrics.reward_mean
              << " policy_loss=" << metrics.policy_loss
              << " value_loss=" << metrics.value_loss
              << " entropy=" << metrics.entropy
              << " ngp=" << metrics.ngp_label
              << " ngp_promotion_index=" << metrics.ngp_promotion_index
              << " self_play_eval_s=" << metrics.self_play_eval_seconds
              << '\n';
    append_metrics_line(checkpoint_dir, current_update_index, global_step, metrics);
    if (wandb.enabled()) {
      nlohmann::json wandb_metrics = {
          {"update", current_update_index},
          {"global_step", global_step},
          {"collection_agent_steps_per_second", metrics.collection_agent_steps_per_second},
          {"update_agent_steps_per_second", metrics.update_agent_steps_per_second},
          {"overall_agent_steps_per_second", metrics.overall_agent_steps_per_second},
          {"update_seconds", metrics.update_seconds},
          {"reward_mean", metrics.reward_mean},
          {"policy_loss", metrics.policy_loss},
          {"value_loss", metrics.value_loss},
          {"entropy", metrics.entropy},
          {"value_entropy", metrics.value_entropy},
          {"value_variance", metrics.value_variance},
          {"obs_build_seconds", metrics.obs_build_seconds},
          {"mask_build_seconds", metrics.mask_build_seconds},
          {"policy_forward_seconds", metrics.policy_forward_seconds},
          {"action_decode_seconds", metrics.action_decode_seconds},
          {"env_step_seconds", metrics.env_step_seconds},
          {"done_reset_seconds", metrics.done_reset_seconds},
          {"reward_model_seconds", metrics.reward_model_seconds},
          {"rollout_append_seconds", metrics.rollout_append_seconds},
          {"gae_seconds", metrics.gae_seconds},
          {"ppo_forward_backward_seconds", metrics.ppo_forward_backward_seconds},
          {"optimizer_step_seconds", metrics.optimizer_step_seconds},
          {"self_play_eval_seconds", metrics.self_play_eval_seconds},
          {"ngp_promotion_index", metrics.ngp_promotion_index},
          {"ngp_promoted_global_step", metrics.ngp_promoted_global_step},
          {"ngp_source_global_step", metrics.ngp_source_global_step},
          {"ngp_source_update_index", metrics.ngp_source_update_index},
          {"ngp_online_samples_written", metrics.ngp_online_samples_written},
          {"ngp_online_trajectories_written", metrics.ngp_online_trajectories_written},
      };
      for (const auto& [mode, rating] : metrics.elo_ratings) {
        wandb_metrics["elo/" + mode] = rating;
      }
      wandb_metrics["ngp_label"] = metrics.ngp_label;
      wandb_metrics["ngp_checkpoint"] = metrics.ngp_checkpoint;
      wandb_metrics["ngp_config_hash"] = metrics.ngp_config_hash;
      wandb.log(std::move(wandb_metrics));
    }

    if (current_update_index % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(checkpoint_dir, global_step, current_update_index);
    }
    if (metrics.reward_mean > best_reward_mean_) {
      best_reward_mean_ = metrics.reward_mean;
      save_checkpoint_to_directory(std::filesystem::path(checkpoint_dir) / "best", global_step, current_update_index);
    }
    maybe_refresh_ngp_candidate_in_process(global_step, current_update_index, checkpoint_dir);
    maybe_promote_ngp_reward(global_step, current_update_index, checkpoint_dir);
  }
  if (online_ngp_dataset_writer_) {
    online_ngp_dataset_writer_->finish();
  }
  if (ngp_replay_buffer_) {
    ngp_replay_buffer_->close_window();
    maybe_collect_ngp_refresh_result(checkpoint_dir);
    maybe_schedule_ngp_refresh_task(global_step, resumed_update_index_ + updates);
  }
  stop_ngp_refresh_worker();
  maybe_collect_ngp_refresh_result(checkpoint_dir);
  if (updates > 0 &&
      last_completed_update_index % std::max(1, config_.ppo.checkpoint_interval) == 0) {
    save_checkpoint_to_directory(
        std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(last_completed_update_index)),
        global_step,
        last_completed_update_index);
  }
  save_checkpoint_to_directory(
      std::filesystem::path(checkpoint_dir) / "final",
      global_step,
      resumed_update_index_ + updates);
  wandb.finish();
}

}  // namespace pulsar

#endif
