#include "pulsar/training/self_play_manager.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/ppo_math.hpp"

namespace pulsar {
namespace {

torch::Tensor masked_argmax(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  const torch::Tensor masked_logits = logits.masked_fill(action_masks.logical_not(), -1.0e9);
  return masked_logits.argmax(-1);
}

torch::Tensor masked_sample(const torch::Tensor& logits, const torch::Tensor& action_masks) {
  return sample_masked_actions(logits, action_masks, false, nullptr);
}

void update_elo_impl(double& winner, double& loser, double k_factor) {
  const double expected = 1.0 / (1.0 + std::pow(10.0, (loser - winner) / 400.0));
  winner += k_factor * (1.0 - expected);
  loser += k_factor * (expected - 1.0);
}

void update_elo_draw_impl(double& rating_a, double& rating_b, double k_factor) {
  const double expected_a = 1.0 / (1.0 + std::pow(10.0, (rating_b - rating_a) / 400.0));
  rating_a += k_factor * (0.5 - expected_a);
  rating_b += k_factor * (expected_a - 0.5);
}

std::shared_ptr<MutatorSequence> make_eval_reset_mutator(const EnvConfig& config) {
  return std::make_shared<MutatorSequence>(
      std::vector<StateMutatorPtr>{
          std::make_shared<FixedTeamSizeMutator>(config),
          std::make_shared<KickoffMutator>(config),
      });
}

}  // namespace

void update_elo_ratings(double& winner, double& loser, double k_factor) {
  update_elo_impl(winner, loser, k_factor);
}

SelfPlayManager::SelfPlayManager(
    ExperimentConfig config,
    std::filesystem::path snapshot_root,
    ObsBuilderPtr obs_builder,
    ActionParserPtr action_parser,
    torch::Device device)
    : config_(std::move(config)),
      snapshot_root_(std::move(snapshot_root)),
      obs_builder_(std::move(obs_builder)),
      action_parser_(std::move(action_parser)),
      device_(std::move(device)),
      rng_(static_cast<std::mt19937::result_type>(config_.env.seed)) {
  current_ratings_[mode_name()] = config_.self_play_league.elo_initial;
  if (!snapshot_root_.empty()) {
    std::filesystem::create_directories(snapshot_root_);
    load_existing_snapshots();
  }
}

bool SelfPlayManager::enabled() const {
  return config_.self_play_league.enabled;
}

SelfPlayAssignment SelfPlayManager::sample_assignment(std::size_t, std::uint64_t) {
  SelfPlayAssignment assignment{};
  if (!enabled() || snapshots_.empty()) {
    return assignment;
  }

  std::uniform_real_distribution<float> probability(0.0F, 1.0F);
  if (probability(rng_) > config_.self_play_league.opponent_probability) {
    return assignment;
  }

  std::uniform_int_distribution<int> snapshot_dist(0, static_cast<int>(snapshots_.size() - 1));
  std::uniform_int_distribution<int> team_dist(0, 1);
  assignment.enabled = true;
  assignment.snapshot_index = snapshot_dist(rng_);
  assignment.learner_team = team_dist(rng_) == 0 ? Team::Blue : Team::Orange;
  return assignment;
}

bool SelfPlayManager::has_snapshots() const {
  return !snapshots_.empty();
}

void SelfPlayManager::infer_opponent_actions(
    PPOActor&,
    const torch::Tensor& raw_obs,
    const torch::Tensor& action_masks,
    const torch::Tensor& episode_starts,
    const torch::Tensor& snapshot_ids,
    ContinuumState& opponent_state,
    torch::Tensor* out_actions,
    double* inference_seconds) {
  PULSAR_TRACE_SCOPE_CAT("self_play", "infer_opponent_actions");
  if (out_actions == nullptr) {
    throw std::invalid_argument("SelfPlayManager::infer_opponent_actions requires an output tensor.");
  }

  auto actions = torch::zeros(
      {raw_obs.size(0)},
      torch::TensorOptions().dtype(torch::kLong).device(raw_obs.device()));
  const auto start = std::chrono::steady_clock::now();

  const torch::Tensor snapshot_ids_cpu = snapshot_ids.to(torch::kCPU);
  const auto* snapshot_data = snapshot_ids_cpu.data_ptr<std::int64_t>();
  std::vector<std::vector<std::int64_t>> grouped(snapshots_.size());
  for (std::int64_t index = 0; index < snapshot_ids_cpu.size(0); ++index) {
    const std::int64_t snapshot_index = snapshot_data[index];
    if (snapshot_index >= 0 && snapshot_index < static_cast<std::int64_t>(snapshots_.size())) {
      grouped[static_cast<std::size_t>(snapshot_index)].push_back(index);
    }
  }

  torch::NoGradGuard no_grad;
  const bool deterministic = config_.self_play_league.training_opponent_policy != "stochastic";
  for (std::size_t snapshot_index = 0; snapshot_index < grouped.size(); ++snapshot_index) {
    if (grouped[snapshot_index].empty()) {
      continue;
    }

    const torch::Tensor indices =
        torch::tensor(grouped[snapshot_index], torch::TensorOptions().dtype(torch::kLong).device(raw_obs.device()));
    Snapshot& snapshot = snapshots_[snapshot_index];
    const torch::Tensor obs = raw_obs.index_select(0, indices);
    const torch::Tensor masks = action_masks.index_select(0, indices).to(torch::kBool);
    const torch::Tensor starts = episode_starts.index_select(0, indices);
    const torch::Tensor normalized_obs = snapshot.normalizer.normalize(obs);
    ContinuumState state = gather_state(opponent_state, indices);
    const ActorStepOutput output = snapshot.model->forward_step(normalized_obs, std::move(state), starts);
    const torch::Tensor sampled_actions =
        deterministic ? masked_argmax(output.policy_logits, masks) : masked_sample(output.policy_logits, masks);
    actions.index_copy_(0, indices, sampled_actions);
    scatter_state(opponent_state, indices, detach_state(std::move(output.state)));
  }

  if (inference_seconds != nullptr) {
    *inference_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  *out_actions = actions;
}

SelfPlayMetrics SelfPlayManager::on_update(
    PPOActor& current_model,
    const ObservationNormalizer& current_normalizer,
    std::int64_t global_step,
    int update_index) {
  PULSAR_TRACE_SCOPE_CAT("self_play", "on_update");
  SelfPlayMetrics metrics{};
  if (!enabled()) {
    return metrics;
  }

  if (config_.self_play_league.snapshot_interval_updates > 0 &&
      update_index % config_.self_play_league.snapshot_interval_updates == 0) {
    add_snapshot(current_model, current_normalizer, global_step, update_index);
  }
  metrics.snapshot_count = static_cast<int>(snapshots_.size());

  if (!snapshots_.empty() &&
      config_.self_play_league.eval_interval_updates > 0 &&
      update_index % config_.self_play_league.eval_interval_updates == 0) {
    metrics = evaluate_current(current_model, current_normalizer);
    metrics.snapshot_count = static_cast<int>(snapshots_.size());
  } else {
    for (const auto& pair : current_ratings_) {
      metrics.ratings[pair.first] = pair.second;
    }
  }
  return metrics;
}

void SelfPlayManager::load_existing_snapshots() {
  PULSAR_TRACE_SCOPE_CAT("self_play", "load_existing_snapshots");
  snapshots_.clear();
  if (!std::filesystem::exists(snapshot_root_)) {
    return;
  }

  std::vector<std::filesystem::path> directories;
  for (const auto& entry : std::filesystem::directory_iterator(snapshot_root_)) {
    if (entry.is_directory()) {
      directories.push_back(entry.path());
    }
  }
  std::sort(directories.begin(), directories.end());

  for (const auto& directory : directories) {
    const ExperimentConfig snapshot_config = load_experiment_config((directory / "config.json").string());
    const CheckpointMetadata metadata = load_checkpoint_metadata((directory / "metadata.json").string());
    validate_inference_checkpoint_metadata(metadata, snapshot_config);

    Snapshot snapshot{
        .global_step = metadata.global_step,
        .update_index = static_cast<int>(metadata.update_index),
        .model = PPOActor(snapshot_config.model, snapshot_config.critic),
        .normalizer = ObservationNormalizer(snapshot_config.model.observation_dim),
        .ratings = {},
    };
    torch::serialize::InputArchive archive;
    archive.load_from((directory / "model.pt").string());
    snapshot.model->load(archive);
    snapshot.normalizer.load(archive);
    snapshot.model->to(device_);
    snapshot.normalizer.to(device_);
    snapshot.model->eval();

    const auto ratings_path = directory / "ratings.json";
    if (std::filesystem::exists(ratings_path)) {
      std::ifstream input(ratings_path);
      nlohmann::json ratings_json;
      input >> ratings_json;
      snapshot.ratings = ratings_json.get<std::map<std::string, double>>();
    }
    snapshots_.push_back(std::move(snapshot));
  }
}

void SelfPlayManager::save_snapshot(const Snapshot& snapshot) const {
  const std::filesystem::path directory = snapshot_root_ / std::to_string(snapshot.global_step);
  std::filesystem::create_directories(directory);
  save_experiment_config(config_, (directory / "config.json").string());
  save_checkpoint_metadata(
      CheckpointMetadata{
          .schema_version = config_.schema_version,
          .obs_schema_version = config_.obs_schema_version,
          .config_hash = config_hash(config_),
          .action_table_hash = action_table_hash(config_.action_table),
          .architecture_name = "policy_snapshot",
          .device = device_.str(),
          .global_step = snapshot.global_step,
          .update_index = snapshot.update_index,
          .critic_heads = snapshot.model->enabled_critic_heads(),
      },
      (directory / "metadata.json").string());

  torch::serialize::OutputArchive archive;
  snapshot.model->save(archive);
  snapshot.normalizer.save(archive);
  archive.save_to((directory / "model.pt").string());

  std::ofstream output(directory / "ratings.json");
  output << nlohmann::json(snapshot.ratings).dump(2) << '\n';
}

void SelfPlayManager::trim_snapshots() {
  while (static_cast<int>(snapshots_.size()) > config_.self_play_league.max_snapshots) {
    const auto global_step = snapshots_.front().global_step;
    snapshots_.erase(snapshots_.begin());
    std::filesystem::remove_all(snapshot_root_ / std::to_string(global_step));
  }
}

void SelfPlayManager::add_snapshot(
    PPOActor& current_model,
    const ObservationNormalizer& current_normalizer,
    std::int64_t global_step,
    int update_index) {
  PULSAR_TRACE_SCOPE_CAT("self_play", "add_snapshot");
  Snapshot snapshot{
      .global_step = global_step,
      .update_index = update_index,
      .model = clone_ppo_actor(current_model, device_),
      .normalizer = current_normalizer.clone(),
      .ratings = current_ratings_,
  };
  snapshot.model->eval();
  snapshot.normalizer.to(device_);
  snapshots_.push_back(std::move(snapshot));
  save_snapshot(snapshots_.back());
  trim_snapshots();
}

SelfPlayMetrics SelfPlayManager::evaluate_current(
    PPOActor& current_model,
    const ObservationNormalizer& current_normalizer) {
  PULSAR_TRACE_SCOPE_CAT("self_play", "evaluate_current");
  SelfPlayMetrics metrics{};
  metrics.snapshot_count = static_cast<int>(snapshots_.size());
  if (snapshots_.empty()) {
    return metrics;
  }

  const auto start = std::chrono::steady_clock::now();
  const auto reset_mutator = make_eval_reset_mutator(config_.env);
  const std::string mode = mode_name();
  const auto* discrete = dynamic_cast<const DiscreteActionParser*>(action_parser_.get());
  const bool deterministic = config_.self_play_league.eval_policy == "deterministic";
  if (discrete == nullptr) {
    throw std::invalid_argument("SelfPlayManager evaluation requires DiscreteActionParser.");
  }

  torch::NoGradGuard no_grad;
  for (std::size_t snapshot_index = 0; snapshot_index < snapshots_.size(); ++snapshot_index) {
    Snapshot& snapshot = snapshots_[snapshot_index];
    for (int match = 0; match < config_.self_play_league.eval_matches_per_snapshot; ++match) {
      std::vector<std::shared_ptr<RocketSimTransitionEngine>> engines;
      engines.reserve(static_cast<std::size_t>(config_.self_play_league.eval_num_envs));
      for (int env_idx = 0; env_idx < config_.self_play_league.eval_num_envs; ++env_idx) {
        EnvConfig env_config = config_.env;
        env_config.seed += static_cast<std::uint64_t>(snapshot_index * 997 + match * 131 + env_idx);
        engines.push_back(std::make_shared<RocketSimTransitionEngine>(env_config, reset_mutator));
      }

      const std::size_t agents_per_env = engines.front()->num_agents();
      const std::size_t total_agents = engines.size() * agents_per_env;
      torch::Tensor obs = torch::empty(
          {static_cast<long>(total_agents), static_cast<long>(obs_builder_->obs_dim())},
          torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      torch::Tensor masks = torch::empty(
          {static_cast<long>(total_agents), static_cast<long>(discrete->action_table().size())},
          torch::TensorOptions().dtype(torch::kUInt8).device(device_));
      torch::Tensor episode_starts = torch::ones({static_cast<long>(total_agents)}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      ContinuumState current_state = current_model->initial_state(static_cast<std::int64_t>(total_agents), device_);
      ContinuumState snapshot_state = snapshot.model->initial_state(static_cast<std::int64_t>(total_agents), device_);
      const Team current_team = (match % 2 == 0) ? Team::Blue : Team::Orange;
      std::vector<bool> env_done(engines.size(), false);
      int active_envs = static_cast<int>(engines.size());

      for (int tick = 0; tick < config_.env.max_episode_ticks && active_envs > 0; tick += config_.env.tick_skip) {
        std::vector<float> host_obs(total_agents * obs_builder_->obs_dim());
        std::vector<std::uint8_t> host_masks(total_agents * discrete->action_table().size(), 1);
        for (std::size_t env_idx = 0; env_idx < engines.size(); ++env_idx) {
          const std::size_t offset = env_idx * agents_per_env;
          obs_builder_->build_obs_batch(
              engines[env_idx]->state(),
              std::span<float>(
                  host_obs.data() + static_cast<std::ptrdiff_t>(offset * obs_builder_->obs_dim()),
                  agents_per_env * obs_builder_->obs_dim()));
          action_parser_->build_action_mask_batch(
              engines[env_idx]->state(),
              std::span<std::uint8_t>(
                  host_masks.data() + static_cast<std::ptrdiff_t>(offset * discrete->action_table().size()),
                  agents_per_env * discrete->action_table().size()));
        }
        obs.copy_(torch::from_blob(host_obs.data(), {static_cast<long>(total_agents), static_cast<long>(obs_builder_->obs_dim())}, torch::kFloat32).clone().to(device_));
        masks.copy_(torch::from_blob(host_masks.data(), {static_cast<long>(total_agents), static_cast<long>(discrete->action_table().size())}, torch::kUInt8).clone().to(device_));

        const ActorStepOutput current_out =
            current_model->forward_step(current_normalizer.normalize(obs), std::move(current_state), episode_starts);
        const ActorStepOutput snapshot_out =
            snapshot.model->forward_step(snapshot.normalizer.normalize(obs), std::move(snapshot_state), episode_starts);
        current_state = std::move(current_out.state);
        snapshot_state = std::move(snapshot_out.state);
        const torch::Tensor action_masks = masks.to(torch::kBool);
        const torch::Tensor current_actions =
            deterministic ? masked_argmax(current_out.policy_logits, action_masks)
                          : masked_sample(current_out.policy_logits, action_masks);
        const torch::Tensor snapshot_actions =
            deterministic ? masked_argmax(snapshot_out.policy_logits, action_masks)
                          : masked_sample(snapshot_out.policy_logits, action_masks);

        std::vector<std::int64_t> merged(total_agents, 0);
        const torch::Tensor current_cpu = current_actions.to(torch::kCPU);
        const torch::Tensor snapshot_cpu = snapshot_actions.to(torch::kCPU);
        const auto* current_ptr = current_cpu.data_ptr<std::int64_t>();
        const auto* snapshot_ptr = snapshot_cpu.data_ptr<std::int64_t>();
        for (std::size_t env_idx = 0; env_idx < engines.size(); ++env_idx) {
          const auto& state = engines[env_idx]->state();
          const std::size_t offset = env_idx * agents_per_env;
          for (std::size_t local_idx = 0; local_idx < state.cars.size(); ++local_idx) {
            const bool use_current = state.cars[local_idx].team == current_team;
            merged[offset + local_idx] = use_current ? current_ptr[offset + local_idx] : snapshot_ptr[offset + local_idx];
          }
        }

        std::vector<ControllerState> parsed(total_agents);
        discrete->parse_actions_into(merged, parsed);
        for (std::size_t env_idx = 0; env_idx < engines.size(); ++env_idx) {
          if (env_done[env_idx]) continue;
          const std::size_t offset = env_idx * agents_per_env;
          engines[env_idx]->step_inplace(
              std::span<const ControllerState>(
                  parsed.data() + static_cast<std::ptrdiff_t>(offset),
                  agents_per_env));
          const EnvState& state = engines[env_idx]->state();
          if (state.goal_scored) {
            const bool current_won =
                (state.blue_score > state.orange_score && current_team == Team::Blue) ||
                (state.orange_score > state.blue_score && current_team == Team::Orange);
            auto [current_it, _] = current_ratings_.emplace(mode, config_.self_play_league.elo_initial);
            auto [snapshot_it, __] = snapshot.ratings.emplace(mode, config_.self_play_league.elo_initial);
            if (current_won) {
              update_elo_ratings(current_it->second, snapshot_it->second, config_.self_play_league.elo_k);
            } else {
              update_elo_ratings(snapshot_it->second, current_it->second, config_.self_play_league.elo_k);
            }
            env_done[env_idx] = true;
            --active_envs;
          }
        }
        episode_starts.zero_();
      }

      // Handle draws: environments that reached max_episode_ticks without a goal.
      for (std::size_t env_idx = 0; env_idx < engines.size(); ++env_idx) {
        if (!env_done[env_idx]) {
          auto [current_it, _] = current_ratings_.emplace(mode, config_.self_play_league.elo_initial);
          auto [snapshot_it, __] = snapshot.ratings.emplace(mode, config_.self_play_league.elo_initial);
          update_elo_draw_impl(current_it->second, snapshot_it->second, config_.self_play_league.elo_k);
          env_done[env_idx] = true;
        }
      }
    }
  }

  metrics.eval_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  for (const auto& pair : current_ratings_) {
    metrics.ratings[pair.first] = pair.second;
  }
  return metrics;
}

std::string SelfPlayManager::mode_name() const {
  return std::to_string(config_.env.team_size) + "v" + std::to_string(config_.env.team_size);
}

}  // namespace pulsar

#endif
