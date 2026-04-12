#pragma once

#ifdef PULSAR_HAS_TORCH

#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"

namespace pulsar {

void update_elo_ratings(double& winner, double& loser, double k_factor);

struct SelfPlayMetrics {
  double eval_seconds = 0.0;
  std::map<std::string, double> ratings{};
  int snapshot_count = 0;
};

class SelfPlayManager {
 public:
  SelfPlayManager(
      ExperimentConfig config,
      std::filesystem::path snapshot_root,
      ObsBuilderPtr obs_builder,
      ActionParserPtr action_parser,
      torch::Device device);

  [[nodiscard]] bool enabled() const;
  [[nodiscard]] SelfPlayAssignment sample_assignment(std::size_t env_idx, std::uint64_t seed);
  [[nodiscard]] bool has_snapshots() const;

  void infer_opponent_actions(
      SharedActorCritic& current_model,
      const torch::Tensor& raw_obs,
      const torch::Tensor& action_masks,
      const torch::Tensor& episode_starts,
      const torch::Tensor& snapshot_ids,
      ContinuumState& opponent_state,
      torch::Tensor* out_actions,
      double* inference_seconds);

  SelfPlayMetrics on_update(
      SharedActorCritic& current_model,
      const ObservationNormalizer& current_normalizer,
      std::int64_t global_step,
      int update_index);

 private:
  struct Snapshot {
    std::int64_t global_step = 0;
    int update_index = 0;
    SharedActorCritic model{nullptr};
    ObservationNormalizer normalizer{0};
    std::map<std::string, double> ratings{};
  };

  void load_existing_snapshots();
  void save_snapshot(const Snapshot& snapshot) const;
  void trim_snapshots();
  void add_snapshot(
      SharedActorCritic& current_model,
      const ObservationNormalizer& current_normalizer,
      std::int64_t global_step,
      int update_index);
  SelfPlayMetrics evaluate_current(SharedActorCritic& current_model, const ObservationNormalizer& current_normalizer);
  [[nodiscard]] std::string mode_name() const;

  ExperimentConfig config_{};
  std::filesystem::path snapshot_root_{};
  ObsBuilderPtr obs_builder_{};
  ActionParserPtr action_parser_{};
  torch::Device device_{torch::kCPU};
  std::vector<Snapshot> snapshots_{};
  std::map<std::string, double> current_ratings_{};
  mutable std::mt19937 rng_;
};

}  // namespace pulsar

#endif
