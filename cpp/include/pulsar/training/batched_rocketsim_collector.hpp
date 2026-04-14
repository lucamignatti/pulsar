#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/core/parallel_executor.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

struct CollectorTimings {
  double obs_build_seconds = 0.0;
  double mask_build_seconds = 0.0;
  double env_step_seconds = 0.0;
  double done_reset_seconds = 0.0;
};

struct SelfPlayAssignment {
  bool enabled = false;
  Team learner_team = Team::Blue;
  int snapshot_index = -1;
};

class BatchedRocketSimCollector {
 public:
  using AssignmentFn = std::function<SelfPlayAssignment(std::size_t, std::uint64_t)>;

  BatchedRocketSimCollector(
      ExperimentConfig config,
      ObsBuilderPtr obs_builder,
      ActionParserPtr action_parser,
      DoneConditionPtr done_condition,
      bool pin_host_memory);
  BatchedRocketSimCollector(
      ExperimentConfig config,
      std::vector<TransitionEnginePtr> engines,
      ObsBuilderPtr obs_builder,
      ActionParserPtr action_parser,
      DoneConditionPtr done_condition,
      bool pin_host_memory);

  void set_self_play_assignment_fn(AssignmentFn assignment_fn);

  [[nodiscard]] std::size_t num_envs() const;
  [[nodiscard]] std::size_t total_agents() const;
  [[nodiscard]] int obs_dim() const;
  [[nodiscard]] int action_dim() const;

  torch::Tensor collect_observations(CollectorTimings* timings = nullptr);
  void collect_action_masks(CollectorTimings* timings = nullptr);
  void step(std::span<const ControllerState> actions, bool collect_post_step_obs, CollectorTimings* timings = nullptr);

  [[nodiscard]] const torch::Tensor& host_action_masks() const;
  [[nodiscard]] const torch::Tensor& host_learner_active() const;
  [[nodiscard]] const torch::Tensor& host_snapshot_ids() const;
  [[nodiscard]] const torch::Tensor& host_episode_starts() const;
  [[nodiscard]] const torch::Tensor& host_dones() const;
  [[nodiscard]] const torch::Tensor& host_terminated() const;
  [[nodiscard]] const torch::Tensor& host_truncated() const;
  [[nodiscard]] const torch::Tensor& host_terminal_next_goal_labels() const;
  [[nodiscard]] const torch::Tensor& host_post_step_obs() const;

 private:
  struct EnvRuntime {
    TransitionEnginePtr engine;
    SelfPlayAssignment assignment;
    std::uint64_t reset_seed = 0;
  };

  void assign_env(std::size_t env_idx, std::uint64_t seed);

  ExperimentConfig config_{};
  ObsBuilderPtr obs_builder_{};
  ActionParserPtr action_parser_{};
  DoneConditionPtr done_condition_{};
  ParallelExecutor executor_;
  std::vector<EnvRuntime> envs_{};
  std::vector<std::size_t> agent_offsets_{};
  AssignmentFn assignment_fn_{};
  torch::Tensor host_obs_;
  torch::Tensor host_action_masks_;
  torch::Tensor host_learner_active_;
  torch::Tensor host_snapshot_ids_;
  torch::Tensor host_episode_starts_;
  torch::Tensor host_dones_;
  torch::Tensor host_terminated_;
  torch::Tensor host_truncated_;
  torch::Tensor host_terminal_next_goal_labels_;
  torch::Tensor host_post_step_obs_;
  std::size_t total_agents_ = 0;
  int obs_dim_ = 0;
  int action_dim_ = 0;
};

}  // namespace pulsar

#endif
