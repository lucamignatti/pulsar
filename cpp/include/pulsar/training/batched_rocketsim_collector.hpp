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
#include "pulsar/training/reward_engine.hpp"

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
  void update_reward_config(const ExperimentConfig& cfg);
  void update_unlocked_mechanics(const std::vector<std::string>& mechanics);
  void set_mode(const std::string& mode);
  [[nodiscard]] const std::string& mode() const;
  [[nodiscard]] std::int8_t mode_id() const;
  void reset_all(CollectorTimings* timings = nullptr);
  void reset_es_episode(int update_index, int episode_index, int eval_envs_per_member, CollectorTimings* timings = nullptr);

  [[nodiscard]] std::size_t num_envs() const;
  [[nodiscard]] std::size_t total_agents() const;
  [[nodiscard]] int obs_dim() const;
  [[nodiscard]] int action_dim() const;

  void step(std::span<const ControllerState> actions, CollectorTimings* timings = nullptr);
  void step(std::span<const std::int64_t> action_indices, CollectorTimings* timings = nullptr);

  [[nodiscard]] const torch::Tensor& host_observations() const;
  [[nodiscard]] const torch::Tensor& host_action_masks() const;
  [[nodiscard]] const torch::Tensor& host_learner_active() const;
  [[nodiscard]] const torch::Tensor& host_snapshot_ids() const;
  [[nodiscard]] const torch::Tensor& host_episode_starts() const;
  [[nodiscard]] const torch::Tensor& host_dones() const;
  [[nodiscard]] const torch::Tensor& host_terminated() const;
  [[nodiscard]] const torch::Tensor& host_truncated() const;
  [[nodiscard]] const torch::Tensor& host_terminal_outcome_labels() const;
  [[nodiscard]] const torch::Tensor& host_terminal_observations() const;
  [[nodiscard]] const torch::Tensor& host_goal_positions() const;
  [[nodiscard]] const torch::Tensor& host_ball_proximity() const;
  [[nodiscard]] const torch::Tensor& host_episode_ball_touch() const;
  [[nodiscard]] const torch::Tensor& host_episode_ball_touch_count() const;
  [[nodiscard]] const torch::Tensor& host_rewards() const;
  [[nodiscard]] const torch::Tensor& host_gameplay_rewards() const;
  [[nodiscard]] const torch::Tensor& host_mechanic_rewards() const;
  [[nodiscard]] const torch::Tensor& host_env_touched() const;
  [[nodiscard]] const torch::Tensor& host_env_multi_touched() const;
  [[nodiscard]] const torch::Tensor& host_bootstrap_truncated() const;

 private:
  struct HostBuffers {
    torch::Tensor obs{};
    torch::Tensor action_masks{};
    torch::Tensor learner_active{};
    torch::Tensor snapshot_ids{};
    torch::Tensor episode_starts{};
  };

  struct EnvRuntime {
    TransitionEnginePtr engine;
    SelfPlayAssignment assignment;
    std::uint64_t reset_seed = 0;
    std::vector<ControllerState> action_scratch{};
    std::vector<std::uint8_t> terminated_scratch{};
    std::vector<std::uint8_t> truncated_scratch{};
  };

  [[nodiscard]] HostBuffers allocate_host_buffers(bool pin_host_memory) const;
  void assign_env(std::size_t env_idx, std::uint64_t seed);
  void finalize_step(CollectorTimings* timings);
  void initialize(std::vector<TransitionEnginePtr> engines, bool pin_host_memory);
  void rebuild_host_buffers(HostBuffers& buffers, CollectorTimings* timings);
  void rebuild_next_buffers(CollectorTimings* timings);

  ExperimentConfig config_{};
  ObsBuilderPtr obs_builder_{};
  ActionParserPtr action_parser_{};
  DoneConditionPtr done_condition_{};
  ParallelExecutor executor_;
  std::vector<EnvRuntime> envs_{};
  std::vector<std::size_t> agent_offsets_{};
  AssignmentFn assignment_fn_{};
  HostBuffers current_buffers_{};
  HostBuffers next_buffers_{};
  torch::Tensor host_dones_;
  torch::Tensor host_terminated_;
  torch::Tensor host_truncated_;
  torch::Tensor host_terminal_outcome_labels_;
  torch::Tensor host_terminal_observations_;
  torch::Tensor host_goal_positions_;
  torch::Tensor host_ball_proximity_;
  torch::Tensor host_episode_ball_touch_;
  torch::Tensor host_episode_ball_touch_count_;
  torch::Tensor host_rewards_;
  torch::Tensor host_gameplay_rewards_;
  torch::Tensor host_mechanic_rewards_;
  torch::Tensor host_env_touched_;
  torch::Tensor host_env_multi_touched_;
  torch::Tensor host_bootstrap_truncated_;
  std::vector<AgentRewardState> agent_reward_states_;
  std::vector<EnvRewardState> env_reward_states_;
  RewardEngine reward_engine_;
  std::string mode_ = "1v1";
  std::size_t total_agents_ = 0;
  int obs_dim_ = 0;
  int action_dim_ = 0;
};

}  // namespace pulsar

#endif
