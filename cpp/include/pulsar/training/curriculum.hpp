#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"

namespace pulsar {

struct CurriculumState {
  int stage_index = 0;
  std::int64_t agent_steps_in_stage = 0;
  int promotion_counter = 0;
  int demotion_counter = 0;
  std::string current_mode = "1v1";
  std::deque<double> touch_rates{};
  std::deque<double> scored_rates{};
};

class Curriculum {
 public:
  explicit Curriculum(const CurriculumConfig& config);

  bool enabled() const;
  const CurriculumStageConfig& current_stage() const;
  int stage_index() const;
  const std::string& current_mode() const;
  bool mode_changed() const;

  const OutcomeConfig& outcome() const;
  const MechanicRewardConfig& mechanic_rewards() const;
  const DenseRewardConfig& dense_rewards() const;
  const std::vector<std::string>& unlocked_mechanics() const;
  float learning_rate() const;

  bool check_promotion(double touch_episode_rate, double scored_episode_rate, std::int64_t agent_steps);
  bool check_demotion(double scored_episode_rate);
  void initialize_stage();

  const CurriculumState& state() const;
  void restore_state(const CurriculumState& state);

 private:
  CurriculumConfig config_;
  CurriculumState state_;
};

}  // namespace pulsar
