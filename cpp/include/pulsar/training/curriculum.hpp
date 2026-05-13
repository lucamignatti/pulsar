#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"

namespace pulsar {

struct CurriculumState {
  int stage_index = 0;
  int previous_stage_index = -1;
  std::int64_t agent_steps_in_stage = 0;
  int promotion_counter = 0;
  int demotion_counter = 0;
  std::string current_mode = "1v1";
  std::map<std::string, std::deque<double>> mode_touch_rates{};
  std::map<std::string, std::deque<double>> mode_scored_rates{};
};

class Curriculum {
 public:
  explicit Curriculum(const CurriculumConfig& config);

  bool enabled() const;
  const CurriculumStageConfig& current_stage() const;
  int stage_index() const;
  const std::string& current_mode() const;
  bool mode_changed() const;
  bool mode_allocation_changed() const;
  const std::map<std::string, float>& mode_allocation() const;
  std::string primary_mode() const;

  const OutcomeConfig& outcome() const;
  const MechanicRewardConfig& mechanic_rewards() const;
  const DenseRewardConfig& dense_rewards() const;
  const std::vector<std::string>& unlocked_mechanics() const;
  float learning_rate() const;

  bool check_promotion(
      const std::map<std::string, double>& mode_touch_rates,
      const std::map<std::string, double>& mode_scored_rates,
      std::int64_t agent_steps);
  bool check_demotion(const std::map<std::string, double>& mode_scored_rates);
  void initialize_stage();

  const CurriculumState& state() const;
  void restore_state(const CurriculumState& state);

 private:
  CurriculumConfig config_;
  CurriculumState state_;
};

}  // namespace pulsar
