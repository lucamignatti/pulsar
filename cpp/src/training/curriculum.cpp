#include "pulsar/training/curriculum.hpp"

#include <algorithm>
#include <iostream>

namespace pulsar {

Curriculum::Curriculum(const CurriculumConfig& config) : config_(config) {}

bool Curriculum::enabled() const {
  return config_.enabled;
}

const CurriculumStageConfig& Curriculum::current_stage() const {
  const auto& stages = config_.stages;
  const int idx = std::min(state_.stage_index, static_cast<int>(stages.size()) - 1);
  return stages[static_cast<std::size_t>(idx)];
}

int Curriculum::stage_index() const {
  return state_.stage_index;
}

const std::string& Curriculum::current_mode() const {
  return state_.current_mode;
}

bool Curriculum::mode_changed() const {
  return current_stage().mode != state_.current_mode;
}

const OutcomeConfig& Curriculum::outcome() const {
  return current_stage().outcome_override;
}

const MechanicRewardConfig& Curriculum::mechanic_rewards() const {
  return current_stage().mechanic_rewards_override;
}

const DenseRewardConfig& Curriculum::dense_rewards() const {
  return current_stage().dense_rewards_override;
}

const std::vector<std::string>& Curriculum::unlocked_mechanics() const {
  return current_stage().unlocked_mechanics;
}

float Curriculum::learning_rate() const {
  return current_stage().learning_rate;
}

void Curriculum::initialize_stage() {
  if (!config_.enabled) return;

  const auto& stages = config_.stages;
  if (stages.empty()) return;

  if (state_.stage_index < 0) {
    state_.stage_index = 0;
  }
  const int idx = std::min(state_.stage_index, static_cast<int>(stages.size()) - 1);
  state_.stage_index = idx;
  state_.agent_steps_in_stage = 0;
  state_.promotion_counter = 0;
  state_.demotion_counter = 0;
  state_.touch_rates.clear();
  state_.scored_rates.clear();

  const auto& stage = stages[static_cast<std::size_t>(idx)];
  state_.current_mode = stage.mode;

  std::cout << "curriculum_stage=" << idx
            << " name=" << stage.name
            << " mode=" << stage.mode
            << " lr=" << stage.learning_rate
            << " mechanics_unlocked=" << stage.unlocked_mechanics.size()
            << '\n';
}

bool Curriculum::check_promotion(
    double touch_episode_rate, double scored_episode_rate, std::int64_t agent_steps) {
  if (!config_.enabled) return false;

  const auto& stages = config_.stages;
  if (stages.empty()) return false;

  state_.agent_steps_in_stage += agent_steps;

  const int idx = std::min(state_.stage_index, static_cast<int>(stages.size()) - 1);
  if (idx < 0 || idx >= static_cast<int>(stages.size()) - 1) return false;

  const auto& stage = stages[static_cast<std::size_t>(idx)];

  if (state_.agent_steps_in_stage < stage.min_agent_steps) return false;

  const int window_size = stage.rolling_window_size;
  if (window_size <= 0) return false;

  state_.touch_rates.push_back(touch_episode_rate);
  state_.scored_rates.push_back(scored_episode_rate);

  while (static_cast<int>(state_.touch_rates.size()) > window_size) state_.touch_rates.pop_front();
  while (static_cast<int>(state_.scored_rates.size()) > window_size) state_.scored_rates.pop_front();

  if (static_cast<int>(state_.touch_rates.size()) < window_size) return false;

  double avg_touch = 0.0;
  for (double v : state_.touch_rates) avg_touch += v;
  avg_touch /= static_cast<double>(state_.touch_rates.size());

  double avg_scored = 0.0;
  for (double v : state_.scored_rates) avg_scored += v;
  avg_scored /= static_cast<double>(state_.scored_rates.size());

  bool touch_ok = stage.required_touch_episode_rate <= 0.0F ||
                  avg_touch >= static_cast<double>(stage.required_touch_episode_rate);
  bool score_ok = stage.required_scored_episode_rate <= 0.0F ||
                  avg_scored >= static_cast<double>(stage.required_scored_episode_rate);

  if (!touch_ok || !score_ok) {
    state_.promotion_counter = 0;
    return false;
  }

  state_.promotion_counter++;

  if (state_.promotion_counter >= stage.consecutive_success_threshold) {
    state_.stage_index++;
    state_.promotion_counter = 0;
    state_.touch_rates.clear();
    state_.scored_rates.clear();
    initialize_stage();
    std::cout << "curriculum_promoted stage=" << state_.stage_index
              << " mode=" << state_.current_mode << '\n';
    return true;
  }

  return false;
}

bool Curriculum::check_demotion(double scored_episode_rate) {
  if (!config_.enabled) return false;

  const auto& stages = config_.stages;
  if (stages.empty()) return false;

  if (state_.stage_index <= 0) return false;

  const int idx = std::min(state_.stage_index, static_cast<int>(stages.size()) - 1);
  const auto& stage = stages[static_cast<std::size_t>(idx)];

  if (stage.demotion_threshold_rate <= 0.0F) return false;
  if (stage.demotion_window_updates <= 0) return false;
  if (stage.required_scored_episode_rate <= 0.0F) return false;

  const float threshold = stage.required_scored_episode_rate * stage.demotion_threshold_rate;

  if (scored_episode_rate >= static_cast<double>(threshold)) {
    state_.demotion_counter = 0;
    return false;
  }

  state_.demotion_counter++;

  if (state_.demotion_counter >= stage.demotion_window_updates) {
    state_.stage_index--;
    state_.promotion_counter = 0;
    state_.demotion_counter = 0;
    state_.touch_rates.clear();
    state_.scored_rates.clear();
    initialize_stage();
    std::cout << "curriculum_demoted stage=" << state_.stage_index
              << " mode=" << state_.current_mode << '\n';
    return true;
  }

  return false;
}

const CurriculumState& Curriculum::state() const {
  return state_;
}

void Curriculum::restore_state(const CurriculumState& state) {
  state_ = state;
}

}  // namespace pulsar
