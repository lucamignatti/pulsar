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

bool Curriculum::mode_allocation_changed() const {
  const auto& stages = config_.stages;
  if (stages.size() <= 1) return false;
  const int curr = state_.stage_index;
  const int prev = state_.previous_stage_index;
  if (prev < 0 || prev >= static_cast<int>(stages.size())) return false;
  if (curr < 0 || curr >= static_cast<int>(stages.size())) return false;
  return stages[static_cast<std::size_t>(prev)].mode_allocation
      != stages[static_cast<std::size_t>(curr)].mode_allocation;
}

const std::map<std::string, float>& Curriculum::mode_allocation() const {
  return current_stage().mode_allocation;
}

std::string Curriculum::primary_mode() const {
  const auto& alloc = current_stage().mode_allocation;
  float best = -1.0F;
  std::string best_mode = "1v1";
  for (const auto& [mode, frac] : alloc) {
    if (frac > best) {
      best = frac;
      best_mode = mode;
    }
  }
  if (best < 0.0F && !alloc.empty()) {
    best_mode = alloc.begin()->first;
  }
  return best_mode;
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
  state_.mode_touch_rates.clear();
  state_.mode_multi_touch_rates.clear();
  state_.mode_scored_rates.clear();

  const auto& stage = stages[static_cast<std::size_t>(idx)];
  state_.current_mode = primary_mode();

  std::cout << "curriculum_stage=" << idx
            << " name=" << stage.name
            << " primary_mode=" << state_.current_mode
            << " modes=" << stage.mode_allocation.size()
            << " lr=" << stage.learning_rate
            << " mechanics_unlocked=" << stage.unlocked_mechanics.size()
            << '\n';
}

bool Curriculum::check_promotion(
    const std::map<std::string, double>& mode_touch_rates,
    const std::map<std::string, double>& mode_scored_rates,
    std::int64_t agent_steps) {
  static const std::map<std::string, double> kNoMultiTouchRates{};
  return check_promotion(mode_touch_rates, kNoMultiTouchRates, mode_scored_rates, agent_steps);
}

bool Curriculum::check_promotion(
    const std::map<std::string, double>& mode_touch_rates,
    const std::map<std::string, double>& mode_multi_touch_rates,
    const std::map<std::string, double>& mode_scored_rates,
    std::int64_t agent_steps) {
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

  // push per-mode rates into deques
  for (const auto& [mode, frac] : stage.mode_allocation) {
    auto touch_it = mode_touch_rates.find(mode);
    if (touch_it != mode_touch_rates.end()) {
      state_.mode_touch_rates[mode].push_back(touch_it->second);
    } else {
      state_.mode_touch_rates[mode].push_back(0.0);
    }

    auto multi_touch_it = mode_multi_touch_rates.find(mode);
    if (multi_touch_it != mode_multi_touch_rates.end()) {
      state_.mode_multi_touch_rates[mode].push_back(multi_touch_it->second);
    } else {
      state_.mode_multi_touch_rates[mode].push_back(0.0);
    }

    auto scored_it = mode_scored_rates.find(mode);
    if (scored_it != mode_scored_rates.end()) {
      state_.mode_scored_rates[mode].push_back(scored_it->second);
    } else {
      state_.mode_scored_rates[mode].push_back(0.0);
    }

    while (static_cast<int>(state_.mode_touch_rates[mode].size()) > window_size) {
      state_.mode_touch_rates[mode].pop_front();
    }
    while (static_cast<int>(state_.mode_multi_touch_rates[mode].size()) > window_size) {
      state_.mode_multi_touch_rates[mode].pop_front();
    }
    while (static_cast<int>(state_.mode_scored_rates[mode].size()) > window_size) {
      state_.mode_scored_rates[mode].pop_front();
    }
  }

  // check all modes have filled their windows
  for (const auto& [mode, frac] : stage.mode_allocation) {
    if (static_cast<int>(state_.mode_touch_rates[mode].size()) < window_size) return false;
  }

  // every mode must independently pass both thresholds
  bool all_pass = true;
  for (const auto& [mode, frac] : stage.mode_allocation) {
    double avg_touch = 0.0;
    for (double v : state_.mode_touch_rates[mode]) avg_touch += v;
    avg_touch /= static_cast<double>(state_.mode_touch_rates[mode].size());

    double avg_multi_touch = 0.0;
    for (double v : state_.mode_multi_touch_rates[mode]) avg_multi_touch += v;
    avg_multi_touch /= static_cast<double>(state_.mode_multi_touch_rates[mode].size());

    double avg_scored = 0.0;
    for (double v : state_.mode_scored_rates[mode]) avg_scored += v;
    avg_scored /= static_cast<double>(state_.mode_scored_rates[mode].size());

    // Per-mode threshold overrides take precedence over the scalar fields.
    float touch_threshold = stage.required_touch_episode_rate;
    {
      auto it = stage.mode_touch_thresholds.find(mode);
      if (it != stage.mode_touch_thresholds.end()) touch_threshold = it->second;
    }
    float multi_touch_threshold = stage.required_multi_touch_episode_rate;
    {
      auto it = stage.mode_multi_touch_thresholds.find(mode);
      if (it != stage.mode_multi_touch_thresholds.end()) multi_touch_threshold = it->second;
    }
    float score_threshold = stage.required_scored_episode_rate;
    {
      auto it = stage.mode_scored_thresholds.find(mode);
      if (it != stage.mode_scored_thresholds.end()) score_threshold = it->second;
    }
    bool touch_ok = touch_threshold <= 0.0F ||
                    avg_touch >= static_cast<double>(touch_threshold);
    bool multi_touch_ok = multi_touch_threshold <= 0.0F ||
                          avg_multi_touch >= static_cast<double>(multi_touch_threshold);
    bool score_ok = score_threshold <= 0.0F ||
                    avg_scored >= static_cast<double>(score_threshold);

    if (!touch_ok || !multi_touch_ok || !score_ok) {
      all_pass = false;
      break;
    }
  }

  if (!all_pass) {
    state_.promotion_counter = 0;
    return false;
  }

  state_.promotion_counter++;

  if (state_.promotion_counter >= stage.consecutive_success_threshold) {
    state_.previous_stage_index = state_.stage_index;
    state_.stage_index++;
    state_.promotion_counter = 0;
    state_.mode_touch_rates.clear();
    state_.mode_multi_touch_rates.clear();
    state_.mode_scored_rates.clear();
    initialize_stage();
    std::cout << "curriculum_promoted stage=" << state_.stage_index
              << " primary_mode=" << state_.current_mode
              << " alloc_modes=" << stage.mode_allocation.size() << '\n';
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
