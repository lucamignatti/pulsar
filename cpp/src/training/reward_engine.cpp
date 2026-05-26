#include "pulsar/training/reward_engine.hpp"

namespace pulsar {

RewardEngine::RewardEngine(const ExperimentConfig& cfg)
    : outcome_(cfg.outcome) {}

void RewardEngine::update_config(const ExperimentConfig& cfg) {
  outcome_ = cfg.outcome;
}

void RewardEngine::set_unlocked_mechanics(const std::vector<std::string>& mechanics) {
  // No-op
}

float RewardEngine::outcome_score() const { return outcome_.score; }
float RewardEngine::outcome_concede() const { return outcome_.concede; }
float RewardEngine::outcome_neutral() const { return outcome_.neutral; }
float RewardEngine::outcome_neutral_no_touch() const { return outcome_.neutral_no_touch; }
float RewardEngine::outcome_team_spirit() const { return outcome_.team_spirit; }
float RewardEngine::outcome_step_penalty() const { return outcome_.step_penalty; }

RewardBreakdown RewardEngine::compute(
    int global_tick,
    const CarState& car,
    const EnvState& env,
    int env_team_size,
    AgentRewardState& agent_state,
    EnvRewardState& env_state,
    bool done,
    int outcome_label) const {

  RewardBreakdown bd{};

  if (done) {
    if (outcome_label == 0) {
      bd.terminal = outcome_.score;
      bd.terms["terminal.score"] = outcome_.score;
    } else if (outcome_label == 1) {
      bd.terminal = outcome_.concede;
      bd.terms["terminal.concede"] = outcome_.concede;
    } else if (outcome_label == 2) {
      bd.terminal = outcome_.neutral;
      bd.terms["terminal.neutral"] = outcome_.neutral;
    } else if (outcome_label == 3) {
      bd.terminal = outcome_.neutral_no_touch;
      bd.terms["terminal.no_touch"] = outcome_.neutral_no_touch;
    }
  }

  bd.total = bd.terminal + bd.gameplay + bd.mechanic;

  if (!done && outcome_.step_penalty != 0.0F) {
    bd.total += outcome_.step_penalty;
    bd.terms["step_penalty"] = outcome_.step_penalty;
  }

  return bd;
}

}  // namespace pulsar
