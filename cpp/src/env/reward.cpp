#include "pulsar/env/reward.hpp"

#include <algorithm>
#include <cmath>

namespace pulsar {
namespace {

float dot(const Vec3& lhs, const Vec3& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

float magnitude(const Vec3& value) {
  return std::sqrt(dot(value, value));
}

Vec3 normalize(const Vec3& value) {
  const float len = magnitude(value);
  if (len < 1.0e-6F) {
    return {};
  }
  return value * (1.0F / len);
}

Vec3 goal_center_for_team(Team team) {
  return team == Team::Blue ? Vec3{0.0F, 5120.0F, 321.0F} : Vec3{0.0F, -5120.0F, 321.0F};
}

std::vector<float> apply_team_spirit_and_zero_sum(
    const std::vector<float>& base_rewards,
    const EnvState& state,
    float team_spirit,
    float opponent_scale) {
  std::vector<float> adjusted(base_rewards.size(), 0.0F);
  float blue_sum = 0.0F;
  float orange_sum = 0.0F;
  int blue_count = 0;
  int orange_count = 0;

  for (std::size_t i = 0; i < state.cars.size(); ++i) {
    if (state.cars[i].team == Team::Blue) {
      blue_sum += base_rewards[i];
      ++blue_count;
    } else {
      orange_sum += base_rewards[i];
      ++orange_count;
    }
  }

  const float blue_avg = blue_count > 0 ? blue_sum / static_cast<float>(blue_count) : 0.0F;
  const float orange_avg = orange_count > 0 ? orange_sum / static_cast<float>(orange_count) : 0.0F;

  for (std::size_t i = 0; i < state.cars.size(); ++i) {
    const bool blue = state.cars[i].team == Team::Blue;
    const float team_avg = blue ? blue_avg : orange_avg;
    const float opp_avg = blue ? orange_avg : blue_avg;
    adjusted[i] = (base_rewards[i] * (1.0F - team_spirit)) + (team_avg * team_spirit) - (opp_avg * opponent_scale);
  }

  return adjusted;
}

}  // namespace

CombinedRewardFunction::CombinedRewardFunction(RewardConfig config) : config_(std::move(config)) {}

std::vector<float> CombinedRewardFunction::get_rewards(
    const EnvState& previous_state,
    const EnvState& current_state,
    std::span<const std::uint8_t>,
    std::span<const std::uint8_t>) const {
  std::vector<float> rewards(current_state.cars.size(), 0.0F);

  const int blue_delta = current_state.blue_score - previous_state.blue_score;
  const int orange_delta = current_state.orange_score - previous_state.orange_score;

  for (const auto& term : config_.terms) {
    if (term.name == "goal") {
      for (std::size_t i = 0; i < current_state.cars.size(); ++i) {
        const bool blue = current_state.cars[i].team == Team::Blue;
        rewards[i] += term.weight * static_cast<float>((blue ? blue_delta : orange_delta) - (blue ? orange_delta : blue_delta));
      }
      continue;
    }

    if (term.name == "touch") {
      if (current_state.last_touch_agent >= 0 && current_state.last_touch_agent != previous_state.last_touch_agent) {
        rewards.at(static_cast<std::size_t>(current_state.last_touch_agent)) += term.weight;
      }
      continue;
    }

    if (term.name == "speed_to_ball") {
      for (std::size_t i = 0; i < current_state.cars.size(); ++i) {
        const auto& car = current_state.cars[i];
        const Vec3 to_ball = normalize(current_state.ball.position - car.position);
        const float speed_toward_ball = std::max(0.0F, dot(to_ball, car.velocity) / 2300.0F);
        rewards[i] += term.weight * speed_toward_ball;
      }
      continue;
    }

    if (term.name == "ball_to_goal") {
      for (std::size_t i = 0; i < current_state.cars.size(); ++i) {
        const auto& car = current_state.cars[i];
        const Vec3 to_goal = normalize(goal_center_for_team(car.team) - current_state.ball.position);
        rewards[i] += term.weight * (dot(to_goal, current_state.ball.velocity) / 6000.0F);
      }
      continue;
    }

    if (term.name == "face_ball") {
      for (std::size_t i = 0; i < current_state.cars.size(); ++i) {
        const auto& car = current_state.cars[i];
        const Vec3 to_ball = normalize(current_state.ball.position - car.position);
        rewards[i] += term.weight * std::max(0.0F, dot(normalize(car.forward), to_ball));
      }
      continue;
    }

    if (term.name == "in_air") {
      for (std::size_t i = 0; i < current_state.cars.size(); ++i) {
        rewards[i] += term.weight * (current_state.cars[i].on_ground ? 0.0F : 1.0F);
      }
    }
  }

  return apply_team_spirit_and_zero_sum(rewards, current_state, config_.team_spirit, config_.opponent_scale);
}

}  // namespace pulsar
