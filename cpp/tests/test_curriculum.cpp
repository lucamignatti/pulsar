#include <cstdlib>
#include <exception>
#include <iostream>
#include <cmath>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/training/curriculum.hpp"
#include "test_utils.hpp"

namespace {

std::map<std::string, double> mode_map(double v) {
  return {{"1v1", v}};
}

std::map<std::string, double> mode_touch(double v) { return mode_map(v); }
std::map<std::string, double> mode_multi_touch(double v) { return mode_map(v); }
std::map<std::string, double> mode_scored(double v) { return mode_map(v); }

void require_eq(int actual, int expected, const std::string& msg) {
  if (actual != expected) {
    throw std::runtime_error(
        msg + " (expected " + std::to_string(expected) +
        ", got " + std::to_string(actual) + ")");
  }
}

void require_approx(float actual, float expected, const std::string& msg) {
  if (std::fabs(actual - expected) > 1e-5f) {
    throw std::runtime_error(
        msg + " (expected ~" + std::to_string(expected) +
        ", got " + std::to_string(actual) + ")");
  }
}

pulsar::CurriculumConfig make_two_stage_config() {
  pulsar::CurriculumConfig config;
  config.enabled = true;

  pulsar::CurriculumStageConfig stage0;
  stage0.name = "stage0";
  stage0.learning_rate = 0.001f;
  stage0.min_agent_steps = 1000;
  stage0.rolling_window_size = 3;
  stage0.consecutive_success_threshold = 3;
  stage0.required_touch_episode_rate = 0.3f;
  stage0.required_scored_episode_rate = 0.0f;
  stage0.mode_allocation["1v1"] = 1.0F;
  stage0.outcome_override.score = 1.0f;
  stage0.outcome_override.concede = -1.0f;
  stage0.outcome_override.neutral = 0.0f;
  config.stages.push_back(stage0);

  pulsar::CurriculumStageConfig stage1;
  stage1.name = "stage1";
  stage1.learning_rate = 0.0005f;
  stage1.min_agent_steps = 2000;
  stage1.rolling_window_size = 3;
  stage1.consecutive_success_threshold = 3;
  stage1.required_touch_episode_rate = 0.0f;
  stage1.required_scored_episode_rate = 0.1f;
  stage1.mode_allocation["1v1"] = 1.0F;
  stage1.outcome_override.score = 10.0f;
  stage1.outcome_override.concede = -8.0f;
  config.stages.push_back(stage1);

  return config;
}

void test_basic_initialization() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  pulsar::test::require(curriculum.enabled(), "should be enabled");
  require_eq(curriculum.stage_index(), 0, "stage index starts at 0");
  pulsar::test::require(curriculum.current_stage().name == "stage0", "stage name is stage0");
  require_approx(curriculum.outcome().score, 1.0f, "stage0 score override");
  require_approx(curriculum.outcome().concede, -1.0f, "stage0 concede override");
  require_approx(curriculum.learning_rate(), 0.001f, "stage0 learning rate");

  const auto& state = curriculum.state();
  require_eq(state.stage_index, 0, "state stage_index");
  require_eq(state.promotion_counter, 0, "state promotion_counter");
  pulsar::test::require(state.agent_steps_in_stage == 0, "state agent_steps_in_stage");
  pulsar::test::require(state.mode_touch_rates.empty(), "state touch_rates empty");
}

void test_no_promotion_before_min_steps() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 500);
  pulsar::test::require(!promoted, "no promotion when steps < min_agent_steps");
  require_eq(curriculum.stage_index(), 0, "still at stage 0");
}

void test_no_promotion_before_window_filled() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  // window=3, first call: size 1 < 3
  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 2000);
  pulsar::test::require(!promoted, "no promotion with 1 sample (window=3)");

  // second call: size 2 < 3
  promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 0);
  pulsar::test::require(!promoted, "no promotion with 2 samples (window=3)");
  require_eq(curriculum.stage_index(), 0, "still stage 0");
}

void test_promotion_after_consecutive_passing_windows() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  // window=3: need 3 calls to fill window (counter becomes 1),
  // then 2 more passing evaluations to reach counter=3 and promote.
  bool promoted = false;
  for (int i = 0; i < 5; i++) {
    promoted = curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }
  pulsar::test::require(promoted, "promoted after 5 consecutive passing windows");
  require_eq(curriculum.stage_index(), 1, "now at stage 1");
}

void test_stage1_values_after_promotion() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  for (int i = 0; i < 5; i++) {
    curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }

  pulsar::test::require(curriculum.current_stage().name == "stage1", "stage 1 name");
  require_approx(curriculum.outcome().score, 10.0f, "stage1 score");
  require_approx(curriculum.outcome().concede, -8.0f, "stage1 concede");
  require_approx(curriculum.learning_rate(), 0.0005f, "stage1 lr");
}

void test_no_promotion_from_last_stage() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  for (int i = 0; i < 5; i++) {
    curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }
  require_eq(curriculum.stage_index(), 1, "at last stage");

  // last stage returns false before any threshold evaluation
  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 99999);
  pulsar::test::require(!promoted, "last stage never promotes");
  require_eq(curriculum.stage_index(), 1, "still at last stage");
}

void test_disabled_curriculum_never_promotes() {
  pulsar::CurriculumConfig cfg;
  cfg.enabled = false;
  pulsar::CurriculumStageConfig stage;
  stage.name = "only";
  stage.min_agent_steps = 10;
  stage.rolling_window_size = 1;
  stage.consecutive_success_threshold = 1;
  stage.required_touch_episode_rate = 0.0f;
  stage.required_scored_episode_rate = 0.0f;
  stage.mode_allocation["1v1"] = 1.0F;
  cfg.stages.push_back(stage);
  cfg.stages.push_back(pulsar::CurriculumStageConfig{});

  pulsar::Curriculum curriculum(cfg);
  pulsar::test::require(!curriculum.enabled(), "disabled");

  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 99999);
  pulsar::test::require(!promoted, "disabled never promotes");
  require_eq(curriculum.stage_index(), 0, "disabled stays at stage 0");
}

void test_counter_resets_when_touch_threshold_fails() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  // Build counter to 1 (3 calls fill window, all pass)
  for (int i = 0; i < 3; i++) {
    curriculum.check_promotion(mode_touch(0.35), mode_scored(0.0), 2000);
  }
  require_eq(curriculum.state().promotion_counter, 1,
             "counter is 1 after 3 passing calls");

  // Push failing touch rate: (0.35+0.35+0.0)/3 = 0.233 < 0.3
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.0), 0);
  require_eq(curriculum.state().promotion_counter, 0,
             "counter reset after touch threshold failure");
  require_eq(curriculum.stage_index(), 0, "still at stage 0");
}

void test_counter_resets_when_scored_threshold_fails() {
  pulsar::CurriculumConfig cfg;
  cfg.enabled = true;

  pulsar::CurriculumStageConfig stage0;
  stage0.name = "stage0";
  stage0.min_agent_steps = 500;
  stage0.rolling_window_size = 2;
  stage0.consecutive_success_threshold = 2;
  stage0.required_touch_episode_rate = 0.0f;
  stage0.required_scored_episode_rate = 0.4f;
  stage0.mode_allocation["1v1"] = 1.0F;
  cfg.stages.push_back(stage0);

  pulsar::CurriculumStageConfig stage1;
  stage1.name = "stage1";
  stage1.mode_allocation["1v1"] = 1.0F;
  cfg.stages.push_back(stage1);

  pulsar::Curriculum curriculum(cfg);

  // 2 calls fill window=2, both pass scored threshold
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.6), 600);
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.6), 0);
  require_eq(curriculum.state().promotion_counter, 1,
             "counter=1 after 2 passing scored calls");

  // failing scored rate: (0.6+0.1)/2 = 0.35 < 0.4
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.1), 0);
  require_eq(curriculum.state().promotion_counter, 0,
             "counter reset by scored failure");
  require_eq(curriculum.stage_index(), 0, "still stage 0 after scored failure");
}

void test_promotion_after_recovery_from_failure() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  // 3 calls: build counter to 1
  for (int i = 0; i < 3; i++) {
    curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }
  require_eq(curriculum.state().promotion_counter, 1, "counter=1");

  // failing calls: need 2 zero-rate calls to flush both remaining 0.5s from window
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.0), 0);
  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.0), 0);
  require_eq(curriculum.state().promotion_counter, 0, "counter=0 after failure");

  // need more passing calls to refill window and re-promote
  bool promoted = false;
  for (int i = 0; i < 10; i++) {
    if (curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 0)) {
      promoted = true;
      break;
    }
  }
  pulsar::test::require(promoted, "promoted after recovery");
  require_eq(curriculum.stage_index(), 1, "at stage 1");
}

void test_state_resets_after_promotion() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  for (int i = 0; i < 5; i++) {
    curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }

  const auto& state = curriculum.state();
  require_eq(state.stage_index, 1, "state stage_index updated");
  require_eq(state.promotion_counter, 0, "promotion_counter reset");
  pulsar::test::require(state.agent_steps_in_stage == 0, "agent_steps_in_stage reset");
  pulsar::test::require(state.mode_touch_rates.empty(), "touch_rates cleared after promotion");
  pulsar::test::require(state.mode_scored_rates.empty(), "scored_rates cleared after promotion");
}

void test_agent_steps_accumulate_correctly() {
  auto cfg = make_two_stage_config();
  pulsar::Curriculum curriculum(cfg);

  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.0), 300);
  pulsar::test::require(curriculum.state().agent_steps_in_stage == 300,
                        "accumulated 300 steps");

  curriculum.check_promotion(mode_touch(0.0), mode_scored(0.0), 500);
  pulsar::test::require(curriculum.state().agent_steps_in_stage == 800,
                        "accumulated 800 steps");

  // 800 < 1000 min_agent_steps, so no stats collection yet
  require_eq(curriculum.stage_index(), 0, "still stage 0 below min steps");
}

void test_single_stage_never_promotes() {
  pulsar::CurriculumConfig cfg;
  cfg.enabled = true;

  pulsar::CurriculumStageConfig stage;
  stage.name = "only";
  stage.min_agent_steps = 100;
  stage.rolling_window_size = 1;
  stage.consecutive_success_threshold = 1;
  stage.required_touch_episode_rate = 0.0f;
  stage.required_scored_episode_rate = 0.0f;
  stage.mode_allocation["1v1"] = 1.0F;
  cfg.stages.push_back(stage);

  pulsar::Curriculum curriculum(cfg);
  require_eq(curriculum.stage_index(), 0, "single stage index");

  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 99999);
  pulsar::test::require(!promoted, "single stage never promotes");
  require_eq(curriculum.stage_index(), 0, "still at single stage");
}

void test_empty_stages_never_promotes() {
  pulsar::CurriculumConfig cfg;
  cfg.enabled = true;

  pulsar::Curriculum curriculum(cfg);
  pulsar::test::require(curriculum.enabled(), "enabled with empty stages");

  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_scored(1.0), 99999);
  pulsar::test::require(!promoted, "empty stages never promotes");
}

void test_initialize_stage_clamps_index() {
  pulsar::CurriculumConfig cfg;
  cfg.enabled = true;

  pulsar::CurriculumStageConfig stage;
  stage.name = "clamped";
  stage.learning_rate = 0.002f;
  cfg.stages.push_back(stage);

  pulsar::Curriculum curriculum(cfg);
  // force stage_index out of range then re-initialize
  // initialize_stage clamps and resets counters
  curriculum.initialize_stage();

  require_eq(curriculum.stage_index(), 0, "stage index clamped to 0");
  require_approx(curriculum.learning_rate(), 0.002f, "clamped stage lr");
  pulsar::test::require(curriculum.state().agent_steps_in_stage == 0,
                        "initialize_stage resets agent steps");
  require_eq(curriculum.state().promotion_counter, 0,
             "initialize_stage resets promotion counter");
}

void test_multi_touch_threshold_gates_promotion() {
  auto cfg = make_two_stage_config();
  cfg.stages[0].min_agent_steps = 0;
  cfg.stages[0].rolling_window_size = 2;
  cfg.stages[0].consecutive_success_threshold = 1;
  cfg.stages[0].required_touch_episode_rate = 0.0f;
  cfg.stages[0].required_multi_touch_episode_rate = 0.5f;

  pulsar::Curriculum curriculum(cfg);

  curriculum.check_promotion(mode_touch(1.0), mode_multi_touch(0.25), mode_scored(0.0), 0);
  bool promoted = curriculum.check_promotion(mode_touch(1.0), mode_multi_touch(0.25), mode_scored(0.0), 0);
  pulsar::test::require(!promoted, "multi-touch threshold should block promotion");
  require_eq(curriculum.stage_index(), 0, "still stage 0 after low multi-touch");

  promoted = curriculum.check_promotion(mode_touch(1.0), mode_multi_touch(0.75), mode_scored(0.0), 0);
  pulsar::test::require(promoted, "high multi-touch should allow promotion");
  require_eq(curriculum.stage_index(), 1, "promoted after multi-touch threshold passes");
}

void test_mechanic_rewards_and_dense_rewards() {
  auto cfg = make_two_stage_config();

  // stage 0 defaults
  cfg.stages[0].mechanic_rewards_override.speed_flip = 0.25f;
  cfg.stages[0].dense_rewards_override.ball_touch_vel_weight = 0.15f;

  // stage 1 overrides
  cfg.stages[1].mechanic_rewards_override.wavedash = 0.5f;
  cfg.stages[1].dense_rewards_override.face_ball_weight = 0.3f;

  pulsar::Curriculum curriculum(cfg);

  require_approx(curriculum.mechanic_rewards().speed_flip, 0.25f,
                 "stage0 mechanic rewards");
  require_approx(curriculum.dense_rewards().ball_touch_vel_weight, 0.15f,
                 "stage0 dense rewards");

  // promote to stage 1
  for (int i = 0; i < 5; i++) {
    curriculum.check_promotion(mode_touch(0.5), mode_scored(0.0), 2000);
  }

  require_approx(curriculum.mechanic_rewards().wavedash, 0.5f,
                 "stage1 mechanic rewards");
  require_approx(curriculum.dense_rewards().face_ball_weight, 0.3f,
                 "stage1 dense rewards");
}

}  // namespace

int main() {
  try {
    test_basic_initialization();
    test_no_promotion_before_min_steps();
    test_no_promotion_before_window_filled();
    test_promotion_after_consecutive_passing_windows();
    test_stage1_values_after_promotion();
    test_no_promotion_from_last_stage();
    test_disabled_curriculum_never_promotes();
    test_counter_resets_when_touch_threshold_fails();
    test_counter_resets_when_scored_threshold_fails();
    test_promotion_after_recovery_from_failure();
    test_state_resets_after_promotion();
    test_agent_steps_accumulate_correctly();
    test_single_stage_never_promotes();
    test_empty_stages_never_promotes();
    test_initialize_stage_clamps_index();
    test_multi_touch_threshold_gates_promotion();
    test_mechanic_rewards_and_dense_rewards();
    std::cout << "test_curriculum passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "test_curriculum FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
