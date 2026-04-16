#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/ppo_trainer.hpp"
#include "pulsar/training/offline_pretrainer.hpp"

namespace {

std::filesystem::path find_repo_collision_meshes() {
  namespace fs = std::filesystem;
  fs::path current = fs::current_path();
  for (int depth = 0; depth < 6; ++depth) {
    const fs::path candidate = current / "collision_meshes";
    if (fs::exists(candidate)) {
      return fs::canonical(candidate);
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  throw std::runtime_error("failed to locate collision_meshes from test working directory");
}

class FakeTransitionEngine final : public pulsar::TransitionEngine {
 public:
  explicit FakeTransitionEngine(pulsar::EnvConfig config) : config_(std::move(config)) {
    reset(config_.seed);
  }

  void reset(std::uint64_t seed) override {
    state_ = {};
    ticks_ = 0;
    pulse_ = static_cast<int>(seed % 9);
    pulsar::FixedTeamSizeMutator fixed(config_);
    pulsar::KickoffMutator kickoff(config_);
    fixed.apply(state_, seed);
    kickoff.apply(state_, seed);
    state_.last_touch_tick = 0;
    state_.goal_scored = false;
  }

  pulsar::StepResult step(std::span<const pulsar::ControllerState> actions) override {
    step_inplace(actions);
    return {.state = state_};
  }

  void step_inplace(std::span<const pulsar::ControllerState> actions) override {
    state_.goal_scored = false;
    state_.tick += config_.tick_skip;
    ticks_ += config_.tick_skip;
    pulse_ += 1;

    state_.ball.position.y += 110.0F;
    state_.ball.velocity = {0.0F, 110.0F, 0.0F};

    for (std::size_t i = 0; i < state_.cars.size(); ++i) {
      auto& car = state_.cars[i];
      const auto& action = actions[i];
      car.velocity = {action.steer * 150.0F, action.throttle * 300.0F, 0.0F};
      car.position = car.position + car.velocity * 0.016F;
      car.forward = {1.0F, 0.0F, 0.0F};
      car.up = {0.0F, 0.0F, 1.0F};
      car.is_boosting = action.boost;
      car.handbrake = action.handbrake ? 1.0F : 0.0F;
      car.ball_touched = false;
    }

    state_.last_touch_agent = pulse_ % static_cast<int>(state_.cars.size());
    state_.cars[static_cast<std::size_t>(state_.last_touch_agent)].ball_touched = true;
    state_.last_touch_tick = state_.tick;

    if (ticks_ >= config_.tick_skip * 3) {
      state_.blue_score += 1;
      state_.goal_scored = true;
      ticks_ = 0;
    }
  }

  const pulsar::EnvState& state() const override {
    return state_;
  }

  std::size_t num_agents() const override {
    return state_.cars.size();
  }

 private:
  pulsar::EnvConfig config_{};
  pulsar::EnvState state_{};
  int ticks_ = 0;
  int pulse_ = 0;
};

}  // namespace

int main() {
  try {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pulsar_offline_test";
    fs::remove_all(root);
    fs::create_directories(root / "data");

    const int64_t rows = 64;
    const int64_t obs_dim = 132;
    const int64_t action_dim = 90;
    torch::Tensor obs = torch::randn({rows, obs_dim});
    torch::Tensor actions = torch::randint(action_dim, {rows}, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor action_probs = torch::one_hot(actions, action_dim).to(torch::kFloat32);
    torch::Tensor next_goal = torch::randint(3, {rows}, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor weights = torch::ones({rows});
    torch::Tensor episode_starts = torch::zeros({rows});
    torch::Tensor terminated = torch::zeros({rows});
    torch::Tensor truncated = torch::zeros({rows});
    episode_starts[0] = 1.0F;
    episode_starts[32] = 1.0F;
    terminated[31] = 1.0F;
    terminated[63] = 1.0F;

    torch::save(obs, (root / "data" / "obs.pt").string());
    torch::save(actions, (root / "data" / "actions.pt").string());
    torch::save(action_probs, (root / "data" / "action_probs.pt").string());
    torch::save(next_goal, (root / "data" / "next_goal.pt").string());
    torch::save(weights, (root / "data" / "weights.pt").string());
    torch::save(episode_starts, (root / "data" / "episode_starts.pt").string());
    torch::save(terminated, (root / "data" / "terminated.pt").string());
    torch::save(truncated, (root / "data" / "truncated.pt").string());

    std::ofstream manifest(root / "data" / "manifest.json");
    manifest << R"({
  "schema_version": 3,
  "observation_dim": 132,
  "action_dim": 90,
  "next_goal_classes": 3,
  "shards": [
    {
      "obs_path": "obs.pt",
      "actions_path": "actions.pt",
      "action_probs_path": "action_probs.pt",
      "next_goal_path": "next_goal.pt",
      "weights_path": "weights.pt",
      "episode_starts_path": "episode_starts.pt",
      "terminated_path": "terminated.pt",
      "truncated_path": "truncated.pt",
      "samples": 64
    }
  ]
})";
    manifest.close();

    pulsar::ExperimentConfig config;
    config.model.observation_dim = static_cast<int>(obs_dim);
    config.model.encoder_dim = 32;
    config.model.workspace_dim = 32;
    config.model.stm_slots = 8;
    config.model.stm_key_dim = 16;
    config.model.stm_value_dim = 16;
    config.model.ltm_slots = 8;
    config.model.ltm_dim = 16;
    config.model.controller_dim = 32;
    config.model.action_dim = static_cast<int>(action_dim);
    config.ppo.device = "cpu";
    config.ppo.value_num_atoms = 11;
    config.offline_dataset.train_manifest = (root / "data" / "manifest.json").string();
    config.offline_dataset.val_manifest = (root / "data" / "manifest.json").string();
    config.offline_dataset.batch_size = 16;
    config.behavior_cloning.epochs = 1;
    config.next_goal_predictor.epochs = 1;
    config.value_pretraining.epochs = 1;

    pulsar::OfflinePretrainer pretrainer(config);
    pretrainer.train((root / "output").string());

    if (!fs::exists(root / "output" / "model.pt")) {
      throw std::runtime_error("offline checkpoint missing");
    }

    pulsar::ExperimentConfig warm_start_config = config;
    warm_start_config.behavior_cloning.enabled = false;
    warm_start_config.behavior_cloning.epochs = 0;
    warm_start_config.next_goal_predictor.init_checkpoint = (root / "output").string();
    warm_start_config.next_goal_predictor.epochs = 1;
    warm_start_config.next_goal_predictor.reuse_normalizer = true;
    warm_start_config.value_pretraining.epochs = 1;
    pulsar::OfflinePretrainer warm_start_pretrainer(warm_start_config);
    warm_start_pretrainer.train((root / "warm_start").string());
    if (!fs::exists(root / "warm_start" / "model.pt")) {
      throw std::runtime_error("warm-start next goal checkpoint missing");
    }

    pulsar::ExperimentConfig eval_only_config = warm_start_config;
    eval_only_config.next_goal_predictor.epochs = 0;
    eval_only_config.value_pretraining.enabled = false;
    eval_only_config.value_pretraining.epochs = 0;
    pulsar::OfflinePretrainer eval_only_pretrainer(eval_only_config);
    eval_only_pretrainer.train((root / "eval_only").string());
    if (!fs::exists(root / "eval_only" / "offline_metrics.jsonl")) {
      throw std::runtime_error("eval-only offline metrics missing");
    }

    config.reward.ngp_checkpoint = (root / "output").string();
    config.reward.ngp_scale = 1.0F;
    config.ppo.init_checkpoint = (root / "output").string();
    config.ppo.num_envs = 2;
    config.ppo.rollout_length = 4;
    config.ppo.minibatch_size = 8;
    config.ppo.epochs = 1;
    config.ppo.checkpoint_interval = 1;
    config.ppo.sequence_length = 2;
    config.ppo.burn_in = 1;
    config.env.seed = 5;
    config.env.collision_meshes_path = find_repo_collision_meshes().string();

    auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
    auto action_parser =
        std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
    auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
    auto collector = std::make_unique<pulsar::BatchedRocketSimCollector>(
        config,
        obs_builder,
        action_parser,
        done_condition,
        false);

    pulsar::PPOTrainer trainer(
        config,
        std::move(collector),
        nullptr,
        root / "ppo");
    trainer.train(1, (root / "ppo").string());

    if (!fs::exists(root / "ppo" / "update_1" / "model.pt")) {
      throw std::runtime_error("ppo update checkpoint missing");
    }
    if (!fs::exists(root / "ppo" / "best" / "model.pt")) {
      throw std::runtime_error("ppo best checkpoint missing");
    }
    if (!fs::exists(root / "ppo" / "final" / "model.pt")) {
      throw std::runtime_error("ppo final checkpoint missing");
    }

    auto make_trainer = [&](const pulsar::ExperimentConfig& trainer_config) {
      auto local_obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(trainer_config.env);
      auto local_action_parser =
          std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(trainer_config.action_table));
      auto local_done_condition = std::make_shared<pulsar::SimpleDoneCondition>(trainer_config.env);
      auto local_collector = std::make_unique<pulsar::BatchedRocketSimCollector>(
          trainer_config,
          local_obs_builder,
          local_action_parser,
          local_done_condition,
          false);
      return std::make_unique<pulsar::PPOTrainer>(
          trainer_config,
          std::move(local_collector),
          nullptr,
          root / "trainer_factory");
    };

    pulsar::ExperimentConfig bad_init_config = config;
    bad_init_config.model.encoder_dim = 64;
    bool init_mismatch_threw = false;
    try {
      (void)make_trainer(bad_init_config);
    } catch (const std::runtime_error& exc) {
      init_mismatch_threw = std::string(exc.what()).find("Init checkpoint") != std::string::npos;
    }
    if (!init_mismatch_threw) {
      throw std::runtime_error("init checkpoint compatibility mismatch should throw clearly");
    }

    pulsar::ExperimentConfig bad_ngp_config = config;
    bad_ngp_config.model.controller_dim = 48;
    bad_ngp_config.ppo.init_checkpoint.clear();
    bool ngp_mismatch_threw = false;
    try {
      (void)make_trainer(bad_ngp_config);
    } catch (const std::runtime_error& exc) {
      ngp_mismatch_threw = std::string(exc.what()).find("NGP checkpoint") != std::string::npos;
    }
    if (!ngp_mismatch_threw) {
      throw std::runtime_error("NGP checkpoint compatibility mismatch should throw clearly");
    }

    fs::remove_all(root);
    std::cout << "pulsar_offline_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
