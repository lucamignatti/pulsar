#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/lfpo_trainer.hpp"
#include "pulsar/training/offline_pretrainer.hpp"

namespace {

class FakeTransitionEngine final : public pulsar::TransitionEngine {
 public:
  explicit FakeTransitionEngine(pulsar::EnvConfig config) : config_(std::move(config)) {
    reset(config_.seed);
  }

  void reset(std::uint64_t seed) override {
    state_ = {};
    ticks_ = 0;
    pulsar::FixedTeamSizeMutator fixed(config_);
    pulsar::KickoffMutator kickoff(config_);
    fixed.apply(state_, seed);
    kickoff.apply(state_, seed);
    state_.last_touch_tick = 0;
    state_.goal_scored = false;
    state_.last_scoring_team = pulsar::Team::Blue;
  }

  pulsar::StepResult step(std::span<const pulsar::ControllerState> actions) override {
    step_inplace(actions);
    return {.state = state_};
  }

  void step_inplace(std::span<const pulsar::ControllerState> actions) override {
    state_.goal_scored = false;
    state_.tick += config_.tick_skip;
    ticks_ += config_.tick_skip;
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
    if (ticks_ >= config_.tick_skip * 3) {
      state_.blue_score += 1;
      state_.goal_scored = true;
      state_.last_scoring_team = pulsar::Team::Blue;
      ticks_ = 0;
    }
  }

  const pulsar::EnvState& state() const override { return state_; }
  std::size_t num_agents() const override { return state_.cars.size(); }

 private:
  pulsar::EnvConfig config_{};
  pulsar::EnvState state_{};
  int ticks_ = 0;
};

pulsar::ExperimentConfig make_lfpo_smoke_config(const std::filesystem::path& manifest_path) {
  pulsar::ExperimentConfig config;
  config.action_table.builtin = "rlgym_lookup_v1";
  config.model.observation_dim = 132;
  config.model.action_dim = 90;
  config.model.encoder_dim = 16;
  config.model.workspace_dim = 16;
  config.model.stm_slots = 4;
  config.model.stm_key_dim = 8;
  config.model.stm_value_dim = 8;
  config.model.ltm_slots = 4;
  config.model.ltm_dim = 8;
  config.model.controller_dim = 16;
  config.model.action_embedding_dim = 8;
  config.future_evaluator.horizons = {1, 2, 3};
  config.future_evaluator.latent_dim = 8;
  config.future_evaluator.model_dim = 16;
  config.future_evaluator.layers = 1;
  config.future_evaluator.heads = 4;
  config.future_evaluator.feedforward_dim = 32;
  config.future_evaluator.class_weights = {1.0F, 1.0F, 0.25F};
  config.lfpo.device = "cpu";
  config.lfpo.num_envs = 2;
  config.lfpo.collection_workers = 0;
  config.lfpo.rollout_length = 4;
  config.lfpo.minibatch_size = 8;
  config.lfpo.update_epochs = 1;
  config.lfpo.checkpoint_interval = 1;
  config.lfpo.sequence_length = 2;
  config.lfpo.burn_in = 0;
  config.lfpo.candidate_count = 4;
  config.lfpo.evaluator_update_interval = 1;
  config.lfpo.evaluator_target_update_interval = 1;
  config.lfpo.online_window_capacity = 8;
  config.offline_dataset.train_manifest = manifest_path.string();
  config.offline_dataset.val_manifest = manifest_path.string();
  config.offline_dataset.batch_size = 16;
  config.offline_pretraining.evaluator_epochs = 1;
  config.offline_pretraining.actor_epochs = 1;
  config.offline_pretraining.sequence_length = 8;
  config.env.seed = 5;
  config.model.future_latent_dim = config.future_evaluator.latent_dim;
  config.model.future_horizon_count = static_cast<int>(config.future_evaluator.horizons.size());
  return config;
}

void write_manifest_fixture(const std::filesystem::path& root) {
  const std::int64_t rows = 64;
  const std::int64_t obs_dim = 132;
  const std::int64_t action_dim = 90;
  torch::Tensor obs = torch::randn({rows, obs_dim});
  torch::Tensor actions = torch::randint(action_dim, {rows}, torch::TensorOptions().dtype(torch::kLong));
  torch::Tensor action_probs = torch::one_hot(actions, action_dim).to(torch::kFloat32);
  torch::Tensor outcome = torch::cat({
      torch::zeros({rows / 2}, torch::TensorOptions().dtype(torch::kLong)),
      torch::ones({rows / 2}, torch::TensorOptions().dtype(torch::kLong)),
  });
  torch::Tensor outcome_known = torch::ones({rows}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor weights = torch::ones({rows});
  torch::Tensor episode_starts = torch::zeros({rows});
  torch::Tensor terminated = torch::zeros({rows});
  torch::Tensor truncated = torch::zeros({rows});
  episode_starts[0] = 1.0F;
  episode_starts[32] = 1.0F;
  terminated[31] = 1.0F;
  terminated[63] = 1.0F;

  std::filesystem::create_directories(root / "data");
  torch::save(obs, (root / "data" / "obs.pt").string());
  torch::save(actions, (root / "data" / "actions.pt").string());
  torch::save(action_probs, (root / "data" / "action_probs.pt").string());
  torch::save(outcome, (root / "data" / "outcome.pt").string());
  torch::save(outcome_known, (root / "data" / "outcome_known.pt").string());
  torch::save(weights, (root / "data" / "weights.pt").string());
  torch::save(episode_starts, (root / "data" / "episode_starts.pt").string());
  torch::save(terminated, (root / "data" / "terminated.pt").string());
  torch::save(truncated, (root / "data" / "truncated.pt").string());

  std::ofstream manifest(root / "data" / "manifest.json");
  manifest << R"({
  "schema_version": 4,
  "observation_dim": 132,
  "action_dim": 90,
  "outcome_classes": 3,
  "shards": [
    {
      "obs_path": "obs.pt",
      "actions_path": "actions.pt",
      "action_probs_path": "action_probs.pt",
      "outcome_path": "outcome.pt",
      "outcome_known_path": "outcome_known.pt",
      "weights_path": "weights.pt",
      "episode_starts_path": "episode_starts.pt",
      "terminated_path": "terminated.pt",
      "truncated_path": "truncated.pt",
      "samples": 64
    }
  ]
})";
}

std::unique_ptr<pulsar::BatchedRocketSimCollector> make_fake_collector(const pulsar::ExperimentConfig& config) {
  std::vector<pulsar::TransitionEnginePtr> engines;
  for (int env = 0; env < config.lfpo.num_envs; ++env) {
    pulsar::EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(env);
    engines.push_back(std::make_shared<FakeTransitionEngine>(env_config));
  }
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  return std::make_unique<pulsar::BatchedRocketSimCollector>(
      config,
      std::move(engines),
      obs_builder,
      action_parser,
      done_condition,
      false);
}

}  // namespace

int main() {
  try {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pulsar_lfpo_offline_test";
    fs::remove_all(root);
    write_manifest_fixture(root);

    pulsar::ExperimentConfig config = make_lfpo_smoke_config(root / "data" / "manifest.json");
    {
      pulsar::OfflinePretrainer pretrainer(config);
      pretrainer.train((root / "pretrain").string());
    }
    if (!fs::exists(root / "pretrain" / "model.pt") ||
        !fs::exists(root / "pretrain" / "future_evaluator" / "model.pt")) {
      throw std::runtime_error("LFPO pretraining checkpoint missing actor or future evaluator");
    }

    config.lfpo.init_checkpoint = (root / "pretrain").string();
    {
      pulsar::LFPOTrainer trainer(config, make_fake_collector(config), nullptr, root / "online");
      trainer.train(1, (root / "online").string());
    }
    if (!fs::exists(root / "online" / "update_1" / "model.pt") ||
        !fs::exists(root / "online" / "update_1" / "future_evaluator" / "model.pt") ||
        !fs::exists(root / "online" / "final" / "model.pt")) {
      throw std::runtime_error("LFPO online checkpoint missing actor or evaluator");
    }

    pulsar::ExperimentConfig resume_config = config;
    resume_config.lfpo.init_checkpoint = (root / "online" / "update_1").string();
    {
      pulsar::LFPOTrainer resumed(resume_config, make_fake_collector(resume_config), nullptr, root / "resume", false);
      resumed.train(1, (root / "resume").string());
    }
    if (!fs::exists(root / "resume" / "update_2" / "model.pt")) {
      throw std::runtime_error("LFPO resume checkpoint did not continue update numbering");
    }

    fs::remove_all(root);
    std::cout << "pulsar_offline_tests passed\n" << std::flush;
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_tests failed: " << exc.what() << '\n' << std::flush;
    std::_Exit(EXIT_FAILURE);
  }
}
