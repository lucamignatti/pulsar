#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>

#include "pulsar/env/obs_builder.hpp"
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/self_play_manager.hpp"
#include "test_utils.hpp"

namespace {

void test_snapshot_save_load_trim_and_assignment() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 1;
  config.self_play_league.eval_interval_updates = 0;
  config.lfpo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_self_play_test";
  fs::remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::LatentFutureActor model(config.model);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({16, config.model.observation_dim}));

  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    pulsar::test::require(!manager.has_snapshots(), "snapshot pool should start empty");
    auto metrics = manager.on_update(model, normalizer, 10, 1);
    pulsar::test::require(metrics.snapshot_count == 1, "first snapshot should be created");
    pulsar::test::require(fs::exists(root / "10" / "model.pt"), "snapshot model should be written");

    auto second_metrics = manager.on_update(model, normalizer, 20, 2);
    pulsar::test::require(second_metrics.snapshot_count == 1, "snapshot trim should keep one snapshot");
    pulsar::test::require(!fs::exists(root / "10"), "oldest snapshot should be trimmed");
    pulsar::test::require(fs::exists(root / "20" / "ratings.json"), "ratings should be written");
  }

  {
    pulsar::SelfPlayManager reloaded(config, root, obs_builder, action_parser, torch::kCPU);
    pulsar::test::require(reloaded.has_snapshots(), "reloaded manager should see snapshots");
    const auto assignment = reloaded.sample_assignment(0, 7);
    pulsar::test::require(assignment.enabled, "assignment should be enabled with opponent_probability=1");
  }

  pulsar::ExperimentConfig league_changed_config = config;
  league_changed_config.self_play_league.opponent_probability = 0.75F;
  {
    pulsar::SelfPlayManager league_changed_reloaded(
        league_changed_config,
        root,
        obs_builder,
        action_parser,
        torch::kCPU);
    pulsar::test::require(
        league_changed_reloaded.has_snapshots(),
        "self-play snapshot load should tolerate league sampling config changes");
  }
  fs::remove_all(root);
}

void test_opponent_inference_and_elo_math() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 2;
  config.self_play_league.eval_interval_updates = 0;
  config.self_play_league.training_opponent_policy = "stochastic";
  config.lfpo.device = "cpu";

  const auto root = std::filesystem::temp_directory_path() / "pulsar_self_play_infer";
  std::filesystem::remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::LatentFutureActor model(config.model);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({8, config.model.observation_dim}));
  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    manager.on_update(model, normalizer, 10, 1);

    torch::Tensor raw_obs = torch::randn({4, config.model.observation_dim});
    torch::Tensor action_masks = torch::zeros({4, config.model.action_dim}, torch::kBool);
    action_masks.index_put_({1, 7}, true);
    action_masks.index_put_({2, 7}, true);
    action_masks.index_put_({0, 0}, true);
    action_masks.index_put_({3, 0}, true);
    torch::Tensor episode_starts = torch::zeros({4});
    torch::Tensor snapshot_ids = torch::tensor({-1, 0, 0, -1}, torch::kLong);
    auto opponent_state = model->initial_state(4, torch::kCPU);
    torch::Tensor actions;
    double inference_seconds = 0.0;
    manager.infer_opponent_actions(
        model,
        raw_obs,
        action_masks,
        episode_starts,
        snapshot_ids,
        opponent_state,
        &actions,
        &inference_seconds);
    pulsar::test::require(actions[0].item<std::int64_t>() == 0, "non-opponent slot should remain untouched");
    pulsar::test::require(actions[1].item<std::int64_t>() == 7, "opponent slot should obey mask");
    pulsar::test::require(actions[2].item<std::int64_t>() == 7, "opponent slot should obey mask");
    pulsar::test::require(!actions.requires_grad(), "opponent actions should be inference-only");
    pulsar::test::require(!opponent_state.workspace.requires_grad(), "opponent workspace state should be detached");
    pulsar::test::require(!opponent_state.stm_keys.requires_grad(), "opponent STM key state should be detached");
    pulsar::test::require(!opponent_state.stm_values.requires_grad(), "opponent STM value state should be detached");
    pulsar::test::require(inference_seconds >= 0.0, "inference timing should be recorded");
  }
  std::filesystem::remove_all(root);

  double winner = 1000.0;
  double loser = 1000.0;
  pulsar::update_elo_ratings(winner, loser, 32.0);
  pulsar::test::require(winner > 1000.0, "winner rating should increase");
  pulsar::test::require(loser < 1000.0, "loser rating should decrease");
}

}  // namespace

int main() {
  try {
    test_snapshot_save_load_trim_and_assignment();
    test_opponent_inference_and_elo_math();
    std::cout << "pulsar_self_play_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_self_play_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
