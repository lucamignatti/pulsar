#include <cstdlib>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/self_play_manager.hpp"
#include "test_utils.hpp"

namespace {

// Safely remove a directory tree without relying on std::filesystem::remove_all,
// which resolves to a broken implementation in libtorch's bundled libstdc++ on some systems.
void safe_remove_all(const std::filesystem::path& path) {
  // Avoid std::filesystem::remove_all entirely because both the throwing
  // and error_code overloads jump through a null function pointer inside
  // libtorch's bundled libstdc++. Use POSIX rm -rf directly.
  std::system(("rm -rf " + path.string() + " 2>/dev/null").c_str());
}

void test_snapshot_save_load_trim_and_assignment() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 1;
  config.self_play_league.eval_interval_updates = 0;
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_self_play_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
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
  safe_remove_all(root);
}

void test_opponent_inference_and_elo_math() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.team_size = 1;
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 2;
  config.self_play_league.eval_interval_updates = 0;
  config.self_play_league.training_opponent_policy = "stochastic";
  config.ppo.device = "cpu";

  const auto root = std::filesystem::temp_directory_path() / "pulsar_self_play_infer";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
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
    torch::Tensor actions;
    double inference_seconds = 0.0;
    auto encoded = model->forward_step(raw_obs).encoded;
    manager.infer_opponent_actions(
        model,
        raw_obs,
        action_masks,
        episode_starts,
        snapshot_ids,
        encoded,
        &actions,
        &inference_seconds);
    pulsar::test::require(actions[0].item<std::int64_t>() == 0, "non-opponent slot should remain untouched");
    pulsar::test::require(actions[1].item<std::int64_t>() == 7, "opponent slot should obey mask");
    pulsar::test::require(actions[2].item<std::int64_t>() == 7, "opponent slot should obey mask");
    pulsar::test::require(!actions.requires_grad(), "opponent actions should be inference-only");
    pulsar::test::require(inference_seconds >= 0.0, "inference timing should be recorded");

    const double rating_before = manager.current_ratings().at("1v1");
    manager.record_live_outcomes("1v1", {pulsar::SelfPlayLiveOutcome{
        .snapshot_index = 0,
        .learner_result = 1,
    }});
    pulsar::test::require(
        manager.current_ratings().at("1v1") > rating_before,
        "live self-play win should increase current ELO");
  }
  safe_remove_all(root);

  double winner = 1000.0;
  double loser = 1000.0;
  pulsar::update_elo_ratings(winner, loser, 32.0);
  pulsar::test::require(winner > 1000.0, "winner rating should increase");
  pulsar::test::require(loser < 1000.0, "loser rating should decrease");
}

void test_elo_math_detailed() {
  // Equal players draw: ratings should not change
  double a = 1200.0;
  double b = 1200.0;
  pulsar::update_elo_ratings(a, b, 32.0);
  pulsar::test::require(a > b, "winner A should increase over loser B");
  double diff_a = a - 1200.0;

  // Re-run with same ratings: result should be deterministic
  double a2 = 1200.0;
  double b2 = 1200.0;
  pulsar::update_elo_ratings(a2, b2, 32.0);
  pulsar::test::require(std::abs(a2 - a) < 0.001, "ELO should be deterministic");
  pulsar::test::require(std::abs(b2 - b) < 0.001, "ELO should be deterministic");

  // Higher K factor produces larger rating changes
  double a_high_k = 1200.0;
  double b_high_k = 1200.0;
  pulsar::update_elo_ratings(a_high_k, b_high_k, 64.0);
  pulsar::test::require((a_high_k - 1200.0) > diff_a, "higher K should produce larger rating change");

  // Stronger player beating weaker player: smaller rating change
  double strong = 1600.0;
  double weak = 800.0;
  pulsar::update_elo_ratings(strong, weak, 32.0);
  double strong_gain = strong - 1600.0;
  pulsar::test::require(strong_gain > 0.0, "strong winner should still gain rating");
  pulsar::test::require(strong_gain < diff_a, "strong beating weak should gain less than equal match");

  // Underdog beating favorite: larger rating change
  double underdog = 800.0;
  double favorite = 1600.0;
  pulsar::update_elo_ratings(underdog, favorite, 32.0);
  double underdog_gain = underdog - 800.0;
  pulsar::test::require(underdog_gain > strong_gain, "underdog beating favorite should gain more");

  // Sum of ratings is conserved (zero-sum)
  double a_sum = 1000.0;
  double b_sum = 1500.0;
  double sum_before = a_sum + b_sum;
  pulsar::update_elo_ratings(a_sum, b_sum, 32.0);
  double sum_after = a_sum + b_sum;
  pulsar::test::require(std::abs(sum_after - sum_before) < 0.001, "ELO should conserve total rating sum");

  // K=0 produces no change
  double a_k0 = 1000.0;
  double b_k0 = 1000.0;
  pulsar::update_elo_ratings(a_k0, b_k0, 0.0);
  pulsar::test::require(std::abs(a_k0 - 1000.0) < 0.001, "K=0 should not change winner rating");
  pulsar::test::require(std::abs(b_k0 - 1000.0) < 0.001, "K=0 should not change loser rating");
}

void test_snapshot_reload_preserves_actor_config() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 0.5F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 3;
  config.self_play_league.eval_interval_updates = 0;
  config.ppo.device = "cpu";

  config.goal_critic.embedding_dim = 64;
  config.goal_critic.hidden_dim = 256;

  const fs::path root = fs::temp_directory_path() / "pulsar_self_play_actor_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({16, config.model.observation_dim}));

  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    manager.on_update(model, normalizer, 10, 1);
    pulsar::test::require(fs::exists(root / "10" / "model.pt"), "snapshot should be saved");
  }

  {
    pulsar::SelfPlayManager reloaded(config, root, obs_builder, action_parser, torch::kCPU);
    pulsar::test::require(reloaded.has_snapshots(), "reloaded manager should see snapshots");

    const auto metadata = pulsar::load_checkpoint_metadata(
        (root / "10" / "metadata.json").string());
    auto enabled_heads = metadata.critic_heads;
    bool has_extrinsic = false;
    for (const auto& head : enabled_heads) {
      if (head == "extrinsic") has_extrinsic = true;
    }
    pulsar::test::require(has_extrinsic, "snapshot metadata must list extrinsic head");

    pulsar::PPOActor snapshot_model = pulsar::load_ppo_actor(
        (root / "10").string(), "cpu");
    
    auto output = snapshot_model->forward_step(
        torch::randn({2, config.model.observation_dim}));
    pulsar::test::require(
        output.value_win_logits.sizes() ==
            torch::IntArrayRef({2, 1}),
        "snapshot value win head shape matches model config");
    pulsar::test::require(
        snapshot_model->goal_critic()->goal_embedding(torch::zeros({1, config.goal_critic.goal_dim})).size(1) == config.goal_critic.embedding_dim,
        "snapshot goal critic should have matching embedding dim");
  }

  safe_remove_all(root);
}

void test_self_play_eval_is_skipped_and_loaded_snapshots_are_bounded() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.env.max_episode_ticks = 16;
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 2;
  config.self_play_league.eval_interval_updates = 1;
  config.self_play_league.eval_num_envs = 1;
  config.self_play_league.eval_matches_per_snapshot = 1;
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_self_play_eval_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({16, config.model.observation_dim}));

  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    manager.on_update(model, normalizer, 10, 1);
    manager.on_update(model, normalizer, 20, 2);
    manager.on_update(model, normalizer, 30, 3);
  }

  {
    pulsar::SelfPlayManager reloaded(config, root, obs_builder, action_parser, torch::kCPU);
    const auto metrics = reloaded.on_update(model, normalizer, 40, 4);
    pulsar::test::require(metrics.snapshot_count == 2, "loaded snapshot pool should honor max_snapshots");
    pulsar::test::require(metrics.eval_seconds == 0.0, "scheduled self-play evaluation should be skipped");
    pulsar::test::require(!metrics.ratings.empty(), "self-play update should report live ELO ratings");
  }

  safe_remove_all(root);
}

void test_checkpoint_metadata_validation() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "pulsar_metadata_validation_test";
  safe_remove_all(root);
  fs::create_directories(root);

  {
    auto config = pulsar::test::make_test_config();
    pulsar::CheckpointMetadata meta;
    meta.schema_version = config.schema_version;
    meta.obs_schema_version = config.obs_schema_version;
    meta.config_hash = pulsar::config_hash(config);
    meta.action_table_hash = pulsar::action_table_hash(config.action_table);
    meta.architecture_name = "mamba2_goal_appo";
    meta.device = "cpu";
    meta.global_step = 0;
    meta.update_index = 0;
    meta.critic_heads = {"extrinsic"};
    pulsar::save_checkpoint_metadata(meta, (root / "valid.json").string());
  }

  {
    auto config = pulsar::test::make_test_config();
    auto meta = pulsar::load_checkpoint_metadata((root / "valid.json").string());
    meta.schema_version = 999;
    bool threw = false;
    try {
      pulsar::validate_inference_checkpoint_metadata(meta, config);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    pulsar::test::require(threw, "schema_version mismatch should throw");
  }

  {
    auto config = pulsar::test::make_test_config();
    auto meta = pulsar::load_checkpoint_metadata((root / "valid.json").string());
    meta.obs_schema_version = 999;
    bool threw = false;
    try {
      pulsar::validate_inference_checkpoint_metadata(meta, config);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    pulsar::test::require(threw, "obs_schema_version mismatch should throw");
  }

  {
    auto config = pulsar::test::make_test_config();
    auto meta = pulsar::load_checkpoint_metadata((root / "valid.json").string());
    meta.action_table_hash = "deadbeef";
    bool threw = false;
    try {
      pulsar::validate_inference_checkpoint_metadata(meta, config);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    pulsar::test::require(!threw, "action_table_hash mismatch should not throw");
  }

  {
    auto config = pulsar::test::make_test_config();
    auto meta = pulsar::load_checkpoint_metadata((root / "valid.json").string());
    meta.critic_heads = {"unknown_head"};
    pulsar::test::require(
        meta.critic_heads.size() == 1,
        "metadata should preserve loaded critic_heads");
  }

  {
    auto config = pulsar::test::make_test_config();
    auto meta = pulsar::load_checkpoint_metadata((root / "valid.json").string());
    meta.critic_heads = {};
    bool threw = false;
    try {
      pulsar::validate_inference_checkpoint_metadata(meta, config);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    pulsar::test::require(threw, "empty critic heads should throw");
  }

  safe_remove_all(root);
}

void test_self_play_league_default_config() {
  pulsar::SelfPlayLeagueConfig league;
  pulsar::test::require(!league.enabled, "default self_play_league should be disabled");
  pulsar::test::require(league.elo_initial == 1000.0F, "default elo_initial should be 1000");
  pulsar::test::require(league.elo_k == 32.0F, "default elo_k should be 32");
  pulsar::test::require(league.max_snapshots == 8, "default max_snapshots should be 8");
  pulsar::test::require(league.snapshot_interval_updates == 10, "default snapshot_interval_updates should be 10");
  pulsar::test::require(league.training_opponent_policy == "stochastic", "default training_opponent_policy");
  pulsar::test::require(league.eval_policy == "deterministic", "default eval_policy");
}

void test_rng_state_round_trip() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 0.5F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 1;
  config.self_play_league.eval_interval_updates = 0;
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_rng_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({16, config.model.observation_dim}));

  pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
  auto state = manager.rng_state();
  pulsar::test::require(!state.empty(), "rng_state should not be empty");

  // Sample before and after restore should match
  const auto assignment_before = manager.sample_assignment(0, 42);
  manager.restore_rng_state(state);
  const auto assignment_after = manager.sample_assignment(0, 42);
  pulsar::test::require(assignment_before.enabled == assignment_after.enabled,
                        "rng restore should reproduce same assignment");

  safe_remove_all(root);
}

void test_restore_ratings() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.self_play_league.enabled = true;
  config.self_play_league.eval_interval_updates = 0;
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_ratings_restore_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));

  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    std::map<std::string, double> ratings;
    ratings["1v1"] = 1100.0;
    ratings["2v2"] = 1050.0;
    manager.restore_ratings(ratings);
    const auto& restored = manager.current_ratings();
    pulsar::test::require(restored.at("1v1") == 1100.0, "1v1 rating should be 1100");
    pulsar::test::require(restored.at("2v2") == 1050.0, "2v2 rating should be 1050");
  }
  safe_remove_all(root);
}

void test_ratings_include_all_curriculum_modes() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.self_play_league.enabled = true;
  config.self_play_league.elo_initial = 987.0F;
  config.curriculum.enabled = true;
  pulsar::CurriculumStageConfig stage;
  stage.name = "mixed";
  stage.mode_allocation["1v1"] = 0.34F;
  stage.mode_allocation["2v2"] = 0.33F;
  stage.mode_allocation["3v3"] = 0.33F;
  config.curriculum.stages = {stage};
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_ratings_modes_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));

  pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
  const auto& ratings = manager.current_ratings();
  pulsar::test::require(ratings.at("1v1") == 987.0, "1v1 rating should be initialized");
  pulsar::test::require(ratings.at("2v2") == 987.0, "2v2 rating should be initialized");
  pulsar::test::require(ratings.at("3v3") == 987.0, "3v3 rating should be initialized");

  safe_remove_all(root);
}

void test_snapshot_mode_is_preserved() {
  namespace fs = std::filesystem;
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.self_play_league.enabled = true;
  config.self_play_league.opponent_probability = 1.0F;
  config.self_play_league.snapshot_interval_updates = 1;
  config.self_play_league.max_snapshots = 2;
  config.self_play_league.eval_interval_updates = 0;
  config.ppo.device = "cpu";

  const fs::path root = fs::temp_directory_path() / "pulsar_snapshot_mode_test";
  safe_remove_all(root);
  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  pulsar::PPOActor model(config.model, config.goal_critic);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  normalizer.update(torch::randn({16, config.model.observation_dim}));

  {
    pulsar::SelfPlayManager manager(config, root, obs_builder, action_parser, torch::kCPU);
    manager.set_current_mode("2v2");
    manager.on_update(model, normalizer, 10, 1);
  }

  {
    auto metadata = pulsar::load_checkpoint_metadata((root / "10" / "metadata.json").string());
    std::string mode = metadata.extra.value("mode", std::string{"1v1"});
    pulsar::test::require(mode == "2v2", "snapshot mode should be 2v2 after set_current_mode");
  }

  {
    pulsar::SelfPlayManager reloaded(config, root, obs_builder, action_parser, torch::kCPU);
    pulsar::test::require(reloaded.has_snapshots(), "reloaded manager should see snapshots");
    const auto assignment = reloaded.sample_assignment(0, 7);
    pulsar::test::require(assignment.enabled, "assignment should be enabled");
  }

  safe_remove_all(root);
}

void test_collection_applies_snapshot_assignments() {
  pulsar::ExperimentConfig config = pulsar::test::make_test_config();
  config.env.collision_meshes_path = pulsar::test::find_repo_collision_meshes().string();
  config.env.team_size = 1;
  config.ppo.num_envs = 2;
  config.ppo.collection_workers = 0;

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      obs_builder,
      action_parser,
      done_condition,
      false);

  collector.set_self_play_assignment_fn([](std::size_t, std::uint64_t) {
    return pulsar::SelfPlayAssignment{
        .enabled = true,
        .learner_team = pulsar::Team::Blue,
        .snapshot_index = 42,
    };
  });

  const auto require_split_self_play = [&collector](const std::string& context) {
    const torch::Tensor learner_active = collector.host_learner_active();
    const torch::Tensor snapshot_ids = collector.host_snapshot_ids();
    pulsar::test::require(
        learner_active.sum().item<float>() == static_cast<float>(learner_active.numel() / 2),
        context + ": one team should remain learner-active");
    pulsar::test::require(
        ((snapshot_ids == 42).sum().item<int64_t>() == snapshot_ids.numel() / 2),
        context + ": snapshot ids should be assigned to the opponent team");
    pulsar::test::require(
        ((snapshot_ids == -1).sum().item<int64_t>() == snapshot_ids.numel() / 2),
        context + ": learner team should not have snapshot ids");
  };

  require_split_self_play("after assignment hook");

  std::vector<std::int64_t> actions(collector.total_agents(), 0);
  collector.step(std::span<const std::int64_t>(actions.data(), actions.size()));
  require_split_self_play("after environment step");
}

}  // namespace

int main() {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    test_elo_math_detailed();
    test_self_play_league_default_config();
    test_snapshot_save_load_trim_and_assignment();
    test_opponent_inference_and_elo_math();
    test_snapshot_reload_preserves_actor_config();
    test_self_play_eval_is_skipped_and_loaded_snapshots_are_bounded();
    test_checkpoint_metadata_validation();
    test_rng_state_round_trip();
    test_restore_ratings();
    test_ratings_include_all_curriculum_modes();
    test_snapshot_mode_is_preserved();
    test_collection_applies_snapshot_assignments();
    std::cout << "pulsar_self_play_tests passed\n" << std::flush;
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_self_play_tests failed: " << exc.what() << '\n' << std::flush;
    std::_Exit(EXIT_FAILURE);
  }
}
