#include <cstdlib>
#include <cmath>
#include <exception>
#include <iostream>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "test_utils.hpp"

int main() {
  try {
    // Test 1: default construction and validation
    pulsar::ExperimentConfig config = pulsar::test::make_test_config();
    config.ppo.rollout_length = 128;
    config.ppo.minibatch_size = 256;
    config.model.encoder_dim = 512;
    config.model.num_encoder_blocks = 2;
    config.model.sequence_length = 4;
    config.model.max_forward_samples = 1024;
    pulsar::test::require(config.ppo.overlap_collection_update, "default trainer should overlap collection/update");
    pulsar::validate_experiment_config(config);

    // Test 2: JSON round-trip
    auto tmp = std::filesystem::temp_directory_path() / "config_test.json";
    pulsar::save_experiment_config(config, tmp.string());
    auto loaded = pulsar::load_experiment_config(tmp.string());
    pulsar::test::require(loaded.ppo.rollout_length == 128, "round-trip rollout_length");
    pulsar::test::require(loaded.ppo.minibatch_size == 256, "round-trip minibatch");
    pulsar::test::require(loaded.model.encoder_dim == 512, "round-trip encoder_dim");
    std::filesystem::remove(tmp);

    // Test 3: config hash is deterministic
    auto hash1 = pulsar::config_hash(config);
    auto hash2 = pulsar::config_hash(config);
    pulsar::test::require(hash1 == hash2, "hash should be deterministic");
    pulsar::test::require(!hash1.empty(), "hash should not be empty");

    // Test 4: reject invalid rollout_length
    pulsar::ExperimentConfig bad = config;
    bad.ppo.rollout_length = 1;
    bool caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "rollout_length <= 1 should throw");

    // Test 5: reject negative learning_rate
    bad = config;
    bad.ppo.learning_rate = -0.001f;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "negative lr should throw");

    // Test 7: reject zero num_envs
    bad = config;
    bad.ppo.num_envs = 0;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "zero num_envs should throw");

    // Test 8: reject zero minibatch_size
    bad = config;
    bad.ppo.minibatch_size = 0;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "zero minibatch should throw");

    // Test 9: reject invalid team_size
    bad = config;
    bad.env.team_size = 0;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "team_size=0 should throw");
    bad.env.team_size = 5;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "team_size=5 should throw");

    // Test 10: reject negative clip_range
    bad = config;
    bad.ppo.clip_range = -0.1f;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "negative clip_range should throw");

    // Test 11: reject zero max_grad_norm
    bad = config;
    bad.ppo.max_grad_norm = 0.0f;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "zero max_grad_norm should throw");

    // Test 12: reject negative value_loss_delta
    bad = config;
    bad.ppo.value_loss_delta = -1.0f;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "negative value_loss_delta should throw");

    // Test 13: reject invalid es_lora.population_size when antithetic
    bad = config;
    bad.es_lora.population_size = 7;
    bad.es_lora.antithetic_sampling = true;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "odd population_size with antithetic should throw");

    // Test 14: reject collection_shards > num_envs
    bad = config;
    bad.ppo.num_envs = 4;
    bad.ppo.collection_shards = 8;
    caught = false;
    try { pulsar::validate_experiment_config(bad); }
    catch (const std::invalid_argument&) { caught = true; }
    pulsar::test::require(caught, "collection_shards > num_envs should throw");

    // Test 14: load production config
    auto config_path = pulsar::test::find_repo_path("configs") / "2v2_appo.json";
    if (std::filesystem::exists(config_path)) {
      auto prod = pulsar::load_experiment_config(config_path.string());
      pulsar::validate_experiment_config(prod);
      pulsar::test::require(prod.schema_version == 6, "production config schema");
      pulsar::test::require(prod.curriculum.enabled, "curriculum enabled in production");
      pulsar::test::require(prod.curriculum.stages.size() == 3, "three curriculum stages in production");
      pulsar::test::require(prod.curriculum.stages[0].mode_allocation.size() == 3,
                            "production config starts with all ranked team sizes");
      pulsar::test::require(prod.self_play_league.enabled, "self_play_league enabled in production");
      pulsar::test::require(prod.self_play_league.max_snapshots == 4, "max_snapshots in production");
      pulsar::test::require(prod.ppo.overlap_collection_update,
                            "production config should overlap collection/update by default");
    }

    // Test 15: stable_json is deterministic
    auto json1 = pulsar::stable_json(config);
    auto json2 = pulsar::stable_json(config);
    pulsar::test::require(json1 == json2, "stable_json should be deterministic");

    // Test 16: action_table_hash
    auto ath = pulsar::action_table_hash(config.action_table);
    pulsar::test::require(!ath.empty(), "action_table_hash should not be empty");

    // Test 17: CheckpointMetadata round-trip
    {
      pulsar::CheckpointMetadata meta;
      meta.schema_version = 6;
      meta.obs_schema_version = 2;
      meta.config_hash = "abc123";
      meta.action_table_hash = "def456";
      meta.architecture_name = "mamba2_goal_appo";
      meta.device = "cpu";
      meta.global_step = 1000;
      meta.update_index = 50;
      meta.critic_heads = {"extrinsic"};

      auto meta_path = std::filesystem::temp_directory_path() / "meta_test.json";
      pulsar::save_checkpoint_metadata(meta, meta_path.string());
      auto loaded_meta = pulsar::load_checkpoint_metadata(meta_path.string());
      pulsar::test::require(loaded_meta.schema_version == 6, "metadata schema round-trip");
      pulsar::test::require(loaded_meta.config_hash == "abc123", "metadata config_hash round-trip");
      pulsar::test::require(loaded_meta.global_step == 1000, "metadata global_step round-trip");
      std::filesystem::remove(meta_path);
    }

    std::cout << "test_config passed\n" << std::flush;
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "test_config FAILED: " << exc.what() << '\n' << std::flush;
    return EXIT_FAILURE;
  }
}
