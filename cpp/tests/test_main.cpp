#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/rl/action_table.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

pulsar::ExperimentConfig make_config() {
  pulsar::ExperimentConfig config;
  config.action_table.builtin = "rlgym_lookup_v1";
  return config;
}

}  // namespace

int main() {
  try {
    pulsar::ExperimentConfig config = make_config();

    const pulsar::ControllerActionTable action_table(config.action_table);
    require(action_table.size() == 90, "lookup action table size mismatch");
    require(!action_table.hash().empty(), "action table hash should not be empty");
    const pulsar::DiscreteActionParser parser(action_table);

    pulsar::EnvState state;
    pulsar::FixedTeamSizeMutator fixed(config.env);
    pulsar::KickoffMutator kickoff(config.env);
    fixed.apply(state, 7);
    kickoff.apply(state, 7);
    require(state.cars.size() == 4, "2v2 mutator should produce four cars");

    pulsar::PulsarObsBuilder obs_builder(config.env);
    const auto obs = obs_builder.build_obs(state, 0);
    require(obs_builder.obs_dim() == 132, "default obs dimension mismatch");
    require(obs.size() == obs_builder.obs_dim(), "observation dimension mismatch");

    pulsar::SimpleDoneCondition done(config.env);
    pulsar::EnvState next_state = state;
    next_state.blue_score = 1;
    next_state.goal_scored = true;
    const auto [terminated, truncated] = done.is_done(next_state, config.env.max_episode_ticks);
    require(terminated[0] == 1, "goal should terminate the episode");
    require(truncated[0] == 1, "max episode ticks should truncate the episode");

    state.cars.resize(1);
    state.cars[0].id = 0;
    state.cars[0].boost = 0.0F;
    state.cars[0].on_ground = true;
    state.cars[0].has_flip = true;
    std::vector<std::uint8_t> no_boost_mask(action_table.size(), 0);
    parser.build_action_mask_batch(state, no_boost_mask);
    bool saw_masked_boost = false;
    for (std::size_t i = 0; i < action_table.size(); ++i) {
      if (action_table.at(i).boost) {
        saw_masked_boost = saw_masked_boost || (no_boost_mask[i] == 0);
      }
    }
    require(saw_masked_boost, "boost actions should be masked when boost is unavailable");

    state.cars[0].boost = 33.0F;
    state.cars[0].on_ground = true;
    state.cars[0].has_flip = true;
    std::vector<std::uint8_t> grounded_mask(action_table.size(), 0);
    parser.build_action_mask_batch(state, grounded_mask);
    bool saw_ground_jump = false;
    for (std::size_t i = 0; i < action_table.size(); ++i) {
      if (action_table.at(i).jump && grounded_mask[i] != 0) {
        saw_ground_jump = true;
        break;
      }
    }
    require(saw_ground_jump, "jump actions should remain valid on the ground");

    state.cars[0].on_ground = false;
    state.cars[0].has_flip = true;
    std::vector<std::uint8_t> flip_mask(action_table.size(), 0);
    parser.build_action_mask_batch(state, flip_mask);
    bool saw_air_jump = false;
    for (std::size_t i = 0; i < action_table.size(); ++i) {
      if (action_table.at(i).jump && flip_mask[i] != 0) {
        saw_air_jump = true;
        break;
      }
    }
    require(saw_air_jump, "jump actions should remain valid while a flip is available");

    state.cars[0].has_flip = false;
    std::vector<std::uint8_t> no_flip_mask(action_table.size(), 0);
    parser.build_action_mask_batch(state, no_flip_mask);
    bool saw_masked_jump = false;
    for (std::size_t i = 0; i < action_table.size(); ++i) {
      if (action_table.at(i).jump) {
        saw_masked_jump = saw_masked_jump || (no_flip_mask[i] == 0);
      }
    }
    require(saw_masked_jump, "jump actions should be masked when airborne without a flip");

    const auto tmp_dir = std::filesystem::temp_directory_path() / "pulsar_test_metadata.json";
    const pulsar::CheckpointMetadata metadata{
        .schema_version = config.schema_version,
        .obs_schema_version = config.obs_schema_version,
        .config_hash = pulsar::config_hash(config),
        .action_table_hash = pulsar::action_table_hash(config.action_table),
        .architecture_name = "shared_actor_critic",
        .device = "cpu",
        .global_step = 12,
        .update_index = 3,
    };
    pulsar::save_checkpoint_metadata(metadata, tmp_dir.string());
    const auto loaded = pulsar::load_checkpoint_metadata(tmp_dir.string());
    pulsar::validate_checkpoint_metadata(loaded, config);
    std::filesystem::remove(tmp_dir);

    config.ppo.self_play.enabled = true;
    config.ppo.self_play.opponent_probability = 0.5F;
    const auto config_path = std::filesystem::temp_directory_path() / "pulsar_test_config.json";
    pulsar::save_experiment_config(config, config_path.string());
    const auto roundtripped = pulsar::load_experiment_config(config_path.string());
    require(roundtripped.ppo.self_play.enabled, "self-play config should round-trip");
    std::filesystem::remove(config_path);

    std::cout << "pulsar_core_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_core_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
