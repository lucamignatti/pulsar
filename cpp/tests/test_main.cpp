#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/reward.hpp"
#include "pulsar/rl/action_table.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

pulsar::ExperimentConfig make_config() {
  pulsar::ExperimentConfig config;
  config.reward.terms = {
      {"goal", 10.0F},
      {"touch", 0.25F},
      {"speed_to_ball", 0.01F},
      {"ball_to_goal", 0.1F},
      {"face_ball", 0.005F},
  };
  config.reward.team_spirit = 0.2F;
  config.reward.opponent_scale = 1.0F;
  config.action_table.builtin = "rlgym_lookup_v1";
  return config;
}

}  // namespace

int main() {
  try {
    const pulsar::ExperimentConfig config = make_config();

    const pulsar::ControllerActionTable action_table(config.action_table);
    require(action_table.size() == 90, "lookup action table size mismatch");
    require(!action_table.hash().empty(), "action table hash should not be empty");

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

    pulsar::EnvState next_state = state;
    next_state.blue_score = 1;
    next_state.goal_scored = true;
    next_state.last_touch_agent = 0;
    next_state.cars[0].velocity = {1000.0F, 0.0F, 0.0F};
    next_state.cars[0].forward = {1.0F, 0.0F, 0.0F};
    pulsar::CombinedRewardFunction reward_fn(config.reward);
    const auto rewards = reward_fn.get_rewards(state, next_state, {}, {});
    require(rewards.size() == 4, "reward vector size mismatch");
    require(rewards[0] > 0.0F, "scoring agent should receive positive reward");
    require(rewards[2] < rewards[0], "opponents should receive less reward under zero-sum shaping");

    pulsar::SimpleDoneCondition done(config.env);
    const auto [terminated, truncated] = done.is_done(next_state, config.env.max_episode_ticks);
    require(terminated[0] == 1, "goal should terminate the episode");
    require(truncated[0] == 1, "max episode ticks should truncate the episode");

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

    std::cout << "pulsar_core_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_core_tests failed: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
