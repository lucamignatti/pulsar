#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include "pulsar/training/rollout_storage.hpp"

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

}  // namespace

int main() {
  try {
    const int rollout_len = 4;
    const int num_agents = 2;
    const int obs_dim = 8;
    const int action_dim = 3;

    // --- Construction with a single head name ---
    std::vector<std::string> head_names = {"extrinsic"};
    pulsar::RolloutStorage storage(rollout_len, num_agents, obs_dim, action_dim, 8,
                                   torch::kCPU, head_names);

    require(storage.capacity() == rollout_len, "capacity");
    require(storage.num_agents() == num_agents, "num agents");
    require(storage.rollout_length() == 0, "rollout length initially zero");

    // --- Append data ---
    auto obs = torch::ones({num_agents, obs_dim});
    auto ep_starts = torch::zeros({num_agents}, torch::kBool);
    auto action_masks = torch::ones({num_agents, action_dim}, torch::kUInt8);
    auto learner_active = torch::ones({num_agents});
    auto actions = torch::zeros({num_agents}, torch::kInt64);
    auto action_log_probs = torch::zeros({num_agents});
    std::unordered_map<std::string, torch::Tensor> values;
    values["extrinsic"] = torch::zeros({num_agents});
    std::unordered_map<std::string, torch::Tensor> rewards;
    rewards["extrinsic"] = torch::ones({num_agents});
    auto dones = torch::zeros({num_agents});
    auto truncated = torch::zeros({num_agents});
    auto bootstrap = torch::zeros({num_agents});
    auto goal_pos = torch::zeros({num_agents, 3});
    auto labels = torch::zeros({num_agents}, torch::kInt64);
    auto term_obs = torch::zeros({num_agents, obs_dim});

    for (int step = 0; step < rollout_len; step++) {
      storage.append(step, obs, ep_starts, action_masks, learner_active,
                     actions, action_log_probs, values, rewards,
                     dones, truncated, bootstrap, goal_pos, labels, term_obs);
    }

    require(storage.rollout_length() == rollout_len, "rollout length after append");

    // --- Retrieve stored observations ---
    auto stored_obs = storage.obs;
    require(stored_obs.sizes() == torch::IntArrayRef({rollout_len, num_agents, obs_dim}),
            "obs shape");
    require(stored_obs.sum().item<float>() > 0.0f, "obs non-zero");

    // --- Retrieve stored actions ---
    auto stored_actions = storage.actions;
    require(stored_actions.sizes() == torch::IntArrayRef({rollout_len, num_agents}),
            "actions shape");
    require(stored_actions.dtype() == torch::kInt64, "actions dtype");

    // --- Retrieve stored rewards ---
    auto retrieved_reward = storage.reward("extrinsic");
    require(retrieved_reward.sizes() == torch::IntArrayRef({rollout_len, num_agents}),
            "reward shape");
    require(retrieved_reward.sum().item<float>() > 0.0f, "rewards non-zero");

    // --- Retrieve stored values ---
    auto retrieved_value = storage.value("extrinsic");
    require(retrieved_value.sizes() == torch::IntArrayRef({rollout_len, num_agents}),
            "value shape");

    // --- all_values / all_rewards ---
    const auto& all_vals = storage.all_values();
    require(all_vals.size() == 1, "one value head");
    require(all_vals.at("extrinsic").sizes() == torch::IntArrayRef({rollout_len, num_agents}),
            "all_values extrinsic shape");

    const auto& all_rewards = storage.all_rewards();
    require(all_rewards.size() == 1, "one reward stream");
    require(all_rewards.at("extrinsic").sizes() == torch::IntArrayRef({rollout_len, num_agents}),
            "all_rewards extrinsic shape");

    // --- Final value setting ---
    std::unordered_map<std::string, torch::Tensor> final_vals;
    final_vals["extrinsic"] = torch::full({num_agents}, 5.0f);
    storage.set_final_values(final_vals);
    auto fv = storage.final_values();
    require(fv.size() == 1, "final values count");
    require(fv.at("extrinsic").sum().item<float>() == 10.0f, "final values sum");

    // --- set_rewards_at ---
    const int overwrite_step = 0;
    std::unordered_map<std::string, torch::Tensor> new_rewards;
    new_rewards["extrinsic"] = torch::full({num_agents}, 7.0f);
    storage.set_rewards_at(overwrite_step, new_rewards);
    auto updated_reward = storage.reward("extrinsic");
    require(updated_reward[overwrite_step].sum().item<float>() == 14.0f,
            "set_rewards_at step 0");

    // --- Clearing ---
    storage.clear();
    require(storage.rollout_length() == 0, "rollout length zero after clear");
    require(storage.final_values().empty(), "final values empty after clear");

    // --- append_slice (offset-based) ---
    const int total_agents = 6;
    const int slice_offset = 0;
    pulsar::RolloutStorage slice_storage(rollout_len, total_agents, obs_dim, action_dim, 8,
                                         torch::kCPU, head_names);

    auto slice_obs = torch::ones({total_agents, obs_dim});
    auto slice_ep_starts = torch::zeros({total_agents}, torch::kBool);
    auto slice_masks = torch::ones({total_agents, action_dim}, torch::kUInt8);
    auto slice_active = torch::ones({total_agents});
    auto slice_acts = torch::zeros({total_agents}, torch::kInt64);
    auto slice_lps = torch::zeros({total_agents});
    std::unordered_map<std::string, torch::Tensor> slice_vals;
    slice_vals["extrinsic"] = torch::zeros({total_agents});
    std::unordered_map<std::string, torch::Tensor> slice_rewards;
    slice_rewards["extrinsic"] = torch::ones({total_agents});
    auto slice_dones = torch::zeros({total_agents});
    auto slice_truncated = torch::zeros({total_agents});
    auto slice_bootstrap = torch::zeros({total_agents});
    auto slice_goal_pos = torch::zeros({total_agents, 3});
    auto slice_labels = torch::zeros({total_agents}, torch::kInt64);
    auto slice_term_obs = torch::zeros({total_agents, obs_dim});

    for (int step = 0; step < rollout_len; step++) {
      slice_storage.append_slice(step, slice_offset,
                                 slice_obs, slice_ep_starts, slice_masks, slice_active,
                                 slice_acts, slice_lps, slice_vals, slice_rewards,
                                 slice_dones, slice_truncated, slice_bootstrap,
                                 slice_goal_pos, slice_labels, slice_term_obs);
      slice_storage.mark_step_filled(step);
    }

    auto slice_retrieved = slice_storage.reward("extrinsic");
    require(slice_retrieved.sizes() == torch::IntArrayRef({rollout_len, total_agents}),
            "slice reward shape");
    require(slice_retrieved.sum().item<float>() == 24.0f, "slice reward total");

    // --- append_slice with non-zero offset ---
    const int partial_agents = 3;
    const int partial_offset = 2;
    pulsar::RolloutStorage partial_storage(rollout_len, total_agents, obs_dim, action_dim, 8,
                                           torch::kCPU, head_names);

    auto partial_obs = torch::full({partial_agents, obs_dim}, 2.0f);
    auto partial_ep_starts = torch::zeros({partial_agents}, torch::kBool);
    auto partial_masks = torch::ones({partial_agents, action_dim}, torch::kUInt8);
    auto partial_active = torch::ones({partial_agents});
    auto partial_acts = torch::zeros({partial_agents}, torch::kInt64);
    auto partial_lps = torch::zeros({partial_agents});
    std::unordered_map<std::string, torch::Tensor> partial_vals;
    partial_vals["extrinsic"] = torch::zeros({partial_agents});
    std::unordered_map<std::string, torch::Tensor> partial_rewards;
    partial_rewards["extrinsic"] = torch::full({partial_agents}, 3.0f);
    auto partial_dones = torch::zeros({partial_agents});
    auto partial_truncated = torch::zeros({partial_agents});
    auto partial_bootstrap = torch::zeros({partial_agents});

    for (int step = 0; step < rollout_len; step++) {
      partial_storage.append_slice(step, partial_offset,
                                   partial_obs, partial_ep_starts, partial_masks, partial_active,
                                   partial_acts, partial_lps, partial_vals, partial_rewards,
                                   partial_dones, partial_truncated, partial_bootstrap,
                                   slice_goal_pos.narrow(0, 0, partial_agents),
                                   slice_labels.narrow(0, 0, partial_agents),
                                   slice_term_obs.narrow(0, 0, partial_agents));
      partial_storage.mark_step_filled(step);
    }

    auto partial_retrieved = partial_storage.reward("extrinsic");
    // Only agents at offset [2..4] should be non-zero, rest should be 0
    float sum_offset_0_1 = partial_retrieved.slice(1, 0, 2).sum().item<float>();
    float sum_offset_2_5 = partial_retrieved.slice(1, 2, 5).sum().item<float>();
    require(sum_offset_0_1 == 0.0f, "agents 0-1 should be zero");
    require(sum_offset_2_5 > 0.0f, "agents 2-5 should be non-zero");

    std::cout << "test_rollout_storage passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "test_rollout_storage FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
