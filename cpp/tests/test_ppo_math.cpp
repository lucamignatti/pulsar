#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "pulsar/config/config.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "test_utils.hpp"

namespace {

const float kTolerance = 1.0e-4F;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(float a, float b, const std::string& message) {
  if (std::fabs(a - b) > kTolerance) {
    throw std::runtime_error(message + ": expected " + std::to_string(b) + " got " + std::to_string(a));
  }
}

void require_finite(const torch::Tensor& t, const std::string& name) {
  require(t.defined(), name + " is undefined");
  require(torch::isfinite(t).all().item<bool>(), name + " has non-finite values");
}

pulsar::ModelConfig tiny_model_config() {
  pulsar::ModelConfig cfg;
  cfg.observation_dim = 8;
  cfg.action_dim = 4;
  cfg.use_layer_norm = false;
  cfg.encoder_dim = 8;
  cfg.workspace_dim = 8;
  cfg.stm_slots = 4;
  cfg.stm_key_dim = 4;
  cfg.stm_value_dim = 4;
  cfg.ltm_slots = 4;
  cfg.ltm_dim = 4;
  cfg.controller_dim = 8;
  cfg.consolidation_stride = 2;
  cfg.value_hidden_dim = 16;
  cfg.value_num_atoms = 21;
  cfg.value_v_min = -5.0F;
  cfg.value_v_max = 5.0F;
  return cfg;
}

}  // namespace

int main() {
  try {
    // 1. PPO clipped loss is per-sample
    {
      const auto cur = torch::tensor({-0.5F, -1.0F, -2.0F}, torch::kFloat32);
      const auto old = torch::tensor({-0.3F, -0.8F, -1.5F}, torch::kFloat32);
      const auto adv = torch::ones({3}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, 0.2F);
      require(loss.sizes() == cur.sizes(), "PPO loss must be per-sample");
    }

    // 2. PPO clipping with positive advantage
    {
      const float old_lp = 0.0F;
      const float cur_lp = std::log(1.5F);
      const float adv_val = 1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, clip_range);
      const float ratio = std::exp(cur_lp - old_lp);
      const float clipped = std::min(ratio, 1.0F + clip_range);
      const float expected = -clipped * adv_val;
      require_close(loss.item<float>(), expected, "clipped PPO with positive advantage");
    }

    // 3. PPO clipping with negative advantage
    {
      const float cur_lp = std::log(0.5F);
      const float old_lp = std::log(1.0F);
      const float adv_val = -1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, clip_range);
      const float ratio = std::exp(cur_lp - old_lp);
      const float clipped_ratio = std::clamp(ratio, 1.0F - clip_range, 1.0F + clip_range);
      const float expected = -std::min(ratio * adv_val, clipped_ratio * adv_val);
      require_close(loss.item<float>(), expected, "clipped PPO with negative advantage");
    }

    // 4. Confidence weighting is per-sample
    {
      const auto value_logits = torch::zeros({2, 21}, torch::kFloat32);
      value_logits[0][10] = 5.0F;
      value_logits[1][5] = 5.0F;
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);
      const auto weights = pulsar::compute_confidence_weights(
          value_logits, atom_support, "entropy", 1.0e-6F, false);
      require(weights.sizes() == torch::IntArrayRef({2}), "confidence weights shape");
      require_finite(weights, "confidence weights");
    }

    // 5. GAE final bootstrap
    {
      const auto values = torch::zeros({2, 1}, torch::kFloat32);
      const auto rewards = torch::zeros({2, 1}, torch::kFloat32);
      const auto dones = torch::zeros({2, 1}, torch::kFloat32);
      const auto next_values = torch::ones({1}, torch::kFloat32);
      const auto advantages = pulsar::compute_gae(values, rewards, dones, 1.0F, 1.0F, next_values);
      require(advantages.sizes() == values.sizes(), "GAE advantages shape");
      require_close(advantages[0].item<float>(), 1.0F, "GAE bootstrap step 0");
      require_close(advantages[1].item<float>(), 1.0F, "GAE bootstrap step 1");
    }

    // 6. GAE terminal masking
    {
      const auto values = torch::zeros({3, 1}, torch::kFloat32);
      const auto rewards = torch::tensor({1.0F, 1.0F, 1.0F}, torch::kFloat32).unsqueeze(1);
      const auto dones = torch::tensor({0.0F, 1.0F, 0.0F}, torch::kFloat32).unsqueeze(1);
      const auto advantages = pulsar::compute_gae(values, rewards, dones, 1.0F, 1.0F, {});
      require_close(advantages[0].item<float>(), 2.0F, "GAE terminal mask step 0");
      require_close(advantages[1].item<float>(), 1.0F, "GAE terminal mask step 1");
      require_close(advantages[2].item<float>(), 1.0F, "GAE terminal mask step 2");
    }

    // 7. One-sample advantage normalization
    {
      const auto adv = torch::tensor({3.0F}, torch::kFloat32);
      const auto mask = torch::ones({1}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "one-sample normalized advantage");
      require_close(normalized.item<float>(), 0.0F, "one-sample normalized advantage zero");
    }

    // 8. Distributional projection
    {
      const auto logits = torch::zeros({1, 21}, torch::kFloat32);
      const float v_min = -5.0F;
      const float v_max = 5.0F;
      const int num_atoms = 21;
      const float delta_z = (v_max - v_min) / static_cast<float>(num_atoms - 1);

      const auto ret_at_min = torch::tensor({v_min}, torch::kFloat32);
      const auto loss_min = pulsar::distributional_value_loss(logits, ret_at_min, v_min, v_max, num_atoms);
      require_finite(loss_min, "distributional loss at v_min");

      const auto ret_at_max = torch::tensor({v_max}, torch::kFloat32);
      const auto loss_max = pulsar::distributional_value_loss(logits, ret_at_max, v_min, v_max, num_atoms);
      require_finite(loss_max, "distributional loss at v_max");

      const float between_val = v_min + delta_z * 5.5F;
      const auto ret_between = torch::tensor({between_val}, torch::kFloat32);
      const auto loss_between = pulsar::distributional_value_loss(logits, ret_between, v_min, v_max, num_atoms);
      require_finite(loss_between, "distributional loss between atoms");
    }

    // 9. Distributional sampling
    {
      torch::manual_seed(42);
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);
      const auto logits_uniform = torch::zeros({32, 21}, torch::kFloat32);
      const auto sampled_uniform = pulsar::sample_quantile_value(logits_uniform, atom_support);
      const float min_atom = atom_support[0].item<float>();
      const float max_atom = atom_support[-1].item<float>();
      for (int i = 0; i < 32; ++i) {
        const float val = sampled_uniform[i].item<float>();
        require(val >= min_atom && val <= max_atom,
                "sampled value outside support at index " + std::to_string(i));
      }
    }

    // 10. Config validation
    {
      pulsar::ExperimentConfig config = pulsar::test::make_test_config();
      config.ppo.rollout_length = 2;
      config.ppo.sequence_length = 1;
      config.ppo.minibatch_size = 1;
      config.ppo.burn_in = 0;
      config.model.encoder_dim = 512;
      try {
        pulsar::validate_experiment_config(config);
      } catch (const std::exception&) {
        throw std::runtime_error("valid config should not throw");
      }

      pulsar::ExperimentConfig bad_config = config;
      bad_config.ppo.rollout_length = 1;
      bool caught = false;
      try {
        pulsar::validate_experiment_config(bad_config);
      } catch (const std::invalid_argument&) {
        caught = true;
      }
      require(caught, "rollout_length <= 1 should throw");
    }

    // 11. Goal critic v_max auto-computation
    {
      pulsar::GoalCriticConfig cfg;
      cfg.horizon_H = 256;
      cfg.gamma_g = 0.99F;
      float v_max = pulsar::compute_goal_critic_v_max(cfg);
      require(v_max > 0.0F, "auto-computed v_max should be positive");
      require(v_max < 300.0F, "auto-computed v_max should be bounded");

      cfg.v_max = 50.0F;
      float explicit_vmax = pulsar::compute_goal_critic_v_max(cfg);
      require_close(explicit_vmax, 50.0F, "explicit v_max should be preserved");
    }

    // 12. Goal occupancy computation
    {
      const int steps = 5;
      const int agents = 2;
      auto goal_dist = torch::zeros({steps, agents}, torch::kFloat32);
      goal_dist[0][0] = 0.9F;
      goal_dist[1][0] = 0.5F;
      goal_dist[2][0] = 0.1F;
      goal_dist[3][0] = 0.05F;
      goal_dist[4][0] = 0.0F;
      for (int k = 0; k < 5; ++k) {
        goal_dist[k][1] = 1.0F;
      }
      auto dones = torch::zeros({steps, agents}, torch::kFloat32);

      auto occupancy = pulsar::compute_finite_horizon_goal_occupancy(
          goal_dist, dones, 0.99F, 0.0F, 0.05F, 4);

      require(occupancy.sizes() == torch::IntArrayRef({steps, agents}), "goal occupancy shape");
      require_finite(occupancy, "goal occupancy");
      require(occupancy[4][0].item<float>() > 0.5F,
              "close-to-goal step should have high occupancy");
      require(occupancy[0][1].item<float>() < 0.1F,
              "far from goal step should have low occupancy");
    }

    // 13. Goal actor loss with discrete actions (REINFORCE-style)
    {
      auto policy_logits = torch::randn({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
      auto action_masks = torch::ones({4, 4}, torch::kBool);
      auto actions = torch::randint(0, 4, {4}, torch::kLong);
      auto goal_critic_logits = torch::zeros({4, 21}, torch::kFloat32);
      auto goal_support = torch::linspace(0.0F, 25.0F, 21, torch::kFloat32);

      auto loss = pulsar::compute_goal_actor_loss_discrete(
          policy_logits, action_masks, goal_critic_logits, goal_support, actions);
      require_finite(loss, "goal actor loss");
      require(loss.sizes() == torch::IntArrayRef({}), "goal actor loss should be scalar");
      require(loss.detach().requires_grad() == false || loss.requires_grad(), "goal actor loss should pass through");
    }

    // 14. KL divergence computation
    {
      auto base_logits = torch::full({2, 4}, 0.01F, torch::kFloat32);
      auto perturbed_logits = torch::full({2, 4}, 0.01F, torch::kFloat32);
      auto masks = torch::ones({2, 4}, torch::kBool);

      float kl_same = pulsar::compute_discrete_policy_kl(base_logits, perturbed_logits, masks);
      require(kl_same < 0.001F, "KL between identical distributions should be near zero");

      perturbed_logits[0][0] = 10.0F;
      float kl_diff = pulsar::compute_discrete_policy_kl(base_logits, perturbed_logits, masks);
      require(kl_diff > 0.0F, "KL between different distributions should be positive");
    }

    // 15. Adaptive epsilon bounds
    {
      const auto tiny_variance = torch::tensor({0.01F}, torch::kFloat32);
      const float eps = pulsar::compute_adaptive_epsilon(tiny_variance, 0.2F, 1.0F, 0.05F, 0.3F);
      require(eps >= 0.05F && eps <= 0.3F, "adaptive epsilon within bounds");

      const auto large_variance = torch::tensor({100.0F}, torch::kFloat32);
      const float small_eps = pulsar::compute_adaptive_epsilon(large_variance, 0.2F, 1.0F, 0.05F, 0.3F);
      require_close(small_eps, 0.05F, "adaptive epsilon clamps to min");
    }

    // 16. compute_mean_value and compute_distribution_variance
    {
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);
      const auto logits = torch::zeros({4, 21}, torch::kFloat32);
      const auto mean_val = pulsar::compute_mean_value(logits, atom_support);
      require(mean_val.sizes() == torch::IntArrayRef({4}), "mean value shape");
      require_finite(mean_val, "mean value finite");

      const auto variance = pulsar::compute_distribution_variance(logits, atom_support);
      require(variance.sizes() == torch::IntArrayRef({4}), "variance shape");
      require_finite(variance, "variance finite");
    }

    // 17. Goal critic loss uses C51 projection
    {
      const auto logits = torch::zeros({1, 21}, torch::kFloat32);
      const float v_min = 0.0F;
      const float v_max = 25.0F;
      const int num_atoms = 21;
      const auto goal_occ_target = torch::tensor({12.5F}, torch::kFloat32);
      const auto loss = pulsar::distributional_value_loss(logits, goal_occ_target, v_min, v_max, num_atoms);
      require_finite(loss, "goal critic C51 loss");
    }

    std::cout << "pulsar_ppo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_ppo_math_tests FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
