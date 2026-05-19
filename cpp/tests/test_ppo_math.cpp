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
  cfg.value_hidden_dim = 16;
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

    // 4. GAE final bootstrap
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

    // 5. GAE terminal masking
    {
      const auto values = torch::zeros({3, 1}, torch::kFloat32);
      const auto rewards = torch::tensor({1.0F, 1.0F, 1.0F}, torch::kFloat32).unsqueeze(1);
      const auto dones = torch::tensor({0.0F, 1.0F, 0.0F}, torch::kFloat32).unsqueeze(1);
      const auto advantages = pulsar::compute_gae(values, rewards, dones, 1.0F, 1.0F, {});
      require_close(advantages[0].item<float>(), 2.0F, "GAE terminal mask step 0");
      require_close(advantages[1].item<float>(), 1.0F, "GAE terminal mask step 1");
      require_close(advantages[2].item<float>(), 1.0F, "GAE terminal mask step 2");
    }

    // 6. One-sample advantage normalization
    {
      const auto adv = torch::tensor({3.0F}, torch::kFloat32);
      const auto mask = torch::ones({1}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "one-sample normalized advantage");
      require_close(normalized.item<float>(), 0.0F, "one-sample normalized advantage zero");
    }

    // 6b. CUDA/HIP PPO math accelerator parity when available.
    if (torch::cuda::is_available()) {
      auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      auto values_cpu = torch::randn({7, 5}, torch::kFloat32);
      auto rewards_cpu = torch::randn({7, 5}, torch::kFloat32);
      auto dones_cpu = torch::zeros({7, 5}, torch::kFloat32);
      dones_cpu[3][2] = 1.0F;
      auto next_cpu = torch::randn({5}, torch::kFloat32);
      auto boot_cpu = torch::zeros({7, 5}, torch::kFloat32);
      boot_cpu[6].fill_(1.0F);
      auto boot_values_cpu = torch::randn({7, 5}, torch::kFloat32);
      auto expected = pulsar::compute_gae(values_cpu, rewards_cpu, dones_cpu, 0.97F, 0.91F, next_cpu, boot_cpu, boot_values_cpu);
      auto actual = pulsar::compute_gae(
          values_cpu.to(opts),
          rewards_cpu.to(opts),
          dones_cpu.to(opts),
          0.97F,
          0.91F,
          next_cpu.to(opts),
          boot_cpu.to(opts),
          boot_values_cpu.to(opts)).to(torch::kCPU);
      require(torch::allclose(actual, expected, 1.0e-5, 1.0e-5), "accelerated GAE parity");

      auto active_cpu = torch::ones({7, 5}, torch::kFloat32);
      active_cpu[0][0] = 0.0F;
      auto norm_expected = pulsar::normalize_advantage(expected, active_cpu);
      auto norm_actual = pulsar::normalize_advantage(expected.to(opts), active_cpu.to(opts)).to(torch::kCPU);
      require(torch::allclose(norm_actual, norm_expected, 1.0e-5, 1.0e-5), "accelerated advantage normalization parity");

      auto goals = torch::randn({7, 5, 3}, opts);
      auto future = pulsar::sample_future_goal_positions(goals, dones_cpu.to(opts), boot_cpu.to(opts), 4);
      require(future.sizes() == torch::IntArrayRef({7, 5, 3}), "accelerated future goals shape");
      require_finite(future.to(torch::kCPU), "accelerated future goals finite");
    }

    // 7. Config validation
    {
      pulsar::ExperimentConfig config = pulsar::test::make_test_config();
      config.ppo.rollout_length = 2;
      config.ppo.minibatch_size = 1;
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

    // 8. Future goal sampling
    {
      torch::manual_seed(42);
      const int steps = 5;
      const int agents = 2;
      const int goal_dim = 3;
      auto goal_pos = torch::zeros({steps, agents, goal_dim}, torch::kFloat32);
      for (int s = 0; s < steps; ++s) {
        for (int a = 0; a < agents; ++a) {
          goal_pos[s][a][0] = 0.1F * static_cast<float>(s);
          goal_pos[s][a][1] = 0.2F * static_cast<float>(s);
          goal_pos[s][a][2] = 0.3F;
        }
      }
      auto dones = torch::zeros({steps, agents}, torch::kFloat32);
      auto ep_starts = torch::zeros({steps, agents}, torch::kFloat32);
      const int max_future = 64;

      auto future_goals = pulsar::sample_future_goal_positions(
          goal_pos, dones, ep_starts, max_future);

      require(future_goals.sizes() == torch::IntArrayRef({steps, agents, goal_dim}), "future goals shape");
      require_finite(future_goals, "future goals finite");
    }

    // 9. Contrastive loss
    {
      torch::manual_seed(42);
      auto lhs = torch::randn({8, 64}, torch::kFloat32);
      auto rhs = torch::randn({8, 64}, torch::kFloat32);
      auto logits = pulsar::compute_pairwise_negative_l2_logits(lhs, rhs);
      require(logits.sizes() == torch::IntArrayRef({8, 8}), "contrastive logits shape");

      auto loss = pulsar::compute_symmetric_infonce_loss(logits, 0.01F);
      require(loss.sizes() == torch::IntArrayRef({}), "contrastive loss scalar");
      require_finite(loss, "contrastive loss finite");
    }

    // 9b. Pairwise negative L2 logits match known squared distances.
    {
      auto lhs = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}}, torch::kFloat32);
      auto rhs = torch::tensor({{0.0F, 0.0F}, {0.0F, 2.0F}}, torch::kFloat32);
      auto logits = pulsar::compute_pairwise_negative_l2_logits(lhs, rhs);
      require_close(logits[0][0].item<float>(), 0.0F, "pairwise L2 zero distance");
      require_close(logits[0][1].item<float>(), -4.0F, "pairwise L2 row 0 col 1");
      require_close(logits[1][0].item<float>(), -1.0F, "pairwise L2 row 1 col 0");
      require_close(logits[1][1].item<float>(), -5.0F, "pairwise L2 row 1 col 1");
    }

    // 9c. Contrastive loss stays finite for distances that would overflow fp16.
    {
      auto lhs = torch::full({2, 4}, 10000.0F, torch::kFloat32);
      auto rhs = torch::full({2, 4}, -10000.0F, torch::kFloat32);
      auto logits = pulsar::compute_pairwise_negative_l2_logits(lhs, rhs);
      require_finite(logits, "large-distance contrastive logits finite");

      auto loss = pulsar::compute_symmetric_infonce_loss(logits, 0.01F);
      require_finite(loss, "large-distance contrastive loss finite");
    }

    // 10. KL divergence computation
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

    // 11. Action masking should avoid infinities in the autograd graph.
    {
      auto logits = torch::tensor({{1.0F, 2.0F, 3.0F}, {0.5F, -0.5F, 1.5F}}, torch::kFloat32);
      auto masks = torch::tensor({{true, false, true}, {false, true, true}}, torch::kBool);
      auto masked = pulsar::apply_action_mask_to_logits(logits, masks);
      require_finite(masked, "masked logits finite");
      require(masked[0][1].item<float>() < -1.0e8F, "invalid action receives very low finite logit");
      auto entropy = pulsar::masked_action_entropy(logits, masks);
      require_finite(entropy, "masked entropy finite");
    }

    std::cout << "pulsar_ppo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_ppo_math_tests FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
