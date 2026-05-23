#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "pulsar/config/config.hpp"
#include "pulsar/model/vrpo_actor.hpp"
#include "pulsar/training/vrpo_math.hpp"
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
    // 1. VRPO clipped loss is per-sample
    {
      const auto cur = torch::tensor({-0.5F, -1.0F, -2.0F}, torch::kFloat32);
      const auto old = torch::tensor({-0.3F, -0.8F, -1.5F}, torch::kFloat32);
      const auto adv = torch::ones({3}, torch::kFloat32);
      const auto loss = pulsar::clipped_vrpo_policy_loss(cur, old, adv, 0.2F);
      require(loss.sizes() == cur.sizes(), "VRPO loss must be per-sample");
    }

    // 2. VRPO clipping with positive advantage
    {
      const float old_lp = 0.0F;
      const float cur_lp = std::log(1.5F);
      const float adv_val = 1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_vrpo_policy_loss(cur, old, adv, clip_range);
      const float ratio = std::exp(cur_lp - old_lp);
      const float clipped = std::min(ratio, 1.0F + clip_range);
      const float expected = -clipped * adv_val;
      require_close(loss.item<float>(), expected, "clipped VRPO with positive advantage");
    }

    // 3. VRPO clipping with negative advantage
    {
      const float cur_lp = std::log(0.5F);
      const float old_lp = std::log(1.0F);
      const float adv_val = -1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_vrpo_policy_loss(cur, old, adv, clip_range);
      const float ratio = std::exp(cur_lp - old_lp);
      const float clipped_ratio = std::clamp(ratio, 1.0F - clip_range, 1.0F + clip_range);
      const float expected = -std::min(ratio * adv_val, clipped_ratio * adv_val);
      require_close(loss.item<float>(), expected, "clipped VRPO with negative advantage");
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
 
    // 5b. Q-boosted GAE numerical validation
    {
      const auto q_taken = torch::tensor({1.5F, 2.5F, 3.5F}, torch::kFloat32).unsqueeze(1);
      const auto v_from_q = torch::tensor({1.0F, 2.0F, 3.0F}, torch::kFloat32).unsqueeze(1);
      const auto rewards = torch::tensor({0.5F, 1.0F, 1.5F}, torch::kFloat32).unsqueeze(1);
      const auto dones = torch::tensor({0.0F, 0.0F, 1.0F}, torch::kFloat32).unsqueeze(1);
      const auto next_v_from_q = torch::tensor({4.0F}, torch::kFloat32);

      const auto advantages = pulsar::compute_q_boosted_gae(
          q_taken, v_from_q, rewards, dones, 0.9F, 0.95F, next_v_from_q);

      require_close(advantages[2].item<float>(), -1.5F, "Q-boosted GAE step 2");
      require_close(advantages[1].item<float>(), -0.01F, "Q-boosted GAE step 1");
      require_close(advantages[0].item<float>(), 0.86395F, "Q-boosted GAE step 0");
    }

    // 6. One-sample advantage normalization
    {
      const auto adv = torch::tensor({3.0F}, torch::kFloat32);
      const auto mask = torch::ones({1}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "one-sample normalized advantage");
      require_close(normalized.item<float>(), 0.0F, "one-sample normalized advantage zero");
    }

    // 6a. Near-constant advantages should not be amplified into huge policy gradients.
    {
      const auto adv = torch::tensor({1.0F, 1.0F + 1.0e-7F, 1.0F - 1.0e-7F}, torch::kFloat32);
      const auto mask = torch::ones({3}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "near-constant normalized advantage");
      require(normalized.abs().max().item<float>() < 1.0e-4F, "near-constant advantage normalization should stay bounded");
    }

    // 6b. CUDA/HIP VRPO math parity when available.
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

      auto q_cpu = torch::randn({7, 5}, torch::kFloat32);
      auto v_from_q_cpu = torch::randn({7, 5}, torch::kFloat32);
      auto next_v_from_q_cpu = torch::randn({5}, torch::kFloat32);
      auto expected_q_boosted = pulsar::compute_q_boosted_gae(
          q_cpu,
          v_from_q_cpu,
          rewards_cpu,
          dones_cpu,
          0.97F,
          0.91F,
          next_v_from_q_cpu,
          boot_cpu,
          boot_values_cpu);
      auto actual_q_boosted = pulsar::compute_q_boosted_gae(
          q_cpu.to(opts),
          v_from_q_cpu.to(opts),
          rewards_cpu.to(opts),
          dones_cpu.to(opts),
          0.97F,
          0.91F,
          next_v_from_q_cpu.to(opts),
          boot_cpu.to(opts),
          boot_values_cpu.to(opts)).to(torch::kCPU);
      require(torch::allclose(actual_q_boosted, expected_q_boosted, 1.0e-5, 1.0e-5), "accelerated Q-boosted GAE parity");

      auto active_cpu = torch::ones({7, 5}, torch::kFloat32);
      active_cpu[0][0] = 0.0F;
      auto norm_expected = pulsar::normalize_advantage(expected, active_cpu);
      auto norm_actual = pulsar::normalize_advantage(expected.to(opts), active_cpu.to(opts)).to(torch::kCPU);
      require(torch::allclose(norm_actual, norm_expected, 1.0e-5, 1.0e-5), "accelerated advantage normalization parity");

      auto logits_cpu = torch::tensor({{1.0F, 2.0F, -1.0F}, {0.25F, -0.5F, 0.75F}}, torch::kFloat32);
      auto masks_cpu = torch::tensor({{true, false, true}, {true, true, true}}, torch::kBool);
      torch::Tensor expected_log_probs;
      auto expected_actions = pulsar::sample_masked_actions(logits_cpu, masks_cpu, true, &expected_log_probs);
      torch::Tensor actual_log_probs;
      auto actual_actions = pulsar::sample_masked_actions(logits_cpu.to(opts), masks_cpu.to(torch::kCUDA), true, &actual_log_probs).to(torch::kCPU);
      require(torch::equal(actual_actions, expected_actions), "accelerated deterministic masked action parity");
      require(torch::allclose(actual_log_probs.to(torch::kCPU), expected_log_probs, 1.0e-5, 1.0e-5), "accelerated masked action log-prob parity");

      auto entropy_expected = pulsar::masked_action_entropy(logits_cpu, masks_cpu);
      auto entropy_actual = pulsar::masked_action_entropy(logits_cpu.to(opts), masks_cpu.to(torch::kCUDA)).to(torch::kCPU);
      require(torch::allclose(entropy_actual, entropy_expected, 1.0e-5, 1.0e-5), "accelerated masked entropy parity");

      auto cur_cpu = torch::tensor({std::log(1.1F), std::log(1.4F), std::log(0.7F)}, torch::kFloat32).set_requires_grad(true);
      auto old_lp_cpu = torch::zeros({3}, torch::kFloat32);
      auto adv_cpu = torch::tensor({1.0F, 1.0F, -1.0F}, torch::kFloat32);
      auto loss_cpu = pulsar::clipped_vrpo_policy_loss(cur_cpu, old_lp_cpu, adv_cpu, 0.2F).sum();
      loss_cpu.backward();
      auto cur_gpu = cur_cpu.detach().to(opts).set_requires_grad(true);
      auto loss_gpu = pulsar::clipped_vrpo_policy_loss(cur_gpu, old_lp_cpu.to(opts), adv_cpu.to(opts), 0.2F).sum();
      loss_gpu.backward();
      require(torch::allclose(loss_gpu.detach().to(torch::kCPU), loss_cpu.detach(), 1.0e-5, 1.0e-5), "accelerated clipped VRPO loss parity");
      require(torch::allclose(cur_gpu.grad().to(torch::kCPU), cur_cpu.grad(), 1.0e-5, 1.0e-5), "accelerated clipped VRPO grad parity");

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

      auto boundary_goal_pos = torch::tensor(
          {
              {{1.0F, 2.0F}},
              {{11.0F, 12.0F}},
              {{21.0F, 22.0F}},
          },
          torch::kFloat32);
      auto boundary_dones = torch::tensor({{0.0F}, {1.0F}, {0.0F}}, torch::kFloat32);
      auto boundary_starts = torch::zeros({3, 1}, torch::kFloat32);
      auto boundary_future = pulsar::sample_future_goal_positions(boundary_goal_pos, boundary_dones, boundary_starts, 64);
      require_close(boundary_future[0][0][0].item<float>(), 11.0F, "future goal boundary should pick the only valid next sample");
      require_close(boundary_future[0][0][1].item<float>(), 12.0F, "future goal boundary should pick the only valid next sample");
      require_close(boundary_future[1][0][0].item<float>(), 11.0F, "future goal terminal step should keep current goal");
      require_close(boundary_future[2][0][0].item<float>(), 21.0F, "future goal final step should keep current goal");

      if (torch::cuda::is_available()) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto boundary_future_cuda = pulsar::sample_future_goal_positions(
            boundary_goal_pos.to(opts),
            boundary_dones.to(opts),
            boundary_starts.to(opts),
            64).to(torch::kCPU);
        require(torch::allclose(boundary_future_cuda, boundary_future, 1.0e-5, 1.0e-5), "future goal CUDA parity");
      }
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

    // 12. Live Predictors: Target Computation lookahead, horizons, and done-boundary masking
    {
      const int T = 5;
      const int N = 2;
      const int C = 3;
      auto sparse_events = torch::zeros({T, N, C}, torch::kUInt8);
      auto dones = torch::zeros({T, N}, torch::kFloat32);
      auto horizons = torch::tensor({2, 3, 4}, torch::kInt32);

      // Event for Agent 0 at timestep 3, Channel 0 (horizon = 2)
      sparse_events[3][0][0] = 1;

      // Event for Agent 1 at timestep 4, Channel 1 (horizon = 3)
      sparse_events[4][1][1] = 1;

      const auto targets = pulsar::compute_sparse_event_soon_targets(sparse_events, dones, horizons);
      require(targets.sizes() == torch::IntArrayRef({T, N, C}), "compute_sparse_event_soon_targets shape mismatch");

      // Verify lookahead for Agent 0, Channel 0 (horizon = 2)
      // Timestep 0, 1 -> 0
      // Timestep 2 -> 1 (sees event at 3 because lookahead is [2, 3])
      // Timestep 3 -> 1 (sees event at 3 because lookahead is [3, 4])
      // Timestep 4 -> 0
      require(targets[0][0][0].item<int64_t>() == 0, "lookahead target mismatch");
      require(targets[1][0][0].item<int64_t>() == 0, "lookahead target mismatch");
      require(targets[2][0][0].item<int64_t>() == 1, "lookahead target mismatch");
      require(targets[3][0][0].item<int64_t>() == 1, "lookahead target mismatch");
      require(targets[4][0][0].item<int64_t>() == 0, "lookahead target mismatch");

      // Verify lookahead for Agent 1, Channel 1 (horizon = 3)
      // Timestep 0, 1 -> 0
      // Timestep 2 -> 1 (sees event at 4 because lookahead is [2, 3, 4])
      // Timestep 3 -> 1 (sees event at 4 because lookahead is [3, 4, 5])
      // Timestep 4 -> 1 (sees event at 4 because lookahead is [4, 5, 6])
      require(targets[0][1][1].item<int64_t>() == 0, "lookahead target mismatch");
      require(targets[1][1][1].item<int64_t>() == 0, "lookahead target mismatch");
      require(targets[2][1][1].item<int64_t>() == 1, "lookahead target mismatch");
      require(targets[3][1][1].item<int64_t>() == 1, "lookahead target mismatch");
      require(targets[4][1][1].item<int64_t>() == 1, "lookahead target mismatch");

      // Test done-boundary truncation
      // If done occurs at timestep 2 for Agent 0, lookahead at timestep 1 must terminate
      // at timestep 2 and not see the event at timestep 3.
      dones[2][0] = 1.0F;
      const auto truncated_targets = pulsar::compute_sparse_event_soon_targets(sparse_events, dones, horizons);
      require(truncated_targets[1][0][0].item<int64_t>() == 0, "done truncation lookahead mismatch");
    }

    // 13. Live Predictors: Convergence State Machine, Warm-up, Positive Rate, and NaN resilience
    {
      float ema_loss = -1.0F;
      float delta = 1.0F;
      std::uint8_t active = 0;
      
      const float warmup_updates = 3;
      const float convergence_threshold = 0.01F;

      auto run_mock_update = [&](double mean_loss, double positive_rate, int update_index) {
        if (mean_loss <= 0.0 || !std::isfinite(mean_loss)) {
          return;
        }
        if (ema_loss < 0.0F) {
          ema_loss = static_cast<float>(mean_loss);
          delta = std::numeric_limits<float>::infinity();
        } else {
          const float next_ema = 0.9F * ema_loss + 0.1F * static_cast<float>(mean_loss);
          delta = std::abs(next_ema - ema_loss);
          ema_loss = next_ema;
        }
        if (active == 0 &&
            update_index >= warmup_updates &&
            positive_rate > 0.0 &&
            std::isfinite(delta) &&
            delta <= convergence_threshold) {
          active = 1;
        }
      };

      // Update 1: First update initializes the EMA loss
      run_mock_update(1.0, 0.1, 1);
      require_close(ema_loss, 1.0F, "state machine EMA initialization");
      require(std::isinf(delta), "initial delta should be infinite");
      require(active == 0, "should not activate on first update");

      // Update 2: Delta becomes small, but update_index (2) < warmup (3)
      run_mock_update(0.9, 0.1, 2); // next_ema = 0.99, delta = 0.01
      require_close(ema_loss, 0.99F, "EMA update");
      require_close(delta, 0.01F, "delta calculation");
      require(active == 0, "should not activate before warmup updates");

      // Update 3: Within warmup, but positive_rate = 0.0 (no occurrences observed)
      run_mock_update(0.99, 0.0, 3); // delta = 0.0
      require(active == 0, "should not activate when positive_rate is 0");

      // Update 4: Valid convergence and positive rate
      run_mock_update(0.99, 0.1, 4); // delta = 0.0
      require(active == 1, "predictor should activate when all conditions are satisfied");

      // Test NaN loss resilience: updating with NaN should leave state unaffected
      const float saved_ema = ema_loss;
      const float saved_delta = delta;
      const std::uint8_t saved_active = active;

      run_mock_update(std::numeric_limits<double>::quiet_NaN(), 0.1, 5);
      require_close(ema_loss, saved_ema, "NaN loss should not affect EMA loss");
      require_close(delta, saved_delta, "NaN loss should not affect delta");
      require(active == saved_active, "NaN loss should not affect active flag");
    }

    std::cout << "pulsar_vrpo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_vrpo_math_tests FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
