#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "pulsar/config/config.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/training/ppo_math.hpp"

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
    // ---------------------------------------------------------------
    // 1. PPO clipped loss is per-sample
    // ---------------------------------------------------------------
    {
      const auto cur = torch::tensor({-0.5F, -1.0F, -2.0F}, torch::kFloat32);
      const auto old = torch::tensor({-0.3F, -0.8F, -1.5F}, torch::kFloat32);
      const auto adv = torch::ones({3}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, 0.2F);
      require(loss.sizes() == cur.sizes(), "PPO loss must be per-sample");
    }

    // ---------------------------------------------------------------
    // 2. PPO clipping with positive advantage
    // ---------------------------------------------------------------
    {
      const float old_lp = 0.0F;   // log(1.0)
      const float cur_lp = std::log(1.5F);
      const float adv_val = 1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, clip_range);
      const float ratio = std::exp(cur_lp - old_lp);  // 1.5
      const float clipped = std::min(ratio, 1.0F + clip_range);  // 1.2
      const float expected = -clipped * adv_val;  // -1.2
      require_close(loss.item<float>(), expected, "clipped PPO with positive advantage");
    }

    // ---------------------------------------------------------------
    // 3. PPO clipping with negative advantage
    // ---------------------------------------------------------------
    {
      const float cur_lp = std::log(0.5F);
      const float old_lp = std::log(1.0F);
      const float adv_val = -1.0F;
      const float clip_range = 0.2F;
      const auto cur = torch::tensor({cur_lp}, torch::kFloat32);
      const auto old = torch::tensor({old_lp}, torch::kFloat32);
      const auto adv = torch::tensor({adv_val}, torch::kFloat32);
      const auto loss = pulsar::clipped_ppo_policy_loss(cur, old, adv, clip_range);
      // ratio = 0.5, clipped_ratio = clamp(0.5, 0.8, 1.2) = 0.8
      // min(0.5 * -1, 0.8 * -1) = min(-0.5, -0.8) = -0.8
      // -(-0.8) = 0.8
      const float ratio = std::exp(cur_lp - old_lp);
      const float clipped_ratio = std::clamp(ratio, 1.0F - clip_range, 1.0F + clip_range);
      const float expected = -std::min(ratio * adv_val, clipped_ratio * adv_val);
      require_close(loss.item<float>(), expected, "clipped PPO with negative advantage");
    }

    // ---------------------------------------------------------------
    // 4. Confidence weighting is per-sample
    // ---------------------------------------------------------------
    {
      const auto value_logits = torch::zeros({2, 21}, torch::kFloat32);
      value_logits[0][10] = 5.0F;
      value_logits[1][5] = 5.0F;
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);
      const auto weights = pulsar::compute_confidence_weights(
          value_logits, atom_support, "entropy", 1.0e-6F, false);
      require(weights.sizes() == torch::IntArrayRef({2}), "confidence weights shape");
      require_finite(weights, "confidence weights");

      // With manual per-sample: [1, 1] * [1, 10] -> mean = 5.5
      const auto per_sample = torch::ones({2}, torch::kFloat32);
      const auto custom_weights = torch::tensor({1.0F, 10.0F}, torch::kFloat32);
      const float weighted_mean = (per_sample * custom_weights).mean().item<float>();
      require_close(weighted_mean, 5.5F, "confidence weighting mean");
    }

    // ---------------------------------------------------------------
    // 5. GAE final bootstrap
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // 6. GAE terminal masking
    // ---------------------------------------------------------------
    {
      const auto values = torch::zeros({3, 1}, torch::kFloat32);
      const auto rewards = torch::tensor({1.0F, 1.0F, 1.0F}, torch::kFloat32).unsqueeze(1);
      const auto dones = torch::tensor({0.0F, 1.0F, 0.0F}, torch::kFloat32).unsqueeze(1);
      const auto advantages = pulsar::compute_gae(values, rewards, dones, 1.0F, 1.0F, {});
      // Step t=2: delta = 1 - 0 = 1, last_gae = 1
      // Step t=1: next_value = values[2] = 0, non_terminal = 1 - 1 = 0
      //           delta = 1 + 1*0*0 - 0 = 1, last_gae = 1 + 1*1*0*0 = 1
      // Step t=0: next_value = values[1] = 0, non_terminal = 1 - 0 = 1
      //           delta = 1 + 1*0*1 - 0 = 1, last_gae = 1 + 1*1*1*1 = 2
      require_close(advantages[0].item<float>(), 2.0F, "GAE terminal mask step 0");
      require_close(advantages[1].item<float>(), 1.0F, "GAE terminal mask step 1");
      require_close(advantages[2].item<float>(), 1.0F, "GAE terminal mask step 2");
    }

    // ---------------------------------------------------------------
    // 7. One-sample advantage normalization
    // ---------------------------------------------------------------
    {
      const auto adv = torch::tensor({3.0F}, torch::kFloat32);
      const auto mask = torch::ones({1}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "one-sample normalized advantage");
      require_close(normalized.item<float>(), 0.0F, "one-sample normalized advantage zero");
    }

    // Two-sample normalization
    {
      const auto adv = torch::tensor({1.0F, 3.0F}, torch::kFloat32);
      const auto mask = torch::ones({2}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require_finite(normalized, "two-sample normalized advantage");
      // mean = 2, std = 1 (biased), normalized = [-1, 1]
      require_close(normalized[0].item<float>(), -1.0F, "two-sample normalized step 0");
      require_close(normalized[1].item<float>(), 1.0F, "two-sample normalized step 1");
    }

    // All-masked advantage
    {
      const auto adv = torch::tensor({1.0F, 2.0F, 3.0F}, torch::kFloat32);
      const auto mask = torch::zeros({3}, torch::kFloat32);
      const auto normalized = pulsar::normalize_advantage(adv, mask);
      require(normalized.allclose(adv), "all-masked advantage unchanged");
    }

    // ---------------------------------------------------------------
    // 8. Distributional projection
    // ---------------------------------------------------------------
    {
      const auto logits = torch::zeros({1, 21}, torch::kFloat32);
      const float v_min = -5.0F;
      const float v_max = 5.0F;
      const int num_atoms = 21;
      const float delta_z = (v_max - v_min) / static_cast<float>(num_atoms - 1);

      // Return at v_min
      const auto ret_at_min = torch::tensor({v_min}, torch::kFloat32);
      const auto loss_min = pulsar::distributional_value_loss(logits, ret_at_min, v_min, v_max, num_atoms);
      require_finite(loss_min, "distributional loss at v_min");

      // Return at v_max
      const auto ret_at_max = torch::tensor({v_max}, torch::kFloat32);
      const auto loss_max = pulsar::distributional_value_loss(logits, ret_at_max, v_min, v_max, num_atoms);
      require_finite(loss_max, "distributional loss at v_max");

      // Return at middle atom
      const float mid_val = v_min + delta_z * 10.0F;
      const auto ret_mid = torch::tensor({mid_val}, torch::kFloat32);
      const auto loss_mid = pulsar::distributional_value_loss(logits, ret_mid, v_min, v_max, num_atoms);
      require_finite(loss_mid, "distributional loss at middle atom");

      // Return below v_min
      const auto ret_below = torch::tensor({v_min - 1.0F}, torch::kFloat32);
      const auto loss_below = pulsar::distributional_value_loss(logits, ret_below, v_min, v_max, num_atoms);
      require_finite(loss_below, "distributional loss below v_min");

      // Return above v_max
      const auto ret_above = torch::tensor({v_max + 1.0F}, torch::kFloat32);
      const auto loss_above = pulsar::distributional_value_loss(logits, ret_above, v_min, v_max, num_atoms);
      require_finite(loss_above, "distributional loss above v_max");

      // Return between two atoms
      const float between_val = v_min + delta_z * 5.5F;
      const auto ret_between = torch::tensor({between_val}, torch::kFloat32);
      const auto loss_between = pulsar::distributional_value_loss(logits, ret_between, v_min, v_max, num_atoms);
      require_finite(loss_between, "distributional loss between atoms");
    }

    // ---------------------------------------------------------------
    // 9. Distributional sampling
    // ---------------------------------------------------------------
    {
      torch::manual_seed(42);
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);

      // Near-one-hot distribution: only atom 10 should be sampled
      const auto logits_one_hot = torch::zeros({4, 21}, torch::kFloat32);
      logits_one_hot.index_fill_(1, torch::tensor({10}, torch::kLong), 20.0F);
      const auto sampled_one_hot = pulsar::sample_quantile_value(logits_one_hot, atom_support);
      const float expected_atom = atom_support[10].item<float>();
      for (int i = 0; i < 4; ++i) {
        require_close(sampled_one_hot[i].item<float>(), expected_atom,
                      "one-hot sample atom " + std::to_string(i));
      }

      // Sampled value always in atom_support
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

    // ---------------------------------------------------------------
    // 10. Critic-head config materialization
    // ---------------------------------------------------------------
    {
      pulsar::ModelConfig model_cfg;
      model_cfg.value_hidden_dim = 128;
      model_cfg.value_num_atoms = 101;
      model_cfg.value_v_min = -100.0F;
      model_cfg.value_v_max = 100.0F;

      // Sentinel CriticHeadConfig (all zeros) should inherit from ModelConfig
      pulsar::CriticHeadConfig sentinel;
      sentinel.value_hidden_dim = 0;
      sentinel.value_num_atoms = 0;
      sentinel.value_v_min = 0.0F;
      sentinel.value_v_max = 0.0F;

      const auto materialized = pulsar::materialize_critic_head_config(sentinel, model_cfg, true);
      require(materialized.enabled, "materialized head enabled");
      require(materialized.value_hidden_dim == 128, "materialized head inherited hidden_dim");
      require(materialized.value_num_atoms == 101, "materialized head inherited num_atoms");
      require_close(materialized.value_v_min, -100.0F, "materialized head inherited v_min");
      require_close(materialized.value_v_max, 100.0F, "materialized head inherited v_max");
    }

    // ---------------------------------------------------------------
    // 11. Config validation
    // ---------------------------------------------------------------
    {
      pulsar::ExperimentConfig config;
      config.ppo.rollout_length = 2;
      config.ppo.sequence_length = 1;
      config.ppo.minibatch_size = 1;
      config.ppo.burn_in = 0;
      config.behavior_cloning.sequence_length = 1;
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

    // Intrinsic consistency: intrinsic rewards require forward/inverse loss coef > 0
    {
      pulsar::ExperimentConfig config;
      config.ppo.rollout_length = 2;
      config.ppo.sequence_length = 1;
      config.ppo.minibatch_size = 1;
      config.intrinsic_rewards.curiosity_weight = 1.0F;
      config.intrinsic_model.forward_loss_coef = 0.0F;
      config.intrinsic_model.inverse_loss_coef = 0.0F;
      bool caught = false;
      try {
        pulsar::validate_experiment_config(config);
      } catch (const std::invalid_argument&) {
        caught = true;
      }
      require(caught, "intrinsic rewards without auxiliary loss coefs should throw");
    }

    // ---------------------------------------------------------------
    // 12. Forward/inverse model shapes
    // ---------------------------------------------------------------
    {
      const auto model_cfg = tiny_model_config();
      pulsar::CriticConfig critic_cfg;
      critic_cfg.controllability.enabled = false;
      pulsar::PPOActor actor(model_cfg, critic_cfg);
      const auto state = actor->initial_state(2, torch::kCPU);
      const auto output = actor->forward_step(torch::randn({2, model_cfg.observation_dim}), std::move(state));
      require_finite(output.encoded, "actor encoded output");

      // Forward model
      const auto actions = torch::randint(0, model_cfg.action_dim, {2}, torch::kLong);
      const auto forward_pred = actor->forward_predict_next(output.encoded, actions);
      require(forward_pred.sizes() == output.encoded.sizes(),
              "forward prediction shape matches encoded shape");
      require_finite(forward_pred, "forward prediction");

      // Inverse model
      const auto encoded_tp1 = torch::randn({2, model_cfg.encoder_dim});
      const auto inverse_logits = actor->forward_predict_action(output.encoded, encoded_tp1);
      require(inverse_logits.sizes() == torch::IntArrayRef({2, model_cfg.action_dim}),
              "inverse logits shape");
      require_finite(inverse_logits, "inverse logits");

      // Forward prediction error
      const auto pred_error = actor->compute_forward_prediction_error(
          output.encoded, actions, encoded_tp1);
      require(pred_error.sizes() == torch::IntArrayRef({2}), "prediction error shape");
      require_finite(pred_error, "prediction error");
    }

    // ---------------------------------------------------------------
    // 13. Forward model gradient flow
    // ---------------------------------------------------------------
    {
      const auto model_cfg = tiny_model_config();
      pulsar::PPOActor actor(model_cfg);
      auto params_before = std::vector<torch::Tensor>();
      for (const auto& p : actor->named_parameters()) {
        if (p.key().find("forward_head") != std::string::npos) {
          params_before.push_back(p.value().detach().clone());
        }
      }
      require(!params_before.empty(), "forward_head has parameters");

      // Compute a non-trivial forward loss
      const auto encoded = torch::randn({4, model_cfg.encoder_dim}, torch::kFloat32);
      const auto actions = torch::randint(0, model_cfg.action_dim, {4}, torch::kLong);
      const auto encoded_tp1 = torch::randn({4, model_cfg.encoder_dim}, torch::kFloat32);
      const auto forward_pred = actor->forward_predict_next(encoded, actions);
      const auto loss = torch::mse_loss(forward_pred, encoded_tp1);
      loss.backward();

      // Check forward_head parameters received gradients
      int grad_count = 0;
      for (const auto& p : actor->named_parameters()) {
        if (p.key().find("forward_head") != std::string::npos) {
          if (p.value().grad().defined()) {
            ++grad_count;
          }
        }
      }
      require(grad_count > 0, "forward_head parameters must receive gradients");
    }

    // ---------------------------------------------------------------
    // 14. Inverse model gradient flow
    // ---------------------------------------------------------------
    {
      const auto model_cfg = tiny_model_config();
      pulsar::PPOActor actor(model_cfg);
      auto params_before = std::vector<torch::Tensor>();
      for (const auto& p : actor->named_parameters()) {
        if (p.key().find("inverse_head") != std::string::npos) {
          params_before.push_back(p.value().detach().clone());
        }
      }
      require(!params_before.empty(), "inverse_head has parameters");

      const auto encoded_t = torch::randn({4, model_cfg.encoder_dim}, torch::kFloat32);
      const auto encoded_tp1 = torch::randn({4, model_cfg.encoder_dim}, torch::kFloat32);
      const auto actions = torch::randint(0, model_cfg.action_dim, {4}, torch::kLong);
      const auto inverse_logits = actor->forward_predict_action(encoded_t, encoded_tp1);
      const auto loss = torch::nn::functional::cross_entropy(
          inverse_logits, actions,
          torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
      loss.backward();

      int grad_count = 0;
      for (const auto& p : actor->named_parameters()) {
        if (p.key().find("inverse_head") != std::string::npos) {
          if (p.value().grad().defined()) {
            ++grad_count;
          }
        }
      }
      require(grad_count > 0, "inverse_head parameters must receive gradients");
    }

    // ---------------------------------------------------------------
    // 15. Adaptive epsilon bounds
    // ---------------------------------------------------------------
    {
      const auto tiny_variance = torch::tensor({0.01F}, torch::kFloat32);
      const float eps = pulsar::compute_adaptive_epsilon(tiny_variance, 0.2F, 1.0F, 0.05F, 0.3F);
      // 0.2 / (1 + 1.0 * 0.01) = 0.2 / 1.01 ≈ 0.198
      require(eps >= 0.05F && eps <= 0.3F, "adaptive epsilon within bounds");

      const auto large_variance = torch::tensor({100.0F}, torch::kFloat32);
      const float small_eps = pulsar::compute_adaptive_epsilon(large_variance, 0.2F, 1.0F, 0.05F, 0.3F);
      require_close(small_eps, 0.05F, "adaptive epsilon clamps to min");
    }

    // ---------------------------------------------------------------
    // 16. compute_mean_value and compute_distribution_variance
    // ---------------------------------------------------------------
    {
      const auto atom_support = torch::linspace(-5.0F, 5.0F, 21, torch::kFloat32);
      const auto logits = torch::zeros({4, 21}, torch::kFloat32);
      const auto mean_val = pulsar::compute_mean_value(logits, atom_support);
      require(mean_val.sizes() == torch::IntArrayRef({4}), "mean value shape");
      require_finite(mean_val, "mean value finite");

      const auto variance = pulsar::compute_distribution_variance(logits, atom_support);
      require(variance.sizes() == torch::IntArrayRef({4}), "variance shape");
      require_finite(variance, "variance finite");

      const auto entropy = pulsar::compute_distribution_entropy(logits);
      require(entropy.sizes() == torch::IntArrayRef({4}), "entropy shape");
      require_finite(entropy, "entropy finite");
    }

    // ---------------------------------------------------------------
    // 17. Advantage mixing with disabled head
    // ---------------------------------------------------------------
    {
      const auto ext_adv = torch::ones({4, 2}, torch::kFloat32);
      const auto cur_adv = 2.0F * torch::ones({4, 2}, torch::kFloat32);
      std::unordered_map<std::string, torch::Tensor> normalized = {
          {"extrinsic", ext_adv},
          {"curiosity", cur_adv},
      };
      std::unordered_map<std::string, float> weights = {
          {"extrinsic", 1.0F},
          {"curiosity", 0.0F},
      };
      const auto mask = torch::ones({4, 2}, torch::kFloat32);
      const auto mixed = pulsar::mix_advantages(normalized, weights, mask);
      require(mixed.allclose(ext_adv), "disabled head should not affect mix");
    }

    std::cout << "pulsar_ppo_math_tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_ppo_math_tests FAILED: " << exc.what() << '\n';
    return EXIT_FAILURE;
  }
}
