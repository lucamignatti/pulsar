#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include <torch/cuda.h>

#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/training/ppo_math.hpp"

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void synchronize_if_cuda(const torch::Device& device) {
  if (device.is_cuda()) {
    torch::cuda::synchronize(device.index());
  }
}

pulsar::ModelConfig benchmark_model_config() {
  pulsar::ModelConfig config;
  config.observation_dim = 132;
  config.action_dim = 90;
  config.encoder_dim = 64;
  config.workspace_dim = 64;
  config.stm_slots = 8;
  config.stm_key_dim = 16;
  config.stm_value_dim = 16;
  config.ltm_slots = 8;
  config.ltm_dim = 16;
  config.controller_dim = 64;
  config.value_hidden_dim = 128;
  config.value_num_atoms = 51;
  config.value_v_min = -10.0F;
  config.value_v_max = 10.0F;
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  const int updates = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const int batch = argc > 2 ? std::max(1, std::atoi(argv[2])) : 4096;
  const torch::Device device =
      argc > 3 ? torch::Device(argv[3])
               : (torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
  const int sequence = 8;

  const pulsar::ModelConfig model_config = benchmark_model_config();
  pulsar::PPOActor actor(model_config, pulsar::GoalCriticConfig{});
  actor->to(device);
  torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(3.0e-4));

  std::cout << "device=" << device.str() << '\n';

  double policy_forward_seconds = 0.0;
  double ppo_seconds = 0.0;
  std::int64_t samples = 0;

  for (int update = 0; update < updates; ++update) {
    torch::Tensor obs = torch::randn({sequence, batch, model_config.observation_dim}, device);
    torch::Tensor starts = torch::zeros({sequence, batch}, device);
    auto state = actor->initial_state(batch, device);
    synchronize_if_cuda(device);
    auto forward_start = std::chrono::steady_clock::now();
    const pulsar::ActorSequenceOutput output = actor->forward_sequence(obs, std::move(state), starts);
    synchronize_if_cuda(device);
    policy_forward_seconds += seconds_since(forward_start);

    const torch::Tensor logits = output.policy_logits.reshape({sequence * batch, model_config.action_dim});
    const torch::Tensor masks = torch::ones(logits.sizes(), torch::TensorOptions().dtype(torch::kBool).device(device));
    const torch::Tensor actions = pulsar::sample_masked_actions(logits, masks, false);
    const torch::Tensor old_log_probs =
        torch::log_softmax(pulsar::apply_action_mask_to_logits(logits, masks), -1).gather(1, actions.unsqueeze(1)).squeeze(1);
    const torch::Tensor values = torch::zeros({sequence * batch}, device);
    const torch::Tensor rewards = torch::zeros({sequence * batch}, device);
    const torch::Tensor dones = torch::zeros({sequence * batch}, device);
    const torch::Tensor advantages = pulsar::compute_gae(
        values.reshape({sequence, batch}),
        rewards.reshape({sequence, batch}),
        dones.reshape({sequence, batch}),
        0.99F, 0.95F).reshape({sequence * batch});

    synchronize_if_cuda(device);
    auto ppo_start = std::chrono::steady_clock::now();
    const torch::Tensor current_log_probs =
        torch::log_softmax(pulsar::apply_action_mask_to_logits(logits, masks), -1).gather(1, actions.unsqueeze(1)).squeeze(1);
    const torch::Tensor policy_loss =
        pulsar::clipped_ppo_policy_loss(current_log_probs, old_log_probs.detach(), advantages, 0.2F)
            .mean();
    const torch::Tensor entropy_loss = -0.01F * pulsar::masked_action_entropy(logits, masks).mean();
    const torch::Tensor value_logits = output.value_win_logits.reshape({sequence * batch, model_config.value_num_atoms});
    const torch::Tensor value_loss = pulsar::distributional_value_loss(
        value_logits, advantages,
        model_config.value_v_min, model_config.value_v_max, model_config.value_num_atoms);
    const torch::Tensor loss = policy_loss + entropy_loss + 1.0F * value_loss;
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    synchronize_if_cuda(device);
    ppo_seconds += seconds_since(ppo_start);
    samples += sequence * batch;
  }

  const double total_seconds = std::max(policy_forward_seconds + ppo_seconds, 1.0e-9);
  std::cout << "collection_agent_steps_per_second=" << static_cast<double>(samples) / total_seconds << '\n';
  std::cout << "ppo_update_agent_steps_per_second=" << static_cast<double>(samples) / std::max(ppo_seconds, 1.0e-9) << '\n';
  std::cout << "offline_pretrain_samples_per_second=" << static_cast<double>(samples) / total_seconds << '\n';
  std::cout << "offline_pretrain_epoch_seconds=" << total_seconds << '\n';
  std::cout << "policy_forward_seconds=" << policy_forward_seconds << '\n';
  std::cout << "ppo_forward_backward_seconds=" << ppo_seconds << '\n';
  return EXIT_SUCCESS;
}
