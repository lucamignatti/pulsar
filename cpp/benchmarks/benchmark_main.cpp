#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/env/done.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/rl/action_table.hpp"
#include "pulsar/training/batched_rocketsim_collector.hpp"
#include "pulsar/training/ppo_math.hpp"

#ifdef PULSAR_HAS_TORCH
#include <torch/cuda.h>
#endif

namespace {

#ifdef PULSAR_HAS_TORCH

struct InferenceTimings {
  double obs_copy_seconds = 0.0;
  double normalizer_seconds = 0.0;
  double policy_forward_seconds = 0.0;
  double action_decode_seconds = 0.0;
  double reward_model_seconds = 0.0;
};

torch::Device resolve_runtime_device(const std::string& device_name) {
  if (device_name.empty()) {
    if (torch::cuda::is_available()) {
      return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
  }
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

void maybe_synchronize(const torch::Device& device) {
  if (device.is_cuda()) {
    torch::cuda::synchronize(device.index());
  }
}

torch::Tensor ngp_scalar(const torch::Tensor& logits) {
  const torch::Tensor probs = torch::softmax(logits, -1);
  return probs.select(-1, 0) - probs.select(-1, 1);
}

torch::Tensor actions_to_cpu(const torch::Tensor& actions) {
  return actions.contiguous().to(torch::kCPU);
}

#endif

}  // namespace

int main(int argc, char** argv) {
  pulsar::ExperimentConfig config;
  const int num_envs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const std::size_t collection_workers =
      argc > 2 ? static_cast<std::size_t>(std::max(0, std::atoi(argv[2]))) : 0;
  const bool pin_host_memory = argc > 3 ? std::atoi(argv[3]) != 0 : false;
  const std::string device_name = argc > 4 ? argv[4] : std::string{};
  config.ppo.num_envs = num_envs;
  config.ppo.collection_workers = static_cast<int>(collection_workers);

  auto obs_builder = std::make_shared<pulsar::PulsarObsBuilder>(config.env);
  auto action_parser =
      std::make_shared<pulsar::DiscreteActionParser>(pulsar::ControllerActionTable(config.action_table));
  auto done_condition = std::make_shared<pulsar::SimpleDoneCondition>(config.env);
  pulsar::BatchedRocketSimCollector collector(
      config,
      obs_builder,
      action_parser,
      done_condition,
      pin_host_memory);
  std::vector<pulsar::ControllerState> actions(
      collector.total_agents(),
      pulsar::ControllerState{.throttle = 1.0F, .boost = true});

  const int warmup_steps = 256;
  const int steps = 10000;
  const int trainer_warmup_steps =
#ifdef PULSAR_HAS_TORCH
      resolve_runtime_device(device_name).is_cuda() ? warmup_steps : 64;
#else
      64;
#endif
  const int trainer_steps =
#ifdef PULSAR_HAS_TORCH
      resolve_runtime_device(device_name).is_cuda() ? steps : 1024;
#else
      1024;
#endif

  auto run = [&]() {
    pulsar::CollectorTimings timings{};
    for (int i = 0; i < warmup_steps; ++i) {
      collector.step(actions, &timings);
    }

    const auto start = std::chrono::steady_clock::now();
    timings = {};
    for (int i = 0; i < steps; ++i) {
      collector.step(actions, &timings);
    }
    return std::pair{
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        timings,
    };
  };

#ifdef PULSAR_HAS_TORCH
  const torch::Device device = resolve_runtime_device(device_name);
  pulsar::ObservationNormalizer normalizer(config.model.observation_dim);
  pulsar::ObservationNormalizer ngp_normalizer(config.model.observation_dim);
  normalizer.to(device);
  ngp_normalizer.to(device);

  pulsar::SharedActorCritic model(config.model, config.ppo);
  pulsar::SharedActorCritic ngp_model(config.model, config.ppo);
  model->to(device);
  ngp_model->to(device);
  model->eval();
  ngp_model->eval();

  auto run_trainer_like = [&]() {
    pulsar::CollectorTimings timings{};
    InferenceTimings inference{};
    pulsar::ContinuumState policy_state = model->initial_state(static_cast<std::int64_t>(collector.total_agents()), device);
    pulsar::ContinuumState ngp_state = ngp_model->initial_state(static_cast<std::int64_t>(collector.total_agents()), device);

    for (int i = 0; i < trainer_warmup_steps; ++i) {
      torch::Tensor raw_obs_host = collector.host_observations();
      const auto copy_start = std::chrono::steady_clock::now();
      const torch::Tensor raw_obs = raw_obs_host.to(device, pin_host_memory);
      const torch::Tensor episode_starts = collector.host_episode_starts().to(device, pin_host_memory);
      const torch::Tensor action_masks = collector.host_action_masks().to(device, pin_host_memory).to(torch::kBool);
      inference.obs_copy_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_start).count();

      torch::Tensor chosen_actions;
      pulsar::ContinuumState next_ngp_state;
      {
        torch::NoGradGuard no_grad;
        const torch::Tensor normalized_ngp_obs = ngp_normalizer.normalize(raw_obs);
        const pulsar::PolicyOutput ngp_output =
            ngp_model->forward_step(normalized_ngp_obs, std::move(ngp_state), episode_starts);
        next_ngp_state = std::move(ngp_output.state);

        normalizer.update(raw_obs);
        const torch::Tensor normalized_obs = normalizer.normalize(raw_obs);
        const pulsar::PolicyOutput output =
            model->forward_step(normalized_obs, std::move(policy_state), episode_starts);
        policy_state = std::move(output.state);
        chosen_actions = pulsar::sample_masked_actions(output.policy_logits, action_masks, false, nullptr);
      }

      const torch::Tensor action_indices_cpu = actions_to_cpu(chosen_actions);
      collector.step(
          std::span<const std::int64_t>(
              action_indices_cpu.data_ptr<std::int64_t>(),
              static_cast<std::size_t>(action_indices_cpu.numel())),
          &timings);

      const auto reward_copy_start = std::chrono::steady_clock::now();
      const torch::Tensor dones = collector.host_dones().to(device, pin_host_memory);
      const torch::Tensor post_step_obs = collector.host_observations().to(device, pin_host_memory);
      inference.obs_copy_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - reward_copy_start).count();
      const torch::Tensor zero_starts = torch::zeros_like(dones);
      {
        torch::NoGradGuard no_grad;
        const torch::Tensor normalized_post_step_obs = ngp_normalizer.normalize(post_step_obs);
        const pulsar::PolicyOutput ngp_output =
            ngp_model->forward_step(normalized_post_step_obs, std::move(next_ngp_state), zero_starts);
        ngp_state = std::move(ngp_output.state);
      }
    }

    maybe_synchronize(device);
    const auto start = std::chrono::steady_clock::now();
    timings = {};
    inference = {};

    for (int i = 0; i < trainer_steps; ++i) {
      torch::Tensor raw_obs_host = collector.host_observations();
      const auto copy_start = std::chrono::steady_clock::now();
      const torch::Tensor raw_obs = raw_obs_host.to(device, pin_host_memory);
      const torch::Tensor episode_starts = collector.host_episode_starts().to(device, pin_host_memory);
      const torch::Tensor action_masks = collector.host_action_masks().to(device, pin_host_memory).to(torch::kBool);
      inference.obs_copy_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_start).count();

      torch::Tensor chosen_actions;
      torch::Tensor ngp_prev_scalar;
      pulsar::ContinuumState next_ngp_state;

      {
        torch::NoGradGuard no_grad;
        const auto ngp_start = std::chrono::steady_clock::now();
        const torch::Tensor normalized_ngp_obs = ngp_normalizer.normalize(raw_obs);
        const pulsar::PolicyOutput ngp_output =
            ngp_model->forward_step(normalized_ngp_obs, std::move(ngp_state), episode_starts);
        ngp_prev_scalar = ngp_scalar(ngp_output.next_goal_logits);
        next_ngp_state = std::move(ngp_output.state);
        inference.reward_model_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ngp_start).count();

        const auto normalizer_start = std::chrono::steady_clock::now();
        normalizer.update(raw_obs);
        const torch::Tensor normalized_obs = normalizer.normalize(raw_obs);
        inference.normalizer_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - normalizer_start).count();

        const auto policy_start = std::chrono::steady_clock::now();
        const pulsar::PolicyOutput output =
            model->forward_step(normalized_obs, std::move(policy_state), episode_starts);
        policy_state = std::move(output.state);
        chosen_actions = pulsar::sample_masked_actions(output.policy_logits, action_masks, false, nullptr);
        inference.policy_forward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();
      }

      const auto decode_start = std::chrono::steady_clock::now();
      const torch::Tensor action_indices_cpu = actions_to_cpu(chosen_actions);
      collector.step(
          std::span<const std::int64_t>(
              action_indices_cpu.data_ptr<std::int64_t>(),
              static_cast<std::size_t>(action_indices_cpu.numel())),
          &timings);
      inference.action_decode_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

      const auto reward_copy_start = std::chrono::steady_clock::now();
      const torch::Tensor dones = collector.host_dones().to(device, pin_host_memory);
      const torch::Tensor post_step_obs = collector.host_observations().to(device, pin_host_memory);
      inference.obs_copy_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - reward_copy_start).count();
      const torch::Tensor zero_starts = torch::zeros_like(dones);
      {
        torch::NoGradGuard no_grad;
        const auto reward_start = std::chrono::steady_clock::now();
        const torch::Tensor normalized_post_step_obs = ngp_normalizer.normalize(post_step_obs);
        const pulsar::PolicyOutput ngp_output =
            ngp_model->forward_step(normalized_post_step_obs, std::move(next_ngp_state), zero_starts);
        const torch::Tensor ngp_current_scalar = ngp_scalar(ngp_output.next_goal_logits);
        (void)(ngp_current_scalar - ngp_prev_scalar);
        ngp_state = std::move(ngp_output.state);
        inference.reward_model_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - reward_start).count();
      }
    }
    maybe_synchronize(device);
    return std::tuple{
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        timings,
        inference,
    };
  };
#endif

  const auto [collection_seconds, collection_timings] = run();
#ifdef PULSAR_HAS_TORCH
  const auto [trainer_seconds, trainer_timings, inference_timings] = run_trainer_like();
#endif
  const double step_env_steps_per_second =
      collection_timings.env_step_seconds > 0.0
          ? static_cast<double>(steps * num_envs) / collection_timings.env_step_seconds
          : 0.0;
  const double collection_env_steps_per_second = static_cast<double>(steps * num_envs) / collection_seconds;
  const double collection_agent_steps_per_second =
      collection_env_steps_per_second * static_cast<double>(collector.total_agents()) / static_cast<double>(num_envs);
  const double collection_sim_ticks_per_second =
      collection_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double collection_agent_ticks_per_second =
      collection_agent_steps_per_second * static_cast<double>(config.env.tick_skip);
#ifdef PULSAR_HAS_TORCH
  const double trainer_env_steps_per_second = static_cast<double>(trainer_steps * num_envs) / trainer_seconds;
  const double trainer_agent_steps_per_second =
      trainer_env_steps_per_second * static_cast<double>(collector.total_agents()) / static_cast<double>(num_envs);
  const double trainer_sim_ticks_per_second =
      trainer_env_steps_per_second * static_cast<double>(config.env.tick_skip);
  const double trainer_agent_ticks_per_second =
      trainer_agent_steps_per_second * static_cast<double>(config.env.tick_skip);
#endif

  std::cout << "backend="
#ifdef PULSAR_HAS_ROCKETSIM
            << "rocketsim"
#else
            << "placeholder"
#endif
            << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "device=" << device.str() << '\n';
#else
  std::cout << "device=cpu\n";
#endif
  std::cout << "num_envs=" << num_envs << '\n';
  std::cout << "collection_workers=" << collection_workers << '\n';
  std::cout << "pin_host_memory=" << (pin_host_memory ? 1 : 0) << '\n';
  std::cout << "step_only_env_steps_per_second=" << step_env_steps_per_second << '\n';
  std::cout << "collection_env_steps_per_second=" << collection_env_steps_per_second << '\n';
  std::cout << "collection_agent_steps_per_second=" << collection_agent_steps_per_second << '\n';
  std::cout << "collection_sim_ticks_per_second=" << collection_sim_ticks_per_second << '\n';
  std::cout << "collection_agent_ticks_per_second=" << collection_agent_ticks_per_second << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "trainer_like_env_steps_per_second=" << trainer_env_steps_per_second << '\n';
  std::cout << "trainer_like_agent_steps_per_second=" << trainer_agent_steps_per_second << '\n';
  std::cout << "trainer_like_sim_ticks_per_second=" << trainer_sim_ticks_per_second << '\n';
  std::cout << "trainer_like_agent_ticks_per_second=" << trainer_agent_ticks_per_second << '\n';
  std::cout << "trainer_like_steps=" << trainer_steps << '\n';
#endif
  std::cout << "agents=" << collector.total_agents() / static_cast<std::size_t>(num_envs) << '\n';
  std::cout << "tick_skip=" << config.env.tick_skip << '\n';
  std::cout << "obs_dim=" << obs_builder->obs_dim() << '\n';
  std::cout << "obs_build_seconds=" << collection_timings.obs_build_seconds << '\n';
  std::cout << "mask_build_seconds=" << collection_timings.mask_build_seconds << '\n';
  std::cout << "env_step_seconds=" << collection_timings.env_step_seconds << '\n';
  std::cout << "done_reset_seconds=" << collection_timings.done_reset_seconds << '\n';
  std::cout << "step_only_env_step_seconds=" << collection_timings.env_step_seconds << '\n';
#ifdef PULSAR_HAS_TORCH
  std::cout << "trainer_like_obs_build_seconds=" << trainer_timings.obs_build_seconds << '\n';
  std::cout << "trainer_like_mask_build_seconds=" << trainer_timings.mask_build_seconds << '\n';
  std::cout << "trainer_like_env_step_seconds=" << trainer_timings.env_step_seconds << '\n';
  std::cout << "trainer_like_done_reset_seconds=" << trainer_timings.done_reset_seconds << '\n';
  std::cout << "trainer_like_obs_copy_seconds=" << inference_timings.obs_copy_seconds << '\n';
  std::cout << "trainer_like_normalizer_seconds=" << inference_timings.normalizer_seconds << '\n';
  std::cout << "trainer_like_policy_forward_seconds=" << inference_timings.policy_forward_seconds << '\n';
  std::cout << "trainer_like_action_decode_seconds=" << inference_timings.action_decode_seconds << '\n';
  std::cout << "trainer_like_reward_model_seconds=" << inference_timings.reward_model_seconds << '\n';
#endif
  return 0;
}
