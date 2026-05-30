#include "pulsar/training/replay_buffer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <stdexcept>
#include <torch/torch.h>

namespace pulsar {

ReplayBuffer::ReplayBuffer(
    const ReplayBufferConfig& config,
    int total_agents,
    int obs_dim,
    int action_dim,
    int goal_dim,
    int car_goal_dim,
    const torch::Device& /*host_device*/)
    : config_(config),
      total_agents_(total_agents),
      obs_dim_(obs_dim),
      action_dim_(action_dim),
      goal_dim_(goal_dim),
      car_goal_dim_(car_goal_dim),
      max_ep_len_(config.max_episode_length) {

  const int64_t E = static_cast<int64_t>(config_.max_episodes);
  const int64_t T = static_cast<int64_t>(max_ep_len_);
  const int64_t A = static_cast<int64_t>(total_agents_);
  const bool use_pinned = torch::cuda::is_available();

  const auto fpin = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(use_pinned);
  const auto lpin = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(use_pinned);
  const auto ipin = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(use_pinned);

  ep_obs_        = torch::zeros({E, T, obs_dim_}, fpin);
  ep_actions_    = torch::zeros({E, T}, lpin);
  ep_ball_goals_ = torch::zeros({E, T, goal_dim_}, fpin);
  ep_car_goals_  = torch::zeros({E, T, car_goal_dim_}, fpin);
  ep_lengths_    = torch::zeros({E}, ipin);

  // Accumulation buffers — CPU only, no pinning needed (written step-by-step)
  const auto fcpu = torch::TensorOptions().dtype(torch::kFloat32);
  const auto lcpu = torch::TensorOptions().dtype(torch::kInt64);
  const auto icpu = torch::TensorOptions().dtype(torch::kInt32);

  acc_obs_        = torch::zeros({A, T, obs_dim_}, fcpu);
  acc_actions_    = torch::zeros({A, T}, lcpu);
  acc_ball_goals_ = torch::zeros({A, T, goal_dim_}, fcpu);
  acc_car_goals_  = torch::zeros({A, T, car_goal_dim_}, fcpu);
  acc_step_       = torch::zeros({A}, icpu);
}

void ReplayBuffer::push_step(
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& ball_goals,
    const torch::Tensor& car_goals,
    const torch::Tensor& dones) {

  const int A = total_agents_;
  TORCH_CHECK(obs.size(0) == A, "push_step: obs batch size mismatch");

  // All inputs assumed CPU already (collector writes to host tensors)
  const torch::Tensor obs_c  = obs.to(torch::kCPU).contiguous();
  const torch::Tensor act_c  = actions.to(torch::kCPU).contiguous();
  const torch::Tensor bg_c   = ball_goals.to(torch::kCPU).contiguous();
  const torch::Tensor cg_c   = car_goals.to(torch::kCPU).contiguous();
  const torch::Tensor done_c = dones.to(torch::kCPU).contiguous();

  const auto* done_ptr = done_c.data_ptr<float>();
  auto* step_ptr       = acc_step_.data_ptr<int32_t>();

  for (int i = 0; i < A; ++i) {
    const int s = step_ptr[i];
    if (s < max_ep_len_) {
      acc_obs_[i][s]        = obs_c[i];
      acc_actions_[i][s]    = act_c[i];
      acc_ball_goals_[i][s] = bg_c[i];
      acc_car_goals_[i][s]  = cg_c[i];
      step_ptr[i] = s + 1;
    }
    if (done_ptr[i] > 0.5F) {
      {
        std::lock_guard<std::mutex> lock(ep_mutex_);
        flush_agent_locked(i);
      }
      step_ptr[i] = 0;
    }
  }
}

void ReplayBuffer::flush_agent_locked(int i) {
  const int len = acc_step_.data_ptr<int32_t>()[i];
  if (len < 2) return;  // need at least 2 steps for a valid (t, t') pair

  const int slot = head_;
  ep_obs_[slot].narrow(0, 0, len).copy_(acc_obs_[i].narrow(0, 0, len));
  ep_actions_[slot].narrow(0, 0, len).copy_(acc_actions_[i].narrow(0, 0, len));
  ep_ball_goals_[slot].narrow(0, 0, len).copy_(acc_ball_goals_[i].narrow(0, 0, len));
  ep_car_goals_[slot].narrow(0, 0, len).copy_(acc_car_goals_[i].narrow(0, 0, len));
  ep_lengths_.data_ptr<int32_t>()[slot] = len;

  head_   = (head_ + 1) % config_.max_episodes;
  filled_ = std::min(filled_ + 1, config_.max_episodes);
}

int  ReplayBuffer::size()  const { return filled_; }
bool ReplayBuffer::ready() const { return filled_ >= config_.min_episodes_before_sampling; }

ReplayBuffer::CRLBatch ReplayBuffer::sample_crl_batch(int batch_size) {
  TORCH_CHECK(ready(), "ReplayBuffer not ready for sampling");

  const auto lopt = torch::TensorOptions().dtype(torch::kInt64);
  const auto fopt = torch::TensorOptions().dtype(torch::kFloat32);

  torch::Tensor ep_indices, lengths_i, t, t_prime;
  {
    std::lock_guard<std::mutex> lock(ep_mutex_);

    // Sample episode indices
    ep_indices = torch::randint(0, filled_, {batch_size}, lopt);  // [B]
    lengths_i  = ep_lengths_.index_select(0, ep_indices.to(torch::kInt64))
                     .to(torch::kInt64);  // [B]

    // Sample t in [0, length-2] so there is always a future step
    const torch::Tensor t_max = (lengths_i - 2).clamp_min(0);  // [B]
    const torch::Tensor t_rand = torch::rand({batch_size}, fopt);
    t = (t_rand * (t_max + 1).to(torch::kFloat32)).to(torch::kInt64).clamp_max(t_max);  // [B]

    // Sample t' in [t+1, min(t+max_future_horizon, length-1)]
    const torch::Tensor t_fut_max =
        (t + static_cast<int64_t>(config_.max_future_horizon)).min(lengths_i - 1);  // [B]
    const torch::Tensor t_fut_range = (t_fut_max - t).clamp_min(1);  // [B]
    const torch::Tensor t_rand2 = torch::rand({batch_size}, fopt);
    t_prime = t + 1 + (t_rand2 * t_fut_range.to(torch::kFloat32)).to(torch::kInt64)
                  .clamp(torch::zeros({batch_size}, lopt), t_fut_range - 1);  // [B]
  }

  const int64_t T = static_cast<int64_t>(max_ep_len_);

  // Flat indices into [max_episodes * max_ep_len, *] view
  const torch::Tensor flat_t  = ep_indices * T + t;        // [B]
  const torch::Tensor flat_tp = ep_indices * T + t_prime;  // [B]

  CRLBatch batch;
  batch.obs = ep_obs_.view({-1, obs_dim_}).index_select(0, flat_t);
  batch.actions = ep_actions_.view({-1}).index_select(0, flat_t);
  batch.future_ball_goals = ep_ball_goals_.view({-1, goal_dim_}).index_select(0, flat_tp);
  batch.future_car_goals  = ep_car_goals_.view({-1, car_goal_dim_}).index_select(0, flat_tp);
  return batch;
}

}  // namespace pulsar

#endif
