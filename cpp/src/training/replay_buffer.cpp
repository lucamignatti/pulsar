#include "pulsar/training/replay_buffer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <torch/torch.h>

namespace pulsar {

ReplayBuffer::ReplayBuffer(
    const ReplayBufferConfig& config,
    int obs_dim,
    int action_dim,
    int goal_dim,
    const torch::Device& /*host_device*/)
    : config_(config), obs_dim_(obs_dim), action_dim_(action_dim), goal_dim_(goal_dim) {

  const int64_t cap = static_cast<int64_t>(config_.capacity);
  // Pinned memory only on CUDA systems; avoids MPS dispatch errors on macOS
  const bool use_pinned = torch::cuda::is_available();
  const auto fopt = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(use_pinned);
  const auto lopt = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(use_pinned);

  buf_obs_               = torch::zeros({cap, obs_dim_}, fopt);
  buf_next_obs_          = torch::zeros({cap, obs_dim_}, fopt);
  buf_dones_             = torch::zeros({cap}, fopt);
  buf_achieved_goals_    = torch::zeros({cap, goal_dim_}, fopt);
  buf_conditioned_goals_ = torch::zeros({cap, goal_dim_}, fopt);
  buf_actions_           = torch::zeros({cap}, lopt);
  buf_episode_ids_       = torch::zeros({cap}, lopt);
  buf_timestep_ids_      = torch::zeros({cap}, lopt);
}

void ReplayBuffer::push(
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& achieved_goal,
    const torch::Tensor& conditioned_goal,
    const torch::Tensor& next_obs,
    const torch::Tensor& dones,
    const torch::Tensor& episode_ids) {

  const int n = static_cast<int>(obs.size(0));
  if (n <= 0) return;

  // All inputs are CPU tensors (collected on CPU in trainer)
  const torch::Tensor obs_cpu   = obs.to(torch::kCPU).contiguous();
  const torch::Tensor next_cpu  = next_obs.to(torch::kCPU).contiguous();
  const torch::Tensor act_cpu   = actions.to(torch::kCPU).contiguous();
  const torch::Tensor done_cpu  = dones.to(torch::kCPU).contiguous();
  const torch::Tensor ach_cpu   = achieved_goal.to(torch::kCPU).contiguous();
  const torch::Tensor cond_cpu  = conditioned_goal.to(torch::kCPU).contiguous();
  const torch::Tensor ep_cpu    = episode_ids.to(torch::kCPU).contiguous();

  const int cap = config_.capacity;

  // Check if we can write contiguously (no wraparound)
  if (head_ + n <= cap) {
    const torch::Tensor slots = torch::arange(head_, head_ + n, torch::TensorOptions().dtype(torch::kInt64));
    buf_obs_.index_put_({slots}, obs_cpu);
    buf_next_obs_.index_put_({slots}, next_cpu);
    buf_actions_.index_put_({slots}, act_cpu);
    buf_dones_.index_put_({slots}, done_cpu);
    buf_achieved_goals_.index_put_({slots}, ach_cpu);
    buf_conditioned_goals_.index_put_({slots}, cond_cpu);
    buf_episode_ids_.index_put_({slots}, ep_cpu);
    // Timestep IDs: use a global counter for ordering within episodes
    const torch::Tensor ts = torch::arange(
        static_cast<int64_t>(filled_),
        static_cast<int64_t>(filled_ + n),
        torch::TensorOptions().dtype(torch::kInt64));
    buf_timestep_ids_.index_put_({slots}, ts);
  } else {
    // Wrap around: write in two chunks
    const int first = cap - head_;
    const int second = n - first;

    if (first > 0) {
      const torch::Tensor s1 = torch::arange(head_, head_ + first, torch::TensorOptions().dtype(torch::kInt64));
      buf_obs_.index_put_({s1}, obs_cpu.narrow(0, 0, first));
      buf_next_obs_.index_put_({s1}, next_cpu.narrow(0, 0, first));
      buf_actions_.index_put_({s1}, act_cpu.narrow(0, 0, first));
      buf_dones_.index_put_({s1}, done_cpu.narrow(0, 0, first));
      buf_achieved_goals_.index_put_({s1}, ach_cpu.narrow(0, 0, first));
      buf_conditioned_goals_.index_put_({s1}, cond_cpu.narrow(0, 0, first));
      buf_episode_ids_.index_put_({s1}, ep_cpu.narrow(0, 0, first));
      const torch::Tensor ts1 = torch::arange(
          static_cast<int64_t>(filled_), static_cast<int64_t>(filled_ + first),
          torch::TensorOptions().dtype(torch::kInt64));
      buf_timestep_ids_.index_put_({s1}, ts1);
    }
    if (second > 0) {
      const torch::Tensor s2 = torch::arange(0, second, torch::TensorOptions().dtype(torch::kInt64));
      buf_obs_.index_put_({s2}, obs_cpu.narrow(0, first, second));
      buf_next_obs_.index_put_({s2}, next_cpu.narrow(0, first, second));
      buf_actions_.index_put_({s2}, act_cpu.narrow(0, first, second));
      buf_dones_.index_put_({s2}, done_cpu.narrow(0, first, second));
      buf_achieved_goals_.index_put_({s2}, ach_cpu.narrow(0, first, second));
      buf_conditioned_goals_.index_put_({s2}, cond_cpu.narrow(0, first, second));
      buf_episode_ids_.index_put_({s2}, ep_cpu.narrow(0, first, second));
      const torch::Tensor ts2 = torch::arange(
          static_cast<int64_t>(filled_ + first), static_cast<int64_t>(filled_ + n),
          torch::TensorOptions().dtype(torch::kInt64));
      buf_timestep_ids_.index_put_({s2}, ts2);
    }
  }

  head_ = (head_ + n) % cap;
  filled_ = std::min(filled_ + n, cap);
}

int ReplayBuffer::size() const { return filled_; }

bool ReplayBuffer::ready() const {
  return filled_ >= config_.min_fill_before_sampling;
}

torch::Tensor ReplayBuffer::compute_goal_reward(
    const torch::Tensor& achieved_goals,
    const torch::Tensor& target_goals) const {
  torch::Tensor diff = achieved_goals - target_goals;
  torch::Tensor dist_sq;
  if (config_.epsilon_position_only && goal_dim_ >= 3) {
    dist_sq = diff.narrow(-1, 0, 3).square().sum(-1);
  } else {
    dist_sq = diff.square().sum(-1);
  }
  const float eps = config_.goal_reach_epsilon;
  return (dist_sq < (eps * eps)).to(torch::kFloat32);
}

std::pair<torch::Tensor, torch::Tensor> ReplayBuffer::her_relabel_batch(
    const torch::Tensor& achieved_goals,
    const torch::Tensor& conditioned_goals,
    const torch::Tensor& episode_ids,
    const torch::Tensor& timestep_ids,
    const torch::Tensor& sampled_indices) {

  const int N = static_cast<int>(achieved_goals.size(0));
  torch::Tensor final_goals = conditioned_goals.clone();

  const torch::Tensor relabel_mask = (torch::rand({N}) < config_.her_relabel_fraction);
  const auto* relabel_ptr  = relabel_mask.data_ptr<bool>();
  const auto* ep_ptr       = episode_ids.data_ptr<int64_t>();
  const auto* ts_ptr       = timestep_ids.data_ptr<int64_t>();
  const auto* idx_ptr      = sampled_indices.data_ptr<int64_t>();
  const auto* buf_ep_ptr   = buf_episode_ids_.data_ptr<int64_t>();
  const auto* buf_ts_ptr   = buf_timestep_ids_.data_ptr<int64_t>();
  auto*       goals_ptr    = final_goals.data_ptr<float>();
  const auto* buf_ach_ptr  = buf_achieved_goals_.data_ptr<float>();
  const int   cap          = std::min(filled_, config_.capacity);
  const int   gd           = goal_dim_;

  for (int i = 0; i < N; ++i) {
    if (!relabel_ptr[i]) continue;

    const int64_t ep_id = ep_ptr[i];
    const int64_t ts_id = ts_ptr[i];
    const int base_slot = static_cast<int>(idx_ptr[i]);

    // Scan forward for a valid future slot from the same episode
    int chosen = -1;
    const int scan_limit = std::min(cap, config_.her_future_k * 32);
    for (int j = 1; j <= scan_limit; ++j) {
      const int cand = (base_slot + j) % cap;
      if (buf_ep_ptr[cand] == ep_id && buf_ts_ptr[cand] > ts_id) {
        chosen = cand;
        // Accept first candidate (could randomize, but first is fine for HER)
        break;
      }
    }
    if (chosen < 0) continue;

    // Replace conditioned goal with future achieved goal
    for (int d = 0; d < gd; ++d) {
      goals_ptr[i * gd + d] = buf_ach_ptr[chosen * gd + d];
    }
  }

  const torch::Tensor rewards = compute_goal_reward(achieved_goals, final_goals);
  return {final_goals, rewards};
}

ReplayBuffer::TransitionBatch ReplayBuffer::sample_transitions(int batch_size) {
  TORCH_CHECK(ready(), "ReplayBuffer not ready");

  const int cap = std::min(filled_, config_.capacity);
  const torch::Tensor indices = torch::randint(0, cap, {batch_size},
      torch::TensorOptions().dtype(torch::kInt64));

  const torch::Tensor obs      = buf_obs_.index_select(0, indices);
  const torch::Tensor actions  = buf_actions_.index_select(0, indices);
  const torch::Tensor next_obs = buf_next_obs_.index_select(0, indices);
  const torch::Tensor dones    = buf_dones_.index_select(0, indices);
  const torch::Tensor achieved = buf_achieved_goals_.index_select(0, indices);
  const torch::Tensor cond     = buf_conditioned_goals_.index_select(0, indices);
  const torch::Tensor ep_ids   = buf_episode_ids_.index_select(0, indices);
  const torch::Tensor ts_ids   = buf_timestep_ids_.index_select(0, indices);

  auto [goals, rewards] = her_relabel_batch(achieved, cond, ep_ids, ts_ids, indices);

  TransitionBatch batch;
  batch.obs      = obs;
  batch.actions  = actions;
  batch.rewards  = rewards;
  batch.next_obs = next_obs;
  batch.dones    = dones;
  batch.goals    = goals;
  return batch;
}

ReplayBuffer::SegmentBatch ReplayBuffer::sample_segments(int num_segments) {
  TORCH_CHECK(ready(), "ReplayBuffer not ready");

  const int L = config_.segment_length;
  const int cap = std::min(filled_, config_.capacity);
  // Leave margin so we don't sample partially-overwritten segments
  const int max_start = std::max(1, cap - L - 1);

  const torch::Tensor start_indices = torch::randint(0, max_start, {num_segments},
      torch::TensorOptions().dtype(torch::kInt64));
  const auto* starts = start_indices.data_ptr<int64_t>();

  std::vector<torch::Tensor> obs_b, act_b, rew_b, next_b, done_b, goal_b, ep_end_b;

  for (int b = 0; b < num_segments; ++b) {
    const int start = static_cast<int>(starts[b]);

    // Build slot index for this segment
    torch::Tensor slots = torch::arange(start, start + L, torch::TensorOptions().dtype(torch::kInt64));
    // Modulo for wraparound
    slots = slots % cap;

    const torch::Tensor seg_obs      = buf_obs_.index_select(0, slots);
    const torch::Tensor seg_next     = buf_next_obs_.index_select(0, slots);
    const torch::Tensor seg_actions  = buf_actions_.index_select(0, slots);
    const torch::Tensor seg_dones    = buf_dones_.index_select(0, slots);
    const torch::Tensor seg_achieved = buf_achieved_goals_.index_select(0, slots);
    const torch::Tensor seg_ep_ids   = buf_episode_ids_.index_select(0, slots);

    // Episode boundary detection: 1 at position l if episode changed from l-1 to l
    const torch::Tensor ep_ends_raw = torch::zeros({L}, torch::TensorOptions().dtype(torch::kFloat32));
    // Use diff of episode_ids; non-zero means boundary
    if (L > 1) {
      const torch::Tensor ep_diff = (seg_ep_ids.narrow(0, 1, L - 1) != seg_ep_ids.narrow(0, 0, L - 1)).to(torch::kFloat32);
      ep_ends_raw.narrow(0, 1, L - 1).copy_(ep_diff);
    }
    const torch::Tensor ep_ends = ep_ends_raw;

    // Segment-level HER: all steps get the same goal
    torch::Tensor seg_goals = buf_conditioned_goals_.index_select(0, slots).clone();
    torch::Tensor seg_rewards = compute_goal_reward(seg_achieved, seg_goals);

    if (torch::rand({1}).item<float>() < config_.her_relabel_fraction) {
      // Try to find a future state from beyond the segment end, same episode as last slot
      const int64_t last_ep = seg_ep_ids[L - 1].item<int64_t>();
      const int future_start = (start + L) % cap;
      int future_slot = -1;
      const int scan = std::min(cap / 4, 512);
      for (int j = 0; j < scan; ++j) {
        const int cand = (future_start + j) % cap;
        if (buf_episode_ids_[cand].item<int64_t>() == last_ep) {
          future_slot = cand;
          break;
        }
      }
      if (future_slot >= 0) {
        const torch::Tensor future_goal = buf_achieved_goals_[future_slot]
            .unsqueeze(0).expand({L, goal_dim_}).clone();
        seg_goals = future_goal;
        seg_rewards = compute_goal_reward(seg_achieved, seg_goals);
      }
    }

    obs_b.push_back(seg_obs.unsqueeze(1));
    act_b.push_back(seg_actions.unsqueeze(1));
    rew_b.push_back(seg_rewards.unsqueeze(1));
    next_b.push_back(seg_next.unsqueeze(1));
    done_b.push_back(seg_dones.unsqueeze(1));
    goal_b.push_back(seg_goals.unsqueeze(1));
    ep_end_b.push_back(ep_ends.unsqueeze(1));
  }

  SegmentBatch batch;
  batch.obs          = torch::cat(obs_b, 1);      // [L, num_segments, obs_dim]
  batch.actions      = torch::cat(act_b, 1);
  batch.rewards      = torch::cat(rew_b, 1);
  batch.next_obs     = torch::cat(next_b, 1);
  batch.dones        = torch::cat(done_b, 1);
  batch.goals        = torch::cat(goal_b, 1);
  batch.episode_ends = torch::cat(ep_end_b, 1);
  return batch;
}

}  // namespace pulsar

#endif
