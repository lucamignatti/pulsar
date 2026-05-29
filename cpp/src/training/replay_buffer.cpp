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

  const int N    = static_cast<int>(achieved_goals.size(0));
  const int cap  = std::min(filled_, config_.capacity);
  // K candidates per sample: scan K slots forward for a future same-episode step
  const int K    = std::min(cap, config_.her_future_k * 16);

  // Build future_slots[N, K]: for sample i, check slots (idx[i]+1), ..., (idx[i]+K) mod cap
  const torch::Tensor offsets = torch::arange(1, K + 1,
      torch::TensorOptions().dtype(torch::kInt64));  // [K]
  // future_slots: [N, K]
  const torch::Tensor future_slots =
      (sampled_indices.unsqueeze(1) + offsets.unsqueeze(0)) % cap;

  // Gather episode_ids and timestep_ids for all candidate slots (one index_select)
  const torch::Tensor fut_ep = buf_episode_ids_
      .index_select(0, future_slots.flatten()).reshape({N, K});  // [N, K]
  const torch::Tensor fut_ts = buf_timestep_ids_
      .index_select(0, future_slots.flatten()).reshape({N, K});  // [N, K]

  // Valid: same episode AND strictly later timestep
  const torch::Tensor same_ep  = (fut_ep == episode_ids.unsqueeze(1));   // [N, K] bool
  const torch::Tensor future_t = (fut_ts > timestep_ids.unsqueeze(1));   // [N, K] bool
  const torch::Tensor valid    = same_ep & future_t;                      // [N, K] bool

  const torch::Tensor has_valid = valid.any(1);                           // [N] bool
  // First valid k index per sample (argmax of bool tensor along dim=1)
  const torch::Tensor first_valid = valid.to(torch::kFloat32).argmax(1);  // [N]

  // Gather the chosen future slots
  const torch::Tensor chosen_slots = future_slots.gather(
      1, first_valid.unsqueeze(1)).squeeze(1);  // [N]

  // Gather future achieved goals: [N, goal_dim]
  const torch::Tensor future_achieved = buf_achieved_goals_
      .index_select(0, chosen_slots);  // [N, goal_dim]

  // HER relabel mask: has_valid AND random draw
  const torch::Tensor relabel = has_valid &
      (torch::rand({N}) < config_.her_relabel_fraction);  // [N] bool

  // Apply relabeling
  const torch::Tensor final_goals = torch::where(
      relabel.unsqueeze(1).expand({N, goal_dim_}),
      future_achieved,
      conditioned_goals);

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

  const int L   = config_.segment_length;
  const int B   = num_segments;
  const int cap = std::min(filled_, config_.capacity);
  const int max_start = std::max(1, cap - L - 1);
  const auto lopt = torch::TensorOptions().dtype(torch::kInt64);

  // Sample start positions for each segment: [B]
  const torch::Tensor starts = torch::randint(0, max_start, {B}, lopt);

  // Build all slots at once: [B, L] → index all buffer arrays in one shot
  // slots[b, l] = (starts[b] + l) % cap
  const torch::Tensor l_range = torch::arange(L, lopt).unsqueeze(0).expand({B, L});  // [B, L]
  const torch::Tensor all_slots = (starts.unsqueeze(1) + l_range) % cap;              // [B, L]
  const torch::Tensor flat_slots = all_slots.flatten();                                // [B*L]

  // Single index_select per buffer array — avoids B separate calls
  const torch::Tensor all_obs      = buf_obs_.index_select(0, flat_slots).reshape({B, L, obs_dim_});
  const torch::Tensor all_next     = buf_next_obs_.index_select(0, flat_slots).reshape({B, L, obs_dim_});
  const torch::Tensor all_actions  = buf_actions_.index_select(0, flat_slots).reshape({B, L});
  const torch::Tensor all_dones    = buf_dones_.index_select(0, flat_slots).reshape({B, L});
  const torch::Tensor all_achieved = buf_achieved_goals_.index_select(0, flat_slots).reshape({B, L, goal_dim_});
  const torch::Tensor all_ep_ids   = buf_episode_ids_.index_select(0, flat_slots).reshape({B, L});
  torch::Tensor all_goals = buf_conditioned_goals_.index_select(0, flat_slots).reshape({B, L, goal_dim_}).clone();

  // Episode boundary detection — vectorized over all [B, L]
  // ep_ends[b, l] = 1 if ep_ids[b, l] != ep_ids[b, l-1]
  torch::Tensor ep_ends = torch::zeros({B, L}, all_dones.options());
  if (L > 1) {
    ep_ends.narrow(1, 1, L - 1) = (all_ep_ids.narrow(1, 1, L - 1) !=
                                    all_ep_ids.narrow(1, 0, L - 1)).to(torch::kFloat32);
  }

  // Segment-level HER: for her_relabel_fraction of segments, replace all steps' goal
  // with a future achieved_goal from beyond the segment end.
  const torch::Tensor do_relabel = (torch::rand({B}) < config_.her_relabel_fraction);  // [B] bool
  if (do_relabel.any().item<bool>()) {
    // For relabeled segments, find a future slot from the same episode
    const int K_seg = std::min(cap, 128);
    // last slot per segment: all_slots[:, L-1]  → [B]
    const torch::Tensor last_slots = all_slots.select(1, L - 1);     // [B]
    const torch::Tensor last_ep    = all_ep_ids.select(1, L - 1);    // [B]

    // Future candidate slots: [B, K_seg]
    const torch::Tensor fut_offsets = torch::arange(L, L + K_seg, lopt);  // [K_seg]
    const torch::Tensor fut_slots_all =
        (last_slots.unsqueeze(1) + fut_offsets.unsqueeze(0)) % cap;  // [B, K_seg]
    const torch::Tensor fut_ep_all = buf_episode_ids_
        .index_select(0, fut_slots_all.flatten()).reshape({B, K_seg});  // [B, K_seg]

    // Match: same episode as last slot
    const torch::Tensor ep_match = (fut_ep_all == last_ep.unsqueeze(1));     // [B, K_seg] bool
    const torch::Tensor has_fut  = ep_match.any(1);                           // [B] bool
    const torch::Tensor first_k  = ep_match.to(torch::kFloat32).argmax(1);   // [B]
    const torch::Tensor chosen_slots = fut_slots_all.gather(1, first_k.unsqueeze(1)).squeeze(1);  // [B]

    // Gather future achieved goals: [B, goal_dim]
    const torch::Tensor fut_goals = buf_achieved_goals_.index_select(0, chosen_slots);

    // Apply: relabeled[b] = do_relabel[b] AND has_fut[b]
    const torch::Tensor apply = (do_relabel & has_fut).unsqueeze(1).unsqueeze(2)
        .expand({B, L, goal_dim_});  // [B, L, goal_dim]
    const torch::Tensor fut_goals_expanded = fut_goals.unsqueeze(1).expand({B, L, goal_dim_});
    all_goals = torch::where(apply, fut_goals_expanded, all_goals);
  }

  // Compute rewards for all (B, L) pairs at once
  const torch::Tensor all_rewards = compute_goal_reward(
      all_achieved.reshape({B * L, goal_dim_}),
      all_goals.reshape({B * L, goal_dim_})).reshape({B, L});

  // Transpose from [B, L, ...] to [L, B, ...]
  SegmentBatch batch;
  batch.obs          = all_obs.permute({1, 0, 2}).contiguous();    // [L, B, obs_dim]
  batch.actions      = all_actions.t().contiguous();                // [L, B]
  batch.rewards      = all_rewards.t().contiguous();               // [L, B]
  batch.next_obs     = all_next.permute({1, 0, 2}).contiguous();   // [L, B, obs_dim]
  batch.dones        = all_dones.t().contiguous();                  // [L, B]
  batch.goals        = all_goals.permute({1, 0, 2}).contiguous();  // [L, B, goal_dim]
  batch.episode_ends = ep_ends.t().contiguous();                    // [L, B]
  return batch;
}

}  // namespace pulsar

#endif
