#include "pulsar/training/future_window_builder.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <stdexcept>

namespace pulsar {
namespace {

bool valid_label(std::int64_t label, int outcome_classes) {
  return label >= 0 && label < outcome_classes;
}

bool tensor_defined_nonempty(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

FutureWindowBatch allocate_batch(
    std::int64_t row_count,
    int max_horizon,
    int horizon_count,
    std::int64_t obs_dim,
    const torch::TensorOptions& obs_options,
    bool has_actions,
    const torch::TensorOptions& action_options,
    bool has_action_probs,
    std::int64_t action_dim,
    const torch::TensorOptions& action_prob_options) {
  FutureWindowBatch out;
  out.windows = torch::zeros({row_count, max_horizon + 1, obs_dim}, obs_options);
  out.labels = torch::full({row_count}, 2, torch::TensorOptions().dtype(torch::kLong));
  out.weights = torch::ones({row_count}, torch::TensorOptions().dtype(torch::kFloat32));
  out.future_horizon_mask =
      torch::zeros({row_count, horizon_count}, torch::TensorOptions().dtype(torch::kBool));
  out.outcome_horizon_mask =
      torch::zeros({row_count, horizon_count}, torch::TensorOptions().dtype(torch::kBool));
  out.time_indices = torch::zeros({row_count}, torch::TensorOptions().dtype(torch::kLong));
  out.column_indices = torch::zeros({row_count}, torch::TensorOptions().dtype(torch::kLong));
  if (has_actions) {
    out.actions = torch::zeros({row_count}, action_options);
  }
  if (has_action_probs) {
    out.action_probs = torch::zeros({row_count, action_dim}, action_prob_options);
  }
  return out;
}

int first_terminal_step(
    const torch::Tensor& terminated,
    const torch::Tensor& truncated,
    std::int64_t t,
    std::int64_t column,
    std::int64_t length) {
  if (!terminated.defined() && !truncated.defined()) {
    return -1;
  }
  for (std::int64_t step = t; step < length; ++step) {
    const bool is_terminated =
        terminated.defined() && terminated[step][column].item<float>() > 0.5F;
    const bool is_truncated =
        truncated.defined() && truncated[step][column].item<float>() > 0.5F;
    if (is_terminated || is_truncated) {
      return static_cast<int>(step);
    }
  }
  return -1;
}

int first_rollout_done_step(
    const torch::Tensor& dones,
    std::int64_t t,
    std::int64_t agent,
    std::int64_t time) {
  if (!dones.defined()) {
    return -1;
  }
  for (std::int64_t step = t; step < time; ++step) {
    if (dones[step][agent].item<float>() > 0.5F) {
      return static_cast<int>(step);
    }
  }
  return -1;
}

bool packed_horizon_valid(std::int64_t t, int horizon, std::int64_t length, int terminal_step) {
  const std::int64_t target = t + horizon;
  if (terminal_step >= 0 && terminal_step < target) {
    return true;
  }
  return target < length;
}

bool rollout_horizon_valid(
    const torch::Tensor& trajectory_ids,
    std::int64_t t,
    std::int64_t agent,
    int horizon,
    std::int64_t time,
    int terminal_step,
    bool has_final_obs) {
  const std::int64_t target = t + horizon;
  if (terminal_step >= 0 && terminal_step < target) {
    return true;
  }
  if (target < time) {
    if (trajectory_ids.defined()) {
      const std::int64_t base_id = trajectory_ids[t][agent].item<std::int64_t>();
      const std::int64_t target_id = trajectory_ids[target][agent].item<std::int64_t>();
      if (base_id != target_id) {
        return false;
      }
    }
    return true;
  }
  return target == time && has_final_obs && terminal_step < 0;
}

}  // namespace

int future_window_max_horizon(const FutureEvaluatorConfig& config) {
  if (config.horizons.empty()) {
    throw std::invalid_argument("FutureEvaluatorConfig.horizons must not be empty.");
  }
  return *std::max_element(config.horizons.begin(), config.horizons.end());
}

FutureWindowBatch build_future_windows_from_packed_batch(
    const OfflineTensorPackedBatch& batch,
    const FutureEvaluatorConfig& evaluator_config) {
  const int max_h = future_window_max_horizon(evaluator_config);
  const int horizon_count = static_cast<int>(evaluator_config.horizons.size());
  const auto obs_dim = batch.obs.size(2);
  const bool has_actions = batch.actions.defined();
  const bool has_action_probs = batch.action_probs.defined();
  const std::int64_t action_dim = has_action_probs ? batch.action_probs.size(2) : 0;

  std::vector<std::pair<std::int64_t, std::int64_t>> rows;
  rows.reserve(static_cast<std::size_t>(batch.valid_mask.sum().item<std::int64_t>()));
  for (std::int64_t column = 0; column < static_cast<std::int64_t>(batch.lengths.size()); ++column) {
    const std::int64_t length = batch.lengths[static_cast<std::size_t>(column)];
    for (std::int64_t t = 0; t < length; ++t) {
      const int terminal_step = first_terminal_step(batch.terminated, batch.truncated, t, column, length);
      bool any_future = false;
      for (const int horizon : evaluator_config.horizons) {
        any_future = any_future || packed_horizon_valid(t, horizon, length, terminal_step);
      }
      if (any_future) {
        rows.emplace_back(t, column);
      }
    }
  }

  FutureWindowBatch out = allocate_batch(
      static_cast<std::int64_t>(rows.size()),
      max_h,
      horizon_count,
      obs_dim,
      batch.obs.options(),
      has_actions,
      has_actions ? batch.actions.options() : torch::TensorOptions().dtype(torch::kLong),
      has_action_probs,
      action_dim,
      has_action_probs ? batch.action_probs.options() : torch::TensorOptions().dtype(torch::kFloat32));

  for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows.size()); ++row) {
    const auto [t, column] = rows[static_cast<std::size_t>(row)];
    const std::int64_t length = batch.lengths[static_cast<std::size_t>(column)];
    const int terminal_step = first_terminal_step(batch.terminated, batch.truncated, t, column, length);
    const std::int64_t label =
        batch.outcome.defined() ? batch.outcome[t][column].item<std::int64_t>() : 2;
    const bool outcome_known =
        batch.outcome_known.defined() && batch.outcome_known[t][column].item<float>() > 0.5F &&
        valid_label(label, evaluator_config.outcome_classes);

    out.labels[row] = label;
    out.weights[row] = batch.weights.defined() ? batch.weights[t][column].item<float>() : 1.0F;
    out.time_indices[row] = t;
    out.column_indices[row] = column;

    for (int dt = 0; dt <= max_h; ++dt) {
      const std::int64_t target = t + dt;
      const std::int64_t src_t =
          (terminal_step >= 0 && terminal_step < target)
              ? terminal_step
              : std::min<std::int64_t>(target, length - 1);
      out.windows[row][dt].copy_(batch.obs[src_t][column]);
    }

    for (int h = 0; h < horizon_count; ++h) {
      const bool valid =
          packed_horizon_valid(t, evaluator_config.horizons[static_cast<std::size_t>(h)], length, terminal_step);
      if (valid) {
        out.future_horizon_mask[row][h] = true;
      }
      if (valid && outcome_known) {
        out.outcome_horizon_mask[row][h] = true;
      }
    }
    if (out.actions.defined()) {
      out.actions[row] = batch.actions[t][column].item<std::int64_t>();
    }
    if (out.action_probs.defined()) {
      out.action_probs[row].copy_(batch.action_probs[t][column]);
    }
  }

  return out;
}

FutureWindowBatch build_future_windows_from_rollout(
    const RolloutWindowSource& source,
    const FutureEvaluatorConfig& evaluator_config,
    int outcome_classes) {
  if (!source.obs.defined() || source.obs.dim() != 3) {
    throw std::invalid_argument("RolloutWindowSource.obs must have shape [time, agents, obs_dim].");
  }
  const int max_h = future_window_max_horizon(evaluator_config);
  const int horizon_count = static_cast<int>(evaluator_config.horizons.size());
  const auto time = source.obs.size(0);
  const auto agents = source.obs.size(1);
  const auto obs_dim = source.obs.size(2);
  const bool has_final_obs = tensor_defined_nonempty(source.final_obs);

  std::vector<std::pair<std::int64_t, std::int64_t>> rows;
  rows.reserve(static_cast<std::size_t>(time * agents));
  for (std::int64_t agent = 0; agent < agents; ++agent) {
    for (std::int64_t t = 0; t < time; ++t) {
      const int done_step = first_rollout_done_step(source.dones, t, agent, time);
      bool any_future = false;
      for (const int horizon : evaluator_config.horizons) {
        any_future = any_future ||
                     rollout_horizon_valid(
                         source.trajectory_ids,
                         t,
                         agent,
                         horizon,
                         time,
                         done_step,
                         has_final_obs);
      }
      if (any_future) {
        rows.emplace_back(t, agent);
      }
    }
  }

  FutureWindowBatch out = allocate_batch(
      static_cast<std::int64_t>(rows.size()),
      max_h,
      horizon_count,
      obs_dim,
      source.obs.options(),
      false,
      torch::TensorOptions().dtype(torch::kLong),
      false,
      0,
      torch::TensorOptions().dtype(torch::kFloat32));

  for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows.size()); ++row) {
    const auto [t, agent] = rows[static_cast<std::size_t>(row)];
    const int done_step = first_rollout_done_step(source.dones, t, agent, time);
    const std::int64_t label =
        (done_step >= 0 && source.terminal_outcomes.defined())
            ? source.terminal_outcomes[done_step][agent].item<std::int64_t>()
            : 2;
    const bool outcome_known = done_step >= 0 && valid_label(label, outcome_classes);

    out.labels[row] = label;
    out.time_indices[row] = t;
    out.column_indices[row] = agent;

    for (int dt = 0; dt <= max_h; ++dt) {
      const std::int64_t target = t + dt;
      if (done_step >= 0 && done_step < target && source.terminal_obs.defined()) {
        out.windows[row][dt].copy_(source.terminal_obs[done_step][agent]);
      } else if (target < time) {
        out.windows[row][dt].copy_(source.obs[target][agent]);
      } else if (target == time && has_final_obs) {
        out.windows[row][dt].copy_(source.final_obs[agent]);
      } else if (done_step >= 0 && source.terminal_obs.defined()) {
        out.windows[row][dt].copy_(source.terminal_obs[done_step][agent]);
      } else {
        out.windows[row][dt].copy_(source.obs[std::min<std::int64_t>(target, time - 1)][agent]);
      }
    }

    for (int h = 0; h < horizon_count; ++h) {
      const bool valid =
          rollout_horizon_valid(
              source.trajectory_ids,
              t,
              agent,
              evaluator_config.horizons[static_cast<std::size_t>(h)],
              time,
              done_step,
              has_final_obs);
      if (valid) {
        out.future_horizon_mask[row][h] = true;
      }
      if (valid && outcome_known) {
        out.outcome_horizon_mask[row][h] = true;
      }
    }
  }

  return out;
}

FutureWindowBatch build_future_windows_from_completed_trajectory(
    const torch::Tensor& obs_cpu,
    std::int64_t outcome,
    const FutureEvaluatorConfig& evaluator_config,
    int outcome_classes) {
  if (!obs_cpu.defined() || obs_cpu.dim() != 2) {
    throw std::invalid_argument("completed trajectory obs must have shape [time, obs_dim].");
  }
  OfflineTensorPackedBatch batch;
  batch.obs = obs_cpu.unsqueeze(1);
  batch.outcome = torch::full(
      {obs_cpu.size(0), 1},
      outcome,
      torch::TensorOptions().dtype(torch::kLong));
  const bool known = valid_label(outcome, outcome_classes);
  batch.outcome_known = torch::full({obs_cpu.size(0), 1}, known ? 1.0F : 0.0F);
  batch.weights = torch::ones({obs_cpu.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32));
  batch.episode_starts = torch::zeros({obs_cpu.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32));
  if (obs_cpu.size(0) > 0) {
    batch.episode_starts[0][0] = 1.0F;
  }
  batch.terminated = torch::zeros({obs_cpu.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32));
  batch.truncated = torch::zeros({obs_cpu.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32));
  if (obs_cpu.size(0) > 0) {
    batch.terminated[obs_cpu.size(0) - 1][0] = 1.0F;
  }
  batch.valid_mask = torch::ones({obs_cpu.size(0), 1}, torch::TensorOptions().dtype(torch::kBool));
  batch.lengths = {obs_cpu.size(0)};
  return build_future_windows_from_packed_batch(batch, evaluator_config);
}

torch::Tensor future_delta_targets(
    const torch::Tensor& normalized_windows,
    const FutureEvaluatorConfig& evaluator_config) {
  if (normalized_windows.dim() != 3) {
    throw std::invalid_argument("future_delta_targets expects [batch, time, obs_dim] windows.");
  }
  std::vector<torch::Tensor> targets;
  targets.reserve(evaluator_config.horizons.size());
  const torch::Tensor current = normalized_windows.select(1, 0);
  for (const int horizon : evaluator_config.horizons) {
    if (normalized_windows.size(1) <= horizon) {
      throw std::invalid_argument("future_delta_targets window is shorter than a configured horizon.");
    }
    targets.push_back(normalized_windows.select(1, horizon) - current);
  }
  return torch::stack(targets, 1);
}

torch::Tensor masked_future_delta_loss(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    const torch::Tensor& future_horizon_mask,
    const torch::Tensor& weights) {
  if (predictions.numel() == 0 || future_horizon_mask.sum().item<std::int64_t>() == 0) {
    return torch::zeros({}, predictions.options());
  }
  torch::Tensor mask = future_horizon_mask.unsqueeze(-1).to(predictions.dtype());
  if (weights.defined()) {
    const torch::Tensor normalized_weights = weights / weights.mean().clamp_min(1.0e-6);
    mask = mask * normalized_weights.view({-1, 1, 1}).to(predictions.device()).to(predictions.dtype());
  }
  return ((predictions - targets).pow(2) * mask).sum() /
         mask.sum().clamp_min(1.0).mul(predictions.size(-1));
}

}  // namespace pulsar

#endif
