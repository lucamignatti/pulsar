#pragma once

#ifdef PULSAR_HAS_TORCH

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "pulsar/config/config.hpp"
#include "pulsar/training/offline_dataset.hpp"

namespace pulsar {

struct FutureWindowBatch {
  torch::Tensor windows{};
  torch::Tensor labels{};
  torch::Tensor weights{};
  torch::Tensor future_horizon_mask{};
  torch::Tensor outcome_horizon_mask{};
  torch::Tensor time_indices{};
  torch::Tensor column_indices{};
  torch::Tensor actions{};
  torch::Tensor action_probs{};
};

struct RolloutWindowSource {
  torch::Tensor obs{};
  torch::Tensor final_obs{};
  torch::Tensor terminal_obs{};
  torch::Tensor dones{};
  torch::Tensor trajectory_ids{};
  torch::Tensor terminal_outcomes{};
};

[[nodiscard]] int future_window_max_horizon(const FutureEvaluatorConfig& config);

[[nodiscard]] FutureWindowBatch build_future_windows_from_packed_batch(
    const OfflineTensorPackedBatch& batch,
    const FutureEvaluatorConfig& evaluator_config);

[[nodiscard]] FutureWindowBatch build_future_windows_from_rollout(
    const RolloutWindowSource& source,
    const FutureEvaluatorConfig& evaluator_config,
    int outcome_classes);

[[nodiscard]] FutureWindowBatch build_future_windows_from_completed_trajectory(
    const torch::Tensor& obs_cpu,
    std::int64_t outcome,
    const FutureEvaluatorConfig& evaluator_config,
    int outcome_classes);

[[nodiscard]] torch::Tensor future_delta_targets(
    const torch::Tensor& normalized_windows,
    const FutureEvaluatorConfig& evaluator_config);

[[nodiscard]] torch::Tensor masked_future_delta_loss(
    const torch::Tensor& predictions,
    const torch::Tensor& targets,
    const torch::Tensor& future_horizon_mask,
    const torch::Tensor& weights = {});

}  // namespace pulsar

#endif
