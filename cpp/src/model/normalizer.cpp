#include "pulsar/model/normalizer.hpp"

#ifdef PULSAR_HAS_TORCH

#include "pulsar/tracing/tracing.hpp"

namespace pulsar {

ObservationNormalizer::ObservationNormalizer(int obs_dim) {
  mean_ = torch::zeros({obs_dim});
  var_ = torch::ones({obs_dim});
  count_ = torch::tensor(0.0F);
}

void ObservationNormalizer::to(const torch::Device& device) {
  mean_ = mean_.to(device, /*non_blocking=*/false, /*copy=*/true);
  var_ = var_.to(device, /*non_blocking=*/false, /*copy=*/true);
  count_ = count_.to(device, /*non_blocking=*/false, /*copy=*/true);
}

void ObservationNormalizer::update(const torch::Tensor& obs) {
  PULSAR_TRACE_SCOPE_CAT("normalizer", "update");
  const torch::Tensor batch_mean = obs.mean(0);
  const torch::Tensor batch_var = obs.var(0, false);
  const torch::Tensor batch_count =
      torch::tensor(static_cast<float>(obs.size(0)), torch::TensorOptions().device(obs.device()));

  const torch::Tensor delta = batch_mean - mean_;
  const torch::Tensor total_count = count_ + batch_count;
  const torch::Tensor new_mean = mean_ + delta * (batch_count / total_count);

  const torch::Tensor m_a = var_ * count_;
  const torch::Tensor m_b = batch_var * batch_count;
  const torch::Tensor correction = delta.pow(2) * (count_ * batch_count / total_count);
  const torch::Tensor new_var = (m_a + m_b + correction) / total_count;

  mean_ = new_mean.detach();
  var_ = new_var.detach().clamp_min(1.0e-6);
  count_ = total_count.detach();
}

torch::Tensor ObservationNormalizer::normalize(const torch::Tensor& obs) const {
  PULSAR_TRACE_SCOPE_CAT("normalizer", "normalize");
  return (obs - mean_) / torch::sqrt(var_ + 1.0e-6);
}

ObservationNormalizer ObservationNormalizer::clone() const {
  ObservationNormalizer copy(0);
  copy.mean_ = mean_.clone();
  copy.var_ = var_.clone();
  copy.count_ = count_.clone();
  return copy;
}

void ObservationNormalizer::merge(const ObservationNormalizer& other) {
  if (other.count_.item<float>() <= 0.0F) {
    return;
  }
  if (count_.item<float>() <= 0.0F) {
    mean_ = other.mean_.detach().clone();
    var_ = other.var_.detach().clone();
    count_ = other.count_.detach().clone();
    return;
  }
  const torch::Tensor new_count = count_ + other.count_;
  const torch::Tensor delta = other.mean_ - mean_;
  const torch::Tensor new_mean = mean_ + delta * (other.count_ / new_count);
  const torch::Tensor m_a = var_ * count_;
  const torch::Tensor m_b = other.var_ * other.count_;
  const torch::Tensor correction = delta.pow(2) * (count_ * other.count_ / new_count);
  mean_ = new_mean.detach();
  var_ = ((m_a + m_b + correction) / new_count).detach().clamp_min(1.0e-6);
  count_ = new_count.detach();
}

void ObservationNormalizer::save(torch::serialize::OutputArchive& archive) const {
  archive.write("norm_mean", mean_);
  archive.write("norm_var", var_);
  archive.write("norm_count", count_);
}

void ObservationNormalizer::load(torch::serialize::InputArchive& archive) {
  archive.read("norm_mean", mean_);
  archive.read("norm_var", var_);
  archive.read("norm_count", count_);
}

}  // namespace pulsar

#endif
