#include "pulsar/model/normalizer.hpp"

#ifdef PULSAR_HAS_TORCH

namespace pulsar {

ObservationNormalizer::ObservationNormalizer(int obs_dim) {
  mean_ = torch::zeros({obs_dim});
  var_ = torch::ones({obs_dim});
  count_ = torch::tensor(1.0F);
}

void ObservationNormalizer::to(const torch::Device& device) {
  mean_ = mean_.to(device);
  var_ = var_.to(device);
  count_ = count_.to(device);
}

void ObservationNormalizer::update(const torch::Tensor& obs) {
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
  return (obs - mean_) / torch::sqrt(var_ + 1.0e-6);
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
