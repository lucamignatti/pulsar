#pragma once

#include <string>

#ifdef PULSAR_HAS_TORCH
#include <torch/torch.h>
#endif

namespace pulsar {

#ifdef PULSAR_HAS_TORCH

class ObservationNormalizer {
 public:
  explicit ObservationNormalizer(int obs_dim);

  void to(const torch::Device& device);
  void update(const torch::Tensor& obs);
  torch::Tensor normalize(const torch::Tensor& obs) const;
  [[nodiscard]] ObservationNormalizer clone() const;
  void save(torch::serialize::OutputArchive& archive) const;
  void load(torch::serialize::InputArchive& archive);

 private:
  torch::Tensor mean_;
  torch::Tensor var_;
  torch::Tensor count_;
};

#else

class ObservationNormalizer {
 public:
  explicit ObservationNormalizer(int) {}
};

#endif

}  // namespace pulsar
