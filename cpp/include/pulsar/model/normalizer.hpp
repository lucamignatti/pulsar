#pragma once

#include <string>
#include <shared_mutex>
#include <mutex>

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
  void merge(const ObservationNormalizer& other);
  void save(torch::serialize::OutputArchive& archive) const;
  void load(torch::serialize::InputArchive& archive);

 private:
  torch::Tensor mean_;
  torch::Tensor var_;
  torch::Tensor count_;
};

class PopArtNormalizer {
 public:
  PopArtNormalizer(double beta = 0.0001, double epsilon = 1e-4)
      : beta_(beta), epsilon_(epsilon), mean_(0.0), var_(1.0), std_(1.0) {}

  PopArtNormalizer(const PopArtNormalizer& other) {
    std::shared_lock<std::shared_mutex> lock(other.mutex_);
    beta_ = other.beta_;
    epsilon_ = other.epsilon_;
    mean_ = other.mean_;
    var_ = other.var_;
    std_ = other.std_;
  }

  PopArtNormalizer& operator=(const PopArtNormalizer& other) {
    if (this != &other) {
      std::unique_lock<std::shared_mutex> lock_this(mutex_, std::defer_lock);
      std::shared_lock<std::shared_mutex> lock_other(other.mutex_, std::defer_lock);
      std::lock(lock_this, lock_other);
      beta_ = other.beta_;
      epsilon_ = other.epsilon_;
      mean_ = other.mean_;
      var_ = other.var_;
      std_ = other.std_;
    }
    return *this;
  }

  void update(const torch::Tensor& targets, torch::nn::LinearImpl& final_layer) {
    torch::NoGradGuard no_grad;
    double batch_mean = targets.mean().item<double>();
    double batch_var = targets.var(false).item<double>();

    std::unique_lock<std::shared_mutex> lock(mutex_);
    double old_mean = mean_;
    double old_std = std_;

    mean_ = (1.0 - beta_) * mean_ + beta_ * batch_mean;
    var_ = (1.0 - beta_) * var_ + beta_ * batch_var;
    std_ = std::sqrt(var_ + epsilon_);

    double scale = old_std / std_;
    final_layer.weight.copy_(final_layer.weight * scale);
    final_layer.bias.copy_((old_std * final_layer.bias + old_mean - mean_) / std_);
  }

  torch::Tensor normalize(const torch::Tensor& targets) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return (targets - mean_) / std_;
  }

  torch::Tensor denormalize(const torch::Tensor& normalized_values) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return normalized_values * std_ + mean_;
  }

  void save(torch::serialize::OutputArchive& archive, const std::string& prefix = "popart_") const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    archive.write(prefix + "mean", torch::tensor({mean_}, torch::kFloat64));
    archive.write(prefix + "var", torch::tensor({var_}, torch::kFloat64));
    archive.write(prefix + "std", torch::tensor({std_}, torch::kFloat64));
  }

  void load(torch::serialize::InputArchive& archive, const std::string& prefix = "popart_") {
    torch::Tensor t_mean, t_var, t_std;
    archive.read(prefix + "mean", t_mean);
    archive.read(prefix + "var", t_var);
    archive.read(prefix + "std", t_std);
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    mean_ = t_mean.item<double>();
    var_ = t_var.item<double>();
    std_ = t_std.item<double>();
  }

  [[nodiscard]] double mean() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return mean_;
  }

  [[nodiscard]] double std() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return std_;
  }

 private:
  double beta_;
  double epsilon_;
  double mean_;
  double var_;
  double std_;
  mutable std::shared_mutex mutex_;
};

#else

class ObservationNormalizer {
 public:
  explicit ObservationNormalizer(int) {}
};

#endif

}  // namespace pulsar
