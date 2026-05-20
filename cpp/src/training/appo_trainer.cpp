#include "pulsar/training/appo_trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

#if defined(__linux__)
#include <malloc.h>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

#include <nlohmann/json.hpp>

#include "pulsar/env/done.hpp"
#include "pulsar/env/mutators.hpp"
#include "pulsar/env/obs_builder.hpp"
#include "pulsar/env/rocketsim_engine.hpp"
#include "pulsar/training/cuda_utils.hpp"
#include "pulsar/training/curriculum.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace pulsar {
namespace {

constexpr int kEsLoraMinStage = 0;
constexpr int kSelfPlayMinStage = 0;
constexpr int kTrainerStateVersion = 2;

double current_process_rss_mb() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  long pages = 0;
  long resident = 0;
  if (statm >> pages >> resident) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) {
      return static_cast<double>(resident) * static_cast<double>(page_size) / (1024.0 * 1024.0);
    }
  }
#elif defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
  }
#endif
  return 0.0;
}

double current_process_peak_rss_mb() {
#if defined(__linux__) || defined(__APPLE__)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
  return 0.0;
#endif
}

struct CgroupMemoryStats {
  double current_mb = 0.0;
  double limit_mb = 0.0;
};

double read_memory_bytes_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string value;
  if (!(input >> value) || value.empty() || value == "max") {
    return 0.0;
  }
  try {
    const double bytes = std::stod(value);
    constexpr double kLikelyUnlimitedBytes = 1.0e18;
    return bytes >= kLikelyUnlimitedBytes ? 0.0 : bytes;
  } catch (...) {
    return 0.0;
  }
}

CgroupMemoryStats current_cgroup_memory_stats() {
  CgroupMemoryStats stats{};
#if defined(__linux__)
  stats.current_mb = read_memory_bytes_file("/sys/fs/cgroup/memory.current") / (1024.0 * 1024.0);
  stats.limit_mb = read_memory_bytes_file("/sys/fs/cgroup/memory.max") / (1024.0 * 1024.0);
  if (stats.current_mb <= 0.0 && stats.limit_mb <= 0.0) {
    stats.current_mb = read_memory_bytes_file("/sys/fs/cgroup/memory/memory.usage_in_bytes") / (1024.0 * 1024.0);
    stats.limit_mb = read_memory_bytes_file("/sys/fs/cgroup/memory/memory.limit_in_bytes") / (1024.0 * 1024.0);
  }
#endif
  return stats;
}

void sample_cuda_memory_stats(TrainerMetrics& metrics, const torch::Device& device) noexcept {
#ifdef PULSAR_HAS_CUDA
  if (!device.is_cuda()) {
    return;
  }
  try {
    const auto device_index = static_cast<c10::DeviceIndex>(device.index());
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
    constexpr std::size_t aggregate =
        static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    constexpr double mb = 1024.0 * 1024.0;
    metrics.cuda_memory_allocated_mb =
        static_cast<double>(stats.allocated_bytes[aggregate].current) / mb;
    metrics.cuda_memory_reserved_mb =
        static_cast<double>(stats.reserved_bytes[aggregate].current) / mb;
    metrics.cuda_max_memory_allocated_mb =
        static_cast<double>(stats.allocated_bytes[aggregate].peak) / mb;
    metrics.cuda_max_memory_reserved_mb =
        static_cast<double>(stats.reserved_bytes[aggregate].peak) / mb;
    metrics.cuda_alloc_retries = stats.num_alloc_retries;
    metrics.cuda_ooms = stats.num_ooms;
  } catch (const std::exception& exc) {
    std::cerr << "cuda memory stats unavailable: " << exc.what() << '\n';
  }
#else
  (void)metrics;
  (void)device;
#endif
}

void sample_cuda_memory_stats(TrainerMetrics& metrics, const std::vector<torch::Device>& devices) noexcept {
  TrainerMetrics aggregate{};
  for (const torch::Device& device : devices) {
    TrainerMetrics per_device{};
    sample_cuda_memory_stats(per_device, device);
    aggregate.cuda_memory_allocated_mb += per_device.cuda_memory_allocated_mb;
    aggregate.cuda_memory_reserved_mb += per_device.cuda_memory_reserved_mb;
    aggregate.cuda_max_memory_allocated_mb += per_device.cuda_max_memory_allocated_mb;
    aggregate.cuda_max_memory_reserved_mb += per_device.cuda_max_memory_reserved_mb;
    aggregate.cuda_alloc_retries += per_device.cuda_alloc_retries;
    aggregate.cuda_ooms += per_device.cuda_ooms;
  }
  metrics.cuda_memory_allocated_mb = aggregate.cuda_memory_allocated_mb;
  metrics.cuda_memory_reserved_mb = aggregate.cuda_memory_reserved_mb;
  metrics.cuda_max_memory_allocated_mb = aggregate.cuda_max_memory_allocated_mb;
  metrics.cuda_max_memory_reserved_mb = aggregate.cuda_max_memory_reserved_mb;
  metrics.cuda_alloc_retries = aggregate.cuda_alloc_retries;
  metrics.cuda_ooms = aggregate.cuda_ooms;
}

void trim_released_host_memory() noexcept {
#if defined(__linux__)
  malloc_trim(0);
#endif
}

#if defined(PULSAR_HAS_CUDA) && !defined(USE_ROCM)
class OptionalCudaAutocastGuard {
 public:
  explicit OptionalCudaAutocastGuard(bool enabled)
      : enabled_(enabled),
        previous_enabled_(enabled ? at::autocast::is_autocast_enabled(at::kCUDA) : false),
        previous_dtype_(enabled ? at::autocast::get_autocast_dtype(at::kCUDA) : at::kFloat) {
    if (enabled_) {
      at::autocast::set_autocast_dtype(at::kCUDA, at::kHalf);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
    }
  }

  ~OptionalCudaAutocastGuard() {
    if (enabled_) {
      at::autocast::set_autocast_enabled(at::kCUDA, previous_enabled_);
      at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
    }
  }

 private:
  bool enabled_;
  bool previous_enabled_;
  at::ScalarType previous_dtype_;
};
#else
class OptionalCudaAutocastGuard {
 public:
  explicit OptionalCudaAutocastGuard(bool) {}
};
#endif

torch::Device resolve_runtime_device(const std::string& device_name) {
  torch::Device device(device_name);
  if (device.is_cuda() && !device.has_index()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return device;
}

std::vector<torch::Device> resolve_runtime_devices(const std::string& device_name) {
  torch::Device requested(device_name);
  if (!requested.is_cuda()) {
    return {requested};
  }
  if (requested.has_index()) {
    return {requested};
  }
#ifdef PULSAR_HAS_CUDA
  const int device_count = static_cast<int>(torch::cuda::device_count());
  if (device_count > 0) {
    std::vector<torch::Device> devices;
    devices.reserve(static_cast<std::size_t>(device_count));
    for (int index = 0; index < device_count; ++index) {
      devices.emplace_back(torch::kCUDA, index);
    }
    return devices;
  }
#endif
  return {torch::Device(torch::kCUDA, 0)};
}

std::string join_device_list(const std::vector<torch::Device>& devices) {
  std::string joined;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i > 0) {
      joined += ",";
    }
    joined += devices[i].str();
  }
  return joined;
}

std::vector<torch::Device> assign_shard_devices(
    const std::vector<torch::Device>& devices,
    std::size_t num_shards) {
  std::vector<torch::Device> result;
  result.reserve(num_shards);
  for (std::size_t shard = 0; shard < num_shards; ++shard) {
    result.push_back(devices[shard % devices.size()]);
  }
  return result;
}

void synchronize_cuda_if_needed(const torch::Device& device, const char* context) noexcept {
  if (!device.is_cuda()) {
    return;
  }
  try {
    torch::cuda::synchronize();
  } catch (const std::exception& exc) {
    std::cerr << "cuda synchronize failed during " << context << ": " << exc.what() << '\n';
  }
}

void synchronize_cuda_if_needed(const std::vector<torch::Device>& devices, const char* context) noexcept {
  for (const torch::Device& device : devices) {
    synchronize_cuda_if_needed(device, context);
  }
}

RolloutStorage make_rollout_storage(
    const ExperimentConfig& config,
    int num_agents,
    int action_dim,
    bool pin_memory) {
  return RolloutStorage(
      config.ppo.rollout_length,
      num_agents,
      config.model.observation_dim,
      action_dim,
      config.model.encoder_dim,
      torch::Device(torch::kCPU),
      {"extrinsic"},
      pin_memory);
}

void require_finite(const torch::Tensor& tensor, const std::string& name) {
  if (tensor.defined() && !torch::isfinite(tensor).all().item<bool>()) {
    throw std::runtime_error("Non-finite tensor: " + name);
  }
}

void shrink_perturb_parameters(torch::nn::Module& module, float shrink, float noise) {
  torch::NoGradGuard no_grad;
  for (auto& param : module.named_parameters()) {
    if (!param.value().requires_grad() || param.value().dim() < 2) {
      continue;
    }
    const std::string& name = param.key();
    if (name.find("lora_") != std::string::npos) {
      continue;
    }
    if (name == "pos" || name.rfind(".pos") == name.size() - 4) {
      continue;
    }
    param.value().mul_(shrink);
    param.value().add_(torch::randn_like(param.value()) * noise);
  }
}

struct CapturedGrad {
  torch::Tensor param;
  torch::Tensor grad;
};

void accumulate_gradients(torch::nn::Module& module, std::vector<CapturedGrad>& accumulated) {
  if (accumulated.empty()) {
    for (auto& p : module.parameters()) {
      accumulated.push_back({p, torch::Tensor{}});
    }
  }
  size_t i = 0;
  for (auto& p : module.parameters()) {
    if (i >= accumulated.size()) {
      accumulated.push_back({p, torch::Tensor{}});
    }
    if (p.grad().defined()) {
      if (accumulated[i].grad.defined()) {
        accumulated[i].grad.add_(p.grad().detach());
      } else {
        accumulated[i].grad = p.grad().detach().clone();
      }
    }
    ++i;
  }
}

void reduce_captured_gradients(
    torch::nn::Module& module,
    std::vector<CapturedGrad>& dst,
    const std::vector<CapturedGrad>& src,
    const torch::Device& device) {
  if (src.empty()) {
    return;
  }
  if (dst.empty()) {
    for (auto& p : module.parameters()) {
      dst.push_back({p, torch::Tensor{}});
    }
  }
  for (size_t i = 0; i < src.size() && i < dst.size(); ++i) {
    if (!src[i].grad.defined()) {
      continue;
    }
    const torch::Tensor grad = src[i].grad.to(device);
    if (dst[i].grad.defined()) {
      dst[i].grad.add_(grad);
    } else {
      dst[i].grad = grad.clone();
    }
  }
}

void zero_existing_gradients(torch::nn::Module& module) {
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.zero_();
    }
  }
}

bool gradients_are_finite(const torch::nn::Module& module) {
  for (const auto& p : module.parameters()) {
    const torch::Tensor grad = p.grad();
    if (grad.defined() && !torch::isfinite(grad).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

struct GradientSanitizeResult {
  bool changed = false;
  std::string first_parameter;
};

GradientSanitizeResult zero_nonfinite_gradients(torch::nn::Module& module) {
  GradientSanitizeResult result;
  for (auto& item : module.named_parameters(true)) {
    torch::Tensor grad = item.value().mutable_grad();
    if (!grad.defined()) {
      continue;
    }
    const torch::Tensor finite = torch::isfinite(grad);
    if (!finite.all().item<bool>()) {
      if (!result.changed) {
        result.first_parameter = item.key();
      }
      grad.masked_fill_(finite.logical_not(), 0.0);
      result.changed = true;
    }
  }
  return result;
}

double clip_existing_gradients(torch::nn::Module& module, double max_norm) {
  double max_abs = 0.0;
  for (auto& p : module.parameters()) {
    const torch::Tensor grad = p.grad();
    if (!grad.defined()) {
      continue;
    }
    const double param_max = grad.detach().abs().max().item<double>();
    if (!std::isfinite(param_max)) {
      return param_max;
    }
    max_abs = std::max(max_abs, param_max);
  }
  if (max_abs == 0.0) {
    return 0.0;
  }
  double scaled_sq_sum = 0.0;
  for (auto& p : module.parameters()) {
    const torch::Tensor grad = p.grad();
    if (!grad.defined()) {
      continue;
    }
    const torch::Tensor scaled = grad.detach().to(torch::kFloat32) / max_abs;
    const double param_scaled_sq = scaled.square().sum().item<double>();
    if (!std::isfinite(param_scaled_sq)) {
      return param_scaled_sq;
    }
    scaled_sq_sum += param_scaled_sq;
  }
  const double total_norm = max_abs * std::sqrt(scaled_sq_sum);
  if (!std::isfinite(total_norm) || max_norm <= 0.0 || total_norm <= max_norm) {
    return total_norm;
  }
  const double scale = max_norm / (total_norm + 1.0e-6);
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.mul_(scale);
    }
  }
  return total_norm;
}

bool captured_group_has_grad(const std::vector<CapturedGrad>& group) {
  for (const auto& captured : group) {
    if (captured.grad.defined()) {
      return true;
    }
  }
  return false;
}

bool captured_group_gradients_are_finite(const std::vector<CapturedGrad>& group) {
  for (const auto& captured : group) {
    if (captured.grad.defined() && !torch::isfinite(captured.grad).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

void scale_existing_gradients(torch::nn::Module& module, double scale) {
  if (scale == 1.0) {
    return;
  }
  for (auto& p : module.parameters()) {
    torch::Tensor grad = p.mutable_grad();
    if (grad.defined()) {
      grad.div_(scale);
    }
  }
}

torch::Tensor smooth_l1_value_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float delta) {
  if (delta <= 0.0F) {
    return torch::mse_loss(prediction, target, torch::Reduction::Mean);
  }
  const torch::Tensor error = prediction - target;
  const torch::Tensor abs_error = error.abs();
  const torch::Tensor delta_tensor = torch::full_like(abs_error, delta);
  const torch::Tensor quadratic = 0.5F * error.square() / delta;
  const torch::Tensor linear = abs_error - 0.5F * delta;
  return torch::where(abs_error < delta_tensor, quadratic, linear).mean();
}

// Multi-group PCGrad: project each group's gradient against all others.
// groups[idx][param] = gradient tensor for that group+parameter.
void apply_pcgrad_multi(std::vector<std::vector<CapturedGrad>>& groups) {
  if (groups.size() < 2) return;

  // Flatten each group into the full parameter vector. Objective-level groups
  // touch different heads, so missing gradients must participate as zeros.
  std::vector<torch::Tensor> flats;
  std::vector<std::pair<size_t, int64_t>> layout;
  std::vector<bool> active_params(groups[0].size(), false);
  flats.reserve(groups.size());
  layout.reserve(groups[0].size());

  for (size_t i = 0; i < groups[0].size(); ++i) {
    layout.push_back({i, groups[0][i].param.numel()});
  }
  std::vector<torch::Tensor> zero_parts(groups[0].size());
  for (auto& group : groups) {
    std::vector<torch::Tensor> parts;
    parts.reserve(group.size());
    for (size_t i = 0; i < groups[0].size(); ++i) {
      if (group[i].grad.defined()) {
        active_params[i] = true;
        parts.push_back(group[i].grad.view({-1}));
      } else {
        if (!zero_parts[i].defined()) {
          zero_parts[i] = torch::zeros({group[i].param.numel()}, group[i].param.options());
        }
        parts.push_back(zero_parts[i]);
      }
    }
    flats.push_back(torch::cat(parts, 0));
  }

  // Standard PCGrad sweep: project each group once against every other group.
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (i == j) continue;
      const torch::Tensor dot = flats[i].dot(flats[j]);
      if (!torch::isfinite(dot).item<bool>()) {
        continue;
      }
      if (dot.item<float>() < 0.0F) {
        const torch::Tensor norm_j_sq = flats[j].dot(flats[j]).clamp_min(1.0e-12F);
        if (!torch::isfinite(norm_j_sq).item<bool>()) {
          continue;
        }
        const torch::Tensor coeff = dot / norm_j_sq;
        flats[i] = flats[i] - coeff * flats[j];
      }
    }
  }

  // Unflatten back into per-parameter gradients.
  for (size_t g = 0; g < groups.size(); ++g) {
    int64_t offset = 0;
    for (const auto& [param_idx, sz] : layout) {
      if (active_params[param_idx]) {
        groups[g][param_idx].grad =
            flats[g].slice(0, offset, offset + sz).view(groups[g][param_idx].param.sizes()).clone();
      } else {
        groups[g][param_idx].grad = torch::Tensor{};
      }
      offset += sz;
    }
  }
}

// Reduce CapturedGrad groups from GPU replicas into the primary groups.
// replica_groups_list[replica_idx][group_idx][param_idx] -> added to primary_groups.
void reduce_captured_grad_groups(
    std::vector<std::vector<CapturedGrad>>& primary_groups,
    const std::vector<std::vector<std::vector<CapturedGrad>>>& replica_groups_list,
    const torch::Device& primary_device) {
  for (const auto& replica_groups : replica_groups_list) {
    if (replica_groups.empty()) continue;
    for (size_t g = 0; g < replica_groups.size(); ++g) {
      if (g >= primary_groups.size()) break;
      for (size_t p = 0; p < replica_groups[g].size(); ++p) {
        if (p >= primary_groups[g].size()) break;
        if (replica_groups[g][p].grad.defined()) {
          if (primary_groups[g][p].grad.defined()) {
            primary_groups[g][p].grad.add_(replica_groups[g][p].grad.to(primary_device));
          } else {
            primary_groups[g][p].grad = replica_groups[g][p].grad.to(primary_device);
          }
        }
      }
    }
  }
}

// Reduce regular (non-PCGrad) gradients from replica actors into the primary actor.
void reduce_gradients_from_replicas(
    PPOActor& primary,
    const std::vector<PPOActor>& replicas) {
  if (!primary) return;
  auto primary_params = primary->named_parameters(true);
  if (primary_params.size() == 0) return;
  const torch::Device primary_device = primary->parameters().front().device();
  for (const auto& replica : replicas) {
    if (!replica) continue;
    auto replica_params = replica->named_parameters(true);
    for (const auto& item : replica_params) {
      torch::Tensor* primary_tensor = primary_params.find(item.key());
      if (primary_tensor == nullptr) continue;
      torch::Tensor replica_grad = item.value().mutable_grad();
      if (!replica_grad.defined()) continue;
      torch::Tensor primary_grad = primary_tensor->mutable_grad();
      if (primary_grad.defined()) {
        primary_grad.add_(replica_grad.to(primary_device));
      } else {
        primary_tensor->mutable_grad() = replica_grad.to(primary_device);
      }
    }
  }
}

// Sync weights from primary actor to all replica actors.
void sync_actor_to_replicas(
    const PPOActor& primary,
    std::vector<PPOActor>& replicas) {
  if (!primary) return;
  for (auto& replica : replicas) {
    if (replica) {
      copy_ppo_actor_tensors_to(primary, replica, replica->parameters().front().device());
    }
  }
}

torch::Tensor policy_goal_values_like(const torch::Tensor& obs, int goal_dim) {
  const auto options = obs.options().dtype(torch::kFloat32);
  if (obs.dim() == 3) {
    return torch::zeros({obs.size(0), obs.size(1), goal_dim}, options);
  }
  return torch::zeros({obs.size(0), goal_dim}, options);
}

class ModuleRequiresGradGuard {
 public:
  explicit ModuleRequiresGradGuard(torch::nn::Module& module, bool requires_grad) {
    for (auto& param : module.parameters()) {
      previous_.push_back({param, param.requires_grad()});
      param.set_requires_grad(requires_grad);
    }
  }

  ~ModuleRequiresGradGuard() {
    for (auto& [param, requires_grad] : previous_) {
      param.set_requires_grad(requires_grad);
    }
  }

 private:
  std::vector<std::pair<torch::Tensor, bool>> previous_;
};

torch::Tensor sample_masked_gumbel_softmax(
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    float temperature) {
  const torch::Tensor masked_logits = apply_action_mask_to_logits(logits, masks);
  const torch::Tensor uniform = torch::rand_like(masked_logits).clamp(1.0e-6F, 1.0F - 1.0e-6F);
  const torch::Tensor gumbel = -torch::log(-torch::log(uniform));
  return torch::softmax((masked_logits + gumbel) / std::max(temperature, 1.0e-3F), -1);
}

torch::Tensor goal_actor_critic_loss(
    GoalCritic& goal_critic,
    const torch::Tensor& features,
    const torch::Tensor& logits,
    const torch::Tensor& masks,
    const torch::Tensor& future_goals,
    int contrastive_batch_size) {
  const auto active_count = features.size(0);
  if (active_count <= 0) {
    return torch::zeros({}, logits.options().dtype(torch::kFloat32));
  }

  torch::Tensor selected_features = features;
  torch::Tensor selected_logits = logits;
  torch::Tensor selected_masks = masks;
  torch::Tensor selected_goals = future_goals;
  const int bounded_batch = std::max(1, contrastive_batch_size);
  if (active_count > static_cast<c10::IntArrayRef::value_type>(bounded_batch)) {
    const torch::Tensor idx = torch::randperm(
        active_count,
        torch::TensorOptions().dtype(torch::kLong).device(logits.device()))
        .narrow(0, 0, bounded_batch);
    selected_features = selected_features.index({idx});
    selected_logits = selected_logits.index({idx});
    selected_masks = selected_masks.index({idx});
    selected_goals = selected_goals.index({idx});
  }

  const torch::Tensor action_probs = sample_masked_gumbel_softmax(
      selected_logits,
      selected_masks,
      1.0F);
  ModuleRequiresGradGuard freeze_goal_critic(*goal_critic, false);
  return -goal_critic->forward(
      selected_features.detach(),
      action_probs,
      selected_goals.detach()).to(torch::kFloat32).mean();
}

int cuda_mamba2_autograd_forward_sample_cap(const ModelConfig& config) {
  constexpr std::int64_t kProjectedActivationBudgetBytes = 2LL * 1024LL * 1024LL * 1024LL;
  const auto projected_dim = static_cast<std::int64_t>(std::max(1, config.encoder_dim)) * 5;
  const std::int64_t bytes_per_sample = projected_dim * static_cast<std::int64_t>(sizeof(float));
  if (bytes_per_sample <= 0) {
    return std::max(1, config.max_forward_samples);
  }
  const std::int64_t sample_cap = std::max<std::int64_t>(1, kProjectedActivationBudgetBytes / bytes_per_sample);
  return static_cast<int>(std::min<std::int64_t>(
      sample_cap,
      static_cast<std::int64_t>(std::numeric_limits<int>::max())));
}

int effective_max_forward_samples(const ModelConfig& config, const torch::Device& device) {
  constexpr int kUnlimitedCap = 524288;
  if (!device.is_cuda()) {
    return config.max_forward_samples == 0
        ? kUnlimitedCap
        : std::max(1, config.max_forward_samples);
  }
  const int cap = cuda_mamba2_autograd_forward_sample_cap(config);
  return config.max_forward_samples == 0
      ? cap
      : std::max(1, std::min(std::max(1, config.max_forward_samples), cap));
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics) {
  nlohmann::json line = {
      {"update", update_index},
      {"global_step", global_step},
      {"collection_agent_steps_per_second", metrics.collection_agent_steps_per_second},
      {"update_agent_steps_per_second", metrics.update_agent_steps_per_second},
      {"overall_agent_steps_per_second", metrics.overall_agent_steps_per_second},
      {"update_seconds", metrics.update_seconds},
      {"policy_loss", metrics.policy_loss},
      {"value_loss", metrics.value_loss},
      {"entropy", metrics.entropy},
      {"grad_norm", metrics.grad_norm},
      {"policy_approx_kl", metrics.policy_approx_kl},
      {"policy_clip_fraction", metrics.policy_clip_fraction},
      {"policy_log_ratio_abs_max", metrics.policy_log_ratio_abs_max},
      {"nonfinite_loss_skips", metrics.nonfinite_loss_skips},
      {"nonfinite_grad_norm_skips", metrics.nonfinite_grad_norm_skips},
      {"kl_guard_skips", metrics.kl_guard_skips},
      {"grad_norm_guard_skips", metrics.grad_norm_guard_skips},
      {"total_reward_mean", metrics.total_reward_mean},
      {"gameplay_reward_mean", metrics.gameplay_reward_mean},
      {"mechanic_reward_mean", metrics.mechanic_reward_mean},
      {"sampled_value_win_mean", metrics.sampled_value_win_mean},
      {"rollout_steps", metrics.rollout_steps},
      {"completed_episodes", metrics.completed_episodes},
      {"scored_episodes", metrics.scored_episodes},
      {"conceded_episodes", metrics.conceded_episodes},
      {"neutral_episodes", metrics.neutral_episodes},
      {"no_touch_episodes", metrics.no_touch_episodes},
      {"truncated_episodes", metrics.truncated_episodes},
      {"goal_critic_loss", metrics.goal_critic_loss},
      {"mean_goal_score", metrics.mean_goal_score},
      {"mean_sampled_goal_distance", metrics.mean_sampled_goal_distance},
      {"mean_goal_distance", metrics.mean_goal_distance},
      {"min_goal_distance", metrics.min_goal_distance},
      {"ball_proximity_rate", metrics.ball_proximity_rate},
      {"goals_scored", metrics.goals_scored},
      {"goals_conceded", metrics.goals_conceded},
      {"obs_build_seconds", metrics.obs_build_seconds},
      {"mask_build_seconds", metrics.mask_build_seconds},
      {"policy_forward_seconds", metrics.policy_forward_seconds},
      {"action_decode_seconds", metrics.action_decode_seconds},
      {"env_step_seconds", metrics.env_step_seconds},
      {"done_reset_seconds", metrics.done_reset_seconds},
      {"forward_backward_seconds", metrics.forward_backward_seconds},
      {"optimizer_step_seconds", metrics.optimizer_step_seconds},
      {"self_play_eval_seconds", metrics.self_play_eval_seconds},
      {"process_rss_mb", metrics.process_rss_mb},
      {"process_peak_rss_mb", metrics.process_peak_rss_mb},
      {"cgroup_memory_current_mb", metrics.cgroup_memory_current_mb},
      {"cgroup_memory_limit_mb", metrics.cgroup_memory_limit_mb},
      {"cuda_memory_allocated_mb", metrics.cuda_memory_allocated_mb},
      {"cuda_memory_reserved_mb", metrics.cuda_memory_reserved_mb},
      {"cuda_max_memory_allocated_mb", metrics.cuda_max_memory_allocated_mb},
      {"cuda_max_memory_reserved_mb", metrics.cuda_max_memory_reserved_mb},
      {"cuda_alloc_retries", metrics.cuda_alloc_retries},
      {"cuda_ooms", metrics.cuda_ooms},
      {"es_fitness_mean", metrics.es_fitness_mean},
      {"es_fitness_std", metrics.es_fitness_std},
      {"es_fitness_best", metrics.es_fitness_best},
      {"es_reward_mean", metrics.es_reward_mean},
      {"es_winrate_mean", metrics.es_winrate_mean},
      {"es_kl_mean", metrics.es_kl_mean},
      {"es_update_norm", metrics.es_update_norm},
      {"es_lora_a_norm", metrics.es_lora_a_norm},
      {"es_lora_b_norm", metrics.es_lora_b_norm},
      {"es_seconds", metrics.es_seconds},
      {"scored_episode_rate", metrics.scored_episode_rate},
      {"conceded_episode_rate", metrics.conceded_episode_rate},
      {"neutral_episode_rate", metrics.neutral_episode_rate},
      {"no_touch_episode_rate", metrics.no_touch_episode_rate},
      {"truncated_episode_rate", metrics.truncated_episode_rate},
      {"touch_episode_rate", metrics.touch_episode_rate},
      {"multi_touch_episode_rate", metrics.multi_touch_episode_rate},
      {"effective_entropy_coef", metrics.effective_entropy_coef},
      {"self_play_snapshot_count", metrics.self_play_snapshot_count},
  };
  for (const auto& [mode, rate] : metrics.mode_touch_rates) {
    line["mode_" + mode + "_touch_episode_rate"] = rate;
  }
  for (const auto& [mode, rate] : metrics.mode_multi_touch_rates) {
    line["mode_" + mode + "_multi_touch_episode_rate"] = rate;
  }
  for (const auto& [mode, rate] : metrics.mode_scored_rates) {
    line["mode_" + mode + "_scored_episode_rate"] = rate;
  }
  for (const auto& [mode, count] : metrics.mode_completed_episodes) {
    line["mode_" + mode + "_completed_episodes"] = count;
  }
  for (const auto& [mode, rating] : metrics.elo_ratings) {
    line["elo_" + mode] = rating;
  }
  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

nlohmann::json wandb_section_order() {
  return nlohmann::json::array({
      "Tables",
      "1v1",
      "2v2",
      "3v3",
      "Rewards",
      "GCRL",
      "ES-LoRA",
      "Optimization",
      "Charts",
      "System",
      "Hidden Panels",
  });
}

void register_wandb_metric_section(
    nlohmann::json& sections,
    const std::string& section,
    const std::string& key) {
  if (!section.empty() && !key.empty()) {
    sections[key] = section;
  }
}

void add_wandb_metric(
    nlohmann::json& payload,
    nlohmann::json& sections,
    const std::string& section,
    const std::string& key,
    nlohmann::json value) {
  register_wandb_metric_section(sections, section, key);
  payload[key] = std::move(value);
}

std::vector<std::string> configured_wandb_modes(const ExperimentConfig& config) {
  std::vector<std::string> modes;
  const auto add_mode = [&modes](const std::string& mode) {
    if (!mode.empty() && std::find(modes.begin(), modes.end(), mode) == modes.end()) {
      modes.push_back(mode);
    }
  };
  add_mode(std::to_string(config.env.team_size) + "v" + std::to_string(config.env.team_size));
  for (const auto& stage : config.curriculum.stages) {
    add_mode(stage.mode);
    for (const auto& [mode, _] : stage.mode_allocation) {
      add_mode(mode);
    }
  }
  return modes;
}

void register_mode_wandb_sections(nlohmann::json& sections, const std::string& mode) {
  register_wandb_metric_section(sections, mode, "elo_" + mode);
  register_wandb_metric_section(sections, mode, "mode_" + mode + "_touch_episode_rate");
  register_wandb_metric_section(sections, mode, "mode_" + mode + "_multi_touch_episode_rate");
  register_wandb_metric_section(sections, mode, "mode_" + mode + "_scored_episode_rate");
  register_wandb_metric_section(sections, mode, "mode_" + mode + "_completed_episodes");
}

std::shared_ptr<MutatorSequence> make_es_eval_reset_mutator(const EnvConfig& config) {
  return std::make_shared<MutatorSequence>(
      std::vector<StateMutatorPtr>{
          std::make_shared<FixedTeamSizeMutator>(config),
          std::make_shared<KickoffMutator>(config),
      });
}

std::unique_ptr<BatchedRocketSimCollector> make_es_eval_collector(
    const ExperimentConfig& config,
    int total_envs,
    int eval_envs_per_member,
    int update_index,
    int episode_index,
    bool pin_host_memory) {
  ExperimentConfig eval_config = config;
  eval_config.ppo.num_envs = total_envs;
  eval_config.ppo.collection_workers = std::min(config.ppo.collection_workers, total_envs);

  const auto reset_mutator = make_es_eval_reset_mutator(config.env);
  std::vector<TransitionEnginePtr> engines;
  engines.reserve(static_cast<std::size_t>(total_envs));
  for (int env_idx = 0; env_idx < total_envs; ++env_idx) {
    const int local_env = env_idx % eval_envs_per_member;
    EnvConfig env_config = config.env;
    env_config.seed += static_cast<std::uint64_t>(
        1'000'003 + update_index * 65'537 + episode_index * 8'191 + local_env);
    engines.push_back(std::make_shared<RocketSimTransitionEngine>(env_config, reset_mutator));
  }

  auto obs_builder_cfg = config.env;
  obs_builder_cfg.team_size = 3;
  return std::make_unique<BatchedRocketSimCollector>(
      eval_config,
      std::move(engines),
      std::make_shared<PulsarObsBuilder>(obs_builder_cfg),
      std::make_shared<DiscreteActionParser>(ControllerActionTable(config.action_table)),
      std::make_shared<SimpleDoneCondition>(config.env),
      pin_host_memory);
}

std::vector<std::unique_ptr<BatchedRocketSimCollector>> make_collector_vector(
    std::unique_ptr<BatchedRocketSimCollector> collector) {
  std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors;
  collectors.push_back(std::move(collector));
  return collectors;
}

std::size_t total_agents_for_collectors(
    const std::vector<std::unique_ptr<BatchedRocketSimCollector>>& collectors) {
  std::size_t total = 0;
  for (const auto& collector : collectors) {
    if (collector) {
      total += collector->total_agents();
    }
  }
  return total;
}

int action_dim_for_collectors(
    const std::vector<std::unique_ptr<BatchedRocketSimCollector>>& collectors) {
  for (const auto& collector : collectors) {
    if (collector) {
      return collector->action_dim();
    }
  }
  return 0;
}

void accumulate_timings(CollectorTimings& dst, const CollectorTimings& src) {
  dst.obs_build_seconds += src.obs_build_seconds;
  dst.mask_build_seconds += src.mask_build_seconds;
  dst.env_step_seconds += src.env_step_seconds;
  dst.done_reset_seconds += src.done_reset_seconds;
}

}  // namespace

APPOTrainer::APPOTrainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : APPOTrainer(
          std::move(config),
          make_collector_vector(std::move(collector)),
          std::move(self_play_manager),
          std::move(run_output_root),
          log_initialization) {}

APPOTrainer::APPOTrainer(
    ExperimentConfig config,
    std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors,
    std::unique_ptr<SelfPlayManager> self_play_manager,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : config_(std::move(config)),
      collectors_(std::move(collectors)),
      self_play_manager_(std::move(self_play_manager)),
      curriculum_(config_.curriculum),
      action_table_(config_.action_table),
      actor_(PPOActor(config_.model, config_.goal_critic, config_.es_lora)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(actor_->parameters(), torch::optim::AdamOptions(config_.ppo.learning_rate).eps(1.0e-5F)),
      device_(resolve_runtime_device(config_.ppo.device)),
      compute_devices_(resolve_runtime_devices(config_.ppo.device)),
      shard_devices_(assign_shard_devices(compute_devices_, collectors_.size())),
      rollout_(make_rollout_storage(
          config_,
          static_cast<int>(total_agents_for_collectors(collectors_)),
          action_dim_for_collectors(collectors_),
          false)),
      rollout_B_(make_rollout_storage(
          config_,
          static_cast<int>(total_agents_for_collectors(collectors_)),
          action_dim_for_collectors(collectors_),
          false)),
      run_output_root_(std::move(run_output_root)),
      log_initialization_(log_initialization) {
  validate_experiment_config(config_);
  if (config_.ppo.cuda_amp) {
#ifdef USE_ROCM
    std::cerr << "AMP requested but disabled on ROCm (unsupported)." << '\n';
    config_.ppo.cuda_amp = false;
#endif
  }
  if (collectors_.empty()) {
    throw std::invalid_argument("APPOTrainer requires at least one collector.");
  }
  if (actor_->policy_lora()->out_features() != action_dim_for_collectors(collectors_)) {
    throw std::invalid_argument("model.action_dim must match the action table size.");
  }
  if (config_.model.observation_dim != collectors_.front()->obs_dim()) {
    throw std::invalid_argument("model.observation_dim must match obs builder output.");
  }
  total_agents_ = total_agents_for_collectors(collectors_);
  if (total_agents_ == 0) {
    throw std::invalid_argument("APPOTrainer collectors must contain agents.");
  }
  seed_everything(config_.env.seed);
  const torch::Device primary_device = compute_devices_.front();
  device_ = primary_device;
  for (const auto& compute_device : compute_devices_) {
    configure_cuda_runtime(compute_device);
  }
  use_pinned_host_buffers_ = device_.is_cuda();
#ifdef PULSAR_HAS_CUDA
  if (device_.is_cuda()) {
    training_stream_.emplace(at::cuda::getStreamFromPool(false, device_.index()));
    const std::size_t num_shards = collectors_.size();
    shard_collection_streams_.reserve(num_shards);
    for (std::size_t i = 0; i < num_shards; ++i) {
      const torch::Device shard_device = shard_devices_[i];
      shard_collection_streams_.emplace_back(
          at::cuda::getStreamFromPool(false, shard_device.index()));
    }
  }
#endif
  actor_->to(primary_device);
  actor_normalizer_.to(primary_device);

  maybe_initialize_from_checkpoint();
  actor_snapshot_ = clone_ppo_actor(actor_, device_);
  actor_snapshot_->eval();

  // Clone actor replicas to each additional compute GPU for data-parallel updates.
  for (size_t i = 1; i < compute_devices_.size(); ++i) {
    auto replica = clone_ppo_actor(actor_, compute_devices_[i]);
    replica->train();
    compute_actors_.push_back(std::move(replica));
  }

  // Persistent collection actors: one per shard device for policy inference during rollout.
  for (size_t i = 0; i < shard_devices_.size(); ++i) {
    if (shard_devices_[i] == device_) {
      collection_actors_.push_back(actor_snapshot_);  // shared, don't clone
    } else {
      collection_actors_.push_back(clone_ppo_actor(actor_snapshot_, shard_devices_[i]));
      collection_actors_.back()->eval();
    }
  }

  shard_agent_offsets_.clear();
  std::int64_t agent_offset = 0;
  for (const auto& collector : collectors_) {
    if (!collector) {
      throw std::invalid_argument("APPOTrainer collectors must be non-null.");
    }
    shard_agent_offsets_.push_back(agent_offset);
    const auto shard_agents = static_cast<std::int64_t>(collector->total_agents());
    agent_offset += shard_agents;
  }

  if (log_initialization_) {
    std::cout << "compute_devices=" << join_device_list(compute_devices_)
              << " collection_shards=" << collectors_.size()
              << " collection_workers=" << config_.ppo.collection_workers
              << '\n';
  }
}

APPOTrainer::~APPOTrainer() {
  synchronize_cuda_if_needed(compute_devices_, "trainer shutdown");
}

std::int64_t APPOTrainer::model_parameter_count() const {
  std::int64_t total = 0;
  for (const auto& param : actor_->parameters()) {
    total += param.numel();
  }
  return total;
}

void APPOTrainer::apply_curriculum_to_collectors() {
  if (!curriculum_.enabled()) return;
  auto cfg = config_;
  cfg.outcome = curriculum_.outcome();
  cfg.mechanic_rewards = curriculum_.mechanic_rewards();
  cfg.dense_rewards = curriculum_.dense_rewards();
  for (auto& collector : collectors_) {
    if (collector) {
      collector->update_reward_config(cfg);
      collector->update_unlocked_mechanics(curriculum_.unlocked_mechanics());
    }
  }
  if (self_play_manager_) {
    self_play_manager_->set_curriculum_stage(curriculum_.stage_index());
    self_play_manager_->set_current_mode(curriculum_.primary_mode());
  }
}

int team_size_from_mode(const std::string& mode) {
  if (mode == "1v1") return 1;
  if (mode == "2v2") return 2;
  if (mode == "3v3") return 3;
  return 1;
}

void APPOTrainer::rebuild_collectors() {
  const auto& alloc = curriculum_.mode_allocation();
  if (alloc.empty()) return;

  // compute max team size across ALL curriculum stages so the obs builder
  // produces constant-dimension observations regardless of mode (matches
  // the global kObsMaxTeamSize in train_main.cpp)
  int max_team_size = 1;
  for (const auto& stage : config_.curriculum.stages) {
    for (const auto& [mode, frac] : stage.mode_allocation) {
      max_team_size = std::max(max_team_size, team_size_from_mode(mode));
    }
  }

  // create obs builder with max team size so obs_dim is constant across modes
  auto obs_builder_cfg = config_.env;
  obs_builder_cfg.team_size = max_team_size;
  auto obs_builder = std::make_shared<PulsarObsBuilder>(obs_builder_cfg);
  auto action_parser = std::make_shared<DiscreteActionParser>(ControllerActionTable(config_.action_table));
  auto done_condition = std::make_shared<SimpleDoneCondition>(config_.env);
  const bool pin_host = device_.is_cuda();

  const int total_envs = config_.ppo.num_envs;
  const int requested_shards = std::max(1, std::min(config_.ppo.collection_shards, total_envs));

  // allocate envs per mode, rounding to nearest integer
  std::vector<std::pair<std::string, int>> mode_envs;
  int allocated = 0;
  for (const auto& [mode, frac] : alloc) {
    int envs = static_cast<int>(std::round(static_cast<float>(total_envs) * frac));
    if (envs <= 0) envs = 1;
    mode_envs.emplace_back(mode, envs);
    allocated += envs;
  }
  // adjust to match total envs exactly (last mode absorbs the difference)
  if (!mode_envs.empty()) {
    mode_envs.back().second += (total_envs - allocated);
    if (mode_envs.back().second <= 0) mode_envs.back().second = 1;
  }

  // Distribute shards across modes: at least 1 shard per mode, distribute remainder
  // proportional to env count.
  const int num_modes = static_cast<int>(mode_envs.size());
  int remaining_shards = requested_shards - num_modes;
  std::vector<int> mode_shard_counts(num_modes, 1);
  if (remaining_shards > 0) {
    // Allocate extra shards proportional to env ratio
    for (int i = 0; i < num_modes && remaining_shards > 0; ++i) {
      int extra = static_cast<int>(std::round(
          static_cast<float>(remaining_shards) *
          static_cast<float>(mode_envs[i].second) / static_cast<float>(total_envs)));
      extra = std::min(extra, remaining_shards);
      mode_shard_counts[i] += extra;
      remaining_shards -= extra;
    }
    // Any remaining shards go to the mode with most envs
    if (remaining_shards > 0) {
      int best = 0;
      for (int i = 1; i < num_modes; ++i)
        if (mode_envs[i].second > mode_envs[best].second) best = i;
      mode_shard_counts[best] += remaining_shards;
    }
  }

  // destroy old collectors
  collectors_.clear();

  // Create per-shard collectors within each mode
  int env_seed_offset = 0;
  for (int mi = 0; mi < num_modes; ++mi) {
    const auto& [mode, mode_total_envs] = mode_envs[mi];
    const int n_shards = mode_shard_counts[mi];

    for (int si = 0; si < n_shards; ++si) {
      const int base_envs = mode_total_envs / n_shards;
      const int extra_envs = (si < (mode_total_envs % n_shards)) ? 1 : 0;
      const int shard_envs = base_envs + extra_envs;
      if (shard_envs <= 0) continue;

      auto mode_cfg = config_;
      mode_cfg.env.team_size = team_size_from_mode(mode);
      mode_cfg.env.spawn_opponents = (mode_cfg.env.team_size >= 1);
      mode_cfg.ppo.num_envs = shard_envs;
      mode_cfg.env.seed += static_cast<std::uint64_t>(env_seed_offset);
      env_seed_offset += shard_envs;

      // Distribute collection_workers across shards
      if (config_.ppo.collection_workers > 0) {
        const int total_shards = [&]() { int s = 0; for (int c : mode_shard_counts) s += c; return s; }();
        const int base_w = config_.ppo.collection_workers / total_shards;
        const int extra_w = (si < (config_.ppo.collection_workers % total_shards)) ? 1 : 0;
        mode_cfg.ppo.collection_workers = std::max(1, base_w + extra_w);
      }

      auto collector = std::make_unique<BatchedRocketSimCollector>(
          mode_cfg, obs_builder, action_parser, done_condition, pin_host);
      collector->set_mode(mode);
      collectors_.push_back(std::move(collector));
    }
  }

  // recompute agent counts
  total_agents_ = total_agents_for_collectors(collectors_);
  if (total_agents_ == 0) {
    throw std::invalid_argument("rebuild_collectors produced zero agents");
  }

  shard_devices_ = assign_shard_devices(compute_devices_, collectors_.size());

#ifdef PULSAR_HAS_CUDA
  if (device_.is_cuda()) {
    shard_collection_streams_.clear();
    for (std::size_t i = 0; i < collectors_.size(); ++i) {
      const torch::Device shard_device = shard_devices_[i];
      shard_collection_streams_.push_back(
          at::cuda::getStreamFromPool(false, shard_device.index()));
    }
  }
#endif

  // Rebuild persistent collection actors for the new shard count and device layout.
  collection_actors_.clear();
  collection_actors_.reserve(shard_devices_.size());
  for (std::size_t i = 0; i < shard_devices_.size(); ++i) {
    if (shard_devices_[i] == device_) {
      collection_actors_.push_back(actor_snapshot_);
    } else {
      collection_actors_.push_back(clone_ppo_actor(actor_snapshot_, shard_devices_[i]));
      collection_actors_.back()->eval();
    }
  }

  // rebuild rollout storage
  const int action_dim = action_dim_for_collectors(collectors_);
  rollout_ = make_rollout_storage(config_, static_cast<int>(total_agents_), action_dim, false);
  rollout_B_ = make_rollout_storage(config_, static_cast<int>(total_agents_), action_dim, false);

  // recompute shard agent offsets
  shard_agent_offsets_.clear();
  std::int64_t agent_offset = 0;
  for (const auto& collector : collectors_) {
    if (!collector) continue;
    shard_agent_offsets_.push_back(agent_offset);
    agent_offset += static_cast<std::int64_t>(collector->total_agents());
  }

  // Pre-allocate pinned action index buffers per shard.
  shard_action_buffers_cpu_.clear();
  shard_action_buffers_cpu_.reserve(collectors_.size());
  for (const auto& collector : collectors_) {
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    if (use_pinned_host_buffers_) {
      opts = opts.pinned_memory(true);
    }
    shard_action_buffers_cpu_.push_back(
        torch::zeros({static_cast<long>(collector->total_agents())}, opts));
  }

  // re-apply curriculum
  apply_curriculum_to_collectors();

  std::cout << "rebuilt_collectors collectors=" << collectors_.size()
            << " modes=" << mode_envs.size()
            << " total_agents=" << total_agents_
            << " total_envs=" << total_envs
            << " devices=" << join_device_list(shard_devices_)
            << " rollout_length=" << config_.ppo.rollout_length << '\n';
}

void APPOTrainer::apply_curriculum_lr() {
  float lr = curriculum_.learning_rate();
  for (auto& opt_group : actor_optimizer_.param_groups()) {
    opt_group.options().set_lr(lr);
  }
}

void APPOTrainer::maybe_initialize_from_checkpoint() {
  std::filesystem::path base;
  const bool explicit_checkpoint = !config_.ppo.init_checkpoint.empty();
  if (explicit_checkpoint) {
    base = std::filesystem::path(config_.ppo.init_checkpoint);
  } else {
    auto latest = find_latest_checkpoint(run_output_root_);
    if (!latest.has_value()) return;
    base = std::move(*latest);
  }
  const ExperimentConfig checkpoint_config = load_experiment_config((base / "config.json").string());
  const CheckpointMetadata metadata = load_checkpoint_metadata((base / "metadata.json").string());
  if (explicit_checkpoint) {
    validate_inference_checkpoint_metadata(metadata, checkpoint_config);
    if (log_initialization_) {
      const bool config_matches = metadata.config_hash == config_hash(config_);
      const int checkpoint_state_version = metadata.extra.value("trainer_state_version", 0);
      if (!config_matches || checkpoint_state_version != kTrainerStateVersion) {
        std::cerr << "warning: explicit checkpoint " << base.string()
                  << " differs from active training state"
                  << " config_match=" << (config_matches ? 1 : 0)
                  << " trainer_state_version=" << checkpoint_state_version
                  << " expected_trainer_state_version=" << kTrainerStateVersion
                  << '\n';
      }
    }
  } else {
    try {
      validate_checkpoint_metadata(metadata, config_);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "Refusing to auto-resume checkpoint " + base.string() +
          ": " + e.what() +
          " Use a fresh output directory for a new run, or set ppo.init_checkpoint explicitly if you intentionally want this checkpoint.");
    }
    const int checkpoint_state_version = metadata.extra.value("trainer_state_version", 0);
    if (checkpoint_state_version != kTrainerStateVersion) {
      throw std::runtime_error(
          "Refusing to auto-resume checkpoint " + base.string() +
          ": trainer_state_version=" + std::to_string(checkpoint_state_version) +
          " expected=" + std::to_string(kTrainerStateVersion) +
          ". Use a fresh output directory for a new run, or set ppo.init_checkpoint explicitly if you intentionally want this older checkpoint.");
    }
  }

  const std::filesystem::path state_path = base / "state.pt";
  if (std::filesystem::exists(state_path)) {
    load_training_state(state_path);
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  } else {
    torch::serialize::InputArchive actor_archive;
    actor_archive.load_from((base / "model.pt").string(), device_);
    actor_->load(actor_archive);
    actor_normalizer_.load(actor_archive);
    actor_->to(device_);
    actor_normalizer_.to(device_);
    if (std::filesystem::exists(base / "actor_optimizer.pt")) {
      torch::serialize::InputArchive optimizer_archive;
      optimizer_archive.load_from((base / "actor_optimizer.pt").string(), device_);
      actor_optimizer_.load(optimizer_archive);
    }
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  }

  if (metadata.extra.contains("recent_scored_rates")) {
    recent_scored_rates_.clear();
    for (const auto& v : metadata.extra["recent_scored_rates"]) {
      recent_scored_rates_.push_back(v.get<double>());
    }
  }
  if (self_play_manager_ && self_play_manager_->enabled()) {
    if (metadata.extra.contains("self_play_rng_state")) {
      self_play_manager_->restore_rng_state(metadata.extra["self_play_rng_state"].get<std::string>());
    }
    if (metadata.extra.contains("self_play_ratings")) {
      std::map<std::string, double> ratings;
      for (const auto& [mode, v] : metadata.extra["self_play_ratings"].items()) {
        ratings[mode] = v.get<double>();
      }
      self_play_manager_->restore_ratings(ratings);
    }
  }
  if (metadata.extra.contains("curriculum_stage") && curriculum_.enabled()) {
    CurriculumState restored;
    restored.stage_index = metadata.extra["curriculum_stage"].get<int>();
    restored.previous_stage_index = metadata.extra.value("curriculum_previous_stage", -1);
    restored.agent_steps_in_stage = metadata.extra.value("curriculum_agent_steps", 0LL);
    restored.promotion_counter = metadata.extra.value("curriculum_promotion_counter", 0);
    restored.current_mode = metadata.extra.value("curriculum_current_mode", std::string{"1v1"});
    if (metadata.extra.contains("curriculum_mode_touch_rates")) {
      for (const auto& [mode, arr] : metadata.extra["curriculum_mode_touch_rates"].items()) {
        std::deque<double> deq;
        for (const auto& v : arr) deq.push_back(v.get<double>());
        restored.mode_touch_rates[mode] = deq;
      }
    }
    if (metadata.extra.contains("curriculum_mode_multi_touch_rates")) {
      for (const auto& [mode, arr] : metadata.extra["curriculum_mode_multi_touch_rates"].items()) {
        std::deque<double> deq;
        for (const auto& v : arr) deq.push_back(v.get<double>());
        restored.mode_multi_touch_rates[mode] = deq;
      }
    }
    if (metadata.extra.contains("curriculum_mode_scored_rates")) {
      for (const auto& [mode, arr] : metadata.extra["curriculum_mode_scored_rates"].items()) {
        std::deque<double> deq;
        for (const auto& v : arr) deq.push_back(v.get<double>());
        restored.mode_scored_rates[mode] = deq;
      }
    }
    // backward compat: old single-mode deques
    if (metadata.extra.contains("curriculum_touch_rates") && restored.mode_touch_rates.empty()) {
      const auto& arr = metadata.extra["curriculum_touch_rates"];
      std::deque<double> deq;
      for (const auto& v : arr) deq.push_back(v.get<double>());
      restored.mode_touch_rates[restored.current_mode] = deq;
    }
    if (metadata.extra.contains("curriculum_scored_rates") && restored.mode_scored_rates.empty()) {
      const auto& arr = metadata.extra["curriculum_scored_rates"];
      std::deque<double> deq;
      for (const auto& v : arr) deq.push_back(v.get<double>());
      restored.mode_scored_rates[restored.current_mode] = deq;
    }
    curriculum_.restore_state(restored);
    std::cout << "restored_curriculum stage=" << restored.stage_index
              << " mode=" << restored.current_mode
              << " steps_in_stage=" << restored.agent_steps_in_stage << '\n';
  }
  if (log_initialization_) {
    std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
  }
}

TrainerMetrics APPOTrainer::update_actor(RolloutStorage& rollout) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "update_actor");
#ifdef PULSAR_HAS_CUDA
  std::optional<c10::cuda::CUDAStream> prev_train_stream;
  if (training_stream_.has_value()) {
    prev_train_stream = c10::cuda::getCurrentCUDAStream(device_.index());
    c10::cuda::setCurrentCUDAStream(*training_stream_);
  }
#endif
  const auto update_start = std::chrono::steady_clock::now();
  TrainerMetrics metrics{};

  float effective_entropy_coef = config_.ppo.entropy_coef;
  float effective_entropy_floor_coef = config_.ppo.entropy_floor_coef;
  if (config_.ppo.adaptive_entropy && !recent_scored_rates_.empty()) {
    double sum = 0.0;
    for (double v : recent_scored_rates_) sum += v;
    double recent_score = sum / static_cast<double>(recent_scored_rates_.size());
    double progress = std::clamp(
        (recent_score - config_.ppo.entropy_decay_score) /
            std::max(1.0e-6, 1.0 - config_.ppo.entropy_decay_score),
        0.0, 1.0);
    effective_entropy_coef = static_cast<float>(
        config_.ppo.entropy_coef + progress * (config_.ppo.entropy_low_coef - config_.ppo.entropy_coef));
    effective_entropy_floor_coef = config_.ppo.entropy_floor_coef * static_cast<float>(1.0 - progress);
  }

  const int seq_len = std::max(1, config_.ppo.rollout_length);
  const int max_forward_samples = effective_max_forward_samples(config_.model, device_);
  const int agents_per_forward = std::max(1, max_forward_samples / seq_len);
  const int requested_logical_agents_per_batch = std::max(1, config_.ppo.minibatch_size / seq_len);
  const int total_agents = rollout.num_agents();
  constexpr int kMaxLogicalMinibatchesPerUpdate = 25;
  const int update_epochs = std::max(1, config_.ppo.update_epochs);
  const int min_agents_for_minibatch_cap = std::max(
      1,
      static_cast<int>(
          (static_cast<std::int64_t>(total_agents) * update_epochs + kMaxLogicalMinibatchesPerUpdate - 1) /
          kMaxLogicalMinibatchesPerUpdate));
  const int logical_agents_per_batch = std::min(
      total_agents,
      std::max(requested_logical_agents_per_batch, min_agents_for_minibatch_cap));
  const int rollout_steps = rollout.rollout_length();
  const bool use_cuda_amp = device_.is_cuda() && config_.ppo.cuda_amp;
  const double cuda_amp_loss_scale = 1.0;
  const int optimizer_accumulation_steps = std::max(1, config_.ppo.optimizer_accumulation_steps);
  int minibatches_per_epoch = 0;
  int microbatches_per_epoch = 0;
  for (int offset = 0; offset < total_agents; offset += logical_agents_per_batch) {
    const int count = std::min(logical_agents_per_batch, total_agents - offset);
    ++minibatches_per_epoch;
    microbatches_per_epoch += (count + agents_per_forward - 1) / agents_per_forward;
  }
  if (benchmark_progress_) {
    std::cout << "bench_update_phase_start"
              << " rollout_steps=" << rollout_steps
              << " total_agents=" << total_agents
              << " update_epochs=" << config_.ppo.update_epochs
              << " logical_agents_per_batch=" << logical_agents_per_batch
              << " effective_forward_samples=" << max_forward_samples
              << " agents_per_forward=" << agents_per_forward
              << " minibatches_per_epoch=" << minibatches_per_epoch
              << " microbatches_per_epoch=" << microbatches_per_epoch
              << " optimizer_accumulation_steps=" << optimizer_accumulation_steps
              << " cuda_amp=" << (use_cuda_amp ? 1 : 0)
              << " loss_scale=" << cuda_amp_loss_scale
              << " pcgrad=" << (config_.ppo.pcgrad ? 1 : 0)
              << '\n' << std::flush;
  }
  std::int64_t metric_steps = 0;
  torch::Tensor policy_loss_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor value_loss_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor entropy_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor grad_norm_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor policy_approx_kl_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor policy_clip_fraction_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor goal_critic_loss_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor goal_score_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  torch::Tensor sampled_goal_distance_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  double policy_log_ratio_abs_max = 0.0;
  int accumulated_minibatches = 0;
  bool accumulated_has_backward = false;
  double accumulated_total_active = 0.0;
  double accumulated_policy_kl_sum = 0.0;
  double accumulated_policy_kl_count = 0.0;

  const auto& all_values = rollout.all_values();
  const auto& all_rewards = rollout.all_rewards();
  if (rollout_steps <= 0) {
    metrics.effective_entropy_coef = static_cast<double>(effective_entropy_coef);
    return metrics;
  }
  const torch::Tensor extrinsic_values = all_values.at("extrinsic").narrow(0, 0, rollout_steps).to(device_);
  const torch::Tensor extrinsic_rewards = all_rewards.at("extrinsic").narrow(0, 0, rollout_steps).to(device_);
  const torch::Tensor rollout_dones = rollout.dones.narrow(0, 0, rollout_steps).to(device_);
  const torch::Tensor rollout_bootstrap_truncated = rollout.bootstrap_truncated.narrow(0, 0, rollout_steps).to(device_);

  torch::Tensor active_mask = rollout.learner_active.narrow(0, 0, rollout_steps).to(device_) > 0.5F;
  torch::Tensor sparse_advantages;
  torch::Tensor normalized_advantages;
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_gae");
    const auto gae_start = std::chrono::steady_clock::now();
    if (benchmark_progress_) {
      std::cout << "bench_update_gae_start\n" << std::flush;
    }
    torch::Tensor terminal_values;
    if (rollout_bootstrap_truncated.any().item<bool>()) {
      torch::NoGradGuard no_grad;
      torch::Tensor term_obs = rollout.terminal_observations.narrow(0, 0, rollout_steps);
      auto term_flat = term_obs.reshape({rollout_steps * total_agents, config_.model.observation_dim});
      const int total_term_samples = rollout_steps * total_agents;
      const int max_term_batch = effective_max_forward_samples(config_.model, device_);
      std::vector<torch::Tensor> term_value_chunks;
      for (int offset = 0; offset < total_term_samples; offset += max_term_batch) {
        int batch = std::min(max_term_batch, total_term_samples - offset);
        auto chunk = actor_normalizer_.normalize(term_flat.slice(0, offset, offset + batch).to(device_));
        auto chunk_goal = policy_goal_values_like(chunk, config_.goal_critic.goal_dim);
        auto chunk_out = actor_->forward_step(chunk, chunk_goal).value_win_logits.squeeze(-1);
        term_value_chunks.push_back(chunk_out);
      }
      auto term_values_flat = torch::cat(term_value_chunks, 0);
      terminal_values = term_values_flat.reshape({rollout_steps, total_agents});
    }
    auto final_values_map = rollout.final_values();
    torch::Tensor final_extrinsic_values = final_values_map.count("extrinsic") && final_values_map.at("extrinsic").defined()
        ? final_values_map.at("extrinsic").to(device_)
        : torch::Tensor{};
    sparse_advantages = compute_gae(
      extrinsic_values,
      extrinsic_rewards,
      rollout_dones,
      config_.ppo.gamma,
      config_.ppo.gae_lambda,
      final_extrinsic_values,
      rollout_bootstrap_truncated,
      terminal_values);
    normalized_advantages = normalize_advantage(sparse_advantages, active_mask);
    if (benchmark_progress_) {
      const double gae_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - gae_start).count();
      std::cout << "bench_update_gae_done seconds=" << gae_seconds << '\n' << std::flush;
    }
  }
  torch::Tensor sparse_returns = sparse_advantages + extrinsic_values.detach();

  int completed_minibatches = 0;
  const size_t num_update_gpus = compute_devices_.size();
  for (int epoch = 0; epoch < config_.ppo.update_epochs; ++epoch) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "update_epoch");
    const torch::Tensor perm = torch::randperm(total_agents, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    for (int agent_offset = 0; agent_offset < total_agents; agent_offset += logical_agents_per_batch) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "update_minibatch");
      const auto minibatch_start = std::chrono::steady_clock::now();

      const int count = std::min(logical_agents_per_batch, total_agents - agent_offset);
      const torch::Tensor agent_indices = perm.narrow(0, agent_offset, count);

      // Determine if this minibatch spans multiple modes.
      // Fast-path: if the entire rollout has only one mode (common in curriculum),
      // skip expensive mode detection and splitting.
      torch::Tensor agent_mode_ids = rollout.mode_ids[0].index_select(0, agent_indices);
      const bool use_pcgrad = config_.ppo.pcgrad;
      int num_modes_present = 1;
      bool has_1v1 = false, has_2v2 = false, has_3v3 = false;
      bool split_modes_for_pcgrad = false;

      // Only check mode splits if PCGrad is enabled AND multiple modes are possible
      if (use_pcgrad) {
        // Quick check: if first agent's mode matches all others (single-mode curriculum stage),
        // skip the expensive per-mode .any() checks.
        const auto first_mode = agent_mode_ids[0].item<std::int8_t>();
        const bool all_same = (agent_mode_ids != first_mode).sum().item<int>() == 0;
        if (!all_same) {
          has_1v1 = (agent_mode_ids == 1).any().item<bool>();
          has_2v2 = (agent_mode_ids == 2).any().item<bool>();
          has_3v3 = (agent_mode_ids == 3).any().item<bool>();
          num_modes_present = (has_1v1 ? 1 : 0) + (has_2v2 ? 1 : 0) + (has_3v3 ? 1 : 0);
          split_modes_for_pcgrad = num_modes_present > 1;
        }
      }
      std::vector<torch::Tensor> mode_agent_indices_list;
      if (split_modes_for_pcgrad) {
        if (has_1v1) {
          torch::Tensor mask = agent_mode_ids == 1;
          mode_agent_indices_list.push_back(agent_indices.index({mask}));
        }
        if (has_2v2) {
          torch::Tensor mask = agent_mode_ids == 2;
          mode_agent_indices_list.push_back(agent_indices.index({mask}));
        }
        if (has_3v3) {
          torch::Tensor mask = agent_mode_ids == 3;
          mode_agent_indices_list.push_back(agent_indices.index({mask}));
        }
      } else {
        mode_agent_indices_list.push_back(agent_indices);
      }

      // Process each mode group (or the full batch if single mode).
      std::vector<std::vector<CapturedGrad>> pcgrad_groups;
      double combined_total_active = 0.0;

      for (const torch::Tensor& mode_agent_indices : mode_agent_indices_list) {
        const int mode_count = static_cast<int>(mode_agent_indices.numel());
        if (mode_count == 0) continue;

        // Fast-path: if all agents are learners (common in curriculum without self-play),
        // mode_total_active is just mode_count * rollout_length. Skip expensive tensor sum.
        double mode_total_active;
        bool mode_all_active = false;
        {
          torch::Tensor active_slice = rollout.learner_active.narrow(0, 0, rollout.rollout_length())
              .index_select(1, mode_agent_indices);
          const float active_sum = active_slice.sum().item<float>();
          const float active_all = static_cast<float>(mode_count * rollout.rollout_length());
          mode_total_active = static_cast<double>(active_sum);
          mode_all_active = active_sum >= active_all - 0.5F;
          if (mode_total_active <= 0.0) continue;
        }
        combined_total_active += mode_total_active;

        if (use_pcgrad || accumulated_minibatches == 0) {
          actor_optimizer_.zero_grad();
        }
        std::vector<CapturedGrad> mode_task_grad_group;
        std::vector<CapturedGrad> mode_goal_critic_grad_group;
        std::vector<CapturedGrad> mode_goal_actor_grad_group;

        // Launch one async task per GPU. Each task processes all micro-batches
        // for its chunk of agents, accumulating GPU-local CapturedGrad groups.
        struct GpuTaskResult {
          std::vector<CapturedGrad> task_group;
          std::vector<CapturedGrad> goal_critic_group;
          std::vector<CapturedGrad> goal_actor_group;
          double policy_loss_sum = 0.0;
          double value_loss_sum = 0.0;
          double entropy_sum = 0.0;
          double policy_approx_kl = 0.0;
          double policy_clip_frac = 0.0;
          double policy_log_ratio_max = 0.0;
          double goal_critic_loss = 0.0;
          double goal_score = 0.0;
          double sampled_goal_dist = 0.0;
          double fwd_bwd_seconds = 0.0;
          bool has_backward = false;
          std::int64_t active_count = 0;
        };
        std::vector<std::future<GpuTaskResult>> gpu_futures;
        gpu_futures.reserve(num_update_gpus);

        const int mode_count_local = mode_count;
        const int agents_per_forward_local = agents_per_forward;
        const int seq_len_local = seq_len;
        const int rollout_steps_local = rollout_steps;
        const double mode_total_active_local = mode_total_active;
        const bool mode_all_active_local = mode_all_active;
        const bool use_pcgrad_local = use_pcgrad;
        const bool use_cuda_amp_local = use_cuda_amp;
        const double cuda_amp_loss_scale_local = cuda_amp_loss_scale;
        const int optimizer_accumulation_steps_local = optimizer_accumulation_steps;

        for (size_t g = 0; g < num_update_gpus; ++g) {
          const torch::Device gpu_dev = compute_devices_[g];
          PPOActor gpu_act = (g == 0) ? actor_ : compute_actors_[g - 1];

          gpu_futures.push_back(std::async(std::launch::async, [=, &rollout, &normalized_advantages, &sparse_returns,
              &mode_agent_indices, &effective_entropy_coef, &effective_entropy_floor_coef, this]() mutable -> GpuTaskResult {
            GpuTaskResult result;
            std::vector<CapturedGrad>& gpu_task_group = result.task_group;
            std::vector<CapturedGrad>& gpu_goal_critic_group = result.goal_critic_group;
            std::vector<CapturedGrad>& gpu_goal_actor_group = result.goal_actor_group;
            torch::Tensor policy_loss_metric;
            torch::Tensor value_loss_metric;
            torch::Tensor entropy_metric;
            torch::Tensor policy_approx_kl_metric;
            torch::Tensor policy_clip_frac_metric;
            torch::Tensor policy_log_ratio_max_metric;
            torch::Tensor goal_critic_loss_metric;
            torch::Tensor goal_score_metric;
            torch::Tensor sampled_goal_dist_metric;
            auto add_metric = [](torch::Tensor& dst, const torch::Tensor& value, double weight) {
              const torch::Tensor term = value.detach().to(torch::kFloat32) * weight;
              dst = dst.defined() ? dst + term : term;
            };
            auto max_metric = [](torch::Tensor& dst, const torch::Tensor& value) {
              const torch::Tensor term = value.detach().to(torch::kFloat32);
              dst = dst.defined() ? torch::maximum(dst, term) : term;
            };
            const auto task_compute_start = std::chrono::steady_clock::now();

            for (int micro_agent_offset = 0; micro_agent_offset < mode_count_local; micro_agent_offset += agents_per_forward_local) {
              const int micro_count = std::min(agents_per_forward_local, mode_count_local - micro_agent_offset);
              const int gpu_chunk_start = static_cast<int>(g * micro_count / num_update_gpus);
              const int gpu_chunk_end = static_cast<int>((g + 1) * micro_count / num_update_gpus);
              const int gpu_chunk_count = gpu_chunk_end - gpu_chunk_start;
              if (gpu_chunk_count <= 0) continue;

              torch::Tensor gpu_micro_indices = mode_agent_indices.narrow(0, micro_agent_offset + gpu_chunk_start, gpu_chunk_count);

              torch::Tensor mode_gpu_obs_mb = rollout.obs.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_episode_starts_mb = rollout.episode_starts.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_action_masks_mb = rollout.action_masks.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_learner_active_mb = rollout.learner_active.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_actions_mb = rollout.actions.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_action_log_probs_mb = rollout.action_log_probs.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices).to(gpu_dev);
              torch::Tensor mode_gpu_advantages_mb = normalized_advantages.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices.to(device_)).to(gpu_dev);
              torch::Tensor mode_gpu_returns_mb = sparse_returns.narrow(0, 0, rollout_steps_local).index_select(1, gpu_micro_indices.to(device_)).to(gpu_dev);

              for (int seq_start = 0; seq_start < rollout_steps_local; seq_start += seq_len_local) {
                const int chunk_start = seq_start;
                const int chunk_end = std::min(rollout_steps_local, chunk_start + seq_len_local);
                const int loss_steps = chunk_end - chunk_start;
                if (loss_steps <= 0) continue;

                const torch::Tensor obs = mode_gpu_obs_mb.narrow(0, chunk_start, loss_steps);
                ActorSequenceOutput output;
                {
                  OptionalCudaAutocastGuard autocast_guard(use_cuda_amp_local);
                  const torch::Tensor goal_values = policy_goal_values_like(obs, config_.goal_critic.goal_dim);
                  const torch::Tensor episode_starts = mode_gpu_episode_starts_mb.narrow(0, chunk_start, loss_steps);
                  output = gpu_act->forward_sequence(obs, goal_values, episode_starts);
                }

                torch::Tensor encoded = output.encoded;
                torch::Tensor policy_logits = output.policy_logits;
                torch::Tensor features = output.features;
                const torch::Tensor action_masks = mode_gpu_action_masks_mb.narrow(0, chunk_start, loss_steps).to(torch::kBool);
                const torch::Tensor learner_active = mode_gpu_learner_active_mb.narrow(0, chunk_start, loss_steps);
                const torch::Tensor old_actions = mode_gpu_actions_mb.narrow(0, chunk_start, loss_steps);
                const torch::Tensor old_log_probs = mode_gpu_action_log_probs_mb.narrow(0, chunk_start, loss_steps);
                const torch::Tensor chunk_advantages = mode_gpu_advantages_mb.narrow(0, chunk_start, loss_steps);

                const auto samples = loss_steps * gpu_chunk_count;
                torch::Tensor flat_logits = policy_logits.reshape({samples, config_.model.action_dim});
                torch::Tensor flat_features = features.reshape({samples, static_cast<int64_t>(gpu_act->feature_dim())});
                torch::Tensor flat_masks = action_masks.reshape({samples, config_.model.action_dim});
                torch::Tensor flat_actions = old_actions.reshape({samples});
                torch::Tensor flat_old_log_probs = old_log_probs.reshape({samples});
                torch::Tensor flat_advantages = chunk_advantages.reshape({samples});

                const bool all_active = mode_all_active_local;
                const torch::Tensor flat_active = all_active ? torch::Tensor{} : learner_active.reshape({samples}) > 0.5F;
                const torch::Tensor active_logits = all_active ? flat_logits.to(torch::kFloat32) : flat_logits.index({flat_active}).to(torch::kFloat32);
                const torch::Tensor active_features = all_active ? flat_features.to(torch::kFloat32) : flat_features.index({flat_active}).to(torch::kFloat32);
                const torch::Tensor active_masks = all_active ? flat_masks : flat_masks.index({flat_active});
                const torch::Tensor active_actions = all_active ? flat_actions : flat_actions.index({flat_active});
                const torch::Tensor active_old_log_probs = all_active ? flat_old_log_probs.to(torch::kFloat32) : flat_old_log_probs.index({flat_active}).to(torch::kFloat32);
                const torch::Tensor active_advantages = all_active ? flat_advantages.to(torch::kFloat32) : flat_advantages.index({flat_active}).to(torch::kFloat32);
                const auto active_sample_count = all_active ? samples : active_logits.size(0);
                if (active_sample_count == 0) continue;
                const auto active_samples = static_cast<double>(active_sample_count);
                result.active_count += active_sample_count;

                const torch::Tensor log_probs = torch::log_softmax(apply_action_mask_to_logits(active_logits, active_masks), -1);
                const torch::Tensor current_log_probs = log_probs.gather(1, active_actions.unsqueeze(1)).squeeze(1);
                torch::Tensor bounded_current_log_probs = current_log_probs;
                const torch::Tensor raw_log_ratio = current_log_probs - active_old_log_probs;
                if (config_.ppo.max_policy_log_ratio > 0.0F) {
                  bounded_current_log_probs = active_old_log_probs + raw_log_ratio.clamp(-config_.ppo.max_policy_log_ratio, config_.ppo.max_policy_log_ratio);
                }
                {
                  const torch::Tensor metric_log_ratio = raw_log_ratio.detach().to(torch::kFloat32).clamp(-20.0F, 20.0F);
                  const torch::Tensor approx_kl = ((torch::exp(metric_log_ratio) - 1.0F) - metric_log_ratio).mean();
                  const torch::Tensor clip_fraction = (raw_log_ratio.detach().abs() > std::log1p(static_cast<double>(config_.ppo.clip_range))).to(torch::kFloat32).mean();
                  add_metric(policy_approx_kl_metric, approx_kl, active_samples);
                  add_metric(policy_clip_frac_metric, clip_fraction, active_samples);
                  max_metric(policy_log_ratio_max_metric, raw_log_ratio.detach().abs().max());
                }

                torch::Tensor policy_loss = clipped_ppo_policy_loss(
                    bounded_current_log_probs,
                    active_old_log_probs,
                    active_advantages,
                    config_.ppo.clip_range).mean();
                const torch::Tensor entropy_values = masked_action_entropy(active_logits, active_masks);
                const torch::Tensor entropy = entropy_values.mean();
                torch::Tensor entropy_floor_loss = torch::zeros({}, active_advantages.options());
                if (config_.ppo.entropy_floor > 0.0F && effective_entropy_floor_coef > 0.0F) {
                  const torch::Tensor entropy_floor_mask = active_masks.to(torch::kFloat32).sum(-1) > 1.0F;
                  const torch::Tensor entropy_floor_count = entropy_floor_mask.to(torch::kFloat32).sum().clamp_min(1.0F);
                  const torch::Tensor entropy_floor_penalty =
                      torch::relu(config_.ppo.entropy_floor - entropy_values).square()
                          * entropy_floor_mask.to(torch::kFloat32);
                  entropy_floor_loss = effective_entropy_floor_coef * entropy_floor_penalty.sum() / entropy_floor_count;
                }

                torch::Tensor chunk_returns = mode_gpu_returns_mb.narrow(0, chunk_start, loss_steps).reshape({samples});
                torch::Tensor active_returns = all_active ? chunk_returns.to(torch::kFloat32) : chunk_returns.index({flat_active}).to(torch::kFloat32);
                torch::Tensor value_win_logits = output.value_win_logits;
                torch::Tensor flat_value_win_logits = value_win_logits.reshape({samples, 1});
                torch::Tensor active_value_win_logits = all_active ? flat_value_win_logits.to(torch::kFloat32) : flat_value_win_logits.index({flat_active}).to(torch::kFloat32);
                torch::Tensor value_loss = smooth_l1_value_loss(active_value_win_logits.squeeze(-1), active_returns, config_.ppo.value_loss_delta);

                torch::Tensor goal_loss = torch::zeros({}, active_advantages.options());
                torch::Tensor actor_goal_loss = torch::zeros({}, active_advantages.options());
                torch::Tensor chunk_goal_score = torch::zeros({}, active_advantages.options());
                torch::Tensor chunk_sampled_goal_norm = torch::zeros({}, active_advantages.options());
                {
                  torch::Tensor chunk_goal_pos =
                      rollout.goal_positions.narrow(0, chunk_start, loss_steps).index_select(1, gpu_micro_indices).to(gpu_dev);
                  torch::Tensor chunk_dones =
                      rollout.dones.narrow(0, chunk_start, loss_steps).index_select(1, gpu_micro_indices).to(gpu_dev);
                  torch::Tensor chunk_ep_starts =
                      rollout.episode_starts.narrow(0, chunk_start, loss_steps).index_select(1, gpu_micro_indices).to(gpu_dev);
                  torch::Tensor future_goal_pos = sample_future_goal_positions(chunk_goal_pos, chunk_dones, chunk_ep_starts, config_.goal_critic.max_future_horizon);
                  torch::Tensor flat_future_goal_pos = future_goal_pos.reshape({samples, config_.goal_critic.goal_dim});
                  torch::Tensor active_future_goal_pos = all_active ? flat_future_goal_pos : flat_future_goal_pos.index({flat_active});

                  const auto active_count = active_features.size(0);
                  const int cb_size = config_.goal_critic.contrastive_batch_size;
                  torch::Tensor selected_features = active_features;
                  torch::Tensor selected_actions = active_actions;
                  torch::Tensor selected_logits = active_logits;
                  torch::Tensor selected_masks = active_masks;
                  torch::Tensor selected_future_goal_pos = active_future_goal_pos;
                  if (active_count > cb_size) {
                    const torch::Tensor idx = torch::randperm(active_count, active_actions.options()).narrow(0, 0, cb_size);
                    selected_features = selected_features.index({idx});
                    selected_actions = selected_actions.index({idx});
                    selected_logits = selected_logits.index({idx});
                    selected_masks = selected_masks.index({idx});
                    selected_future_goal_pos = selected_future_goal_pos.index({idx});
                  }
                  torch::Tensor sa_emb = gpu_act->goal_critic()->sa_embedding(selected_features.detach(), selected_actions);
                  torch::Tensor g_emb = gpu_act->goal_critic()->goal_embedding(selected_future_goal_pos);
                  goal_loss = compute_symmetric_infonce_loss(compute_pairwise_negative_l2_logits(sa_emb, g_emb), config_.goal_critic.logsumexp_penalty_coeff);
                  actor_goal_loss = goal_actor_critic_loss(
                      gpu_act->goal_critic(),
                      active_features,
                      active_logits,
                      active_masks,
                      active_future_goal_pos,
                      config_.goal_critic.contrastive_batch_size);
                  chunk_goal_score = -((sa_emb.detach() - g_emb.detach()).square().sum(-1).clamp_min(1.0e-8F)).mean();
                  chunk_sampled_goal_norm = active_future_goal_pos.norm(2, -1).mean();
                  add_metric(sampled_goal_dist_metric, chunk_sampled_goal_norm, active_samples);
                  add_metric(goal_critic_loss_metric, goal_loss, active_samples);
                  add_metric(goal_score_metric, chunk_goal_score, active_samples);
                }

                const auto sample_weight = active_samples / mode_total_active_local;
                const torch::Tensor task_loss = policy_loss + config_.ppo.value_coef * value_loss + entropy_floor_loss - effective_entropy_coef * entropy;

                if (use_pcgrad_local) {
                  const torch::Tensor weighted_task_loss = task_loss * sample_weight;
                  const torch::Tensor weighted_goal_critic_loss = config_.goal_critic.lambda_Zg * goal_loss * sample_weight;
                  const torch::Tensor weighted_goal_actor_loss = config_.goal_critic.lambda_goal_actor * actor_goal_loss * sample_weight;
                  const int effective_accum = 1;
                  // Collect all active objectives for this chunk.
                  std::vector<std::pair<torch::Tensor, std::vector<CapturedGrad>*>> objective_losses;
                  objective_losses.push_back({weighted_task_loss / static_cast<double>(effective_accum), &gpu_task_group});
                  if (config_.goal_critic.lambda_Zg > 0.0F && weighted_goal_critic_loss.requires_grad()) {
                    objective_losses.push_back({weighted_goal_critic_loss / static_cast<double>(effective_accum), &gpu_goal_critic_group});
                  }
                  if (config_.goal_critic.lambda_goal_actor > 0.0F && weighted_goal_actor_loss.requires_grad()) {
                    objective_losses.push_back({weighted_goal_actor_loss / static_cast<double>(effective_accum), &gpu_goal_actor_group});
                  }

                  // Compute every objective on every GPU chunk. The previous
                  // implementation assigned one objective per GPU, which meant
                  // single-GPU runs trained only the task objective and multi-GPU
                  // runs trained each objective on different data shards. PCGrad
                  // requires per-objective gradients for the same local batch.
                  for (int obj_index = 0; obj_index < static_cast<int>(objective_losses.size()); ++obj_index) {
                    zero_existing_gradients(*gpu_act);
                    const bool retain_graph = obj_index + 1 < static_cast<int>(objective_losses.size());
                    (objective_losses[obj_index].first * cuda_amp_loss_scale_local).backward({}, retain_graph);
                    accumulate_gradients(*gpu_act, *objective_losses[obj_index].second);
                    result.has_backward = true;
                  }
                } else {
                  const torch::Tensor loss =
                      task_loss + config_.goal_critic.lambda_Zg * goal_loss + config_.goal_critic.lambda_goal_actor * actor_goal_loss;
                  (loss * sample_weight / static_cast<double>(optimizer_accumulation_steps_local) * cuda_amp_loss_scale_local).backward();
                  result.has_backward = true;
                }

                add_metric(policy_loss_metric, policy_loss, active_samples);
                add_metric(value_loss_metric, value_loss, active_samples);
                add_metric(entropy_metric, entropy, active_samples);
              }
            }
            const auto metric_opts = torch::TensorOptions().dtype(torch::kFloat32).device(gpu_dev);
            const torch::Tensor zero_metric = torch::zeros({}, metric_opts);
            auto scalar_or_zero = [&](const torch::Tensor& tensor) -> torch::Tensor {
              return tensor.defined() ? tensor.to(torch::kFloat32) : zero_metric;
            };
            const torch::Tensor packed_metrics = torch::stack({
                scalar_or_zero(policy_loss_metric),
                scalar_or_zero(value_loss_metric),
                scalar_or_zero(entropy_metric),
                scalar_or_zero(policy_approx_kl_metric),
                scalar_or_zero(policy_clip_frac_metric),
                scalar_or_zero(policy_log_ratio_max_metric),
                scalar_or_zero(goal_critic_loss_metric),
                scalar_or_zero(goal_score_metric),
                scalar_or_zero(sampled_goal_dist_metric),
            }).to(torch::kCPU);
            const float* metric_data = packed_metrics.data_ptr<float>();
            result.policy_loss_sum = metric_data[0];
            result.value_loss_sum = metric_data[1];
            result.entropy_sum = metric_data[2];
            result.policy_approx_kl = metric_data[3];
            result.policy_clip_frac = metric_data[4];
            result.policy_log_ratio_max = metric_data[5];
            result.goal_critic_loss = metric_data[6];
            result.goal_score = metric_data[7];
            result.sampled_goal_dist = metric_data[8];
            result.fwd_bwd_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - task_compute_start).count();
            return result;
          }));
        }

        // Wait for all GPU tasks and reduce into primary mode groups.
        for (auto& fut : gpu_futures) {
          GpuTaskResult task_result = fut.get();

          reduce_captured_gradients(*actor_, mode_task_grad_group, task_result.task_group, device_);
          reduce_captured_gradients(*actor_, mode_goal_critic_grad_group, task_result.goal_critic_group, device_);
          reduce_captured_gradients(*actor_, mode_goal_actor_grad_group, task_result.goal_actor_group, device_);

          policy_loss_sum = policy_loss_sum + task_result.policy_loss_sum;
          value_loss_sum = value_loss_sum + task_result.value_loss_sum;
          entropy_sum = entropy_sum + task_result.entropy_sum;
          policy_approx_kl_sum = policy_approx_kl_sum + task_result.policy_approx_kl;
          policy_clip_fraction_sum = policy_clip_fraction_sum + task_result.policy_clip_frac;
          accumulated_policy_kl_sum += task_result.policy_approx_kl;
          accumulated_policy_kl_count += static_cast<double>(task_result.active_count);
          policy_log_ratio_abs_max = std::max(policy_log_ratio_abs_max, task_result.policy_log_ratio_max);
          goal_critic_loss_sum = goal_critic_loss_sum + task_result.goal_critic_loss;
          goal_score_sum = goal_score_sum + task_result.goal_score;
          sampled_goal_distance_sum = sampled_goal_distance_sum + task_result.sampled_goal_dist;
          metrics.forward_backward_seconds += task_result.fwd_bwd_seconds;
          accumulated_has_backward = accumulated_has_backward || task_result.has_backward;
          metric_steps += task_result.active_count;
        }

        if (use_pcgrad) {
          if (!mode_task_grad_group.empty() && captured_group_has_grad(mode_task_grad_group)) {
            pcgrad_groups.push_back(std::move(mode_task_grad_group));
          }
          if (!mode_goal_critic_grad_group.empty() && captured_group_has_grad(mode_goal_critic_grad_group)) {
            pcgrad_groups.push_back(std::move(mode_goal_critic_grad_group));
          }
          if (!mode_goal_actor_grad_group.empty() && captured_group_has_grad(mode_goal_actor_grad_group)) {
            pcgrad_groups.push_back(std::move(mode_goal_actor_grad_group));
          }
        }
      }

      accumulated_minibatches++;
      accumulated_total_active += combined_total_active;

      // Apply PCGrad across mode/objective groups, then materialize their sum
      // onto the primary actor's parameters.
      if (use_pcgrad && !pcgrad_groups.empty()) {
        auto removed = std::remove_if(
            pcgrad_groups.begin(),
            pcgrad_groups.end(),
            [](const std::vector<CapturedGrad>& group) {
              return !captured_group_has_grad(group) || !captured_group_gradients_are_finite(group);
            });
        if (removed != pcgrad_groups.end()) {
          metrics.nonfinite_grad_norm_skips += std::distance(removed, pcgrad_groups.end());
          pcgrad_groups.erase(removed, pcgrad_groups.end());
        }
      }
      if (use_pcgrad && pcgrad_groups.empty()) {
        accumulated_has_backward = false;
      }
      if (use_pcgrad && !pcgrad_groups.empty()) {
        zero_existing_gradients(*actor_);
        apply_pcgrad_multi(pcgrad_groups);
        bool pcgrad_finite = true;
        for (const auto& group : pcgrad_groups) {
          if (!captured_group_gradients_are_finite(group)) {
            pcgrad_finite = false;
            break;
          }
        }
        if (!pcgrad_finite) {
          ++metrics.nonfinite_grad_norm_skips;
          pcgrad_groups.clear();
          accumulated_has_backward = false;
        }
      }
      if (use_pcgrad && !pcgrad_groups.empty()) {
        for (size_t i = 0; i < pcgrad_groups[0].size(); ++i) {
          torch::Tensor combined;
          bool has_any = false;
          for (const auto& group : pcgrad_groups) {
            if (group[i].grad.defined()) {
              combined = has_any ? combined + group[i].grad : group[i].grad;
              has_any = true;
            }
          }
          if (has_any) {
            pcgrad_groups[0][i].param.mutable_grad() = combined;
          }
        }
      }

      const bool at_epoch_end = agent_offset + logical_agents_per_batch >= total_agents;
      const bool should_step_optimizer =
          use_pcgrad ||
          accumulated_minibatches >= optimizer_accumulation_steps ||
          at_epoch_end;
      if (!should_step_optimizer) {
        continue;
      }
      if (!accumulated_has_backward) {
        actor_optimizer_.zero_grad();
        accumulated_minibatches = 0;
        accumulated_total_active = 0.0;
        accumulated_policy_kl_sum = 0.0;
        accumulated_policy_kl_count = 0.0;
        continue;
      }
      const double accumulated_policy_kl =
          accumulated_policy_kl_count > 0.0
              ? accumulated_policy_kl_sum / accumulated_policy_kl_count
              : 0.0;
      if (config_.ppo.target_kl > 0.0F &&
          accumulated_policy_kl > static_cast<double>(config_.ppo.target_kl)) {
        ++metrics.kl_guard_skips;
        actor_optimizer_.zero_grad();
        accumulated_minibatches = 0;
        accumulated_has_backward = false;
        accumulated_total_active = 0.0;
        accumulated_policy_kl_sum = 0.0;
        accumulated_policy_kl_count = 0.0;
        if (benchmark_progress_) {
          std::cout << "bench_update_optimizer_skipped reason=target_kl"
                    << " approx_kl=" << accumulated_policy_kl
                    << " target_kl=" << config_.ppo.target_kl
                    << '\n' << std::flush;
        }
        continue;
      }

      // Reduce gradients from replica GPU actors into the primary actor before stepping.
      // Skip when using PCGrad: gradients are already captured and combined via CapturedGrad groups.
      if (num_update_gpus > 1 && !use_pcgrad) {
        reduce_gradients_from_replicas(actor_, compute_actors_);
      }

      const auto optim_start = std::chrono::steady_clock::now();
      double grad_norm = 0.0;
      bool stepped_optimizer = false;
      {
        PULSAR_TRACE_SCOPE_CAT("trainer", "update_optimizer");
        scale_existing_gradients(*actor_, cuda_amp_loss_scale);
        grad_norm = clip_existing_gradients(*actor_, config_.ppo.max_grad_norm);
        const bool nonfinite_before_sanitize = !std::isfinite(grad_norm);
        if (!std::isfinite(grad_norm)) {
          const GradientSanitizeResult sanitized = zero_nonfinite_gradients(*actor_);
          if (sanitized.changed) {
            grad_norm = clip_existing_gradients(*actor_, config_.ppo.max_grad_norm);
            if (std::isfinite(grad_norm)) {
              std::cerr << "zeroed non-finite APPO gradient entries in "
                        << sanitized.first_parameter
                        << "; recovered preclip grad_norm=" << grad_norm << '\n';
            }
          }
        }
        const bool grad_guard_hit =
            config_.ppo.max_preclip_grad_norm > 0.0F &&
            grad_norm > static_cast<double>(config_.ppo.max_preclip_grad_norm);
        if (std::isfinite(grad_norm) && !grad_guard_hit) {
          grad_norm_sum = grad_norm_sum + grad_norm * accumulated_total_active;
          actor_optimizer_.step();
          stepped_optimizer = true;
        } else if (std::isfinite(grad_norm)) {
          ++metrics.grad_norm_guard_skips;
          std::cerr << "skipping APPO optimizer step with preclip grad_norm="
                    << grad_norm
                    << " max_preclip_grad_norm=" << config_.ppo.max_preclip_grad_norm
                    << '\n';
        } else {
          ++metrics.nonfinite_grad_norm_skips;
          std::cerr << "skipping APPO optimizer step with non-finite grad_norm=" << grad_norm << '\n';
        }
      }
      actor_optimizer_.zero_grad();

      // Sync updated primary weights back to all replica GPU actors.
      if (num_update_gpus > 1) {
        sync_actor_to_replicas(actor_, compute_actors_);
        // Clear any residual gradients on replicas after sync.
        for (auto& replica : compute_actors_) {
          if (replica) zero_existing_gradients(*replica);
        }
      }
      metrics.optimizer_step_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - optim_start).count();
      if (stepped_optimizer) {
        metrics.grad_norm += grad_norm * accumulated_total_active;
      }
      accumulated_minibatches = 0;
      accumulated_has_backward = false;
      accumulated_total_active = 0.0;
      accumulated_policy_kl_sum = 0.0;
      accumulated_policy_kl_count = 0.0;
      ++completed_minibatches;
      if (benchmark_progress_) {
        const double minibatch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - minibatch_start).count();
        std::cout << "bench_update_minibatch_done"
                  << " epoch=" << (epoch + 1) << "/" << config_.ppo.update_epochs
                  << " minibatch=" << completed_minibatches << "/"
                  << (minibatches_per_epoch * config_.ppo.update_epochs)
                  << " agents=" << count
                  << " active_samples=" << combined_total_active
                  << " seconds=" << minibatch_seconds
                  << " grad_norm=" << grad_norm
                  << '\n' << std::flush;
      }
    }
  }

  if (metric_steps > 0) {
    const double denom = static_cast<double>(metric_steps);
    metrics.policy_loss = (policy_loss_sum / denom).item<double>();
    metrics.value_loss = (value_loss_sum / denom).item<double>();
    metrics.entropy = (entropy_sum / denom).item<double>();
    metrics.grad_norm = (grad_norm_sum / denom).item<double>();
    metrics.policy_approx_kl = (policy_approx_kl_sum / denom).item<double>();
    metrics.policy_clip_fraction = (policy_clip_fraction_sum / denom).item<double>();
    metrics.policy_log_ratio_abs_max = policy_log_ratio_abs_max;
    metrics.goal_critic_loss = (goal_critic_loss_sum / denom).item<double>();
    metrics.mean_goal_score = (goal_score_sum / denom).item<double>();
    metrics.mean_sampled_goal_distance = (sampled_goal_distance_sum / denom).item<double>();
  }
  metrics.effective_entropy_coef = static_cast<double>(effective_entropy_coef);
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
#ifdef PULSAR_HAS_CUDA
  if (prev_train_stream.has_value()) {
    c10::cuda::setCurrentCUDAStream(*prev_train_stream);
  }
#endif
  return metrics;
}

APPOTrainer::ESPopulationFitness APPOTrainer::evaluate_es_population(
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    int update_index) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "es_evaluate");
  torch::NoGradGuard no_grad_guard;
  const auto& es_cfg = config_.es_lora;
  const int pop = es_cfg.population_size;
  const int eval_envs = es_cfg.eval_num_envs;

  ESPopulationFitness result;
  result.fitness.assign(static_cast<std::size_t>(pop), 0.0F);
  result.reward.assign(static_cast<std::size_t>(pop), 0.0F);
  result.winrate.assign(static_cast<std::size_t>(pop), 0.0F);
  result.kl.assign(static_cast<std::size_t>(pop), 0.0F);

  std::vector<std::pair<std::string, float>> eval_modes;
  if (curriculum_.enabled() && !config_.curriculum.stages.empty()) {
    for (const auto& [mode, frac] : curriculum_.mode_allocation()) {
      if (frac > 0.0F) {
        eval_modes.emplace_back(mode, frac);
      }
    }
  }
  if (eval_modes.empty()) {
    const std::string mode = std::to_string(config_.env.team_size) + "v" +
        std::to_string(config_.env.team_size);
    eval_modes.emplace_back(mode, 1.0F);
  }

  float weight_sum = 0.0F;
  for (const auto& [mode, weight] : eval_modes) {
    (void)mode;
    weight_sum += weight;
  }
  if (weight_sum <= 0.0F) {
    weight_sum = 1.0F;
  }

  for (const auto& [mode, raw_weight] : eval_modes) {
    const float mode_weight = raw_weight / weight_sum;
    ExperimentConfig mode_config = config_;
    mode_config.env.team_size = team_size_from_mode(mode);
    mode_config.env.spawn_opponents = true;
    if (curriculum_.enabled()) {
      mode_config.outcome = curriculum_.outcome();
      mode_config.mechanic_rewards = curriculum_.mechanic_rewards();
      mode_config.dense_rewards = curriculum_.dense_rewards();
    }

    const int total_envs = pop * eval_envs;
    const int team_size = mode_config.env.team_size;
    const int agents_per_env = team_size * 2;
    const int member_agents = eval_envs * agents_per_env;

    std::vector<int> episode_counts(static_cast<std::size_t>(pop), 0);
    std::vector<int> win_counts(static_cast<std::size_t>(pop), 0);

    torch::Tensor reward_sum = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor reward_count = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor kl_sum = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor kl_count = torch::zeros({pop}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    std::vector<std::uint8_t> controlled_host(static_cast<std::size_t>(total_envs * agents_per_env), 0);
    for (int env_idx = 0; env_idx < total_envs; ++env_idx) {
      const int local_env = env_idx % eval_envs;
      const bool perturb_blue = (local_env % 2) == 0;
      for (int local_agent = 0; local_agent < agents_per_env; ++local_agent) {
        const bool is_blue = local_agent < team_size;
        controlled_host[static_cast<std::size_t>(env_idx * agents_per_env + local_agent)] =
            (is_blue == perturb_blue) ? 1 : 0;
      }
    }
    const torch::Tensor controlled_mask = torch::from_blob(
        controlled_host.data(),
        {static_cast<long>(controlled_host.size())},
        torch::TensorOptions().dtype(torch::kUInt8))
        .clone()
        .to(device_)
        .to(torch::kBool);
    const torch::Tensor controlled_float = controlled_mask.to(torch::kFloat32).view({pop, member_agents});

    auto eval_collector = make_es_eval_collector(
        mode_config, total_envs, eval_envs, update_index, 0, use_pinned_host_buffers_);
    if (curriculum_.enabled()) {
      eval_collector->update_unlocked_mechanics(curriculum_.unlocked_mechanics());
    }

    for (int ep = 0; ep < es_cfg.eval_episodes_per_member; ++ep) {
      eval_collector->reset_es_episode(update_index, ep, eval_envs);
      torch::Tensor recurrent_state = actor_->initial_recurrent_state(
          static_cast<int64_t>(total_envs * agents_per_env),
          device_);

      for (int step = 0; step < es_cfg.eval_rollout_length; ++step) {
        torch::Tensor raw_obs = eval_collector->host_observations().to(device_, use_pinned_host_buffers_);
        torch::Tensor episode_starts = eval_collector->host_episode_starts().to(device_, use_pinned_host_buffers_);
        torch::Tensor action_masks = eval_collector->host_action_masks().to(device_, use_pinned_host_buffers_).to(torch::kBool);
        torch::Tensor normalized_obs = actor_normalizer_.normalize(raw_obs);

        const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
        ActorStepOutput output = actor_->forward_step_stateful(
            normalized_obs,
            recurrent_state,
            episode_starts,
            &recurrent_state,
            goal_values);
        torch::Tensor perturbed_logits = actor_->policy_eggroll_logits(
            output.features, A_stack, B_stack, es_cfg.sigma_ES, goal_values);

        torch::Tensor base_actions = sample_masked_actions(output.policy_logits, action_masks, true, nullptr);
        torch::Tensor perturbed_actions = sample_masked_actions(perturbed_logits, action_masks, true, nullptr);
        torch::Tensor actions = torch::where(controlled_mask, perturbed_actions, base_actions);

        const torch::Tensor base_masked = apply_action_mask_to_logits(output.policy_logits, action_masks);
        const torch::Tensor perturbed_masked = apply_action_mask_to_logits(perturbed_logits, action_masks);
        const torch::Tensor base_probs = torch::softmax(base_masked, -1);
        const torch::Tensor perturbed_probs = torch::softmax(perturbed_masked, -1);
        const torch::Tensor kl_values = (
            perturbed_probs * (torch::log(perturbed_probs + 1.0e-8) - torch::log(base_probs + 1.0e-8)))
            .sum(-1)
            .view({pop, member_agents});
        kl_sum += (kl_values * controlled_float).sum(1);
        kl_count += controlled_float.sum(1);

        const torch::Tensor action_indices_cpu = actions.contiguous().to(torch::kCPU);
        eval_collector->step(std::span<const std::int64_t>(
            action_indices_cpu.data_ptr<std::int64_t>(),
            static_cast<std::size_t>(action_indices_cpu.numel())));

        const torch::Tensor rewards = eval_collector->host_rewards()
            .to(device_, use_pinned_host_buffers_)
            .view({pop, member_agents});
        reward_sum += (rewards * controlled_float).sum(1);
        reward_count += controlled_float.sum(1);

        torch::Tensor dones_cpu = eval_collector->host_dones();
        torch::Tensor labels_cpu = eval_collector->host_terminal_outcome_labels();
        const auto* dones_ptr = dones_cpu.data_ptr<float>();
        const auto* labels_ptr = labels_cpu.data_ptr<std::int64_t>();
        for (std::size_t i = 0; i < controlled_host.size(); ++i) {
          if (controlled_host[i] == 0 || dones_ptr[i] <= 0.5F) {
            continue;
          }
          const int env_idx = static_cast<int>(i / static_cast<std::size_t>(agents_per_env));
          const int member = env_idx / eval_envs;
          episode_counts[static_cast<std::size_t>(member)] += 1;
          if (labels_ptr[i] == 0) {
            win_counts[static_cast<std::size_t>(member)] += 1;
          }
        }
      }
    }

    torch::Tensor reward_mean = (reward_sum / reward_count.clamp_min(1.0F)).to(torch::kCPU);
    torch::Tensor kl_mean = (kl_sum / kl_count.clamp_min(1.0F)).to(torch::kCPU);
    const auto* reward_ptr = reward_mean.data_ptr<float>();
    const auto* kl_ptr = kl_mean.data_ptr<float>();
    for (int i = 0; i < pop; ++i) {
      const int denom = std::max(episode_counts[static_cast<std::size_t>(i)], 1);
      const float mode_winrate =
          static_cast<float>(win_counts[static_cast<std::size_t>(i)]) / static_cast<float>(denom);
      result.reward[static_cast<std::size_t>(i)] += mode_weight * reward_ptr[i];
      result.winrate[static_cast<std::size_t>(i)] += mode_weight * mode_winrate;
      result.kl[static_cast<std::size_t>(i)] += mode_weight * kl_ptr[i];
    }
  }

  for (int i = 0; i < pop; ++i) {
    result.fitness[static_cast<std::size_t>(i)] =
        result.reward[static_cast<std::size_t>(i)]
        - es_cfg.beta_KL * result.kl[static_cast<std::size_t>(i)];
  }
  return result;
}

void APPOTrainer::run_es_lora_update(int update_index, TrainerMetrics& metrics) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "es_update");
  const auto es_start = std::chrono::steady_clock::now();
  const auto& es_cfg = config_.es_lora;
  const int pop = es_cfg.population_size;
  const int rank = es_cfg.rank;
  const int in_features = actor_->policy_lora()->in_features();
  const int out_features = actor_->policy_lora()->out_features();

  const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor A_stack;
  torch::Tensor B_stack;
  if (es_cfg.antithetic_sampling) {
    const int half_pop = pop / 2;
    torch::Tensor A_half = torch::randn({half_pop, rank, in_features}, tensor_options);
    torch::Tensor B_half = torch::randn({half_pop, out_features, rank}, tensor_options);
    A_stack = torch::cat({A_half, -A_half}, 0);
    B_stack = torch::cat({B_half, -B_half}, 0);
  } else {
    A_stack = torch::randn({pop, rank, in_features}, tensor_options);
    B_stack = torch::randn({pop, out_features, rank}, tensor_options);
  }

  const auto es_eval_start = std::chrono::steady_clock::now();
  ESPopulationFitness population = evaluate_es_population(A_stack, B_stack, update_index);
  metrics.es_eval_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_eval_start).count();
  std::vector<float>& fitnesses = population.fitness;
  const uint64_t total_members = fitnesses.size();
  float mu = 0.0F;
  for (float f : fitnesses) {
    mu += f;
  }
  mu /= static_cast<float>(total_members);

  float sigma = 0.0F;
  for (float f : fitnesses) {
    sigma += (f - mu) * (f - mu);
  }
  sigma = std::sqrt(sigma / static_cast<float>(total_members));

  std::vector<float> normalized_f;
  for (float f : fitnesses) {
    normalized_f.push_back((f - mu) / (sigma + 1.0e-8F));
  }

  double winrate_mean = 0.0;
  double reward_mean = 0.0;
  double kl_mean = 0.0;
  for (uint64_t i = 0; i < total_members; ++i) {
    reward_mean += population.reward[i];
    winrate_mean += population.winrate[i];
    kl_mean += population.kl[i];
  }
  reward_mean /= static_cast<double>(total_members);
  winrate_mean /= static_cast<double>(total_members);
  kl_mean /= static_cast<double>(total_members);

  const float best_fitness = *std::max_element(fitnesses.begin(), fitnesses.end());
  if (es_cfg.require_fitness_signal && sigma < es_cfg.min_fitness_std) {
    metrics.es_fitness_mean = mu;
    metrics.es_fitness_std = sigma;
    metrics.es_fitness_best = static_cast<double>(best_fitness);
    metrics.es_update_norm = 0.0;
    metrics.es_reward_mean = reward_mean;
    metrics.es_winrate_mean = winrate_mean;
    metrics.es_kl_mean = kl_mean;

    auto lora_params = actor_->es_lora_parameters();
    metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
    metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());
    metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
    return;
  }

  const torch::Tensor fitness_weights = torch::from_blob(
      normalized_f.data(),
      {static_cast<long>(total_members)},
      torch::TensorOptions().dtype(torch::kFloat32))
      .clone()
      .to(device_);
  torch::Tensor grad_A =
      (A_stack * fitness_weights.view({static_cast<long>(total_members), 1, 1})).sum(0) /
      static_cast<float>(total_members);
  torch::Tensor grad_B =
      (B_stack * fitness_weights.view({static_cast<long>(total_members), 1, 1})).sum(0) /
      static_cast<float>(total_members);

  const float step = es_cfg.eta_ES / es_cfg.sigma_ES;
  torch::Tensor delta_A = grad_A * step;
  torch::Tensor delta_B = grad_B * step;

  double update_norm = std::sqrt(
      std::pow(static_cast<double>(delta_A.norm().item<float>()), 2) +
      std::pow(static_cast<double>(delta_B.norm().item<float>()), 2));
  double update_scale = 1.0;
  if (es_cfg.max_kl_mean > 0.0F && kl_mean > static_cast<double>(es_cfg.max_kl_mean)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_kl_mean) / std::max(kl_mean, 1.0e-12));
  }
  if (es_cfg.update_norm_clip && es_cfg.max_update_norm > 0.0F && update_norm > static_cast<double>(es_cfg.max_update_norm)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_update_norm) / std::max(update_norm, 1.0e-12));
  }
  if (update_scale < 1.0) {
    delta_A.mul_(update_scale);
    delta_B.mul_(update_scale);
    update_norm *= update_scale;
  }

  {
    torch::NoGradGuard no_grad;
    auto lora_params = actor_->es_lora_parameters();
    lora_params[0].add_(delta_A);
    lora_params[1].add_(delta_B);
    for (auto& param : lora_params) {
      actor_optimizer_.state().erase(param.unsafeGetTensorImpl());
    }
  }

  metrics.es_fitness_mean = mu;
  metrics.es_fitness_std = sigma;
  metrics.es_fitness_best = static_cast<double>(best_fitness);
  metrics.es_update_norm = update_norm;
  metrics.es_reward_mean = reward_mean;
  metrics.es_winrate_mean = winrate_mean;
  metrics.es_kl_mean = kl_mean;

  auto lora_params = actor_->es_lora_parameters();
  metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
  metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());

  metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
}

void APPOTrainer::collect_rollout(
    RolloutStorage& dest,
    TrainerMetrics& metrics,
    std::int64_t* collected_agent_steps,
    PPOActor rollout_actor,
    ObservationNormalizer& normalizer) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "collect_rollout");
  if (!rollout_actor) {
    throw std::invalid_argument("APPOTrainer::collect_rollout requires a policy snapshot.");
  }
#ifdef PULSAR_HAS_CUDA
  c10::cuda::CUDAStream prev_collect_stream = c10::cuda::getCurrentCUDAStream(device_.index());
  const c10::cuda::CUDAStream default_collect_stream = shard_collection_streams_.empty()
      ? prev_collect_stream
      : shard_collection_streams_[0];
#endif
  dest.clear();
  const auto update_start = std::chrono::steady_clock::now();
  CollectorTimings collector_timings{};
  BatchedRocketSimCollector* collector_ = collectors_.front().get();
  std::int64_t local_collected_steps = 0;

  const auto collection_start = std::chrono::steady_clock::now();

  double total_reward = 0.0;
  double total_gameplay_reward = 0.0;
  double total_mechanic_reward = 0.0;
  int64_t total_steps = 0;
  int64_t total_learner_steps = 0;
  double accumulated_sampled_value = 0.0;
  int64_t accumulated_value_count = 0;
  double total_goal_distance = 0.0;
  double min_goal_distance = 1.0;
  int64_t total_goals_scored = 0;
  int64_t total_goals_conceded = 0;
  int64_t total_ball_proximity_steps = 0;
  int64_t total_ball_proximity_denom = 0;
  int completed_episodes = 0;
  int scored_episodes = 0;
  int conceded_episodes = 0;
  int neutral_episodes = 0;
  int no_touch_episodes = 0;
  int truncated_episodes = 0;
  int touched_episodes = 0;
  int multi_touched_episodes = 0;
  std::map<std::string, int> mode_completed;
  std::map<std::string, int> mode_touched;
  std::map<std::string, int> mode_multi_touched;
  std::map<std::string, int> mode_scored;
  std::vector<torch::Tensor> rollout_recurrent_states;
  rollout_recurrent_states.reserve(collectors_.size());
  std::vector<PPOActor> rollout_actors;
  rollout_actors.reserve(collectors_.size());
  std::vector<ObservationNormalizer> shard_normalizers;
  shard_normalizers.reserve(collectors_.size());
  std::vector<ObservationNormalizer> shard_normalizer_updates;
  shard_normalizer_updates.reserve(collectors_.size());

  // Use persistent collection actors (synced when snapshot changes) instead of cloning per rollout.
  // Fall back to cloning if collection_actors_ is stale (e.g., after a snapshot change without sync).
  const bool use_persistent = collection_actors_.size() == collectors_.size();

  for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
    const auto& collector_ptr = collectors_[shard];
    const torch::Device shard_device = shard_devices_.empty() ? device_ : shard_devices_[shard];
    if (use_persistent) {
      rollout_actors.push_back(collection_actors_[shard]);
    } else if (shard_device == device_) {
      rollout_actors.push_back(rollout_actor);
    } else {
      rollout_actors.push_back(clone_ppo_actor(rollout_actor, shard_device));
      rollout_actors.back()->eval();
    }
    shard_normalizers.push_back(normalizer.clone());
    shard_normalizers.back().to(shard_device);
    shard_normalizer_updates.emplace_back(config_.model.observation_dim);
    shard_normalizer_updates.back().to(shard_device);
    rollout_recurrent_states.push_back(
        rollout_actors.back()->initial_recurrent_state(static_cast<int64_t>(collector_ptr->total_agents()), shard_device));
  }
  const int physics_prefix_ticks =
      config_.env.half_tick_skip > 0 && config_.env.half_tick_skip < config_.env.tick_skip
          ? config_.env.half_tick_skip
          : 0;

  if (collectors_.size() > 1) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "collect_loop_sharded");
    struct PendingShardStep {
      int agent_offset = 0;
      std::size_t shard = 0;
      CollectorTimings timings{};
      double policy_forward_seconds = 0.0;
      double action_decode_seconds = 0.0;
      int goals_scored = 0;
      int goals_conceded = 0;
      int completed_episodes = 0;
      int scored_episodes = 0;
      int conceded_episodes = 0;
      int neutral_episodes = 0;
      int no_touch_episodes = 0;
      int truncated_episodes = 0;
      int touched_episodes = 0;
      int multi_touched_episodes = 0;
      std::int64_t ball_prox_steps = 0;
      std::int64_t ball_prox_denom = 0;
      double sampled_value_sum = 0.0;
      std::int64_t sampled_value_count = 0;
      std::map<std::string, int> mode_completed;
      std::map<std::string, int> mode_touched;
      std::map<std::string, int> mode_multi_touched;
      std::map<std::string, int> mode_scored;
      double total_reward = 0.0;
      double total_gameplay_reward = 0.0;
      double total_mechanic_reward = 0.0;
      int64_t total_steps = 0;
      int64_t total_learner_steps = 0;
      double total_goal_distance = 0.0;
      double min_goal_distance = 1.0;
    };

    // Launch one async task per shard that processes all rollout steps.
    // This reduces std::async launches from (shards × steps) to just shards.
    struct ShardResult {
      std::size_t shard;
      std::vector<PendingShardStep> steps;
      torch::Tensor next_recurrent_state;
    };
    std::vector<std::future<ShardResult>> shard_futures;
    shard_futures.reserve(collectors_.size());

    for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
      BatchedRocketSimCollector* collector_ptr = collectors_[shard].get();
      PPOActor shard_actor = rollout_actors[shard];
      ObservationNormalizer* shard_normalizer = &shard_normalizers[shard];
      ObservationNormalizer* shard_normalizer_update = &shard_normalizer_updates[shard];
      torch::Tensor recurrent_state = rollout_recurrent_states[shard];
      const torch::Device shard_device = shard_devices_.empty() ? device_ : shard_devices_[shard];
      const std::size_t shard_idx = shard;

      shard_futures.push_back(std::async(std::launch::async,
          [&, collector_ptr, shard_actor, shard_normalizer, shard_normalizer_update, recurrent_state, shard_device, shard_idx]() mutable -> ShardResult {
        ShardResult result;
        result.shard = shard_idx;
        result.steps.reserve(config_.ppo.rollout_length);
        auto& collector = *collector_ptr;

#ifdef PULSAR_HAS_CUDA
        std::optional<c10::cuda::CUDAGuard> shard_device_guard;
        if (shard_device.is_cuda()) {
          shard_device_guard.emplace(shard_device);
        }
        if (shard_idx < shard_collection_streams_.size()) {
          c10::cuda::setCurrentCUDAStream(shard_collection_streams_[shard_idx]);
        }
#endif

        for (int step = 0; step < config_.ppo.rollout_length; ++step) {
          if (config_.model.sequence_length > 0 && step % config_.model.sequence_length == 0) {
            recurrent_state.zero_();
          }
          PendingShardStep shard_step;
          shard_step.shard = shard_idx;
          shard_step.agent_offset = static_cast<int>(shard_agent_offsets_[shard_idx]);

          torch::Tensor raw_obs_host = collector.host_observations();
          torch::Tensor episode_starts_host = collector.host_episode_starts();
          torch::Tensor action_masks_host = collector.host_action_masks();
          torch::Tensor learner_active_host = collector.host_learner_active();
          std::future<void> physics_prefix_future;
          if (physics_prefix_ticks > 0) {
            physics_prefix_future = std::async(std::launch::async, [&collector, &shard_step, physics_prefix_ticks]() {
              collector.step_physics_only(physics_prefix_ticks, &shard_step.timings);
            });
          }
          torch::Tensor raw_obs = raw_obs_host.to(shard_device, use_pinned_host_buffers_);
          torch::Tensor episode_starts = episode_starts_host.to(shard_device, use_pinned_host_buffers_);
          torch::Tensor action_masks = action_masks_host.to(shard_device, use_pinned_host_buffers_);  // uint8, sample_masked_actions handles it

          torch::Tensor normalized_obs;
          torch::Tensor actions;
          torch::Tensor action_log_probs;
          ActorStepOutput output;
          const auto policy_start = std::chrono::steady_clock::now();
          {
            torch::NoGradGuard no_grad;
            shard_normalizer->update(raw_obs);
            shard_normalizer_update->update(raw_obs);
            normalized_obs = shard_normalizer->normalize(raw_obs);
            const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
            output = shard_actor->forward_step_stateful(
                normalized_obs, recurrent_state, episode_starts, &recurrent_state, goal_values);
            actions = sample_masked_actions(output.policy_logits, action_masks, false, &action_log_probs);
          }
          if (config_.ppo.synchronize_cuda_timing && device_.is_cuda()) {
            torch::cuda::synchronize();
          }
          shard_step.policy_forward_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

          const auto decode_start = std::chrono::steady_clock::now();
          torch::Tensor action_indices_cpu = actions.contiguous().to(torch::kCPU);
          if (physics_prefix_future.valid()) {
            physics_prefix_future.get();
          }
          collector.step_after_physics_prefix(
              std::span<const std::int64_t>(
                  action_indices_cpu.data_ptr<std::int64_t>(),
                  static_cast<std::size_t>(action_indices_cpu.numel())),
              physics_prefix_ticks,
              &shard_step.timings);

          // Done-reset processing inside the shard task.
          torch::Tensor dones_host = collector.host_dones();
          torch::Tensor truncated_host = collector.host_truncated();
          torch::Tensor terminal_labels = collector.host_terminal_outcome_labels();
          const auto* dones_ptr = dones_host.data_ptr<float>();
          const auto* truncated_ptr = truncated_host.data_ptr<float>();
          const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
          const auto* la_ptr = learner_active_host.data_ptr<float>();
          torch::Tensor env_touch_host = collector.host_env_touched();
          torch::Tensor env_multi_touch_host = collector.host_env_multi_touched();
          const auto* env_touch_ptr = env_touch_host.data_ptr<float>();
          const auto* env_multi_touch_ptr = env_multi_touch_host.data_ptr<float>();
          torch::Tensor ball_prox_host = collector.host_ball_proximity();
          shard_step.ball_prox_steps = ball_prox_host.sum().item<int64_t>();
          shard_step.ball_prox_denom = ball_prox_host.numel();

          const int64_t coll_agents = static_cast<int64_t>(collector.total_agents());
          const int64_t coll_num_envs = static_cast<int64_t>(collector.num_envs());
          const int64_t coll_ape = (coll_num_envs > 0) ? (coll_agents / coll_num_envs) : 2;
          for (int64_t env_agent_begin = 0; env_agent_begin < terminal_labels.numel(); env_agent_begin += coll_ape) {
            const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + coll_ape, terminal_labels.numel());
            bool env_goal_scored = false, env_goal_conceded = false;
            for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
              if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
                if (tl_ptr[i] == 0) env_goal_scored = true;
                if (tl_ptr[i] == 1) env_goal_conceded = true;
              }
            }
            if (env_goal_scored) shard_step.goals_scored++;
            if (env_goal_conceded) shard_step.goals_conceded++;
          }
          for (int64_t env_agent_begin = 0; env_agent_begin < dones_host.numel(); env_agent_begin += coll_ape) {
            bool env_done = false, env_scored = false;
            bool env_conceded = false, env_neutral = false, env_no_touch = false, env_truncated = false;
            const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + coll_ape, dones_host.numel());
            for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
              env_done = env_done || dones_ptr[i] > 0.5F;
              env_scored = env_scored || (dones_ptr[i] > 0.5F && tl_ptr[i] == 0);
              env_conceded = env_conceded || (dones_ptr[i] > 0.5F && tl_ptr[i] == 1);
              env_neutral = env_neutral || (dones_ptr[i] > 0.5F && tl_ptr[i] == 2);
              env_no_touch = env_no_touch || (dones_ptr[i] > 0.5F && tl_ptr[i] == 3);
              env_truncated = env_truncated || (dones_ptr[i] > 0.5F && truncated_ptr[i] > 0.5F);
            }
            if (env_done) {
              shard_step.completed_episodes++;
              if (env_scored) shard_step.scored_episodes++;
              if (env_conceded) shard_step.conceded_episodes++;
              if (env_neutral) shard_step.neutral_episodes++;
              if (env_no_touch) shard_step.no_touch_episodes++;
              if (env_truncated) shard_step.truncated_episodes++;
              const int64_t env_idx = env_agent_begin / coll_ape;
              if (env_touch_ptr[env_idx] > 0.5F) shard_step.touched_episodes++;
              if (env_multi_touch_ptr[env_idx] > 0.5F) shard_step.multi_touched_episodes++;
              const std::string& cmode = collector.mode();
              shard_step.mode_completed[cmode]++;
              if (env_touch_ptr[env_idx] > 0.5F) shard_step.mode_touched[cmode]++;
              if (env_multi_touch_ptr[env_idx] > 0.5F) shard_step.mode_multi_touched[cmode]++;
              if (env_scored) shard_step.mode_scored[cmode]++;
            }
          }

          shard_step.action_decode_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

          // Accumulate reward/goal-distance metrics in the shard task.
          torch::Tensor extrinsic_rewards_host = collector.host_rewards();
          torch::Tensor gameplay_r_host = collector.host_gameplay_rewards();
          torch::Tensor mechanic_r_host = collector.host_mechanic_rewards();
          shard_step.total_reward = (extrinsic_rewards_host * learner_active_host).sum().item<double>();
          shard_step.total_gameplay_reward = (gameplay_r_host * learner_active_host).sum().item<double>();
          shard_step.total_mechanic_reward = (mechanic_r_host * learner_active_host).sum().item<double>();
          shard_step.total_steps = extrinsic_rewards_host.numel();
          shard_step.total_learner_steps = static_cast<std::int64_t>(learner_active_host.sum().item<float>());

          torch::Tensor goal_pos_host = collector.host_goal_positions();
          torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
          float gd_min = goal_norms.min().item<float>();
          float gd_mean = goal_norms.mean().item<float>();
          shard_step.total_goal_distance = static_cast<double>(gd_mean) * static_cast<double>(goal_pos_host.size(0));
          shard_step.min_goal_distance = static_cast<double>(gd_min);

          // Write tensor data directly to rollout storage.
          torch::Tensor sampled_value = output.value_win_logits.squeeze(-1);
          shard_step.sampled_value_sum = sampled_value.sum().item<double>();
          shard_step.sampled_value_count = sampled_value.numel();
          std::unordered_map<std::string, torch::Tensor> all_values;
          all_values["extrinsic"] = sampled_value;
          std::unordered_map<std::string, torch::Tensor> all_rewards;
          all_rewards["extrinsic"] = extrinsic_rewards_host;

          dest.append_slice(
              step,
              shard_step.agent_offset,
              normalized_obs,
              episode_starts_host.to(torch::kBool),
              action_masks_host,
              learner_active_host,
              action_indices_cpu,
              action_log_probs,
              all_values,
              all_rewards,
              dones_host,
              truncated_host,
              collector.host_bootstrap_truncated(),
              goal_pos_host,
              terminal_labels,
              collector.host_terminal_observations());

          auto& coll = collector;
          dest.set_mode_ids_slice(step, shard_step.agent_offset, static_cast<int>(coll.total_agents()), coll.mode_id());

          result.steps.push_back(std::move(shard_step));
        }
        result.next_recurrent_state = recurrent_state;
        return result;
      }));
    }

    // Wait for all shard tasks and aggregate results per step.
    std::vector<std::vector<PendingShardStep>> all_shard_steps(collectors_.size());
    for (auto& fut : shard_futures) {
      ShardResult shard_result = fut.get();
      metrics.policy_forward_seconds += [&] { double s = 0; for (auto& st : shard_result.steps) s += st.policy_forward_seconds; return s; }();
      metrics.action_decode_seconds += [&] { double s = 0; for (auto& st : shard_result.steps) s += st.action_decode_seconds; return s; }();
      if (shard_result.next_recurrent_state.defined()) {
        rollout_recurrent_states[shard_result.shard] = shard_result.next_recurrent_state;
      }
      all_shard_steps[shard_result.shard] = std::move(shard_result.steps);
    }

    // Process results step by step (order matters for rollout storage).
    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
        PendingShardStep& shard_step = all_shard_steps[shard][step];
        accumulate_timings(collector_timings, shard_step.timings);

        total_reward += shard_step.total_reward;
        total_gameplay_reward += shard_step.total_gameplay_reward;
        total_mechanic_reward += shard_step.total_mechanic_reward;
        total_steps += shard_step.total_steps;
        total_learner_steps += shard_step.total_learner_steps;
        total_goal_distance += shard_step.total_goal_distance
            / static_cast<double>(total_agents_);
        if (shard_step.min_goal_distance < min_goal_distance) {
          min_goal_distance = shard_step.min_goal_distance;
        }

        total_goals_scored += shard_step.goals_scored;
        total_goals_conceded += shard_step.goals_conceded;
        completed_episodes += shard_step.completed_episodes;
        scored_episodes += shard_step.scored_episodes;
        conceded_episodes += shard_step.conceded_episodes;
        neutral_episodes += shard_step.neutral_episodes;
        no_touch_episodes += shard_step.no_touch_episodes;
        truncated_episodes += shard_step.truncated_episodes;
        touched_episodes += shard_step.touched_episodes;
        multi_touched_episodes += shard_step.multi_touched_episodes;
        total_ball_proximity_steps += shard_step.ball_prox_steps;
        total_ball_proximity_denom += shard_step.ball_prox_denom;
        for (const auto& [mode, count] : shard_step.mode_completed) mode_completed[mode] += count;
        for (const auto& [mode, count] : shard_step.mode_touched) mode_touched[mode] += count;
        for (const auto& [mode, count] : shard_step.mode_multi_touched) mode_multi_touched[mode] += count;
        for (const auto& [mode, count] : shard_step.mode_scored) mode_scored[mode] += count;

        accumulated_sampled_value += shard_step.sampled_value_sum;
        accumulated_value_count += shard_step.sampled_value_count;

        local_collected_steps += shard_step.total_learner_steps;
      }
      dest.mark_step_filled(step);

    }

    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward_sharded");
      torch::NoGradGuard no_grad;
      std::vector<torch::Tensor> final_values;
      final_values.reserve(collectors_.size());
      for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
        auto& collector = *collectors_[shard];
        PPOActor& shard_actor = rollout_actors[shard];
        ObservationNormalizer& shard_normalizer = shard_normalizers[shard];
        const torch::Device shard_device = shard_devices_.empty() ? device_ : shard_devices_[shard];
#ifdef PULSAR_HAS_CUDA
        std::optional<c10::cuda::CUDAGuard> shard_device_guard;
        if (shard_device.is_cuda()) {
          shard_device_guard.emplace(shard_device);
        }
        if (shard < shard_collection_streams_.size()) {
          c10::cuda::setCurrentCUDAStream(shard_collection_streams_[shard]);
        }
#endif
        torch::Tensor final_raw_obs = collector.host_observations().to(shard_device, use_pinned_host_buffers_);
        torch::Tensor final_normalized = shard_normalizer.normalize(final_raw_obs);
        torch::Tensor final_starts = collector.host_episode_starts().to(shard_device, use_pinned_host_buffers_);
        const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
        ActorStepOutput final_output = shard_actor->forward_step_stateful(
            final_normalized,
            rollout_recurrent_states[shard],
            final_starts,
            nullptr,
            final_goal_values);
        final_values.push_back(final_output.value_win_logits.squeeze(-1).to(device_));
      }
      std::unordered_map<std::string, torch::Tensor> bootstrap_values;
      bootstrap_values["extrinsic"] = torch::cat(final_values, 0);
      dest.set_final_values(bootstrap_values);
    }
  } else {
    PULSAR_TRACE_SCOPE_CAT("trainer", "collect_loop");
#ifdef PULSAR_HAS_CUDA
    c10::cuda::setCurrentCUDAStream(default_collect_stream);
#endif
    for (int step = 0; step < config_.ppo.rollout_length; ++step) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "collect_step");
      if (config_.model.sequence_length > 0 && step % config_.model.sequence_length == 0) {
        rollout_recurrent_states[0].zero_();
      }
      torch::Tensor raw_obs_host = collector_->host_observations();
    torch::Tensor episode_starts_host = collector_->host_episode_starts();
    torch::Tensor action_masks_host = collector_->host_action_masks();
    torch::Tensor learner_active_host = collector_->host_learner_active();
    std::future<void> physics_prefix_future;
    if (physics_prefix_ticks > 0) {
      physics_prefix_future = std::async(std::launch::async, [&]() {
        collector_->step_physics_only(physics_prefix_ticks, &collector_timings);
      });
    }
    torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor episode_starts = episode_starts_host.to(device_, use_pinned_host_buffers_);
    torch::Tensor action_masks = action_masks_host.to(device_, use_pinned_host_buffers_);  // uint8

    torch::Tensor normalized_obs;
    torch::Tensor actions;
    torch::Tensor action_log_probs;
    ActorStepOutput output;
    const auto policy_start = std::chrono::steady_clock::now();
    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "policy_forward");
      torch::NoGradGuard no_grad;
      normalizer.update(raw_obs);
      normalized_obs = normalizer.normalize(raw_obs);
      const torch::Tensor goal_values = policy_goal_values_like(normalized_obs, config_.goal_critic.goal_dim);
      output = rollout_actor->forward_step_stateful(
          normalized_obs,
          rollout_recurrent_states[0],
          episode_starts,
          &rollout_recurrent_states[0],
          goal_values);
      actions = sample_masked_actions(output.policy_logits, action_masks, false, &action_log_probs);
    }
    if (config_.ppo.synchronize_cuda_timing && device_.is_cuda()) {
      torch::cuda::synchronize();
    }
    metrics.policy_forward_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - policy_start).count();

    const auto decode_start = std::chrono::steady_clock::now();
    torch::Tensor action_indices_cpu;
    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "action_decode");
      action_indices_cpu = actions.contiguous().to(torch::kCPU);
      if (physics_prefix_future.valid()) {
        physics_prefix_future.get();
      }
      collector_->step_after_physics_prefix(
          std::span<const std::int64_t>(
              action_indices_cpu.data_ptr<std::int64_t>(),
              static_cast<std::size_t>(action_indices_cpu.numel())),
          physics_prefix_ticks,
          &collector_timings);
    }
    metrics.action_decode_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

    {
      PULSAR_TRACE_SCOPE_CAT("trainer", "collect_post_step");
      torch::Tensor dones_host = collector_->host_dones();
    torch::Tensor truncated_host = collector_->host_truncated();
    torch::Tensor bootstrap_truncated_host = collector_->host_bootstrap_truncated();
    torch::Tensor terminal_labels = collector_->host_terminal_outcome_labels();
      torch::Tensor extrinsic_rewards_host = collector_->host_rewards();
      torch::Tensor gameplay_r_host = collector_->host_gameplay_rewards();
      torch::Tensor mechanic_r_host = collector_->host_mechanic_rewards();
    const auto* dones_ptr = dones_host.data_ptr<float>();

    torch::Tensor ball_prox_host = collector_->host_ball_proximity();
    total_ball_proximity_steps += ball_prox_host.sum().item<int64_t>();
    total_ball_proximity_denom += ball_prox_host.numel();

    const auto* tl_ptr = terminal_labels.data_ptr<std::int64_t>();
    const auto* la_ptr = learner_active_host.data_ptr<float>();
    const auto* truncated_ptr = truncated_host.data_ptr<float>();
    torch::Tensor env_touch_host = collector_->host_env_touched();
    torch::Tensor env_multi_touch_host = collector_->host_env_multi_touched();
    const auto* env_touch_ptr = env_touch_host.data_ptr<float>();
    const auto* env_multi_touch_ptr = env_multi_touch_host.data_ptr<float>();
    const int64_t coll_agents = static_cast<int64_t>(collector_->total_agents());
    const int64_t coll_num_envs = static_cast<int64_t>(collector_->num_envs());
    const int64_t coll_ape = (coll_num_envs > 0) ? (coll_agents / coll_num_envs) : 2;
    for (int64_t env_agent_begin = 0; env_agent_begin < terminal_labels.numel(); env_agent_begin += coll_ape) {
      const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + coll_ape, terminal_labels.numel());
      bool env_goal_scored = false;
      bool env_goal_conceded = false;
      for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
        if (la_ptr[i] > 0.5F && dones_ptr[i] > 0.5F) {
          if (tl_ptr[i] == 0) env_goal_scored = true;
          if (tl_ptr[i] == 1) env_goal_conceded = true;
        }
      }
      if (env_goal_scored) total_goals_scored++;
      if (env_goal_conceded) total_goals_conceded++;
    }
    for (int64_t env_agent_begin = 0; env_agent_begin < dones_host.numel(); env_agent_begin += coll_ape) {
      bool env_done = false;
      bool env_scored = false;
      bool env_conceded = false;
      bool env_neutral = false;
      bool env_no_touch = false;
      bool env_truncated = false;
      const int64_t env_agent_end = std::min<int64_t>(env_agent_begin + coll_ape, dones_host.numel());
      for (int64_t i = env_agent_begin; i < env_agent_end; ++i) {
        env_done = env_done || dones_ptr[i] > 0.5F;
        env_scored = env_scored || (dones_ptr[i] > 0.5F && tl_ptr[i] == 0);
        env_conceded = env_conceded || (dones_ptr[i] > 0.5F && tl_ptr[i] == 1);
        env_neutral = env_neutral || (dones_ptr[i] > 0.5F && tl_ptr[i] == 2);
        env_no_touch = env_no_touch || (dones_ptr[i] > 0.5F && tl_ptr[i] == 3);
        env_truncated = env_truncated || (dones_ptr[i] > 0.5F && truncated_ptr[i] > 0.5F);
      }
      if (env_done) {
        completed_episodes++;
        if (env_scored) {
          scored_episodes++;
        }
        if (env_conceded) conceded_episodes++;
        if (env_neutral) neutral_episodes++;
        if (env_no_touch) no_touch_episodes++;
        if (env_truncated) truncated_episodes++;
        const int64_t env_idx = env_agent_begin / coll_ape;
        if (env_touch_ptr[env_idx] > 0.5F) {
          touched_episodes++;
        }
        if (env_multi_touch_ptr[env_idx] > 0.5F) {
          multi_touched_episodes++;
        }
        const std::string& cmode = collector_->mode();
        mode_completed[cmode]++;
        if (env_touch_ptr[env_idx] > 0.5F) mode_touched[cmode]++;
        if (env_multi_touch_ptr[env_idx] > 0.5F) mode_multi_touched[cmode]++;
        if (env_scored) mode_scored[cmode]++;
      }
    }

    torch::Tensor goal_pos_host = collector_->host_goal_positions();
    torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
    float gd_min = goal_norms.min().item<float>();
    float gd_mean = goal_norms.mean().item<float>();
    total_goal_distance += static_cast<double>(gd_mean);
    if (gd_min < min_goal_distance) {
      min_goal_distance = static_cast<double>(gd_min);
    }

    torch::Tensor terminal_obs_host = collector_->host_terminal_observations();

    const auto learner_step_count = static_cast<std::int64_t>(learner_active_host.sum().item<float>());
    total_reward += (extrinsic_rewards_host * learner_active_host).sum().item<double>();
    total_gameplay_reward += (gameplay_r_host * learner_active_host).sum().item<double>();
    total_mechanic_reward += (mechanic_r_host * learner_active_host).sum().item<double>();
    total_steps += extrinsic_rewards_host.numel();
    total_learner_steps += learner_step_count;

    std::unordered_map<std::string, torch::Tensor> all_values;
    torch::Tensor sampled_value = output.value_win_logits.squeeze(-1);
    all_values["extrinsic"] = sampled_value;
    accumulated_sampled_value += sampled_value.sum().item<double>();
    accumulated_value_count += sampled_value.numel();

    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards_host;

    dest.append(
        step,
        normalized_obs,
        episode_starts_host.to(torch::kBool),
        action_masks_host,
        learner_active_host,
        action_indices_cpu,
        action_log_probs,
        all_values,
        all_rewards,
        dones_host,
        truncated_host,
        bootstrap_truncated_host,
        goal_pos_host,
        terminal_labels,
        terminal_obs_host);
    dest.set_mode_ids_slice(step, 0, collector_->mode_id());

    local_collected_steps += learner_step_count;
    }
  }
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward");
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = normalizer.normalize(final_raw_obs);
    torch::Tensor final_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    const torch::Tensor final_goal_values = policy_goal_values_like(final_normalized, config_.goal_critic.goal_dim);
    ActorStepOutput final_output = rollout_actor->forward_step_stateful(
        final_normalized,
        rollout_recurrent_states[0],
        final_starts,
        nullptr,
        final_goal_values);

    std::unordered_map<std::string, torch::Tensor> bootstrap_values;
    bootstrap_values["extrinsic"] = final_output.value_win_logits.squeeze(-1);
    dest.set_final_values(bootstrap_values);
    }
  }

  if (collectors_.size() > 1) {
    for (auto& update : shard_normalizer_updates) {
      update.to(device_);
      normalizer.merge(update);
    }
  }

  const double collection_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();

  if (total_learner_steps > 0) {
    metrics.total_reward_mean = total_reward / static_cast<double>(total_learner_steps);
    metrics.gameplay_reward_mean = total_gameplay_reward / static_cast<double>(total_learner_steps);
    metrics.mechanic_reward_mean = total_mechanic_reward / static_cast<double>(total_learner_steps);
    metrics.mean_goal_distance = total_goal_distance / static_cast<double>(std::max(dest.rollout_length(), 1));
  }
  metrics.min_goal_distance = min_goal_distance;
  metrics.goals_scored = total_goals_scored;
  metrics.goals_conceded = total_goals_conceded;
  metrics.rollout_steps = dest.rollout_length();
  metrics.completed_episodes = completed_episodes;
  metrics.scored_episodes = scored_episodes;
  metrics.conceded_episodes = conceded_episodes;
  metrics.neutral_episodes = neutral_episodes;
  metrics.no_touch_episodes = no_touch_episodes;
  metrics.truncated_episodes = truncated_episodes;
  metrics.scored_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(scored_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.conceded_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(conceded_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.neutral_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(neutral_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.no_touch_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(no_touch_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.truncated_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(truncated_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.touch_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(touched_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  metrics.multi_touch_episode_rate =
      completed_episodes > 0
          ? static_cast<double>(multi_touched_episodes) / static_cast<double>(completed_episodes)
          : 0.0;
  for (const auto& [mode, comp] : mode_completed) {
    metrics.mode_completed_episodes[mode] = comp;
    metrics.mode_touch_rates[mode] = comp > 0
        ? static_cast<double>(mode_touched[mode]) / static_cast<double>(comp)
        : 0.0;
    metrics.mode_multi_touch_rates[mode] = comp > 0
        ? static_cast<double>(mode_multi_touched[mode]) / static_cast<double>(comp)
        : 0.0;
    metrics.mode_scored_rates[mode] = comp > 0
        ? static_cast<double>(mode_scored[mode]) / static_cast<double>(comp)
        : 0.0;
  }
  if (total_ball_proximity_denom > 0) {
    metrics.ball_proximity_rate = static_cast<double>(total_ball_proximity_steps) / static_cast<double>(total_ball_proximity_denom);
  }
  if (accumulated_value_count > 0) {
    metrics.sampled_value_win_mean = accumulated_sampled_value
        / static_cast<double>(accumulated_value_count);
  }

  metrics.obs_build_seconds = collector_timings.obs_build_seconds;
  metrics.mask_build_seconds = collector_timings.mask_build_seconds;
  metrics.env_step_seconds = collector_timings.env_step_seconds;
  metrics.done_reset_seconds = collector_timings.done_reset_seconds;
  metrics.collection_agent_steps_per_second =
      local_collected_steps > 0 ? static_cast<double>(local_collected_steps) / collection_seconds : 0.0;

#ifdef PULSAR_HAS_CUDA
  c10::cuda::setCurrentCUDAStream(prev_collect_stream);
#endif
  *collected_agent_steps = local_collected_steps;
}

CheckpointMetadata APPOTrainer::make_checkpoint_metadata(std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  nlohmann::json extra = nlohmann::json::object();
  extra["trainer_state_version"] = kTrainerStateVersion;
  if (!wandb_run_id.empty()) {
    extra["wandb_run_id"] = wandb_run_id;
  }
  if (self_play_manager_ && self_play_manager_->enabled()) {
    extra["self_play_ratings"] = self_play_manager_->current_ratings();
    const std::string rng = self_play_manager_->rng_state();
    if (!rng.empty()) {
      extra["self_play_rng_state"] = rng;
    }
  }
  extra["curriculum_stage"] = curriculum_.state().stage_index;
  extra["curriculum_previous_stage"] = curriculum_.state().previous_stage_index;
  extra["curriculum_agent_steps"] = curriculum_.state().agent_steps_in_stage;
  extra["curriculum_promotion_counter"] = curriculum_.state().promotion_counter;
  extra["curriculum_current_mode"] = curriculum_.state().current_mode;
  nlohmann::json mode_touch_json = nlohmann::json::object();
  for (const auto& [mode, deq] : curriculum_.state().mode_touch_rates) {
    nlohmann::json arr = nlohmann::json::array();
    for (double v : deq) arr.push_back(v);
    mode_touch_json[mode] = arr;
  }
  extra["curriculum_mode_touch_rates"] = mode_touch_json;
  nlohmann::json mode_multi_touch_json = nlohmann::json::object();
  for (const auto& [mode, deq] : curriculum_.state().mode_multi_touch_rates) {
    nlohmann::json arr = nlohmann::json::array();
    for (double v : deq) arr.push_back(v);
    mode_multi_touch_json[mode] = arr;
  }
  extra["curriculum_mode_multi_touch_rates"] = mode_multi_touch_json;
  nlohmann::json mode_scored_json = nlohmann::json::object();
  for (const auto& [mode, deq] : curriculum_.state().mode_scored_rates) {
    nlohmann::json arr = nlohmann::json::array();
    for (double v : deq) arr.push_back(v);
    mode_scored_json[mode] = arr;
  }
  extra["curriculum_mode_scored_rates"] = mode_scored_json;
  {
    nlohmann::json recent_json = nlohmann::json::array();
    for (double v : recent_scored_rates_) recent_json.push_back(v);
    extra["recent_scored_rates"] = recent_json;
  }
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "mamba2_goal_appo",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = {"extrinsic"},
      .extra = std::move(extra),
  };
}

void APPOTrainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  PULSAR_TRACE_SCOPE_CAT("trainer", "checkpoint_save");
  synchronize_cuda_if_needed(device_, "checkpoint save start");
  const std::filesystem::path staging = make_checkpoint_staging_directory(directory);
  remove_checkpoint_directory(staging);
  try {
    std::filesystem::create_directories(staging);
    save_experiment_config(config_, (staging / "config.json").string());
    save_checkpoint_metadata(make_checkpoint_metadata(global_step, update_index, wandb_run_id), (staging / "metadata.json").string());
    save_training_state(staging / "state.pt");
    commit_checkpoint_directory(staging, directory);
    synchronize_cuda_if_needed(device_, "checkpoint save end");
  } catch (...) {
    remove_checkpoint_directory(staging);
    throw;
  }
}

TrainerBenchmarkMetrics APPOTrainer::benchmark(int updates) {
  const int bounded_updates = std::max(1, updates);
  TrainerBenchmarkMetrics result{};
  result.updates = bounded_updates;
  const bool previous_progress = benchmark_progress_;
  benchmark_progress_ = true;

  if (curriculum_.enabled()) {
    curriculum_.initialize_stage();
    if (!curriculum_.mode_allocation().empty()) {
      rebuild_collectors();
    }
    apply_curriculum_to_collectors();
    apply_curriculum_lr();
  }

  const auto benchmark_start = std::chrono::steady_clock::now();
  for (int index = 0; index < bounded_updates; ++index) {
    TrainerMetrics collection_metrics{};
    std::int64_t collected_steps = 0;

    std::cout << "bench_update_start update=" << (index + 1)
              << "/" << bounded_updates << '\n' << std::flush;
    const auto collection_start = std::chrono::steady_clock::now();
    collect_rollout(rollout_, collection_metrics, &collected_steps, actor_snapshot_, actor_normalizer_);
    synchronize_cuda_if_needed(device_, "benchmark collection");
    const double collection_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_start).count();
    result.collection_seconds += collection_seconds;
    result.obs_build_seconds += collection_metrics.obs_build_seconds;
    result.mask_build_seconds += collection_metrics.mask_build_seconds;
    result.policy_forward_seconds += collection_metrics.policy_forward_seconds;
    result.action_decode_seconds += collection_metrics.action_decode_seconds;
    result.env_step_seconds += collection_metrics.env_step_seconds;
    result.done_reset_seconds += collection_metrics.done_reset_seconds;
    std::cout << "bench_collection_done update=" << (index + 1)
              << " agent_steps=" << collected_steps
              << " seconds=" << collection_seconds
              << " agent_steps_per_second="
              << (collected_steps > 0 ? static_cast<double>(collected_steps) / std::max(collection_seconds, 1.0e-9) : 0.0)
              << '\n' << std::flush;

    TrainerMetrics update_metrics = update_actor(rollout_);
    result.agent_steps += collected_steps;
    result.update_seconds += update_metrics.update_seconds;
    result.forward_backward_seconds += update_metrics.forward_backward_seconds;
    result.optimizer_step_seconds += update_metrics.optimizer_step_seconds;
    result.policy_loss += update_metrics.policy_loss;
    result.value_loss += update_metrics.value_loss;
    result.entropy += update_metrics.entropy;
    result.grad_norm += update_metrics.grad_norm;
    std::cout << "bench_update_done update=" << (index + 1)
              << " update_seconds=" << update_metrics.update_seconds
              << " forward_backward_seconds=" << update_metrics.forward_backward_seconds
              << " optimizer_step_seconds=" << update_metrics.optimizer_step_seconds
              << " ppo_update_agent_steps_per_second="
              << (collected_steps > 0 ? static_cast<double>(collected_steps) / std::max(update_metrics.update_seconds, 1.0e-9) : 0.0)
              << '\n' << std::flush;

    synchronize_cuda_if_needed(device_, "benchmark snapshot clone");
    actor_snapshot_ = clone_ppo_actor(actor_, device_);
    actor_snapshot_->eval();
    // Sync persistent collection actors with new snapshot
    for (size_t i = 0; i < collection_actors_.size(); ++i) {
      if (collection_actors_[i] && shard_devices_[i] == device_) {
        collection_actors_[i] = actor_snapshot_;
      } else if (collection_actors_[i]) {
        collection_actors_[i] = clone_ppo_actor(actor_snapshot_, shard_devices_[i]);
        collection_actors_[i]->eval();
      }
    }
    // Run ES-LoRA update.  In benchmark mode we ignore curriculum stage
    // and purely follow es_interval so the user can dial ES frequency directly.
    const int update_index = index + 1;
    if (config_.es_lora.es_interval > 0 && update_index % config_.es_lora.es_interval == 0) {
      TrainerMetrics es_metrics{};
      std::cout << "bench_es_update_start update=" << update_index << '/' << bounded_updates << '\n' << std::flush;
      run_es_lora_update(update_index, es_metrics);
      // Sync ES-LoRA weight changes to replica actors.
      if (compute_actors_.size() > 0) {
        sync_actor_to_replicas(actor_, compute_actors_);
      }
      result.es_updates += 1;
      result.es_seconds += es_metrics.es_seconds;
      result.es_eval_seconds += es_metrics.es_eval_seconds;
      std::cout << "bench_es_update_done update=" << update_index
                << " es_seconds=" << es_metrics.es_seconds
                << " es_eval_seconds=" << es_metrics.es_eval_seconds
                << '\n' << std::flush;
    }

    rollout_.clear();
  }

  result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
  const double denom = static_cast<double>(bounded_updates);
  result.policy_loss /= denom;
  result.value_loss /= denom;
  result.entropy /= denom;
  result.grad_norm /= denom;
  benchmark_progress_ = previous_progress;
  return result;
}

void APPOTrainer::save_training_state(const std::filesystem::path& path) const {
  torch::NoGradGuard no_grad;
  torch::serialize::OutputArchive archive;
  actor_->save(archive);
  actor_normalizer_.save(archive);
  actor_optimizer_.save(archive);
  archive.save_to(path.string());
}

void APPOTrainer::load_training_state(const std::filesystem::path& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), device_);
  actor_->load(archive);
  actor_normalizer_.load(archive);
  actor_optimizer_.load(archive);
  actor_->to(device_);
  actor_normalizer_.to(device_);
}

void APPOTrainer::prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const {
  const int max_checkpoints = config_.ppo.max_rolling_checkpoints;
  if (max_checkpoints <= 0) {
    return;
  }
  std::error_code ec;
  if (!std::filesystem::exists(checkpoint_dir, ec)) {
    return;
  }
  std::vector<std::pair<int, std::filesystem::path>> updates;
  for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir, ec)) {
    if (ec) break;
    if (!entry.is_directory(ec)) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    // Never prune stage progression checkpoints.
    if (name.rfind("stage_", 0) == 0) {
      continue;
    }
    if (name.rfind("update_", 0) != 0) {
      continue;
    }
    const std::string suffix = name.substr(7);
    if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) {
      continue;
    }
    try {
      updates.emplace_back(std::stoi(suffix), entry.path());
    } catch (...) {
    }
  }
  std::sort(updates.begin(), updates.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  for (std::size_t i = static_cast<std::size_t>(max_checkpoints); i < updates.size(); ++i) {
    remove_checkpoint_directory(updates[i].second);
  }
}

void APPOTrainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  const bool train_forever = updates <= 0;
  std::cout << "train_start curriculum_enabled=" << (curriculum_.enabled() ? 1 : 0)
            << " stages=" << config_.curriculum.stages.size()
            << '\n';
  std::filesystem::create_directories(checkpoint_dir);
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "dappo_train");
  std::int64_t global_step = resumed_global_step_;

  if (curriculum_.enabled()) {
    if (resumed_update_index_ == 0) {
      curriculum_.initialize_stage();
    }
    if (!curriculum_.mode_allocation().empty()) {
      rebuild_collectors();
    }
    apply_curriculum_to_collectors();
    apply_curriculum_lr();
  }

  TrainerMetrics coll_metrics{};
  std::int64_t coll_steps = 0;
  collect_rollout(rollout_, coll_metrics, &coll_steps, actor_snapshot_, actor_normalizer_);
  global_step += coll_steps;

  for (int index = 0; train_forever || index < updates; ++index) {
    PULSAR_TRACE_SCOPE_CAT("trainer", "train_iteration");
    const auto iter_start = std::chrono::steady_clock::now();
    const int update_index = static_cast<int>(resumed_update_index_) + index + 1;

    TrainerMetrics next_coll_metrics{};
    std::int64_t next_coll_steps = 0;
    const bool has_next = train_forever || index + 1 < updates;
    const bool overlap_collection_update = has_next && config_.ppo.overlap_collection_update;

    std::optional<ObservationNormalizer> collection_normalizer;

    // 1. Main thread CPU work that can overlap with the update when overlap
    // collection is enabled.
    {
      double scored = static_cast<double>(coll_metrics.scored_episodes);
      double completed = static_cast<double>(std::max(coll_metrics.completed_episodes, static_cast<int64_t>(1)));
      double scored_rate = scored / completed;
      recent_scored_rates_.push_back(scored_rate);
      coll_metrics.scored_episode_rate = scored_rate;
      if (static_cast<int>(recent_scored_rates_.size()) > kRecentScoredRateWindow) {
        recent_scored_rates_.pop_front();
      }
    }

    // 2. Update the actor on the main thread so CUDA/Torch thread-local
    // resources do not churn every iteration. Optional overlap only
    // backgrounds collection for the next rollout.
    TrainerMetrics train_metrics{};
    std::future<std::int64_t> collect_future;
    bool discard_overlapped_rollout = false;
    if (overlap_collection_update) {
      collection_normalizer.emplace(actor_normalizer_.clone());
      collect_future = std::async(std::launch::async, [&]() {
        std::int64_t steps = 0;
        collect_rollout(rollout_B_, next_coll_metrics, &steps, actor_snapshot_, *collection_normalizer);
        return steps;
      });
      train_metrics = update_actor(rollout_);
      next_coll_steps = collect_future.get();
      actor_normalizer_ = collection_normalizer->clone();
    } else {
      train_metrics = update_actor(rollout_);
    }

    // 3. Self-play / ES-LoRA / plasticity (touch actor_, safe now)
    if (self_play_manager_ && (!curriculum_.enabled() || curriculum_.stage_index() >= kSelfPlayMinStage)) {
      const SelfPlayMetrics self_play_metrics =
          self_play_manager_->on_update(actor_, actor_normalizer_, global_step, update_index);
      coll_metrics.self_play_eval_seconds = self_play_metrics.eval_seconds;
      coll_metrics.elo_ratings = self_play_metrics.ratings;
      coll_metrics.self_play_snapshot_count = self_play_metrics.snapshot_count;
    }

    if (curriculum_.stage_index() >= kEsLoraMinStage && update_index % config_.es_lora.es_interval == 0) {
      run_es_lora_update(update_index, coll_metrics);
      // Sync ES-LoRA weight changes to replica actors.
      if (compute_actors_.size() > 0) {
        sync_actor_to_replicas(actor_, compute_actors_);
      }
      discard_overlapped_rollout = discard_overlapped_rollout || overlap_collection_update;
    }

    if (config_.ppo.plasticity && update_index % config_.ppo.plasticity_interval == 0) {
      PULSAR_TRACE_SCOPE_CAT("trainer", "plasticity");
      shrink_perturb_parameters(*actor_, config_.ppo.plasticity_shrink, config_.ppo.plasticity_noise);
      actor_->to(device_);
      for (auto& p : actor_->parameters()) {
        if (!p.requires_grad() || p.dim() < 2) continue;
        actor_optimizer_.state().erase(p.unsafeGetTensorImpl());
      }
      // Sync plasticity weight changes to replica actors.
      if (compute_actors_.size() > 0) {
        sync_actor_to_replicas(actor_, compute_actors_);
      }
      discard_overlapped_rollout = discard_overlapped_rollout || overlap_collection_update;
    }

    // 4. Curriculum checks + rebuild (may touch collectors, safe after collection)
    if (curriculum_.enabled()) {
      bool stage_changed = false;
      if (curriculum_.check_promotion(
          coll_metrics.mode_touch_rates,
          coll_metrics.mode_multi_touch_rates,
          coll_metrics.mode_scored_rates,
          coll_metrics.mode_completed_episodes,
          coll_steps)) {
        if (curriculum_.mode_allocation_changed()) {
          rebuild_collectors();
        }
        apply_curriculum_to_collectors();
        apply_curriculum_lr();
        stage_changed = true;
      }
      if (stage_changed) {
        discard_overlapped_rollout = discard_overlapped_rollout || overlap_collection_update;
        save_checkpoint(
            std::filesystem::path(checkpoint_dir) / ("stage_" + std::to_string(curriculum_.state().stage_index) + "_update_" + std::to_string(update_index)),
            global_step, update_index, wandb.run_id());
      }
    }

    // 5. Clone snapshot for next iteration's collection after all work using
    // the previous snapshot has completed.
    PPOActor new_snapshot{nullptr};
    if (has_next) {
      synchronize_cuda_if_needed(device_, "snapshot clone");
      new_snapshot = clone_ppo_actor(actor_, device_);
      new_snapshot->eval();
      // Sync persistent collection actors
      for (size_t i = 0; i < collection_actors_.size(); ++i) {
        if (collection_actors_[i] && shard_devices_[i] == device_) {
          collection_actors_[i] = new_snapshot;
        } else if (collection_actors_[i]) {
          collection_actors_[i] = clone_ppo_actor(new_snapshot, shard_devices_[i]);
          collection_actors_[i]->eval();
        }
      }
    }

    // 6. In the default memory-safe mode, collect after the update with the
    // fresh policy snapshot.
    if (has_next && (!overlap_collection_update || discard_overlapped_rollout)) {
      next_coll_metrics = TrainerMetrics{};
      next_coll_steps = 0;
      collect_rollout(rollout_B_, next_coll_metrics, &next_coll_steps, new_snapshot, actor_normalizer_);
    }

    global_step += next_coll_steps;

    coll_metrics.policy_loss = train_metrics.policy_loss;
    coll_metrics.value_loss = train_metrics.value_loss;
    coll_metrics.entropy = train_metrics.entropy;
    coll_metrics.effective_entropy_coef = train_metrics.effective_entropy_coef;
    coll_metrics.grad_norm = train_metrics.grad_norm;
    coll_metrics.policy_approx_kl = train_metrics.policy_approx_kl;
    coll_metrics.policy_clip_fraction = train_metrics.policy_clip_fraction;
    coll_metrics.policy_log_ratio_abs_max = train_metrics.policy_log_ratio_abs_max;
    coll_metrics.nonfinite_loss_skips = train_metrics.nonfinite_loss_skips;
    coll_metrics.nonfinite_grad_norm_skips = train_metrics.nonfinite_grad_norm_skips;
    coll_metrics.kl_guard_skips = train_metrics.kl_guard_skips;
    coll_metrics.grad_norm_guard_skips = train_metrics.grad_norm_guard_skips;
    coll_metrics.update_seconds = train_metrics.update_seconds;
    coll_metrics.forward_backward_seconds = train_metrics.forward_backward_seconds;
    coll_metrics.optimizer_step_seconds = train_metrics.optimizer_step_seconds;
    coll_metrics.goal_critic_loss = train_metrics.goal_critic_loss;
    coll_metrics.mean_goal_score = train_metrics.mean_goal_score;
    coll_metrics.mean_sampled_goal_distance = train_metrics.mean_sampled_goal_distance;
    coll_metrics.update_agent_steps_per_second =
        next_coll_steps > 0 ? static_cast<double>(next_coll_steps) / std::max(train_metrics.update_seconds, 1.0e-9) : 0.0;

    coll_metrics.overall_agent_steps_per_second =
        next_coll_steps > 0
            ? static_cast<double>(next_coll_steps) /
                  std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count(), 1.0e-9)
            : 0.0;
    trim_released_host_memory();
    coll_metrics.process_rss_mb = current_process_rss_mb();
    coll_metrics.process_peak_rss_mb = current_process_peak_rss_mb();
    const CgroupMemoryStats cgroup_memory = current_cgroup_memory_stats();
    coll_metrics.cgroup_memory_current_mb = cgroup_memory.current_mb;
    coll_metrics.cgroup_memory_limit_mb = cgroup_memory.limit_mb;
    sample_cuda_memory_stats(coll_metrics, compute_devices_);

    append_metrics_line(checkpoint_dir, update_index, global_step, coll_metrics);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " policy_loss=" << coll_metrics.policy_loss
              << " value_loss=" << coll_metrics.value_loss
              << " entropy=" << coll_metrics.entropy
              << " grad_norm=" << coll_metrics.grad_norm
              << " policy_approx_kl=" << coll_metrics.policy_approx_kl
              << " clip_frac=" << coll_metrics.policy_clip_fraction
              << " max_log_ratio=" << coll_metrics.policy_log_ratio_abs_max
              << " nonfinite_loss_skips=" << coll_metrics.nonfinite_loss_skips
              << " nonfinite_grad_skips=" << coll_metrics.nonfinite_grad_norm_skips
              << " kl_guard_skips=" << coll_metrics.kl_guard_skips
              << " grad_guard_skips=" << coll_metrics.grad_norm_guard_skips
              << " total_reward=" << coll_metrics.total_reward_mean
              << " gameplay_reward=" << coll_metrics.gameplay_reward_mean
              << " mechanic_reward=" << coll_metrics.mechanic_reward_mean
              << " rollout_steps=" << coll_metrics.rollout_steps
              << " completed_eps=" << coll_metrics.completed_episodes
              << " scored_eps=" << coll_metrics.scored_episodes
              << " conceded_eps=" << coll_metrics.conceded_episodes
              << " neutral_eps=" << coll_metrics.neutral_episodes
              << " no_touch_eps=" << coll_metrics.no_touch_episodes
              << " trunc_eps=" << coll_metrics.truncated_episodes
              << " touch_rate=" << coll_metrics.touch_episode_rate
              << " multi_touch_rate=" << coll_metrics.multi_touch_episode_rate
              << " no_touch_rate=" << coll_metrics.no_touch_episode_rate
              << " sampled_goal_dist=" << coll_metrics.mean_sampled_goal_distance
              << " mean_goal_dist=" << coll_metrics.mean_goal_distance
              << " ball_prox=" << coll_metrics.ball_proximity_rate
              << " goals=" << coll_metrics.goals_scored << "/" << coll_metrics.goals_conceded
              << " es_fitness=" << coll_metrics.es_fitness_mean
              << " es_reward=" << coll_metrics.es_reward_mean
              << " rss_mb=" << coll_metrics.process_rss_mb
              << " peak_rss_mb=" << coll_metrics.process_peak_rss_mb
              << " cgroup_mem_mb=" << coll_metrics.cgroup_memory_current_mb << "/" << coll_metrics.cgroup_memory_limit_mb
              << " cuda_reserved_mb=" << coll_metrics.cuda_memory_reserved_mb
              << " cuda_max_reserved_mb=" << coll_metrics.cuda_max_memory_reserved_mb
              << " cuda_ooms=" << coll_metrics.cuda_ooms
              << " league_snapshots=" << coll_metrics.self_play_snapshot_count
              << " curriculum=" << curriculum_.state().stage_index
              << " cur_steps=" << curriculum_.state().agent_steps_in_stage
              << " cur_promo=" << curriculum_.state().promotion_counter
              << '\n';
    if (wandb.enabled()) {
      nlohmann::json payload{{"_step", global_step}};
      nlohmann::json sections = nlohmann::json::object();
      const auto add_metric =
          [&payload, &sections](const std::string& section, const std::string& key, nlohmann::json value) {
            add_wandb_metric(payload, sections, section, key, std::move(value));
          };
      const auto register_metric = [&sections](const std::string& section, const std::string& key) {
        register_wandb_metric_section(sections, section, key);
      };

      for (const auto& mode : configured_wandb_modes(config_)) {
        register_mode_wandb_sections(sections, mode);
      }
      register_metric("ES-LoRA", "es_fitness_mean");
      register_metric("ES-LoRA", "es_fitness_std");
      register_metric("ES-LoRA", "es_fitness_best");
      register_metric("ES-LoRA", "es_reward_mean");
      register_metric("ES-LoRA", "es_winrate_mean");
      register_metric("ES-LoRA", "es_kl_mean");
      register_metric("ES-LoRA", "es_update_norm");
      register_metric("ES-LoRA", "es_lora_a_norm");
      register_metric("ES-LoRA", "es_lora_b_norm");

      add_metric("Optimization", "update", update_index);
      add_metric("Optimization", "global_step", global_step);
      add_metric("Optimization", "policy_loss", coll_metrics.policy_loss);
      add_metric("Optimization", "value_loss", coll_metrics.value_loss);
      add_metric("Optimization", "entropy", coll_metrics.entropy);
      add_metric("Optimization", "grad_norm", coll_metrics.grad_norm);
      add_metric("Optimization", "policy_approx_kl", coll_metrics.policy_approx_kl);
      add_metric("Optimization", "policy_clip_fraction", coll_metrics.policy_clip_fraction);
      add_metric("Optimization", "policy_log_ratio_abs_max", coll_metrics.policy_log_ratio_abs_max);
      add_metric("Optimization", "nonfinite_loss_skips", coll_metrics.nonfinite_loss_skips);
      add_metric("Optimization", "nonfinite_grad_norm_skips", coll_metrics.nonfinite_grad_norm_skips);
      add_metric("Optimization", "kl_guard_skips", coll_metrics.kl_guard_skips);
      add_metric("Optimization", "grad_norm_guard_skips", coll_metrics.grad_norm_guard_skips);
      add_metric("Optimization", "rollout_steps", coll_metrics.rollout_steps);
      add_metric("Optimization", "effective_entropy_coef", coll_metrics.effective_entropy_coef);
      add_metric("Optimization", "self_play_snapshot_count", coll_metrics.self_play_snapshot_count);
      add_metric("Optimization", "curriculum_stage", curriculum_.state().stage_index);
      add_metric("Optimization", "curriculum_agent_steps", curriculum_.state().agent_steps_in_stage);
      add_metric("Optimization", "curriculum_promotion_counter", curriculum_.state().promotion_counter);

      add_metric("System", "process_rss_mb", coll_metrics.process_rss_mb);
      add_metric("System", "process_peak_rss_mb", coll_metrics.process_peak_rss_mb);
      add_metric("System", "cgroup_memory_current_mb", coll_metrics.cgroup_memory_current_mb);
      add_metric("System", "cgroup_memory_limit_mb", coll_metrics.cgroup_memory_limit_mb);
      add_metric("System", "cuda_memory_allocated_mb", coll_metrics.cuda_memory_allocated_mb);
      add_metric("System", "cuda_memory_reserved_mb", coll_metrics.cuda_memory_reserved_mb);
      add_metric("System", "cuda_max_memory_allocated_mb", coll_metrics.cuda_max_memory_allocated_mb);
      add_metric("System", "cuda_max_memory_reserved_mb", coll_metrics.cuda_max_memory_reserved_mb);
      add_metric("System", "cuda_alloc_retries", coll_metrics.cuda_alloc_retries);
      add_metric("System", "cuda_ooms", coll_metrics.cuda_ooms);

      add_metric("Rewards", "total_reward_mean", coll_metrics.total_reward_mean);
      add_metric("Rewards", "gameplay_reward_mean", coll_metrics.gameplay_reward_mean);
      add_metric("Rewards", "mechanic_reward_mean", coll_metrics.mechanic_reward_mean);
      add_metric("Rewards", "completed_episodes", coll_metrics.completed_episodes);
      add_metric("Rewards", "conceded_episodes", coll_metrics.conceded_episodes);
      add_metric("Rewards", "neutral_episodes", coll_metrics.neutral_episodes);
      add_metric("Rewards", "no_touch_episodes", coll_metrics.no_touch_episodes);
      add_metric("Rewards", "truncated_episodes", coll_metrics.truncated_episodes);
      add_metric("Rewards", "touch_episode_rate", coll_metrics.touch_episode_rate);
      add_metric("Rewards", "multi_touch_episode_rate", coll_metrics.multi_touch_episode_rate);
      add_metric("Rewards", "scored_episode_rate", coll_metrics.scored_episode_rate);
      add_metric("Rewards", "conceded_episode_rate", coll_metrics.conceded_episode_rate);
      add_metric("Rewards", "neutral_episode_rate", coll_metrics.neutral_episode_rate);
      add_metric("Rewards", "no_touch_episode_rate", coll_metrics.no_touch_episode_rate);
      add_metric("Rewards", "truncated_episode_rate", coll_metrics.truncated_episode_rate);
      add_metric("Rewards", "ball_proximity_rate", coll_metrics.ball_proximity_rate);

      add_metric("GCRL", "sampled_value_win_mean", coll_metrics.sampled_value_win_mean);
      add_metric("GCRL", "goal_critic_loss", coll_metrics.goal_critic_loss);
      add_metric("GCRL", "mean_goal_score", coll_metrics.mean_goal_score);
      add_metric("GCRL", "mean_sampled_goal_distance", coll_metrics.mean_sampled_goal_distance);
      add_metric("GCRL", "mean_goal_distance", coll_metrics.mean_goal_distance);
      add_metric("GCRL", "min_goal_distance", coll_metrics.min_goal_distance);

      for (const auto& [mode, rate] : coll_metrics.mode_touch_rates) {
        add_metric(mode, "mode_" + mode + "_touch_episode_rate", rate);
      }
      for (const auto& [mode, rate] : coll_metrics.mode_multi_touch_rates) {
        add_metric(mode, "mode_" + mode + "_multi_touch_episode_rate", rate);
      }
      for (const auto& [mode, rate] : coll_metrics.mode_scored_rates) {
        add_metric(mode, "mode_" + mode + "_scored_episode_rate", rate);
      }
      for (const auto& [mode, count] : coll_metrics.mode_completed_episodes) {
        add_metric(mode, "mode_" + mode + "_completed_episodes", count);
      }
      if (curriculum_.stage_index() >= kEsLoraMinStage && update_index % config_.es_lora.es_interval == 0) {
        add_metric("ES-LoRA", "es_fitness_mean", coll_metrics.es_fitness_mean);
        add_metric("ES-LoRA", "es_fitness_std", coll_metrics.es_fitness_std);
        add_metric("ES-LoRA", "es_fitness_best", coll_metrics.es_fitness_best);
        add_metric("ES-LoRA", "es_reward_mean", coll_metrics.es_reward_mean);
        add_metric("ES-LoRA", "es_winrate_mean", coll_metrics.es_winrate_mean);
        add_metric("ES-LoRA", "es_kl_mean", coll_metrics.es_kl_mean);
        add_metric("ES-LoRA", "es_update_norm", coll_metrics.es_update_norm);
        add_metric("ES-LoRA", "es_lora_a_norm", coll_metrics.es_lora_a_norm);
        add_metric("ES-LoRA", "es_lora_b_norm", coll_metrics.es_lora_b_norm);
      }
      for (const auto& [mode, rating] : coll_metrics.elo_ratings) {
        add_metric(mode, "elo_" + mode, rating);
      }
      payload["_wandb_sections"] = std::move(sections);
      payload["_wandb_section_order"] = wandb_section_order();
      wandb.log(payload);
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      save_checkpoint(std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index)), global_step, update_index, wandb.run_id());
      prune_old_checkpoints(checkpoint_dir);
    }

    if (has_next) {
      std::swap(rollout_, rollout_B_);
      coll_metrics = std::move(next_coll_metrics);
      actor_snapshot_ = std::move(new_snapshot);
    }
    coll_steps = next_coll_steps;
  }
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + updates, wandb.run_id());
  wandb.finish();
}

}  // namespace pulsar

#endif
