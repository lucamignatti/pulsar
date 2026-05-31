#include "pulsar/training/trainer.hpp"

#ifdef PULSAR_HAS_TORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
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
#include "pulsar/training/optimizers.hpp"
#include "pulsar/training/ppo_math.hpp"
#include "pulsar/tracing/tracing.hpp"

#ifdef PULSAR_HAS_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace pulsar {

namespace {

constexpr int kEsLoraMinStage = 0;
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

std::vector<torch::Tensor> actor_policy_parameters(const Actor& actor) {
  std::vector<torch::Tensor> parameters;
  for (const auto& named_param : actor->named_parameters(/*recurse=*/true)) {
    parameters.push_back(named_param.value());
  }
  return parameters;
}



void write_scalar(torch::serialize::OutputArchive& archive, const std::string& key, float value) {
  archive.write(key, torch::tensor({value}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
}

bool read_scalar(torch::serialize::InputArchive& archive, const std::string& key, float& value) {
  try {
    torch::Tensor tensor;
    archive.read(key, tensor);
    value = tensor.to(torch::kCPU).to(torch::kFloat32).flatten()[0].item<float>();
    return true;
  } catch (...) {
    return false;
  }
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

void synchronize_current_cuda_stream_if_needed(const torch::Device& device, const char* context) noexcept {
  if (!device.is_cuda()) {
    return;
  }
  try {
#ifdef PULSAR_HAS_CUDA
    c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
#else
    (void)context;
#endif
  } catch (const std::exception& exc) {
    std::cerr << "cuda stream synchronize failed during " << context << ": " << exc.what() << '\n';
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
  std::vector<std::string> heads = {"extrinsic", "gameplay", "mechanic"};
  return RolloutStorage(
      config.ppo.rollout_length,
      num_agents,
      config.model.observation_dim,
      action_dim,
      config.model.encoder_dim,
      config.goal_mapping.goal_dim,
      torch::Device(torch::kCPU),
      heads,
      pin_memory);
}

void require_finite(const torch::Tensor& tensor, const std::string& name) {
  if (tensor.defined() && !torch::isfinite(tensor).all().item<bool>()) {
    throw std::runtime_error("Non-finite tensor: " + name);
  }
}

// All CapturedGrad and goal critic loss helpers are modularized in gradient_ops.cpp and gcrl_trainer.cpp.

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

int effective_update_forward_samples(
    const ModelConfig& config,
    const torch::Device& device,
    bool overlap_collection_update,
    std::size_t num_update_gpus,
    int rollout_total_agents,
    int nominal_env_count) {
  (void)overlap_collection_update;
  (void)num_update_gpus;
  (void)rollout_total_agents;
  (void)nominal_env_count;
  return effective_max_forward_samples(config, device);
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics,
    bool es_fired);

struct MetricItem {
  std::string key;
  nlohmann::json value;
  std::string section;
};

std::vector<MetricItem> get_all_metrics(
    const TrainerMetrics& metrics,
    int update_index,
    std::int64_t global_step,
    bool es_fired) {
  std::vector<MetricItem> items;

  // 1. CRL Optimization Metrics
  items.push_back({"update", update_index, "Optimization"});
  items.push_back({"global_step", global_step, "Optimization"});
  items.push_back({"ball_critic_loss", metrics.policy_loss, "Optimization"});
  items.push_back({"grad_norm", metrics.grad_norm_valid_steps > 0 ? nlohmann::json(metrics.grad_norm) : nlohmann::json(nullptr), "Optimization"});
  items.push_back({"grad_norm_valid_steps", metrics.grad_norm_valid_steps, "Optimization"});
  items.push_back({"optimizer_steps", metrics.optimizer_steps, "Optimization"});
  items.push_back({"nonfinite_loss_skips", metrics.nonfinite_loss_skips, "Optimization"});
  items.push_back({"nonfinite_grad_norm_skips", metrics.nonfinite_grad_norm_skips, "Optimization"});
  items.push_back({"rollout_steps", metrics.rollout_steps, "Optimization"});

  // 2. Anchor Evaluation Metrics
  if (metrics.anchor_win_rate >= 0.0) {
    items.push_back({"anchor_win_rate", metrics.anchor_win_rate, "Optimization"});
    items.push_back({"anchor_eval_seconds", metrics.anchor_eval_seconds, "Optimization"});
    items.push_back({"anchor_eval_episodes", metrics.anchor_eval_episodes, "Optimization"});
    items.push_back({"anchor_updated", metrics.anchor_updated ? 1 : 0, "Optimization"});
  }

  // 3. System Metrics
  items.push_back({"process_rss_mb", metrics.process_rss_mb, "System"});
  items.push_back({"process_peak_rss_mb", metrics.process_peak_rss_mb, "System"});
  items.push_back({"cgroup_memory_current_mb", metrics.cgroup_memory_current_mb, "System"});
  items.push_back({"cgroup_memory_limit_mb", metrics.cgroup_memory_limit_mb, "System"});
  items.push_back({"cuda_memory_allocated_mb", metrics.cuda_memory_allocated_mb, "System"});
  items.push_back({"cuda_memory_reserved_mb", metrics.cuda_memory_reserved_mb, "System"});
  items.push_back({"cuda_max_memory_allocated_mb", metrics.cuda_max_memory_allocated_mb, "System"});
  items.push_back({"cuda_max_memory_reserved_mb", metrics.cuda_max_memory_reserved_mb, "System"});
  items.push_back({"cuda_alloc_retries", metrics.cuda_alloc_retries, "System"});
  items.push_back({"cuda_ooms", metrics.cuda_ooms, "System"});

  // 4. Episode Metrics
  items.push_back({"mechanic_reward_mean", metrics.mechanic_reward_mean, "Rewards"});
  items.push_back({"completed_episodes", metrics.completed_episodes, "Rewards"});
  items.push_back({"conceded_episodes", metrics.conceded_episodes, "Rewards"});
  items.push_back({"neutral_episodes", metrics.neutral_episodes, "Rewards"});
  items.push_back({"no_touch_episodes", metrics.no_touch_episodes, "Rewards"});
  items.push_back({"truncated_episodes", metrics.truncated_episodes, "Rewards"});
  items.push_back({"touch_episode_rate", metrics.touch_episode_rate, "Rewards"});
  items.push_back({"multi_touch_episode_rate", metrics.multi_touch_episode_rate, "Rewards"});
  items.push_back({"scored_episode_rate", metrics.scored_episode_rate, "Rewards"});
  items.push_back({"conceded_episode_rate", metrics.conceded_episode_rate, "Rewards"});
  items.push_back({"neutral_episode_rate", metrics.neutral_episode_rate, "Rewards"});
  items.push_back({"no_touch_episode_rate", metrics.no_touch_episode_rate, "Rewards"});
  items.push_back({"truncated_episode_rate", metrics.truncated_episode_rate, "Rewards"});
  items.push_back({"ball_proximity_rate", metrics.ball_proximity_rate, "Rewards"});
  items.push_back({"goals_scored", metrics.goals_scored, "Rewards"});
  items.push_back({"goals_conceded", metrics.goals_conceded, "Rewards"});

  // 5. GCRL Metrics
  items.push_back({"goal_critic_loss", metrics.goal_critic_loss, "GCRL"});
  items.push_back({"mean_goal_score", metrics.mean_goal_score, "GCRL"});
  items.push_back({"mean_sampled_goal_distance", metrics.mean_sampled_goal_distance, "GCRL"});
  items.push_back({"mean_goal_distance", metrics.mean_goal_distance, "GCRL"});
  items.push_back({"min_goal_distance", metrics.min_goal_distance, "GCRL"});

  // 6. Dynamic Shard Mode Metrics
  for (const auto& [mode, rate] : metrics.mode_touch_rates) {
    items.push_back({"mode_" + mode + "_touch_episode_rate", rate, mode});
  }
  for (const auto& [mode, rate] : metrics.mode_multi_touch_rates) {
    items.push_back({"mode_" + mode + "_multi_touch_episode_rate", rate, mode});
  }
  for (const auto& [mode, rate] : metrics.mode_scored_rates) {
    items.push_back({"mode_" + mode + "_scored_episode_rate", rate, mode});
  }
  for (const auto& [mode, count] : metrics.mode_completed_episodes) {
    items.push_back({"mode_" + mode + "_completed_episodes", count, mode});
  }

  // 7. ES-LoRA Metrics
  if (es_fired) {
    items.push_back({"es_fitness_mean", metrics.es_fitness_mean, "ES-LoRA"});
    items.push_back({"es_fitness_std", metrics.es_fitness_std, "ES-LoRA"});
    items.push_back({"es_fitness_best", metrics.es_fitness_best, "ES-LoRA"});
    items.push_back({"es_reward_mean", metrics.es_reward_mean, "ES-LoRA"});
    items.push_back({"es_winrate_mean", metrics.es_winrate_mean, "ES-LoRA"});
    items.push_back({"es_kl_mean", metrics.es_kl_mean, "ES-LoRA"});
    items.push_back({"es_update_norm", metrics.es_update_norm, "ES-LoRA"});
    items.push_back({"es_lora_a_norm", metrics.es_lora_a_norm, "ES-LoRA"});
    items.push_back({"es_lora_b_norm", metrics.es_lora_b_norm, "ES-LoRA"});
    items.push_back({"es_seconds", metrics.es_seconds, "ES-LoRA"});
    items.push_back({"es_eval_seconds", metrics.es_eval_seconds, "ES-LoRA"});
    items.push_back({"es_agent_steps_per_second", metrics.es_agent_steps_per_second, "ES-LoRA"});
    items.push_back({"es_effective_population", metrics.es_effective_population, "ES-LoRA"});
    items.push_back({"es_virtual_population_waves", metrics.es_virtual_population_waves, "ES-LoRA"});
    items.push_back({"es_eval_shards", metrics.es_eval_shards, "ES-LoRA"});
    items.push_back({"es_policy_update_norm", metrics.es_policy_update_norm, "ES-LoRA"});
  }

  // 8. Performance Timing Metrics
  items.push_back({"obs_build_seconds", metrics.obs_build_seconds, "System"});
  items.push_back({"mask_build_seconds", metrics.mask_build_seconds, "System"});
  items.push_back({"policy_forward_seconds", metrics.policy_forward_seconds, "System"});
  items.push_back({"action_decode_seconds", metrics.action_decode_seconds, "System"});
  items.push_back({"env_step_seconds", metrics.env_step_seconds, "System"});
  items.push_back({"done_reset_seconds", metrics.done_reset_seconds, "System"});
  items.push_back({"forward_backward_seconds", metrics.forward_backward_seconds, "System"});
  items.push_back({"optimizer_step_seconds", metrics.optimizer_step_seconds, "System"});

  // 9. CRL Off-Policy Metrics
  items.push_back({"replay_buffer_size", metrics.replay_buffer_size, "CRL"});
  items.push_back({"car_critic_loss", metrics.car_critic_loss, "CRL"});
  items.push_back({"car_actor_loss", metrics.car_actor_loss, "CRL"});
  items.push_back({"car_head_weight", metrics.car_head_weight, "CRL"});

  return items;
}

void append_metrics_line(
    const std::filesystem::path& checkpoint_dir,
    int update_index,
    std::int64_t global_step,
    const TrainerMetrics& metrics,
    bool es_fired) {
  nlohmann::json line = nlohmann::json::object();
  for (const auto& item : get_all_metrics(metrics, update_index, global_step, es_fired)) {
    line[item.key] = item.value;
  }
  
  if (checkpoint_dir.empty()) return;
  std::filesystem::create_directories(checkpoint_dir);
  std::ofstream output(checkpoint_dir / "metrics.jsonl", std::ios::app);
  output << line.dump() << '\n';
}

std::shared_ptr<MutatorSequence> make_es_eval_reset_mutator(const EnvConfig& config) {
  std::vector<StateMutatorPtr> mutators;
  mutators.push_back(std::make_shared<FixedTeamSizeMutator>(config));
  if (config.mode.find("random") != std::string::npos) {
    mutators.push_back(std::make_shared<RandomStateMutator>(config));
  } else {
    mutators.push_back(std::make_shared<KickoffMutator>(config));
  }
  return std::make_shared<MutatorSequence>(std::move(mutators));
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

Trainer::Trainer(
    ExperimentConfig config,
    std::unique_ptr<BatchedRocketSimCollector> collector,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : Trainer(
          std::move(config),
          make_collector_vector(std::move(collector)),
          std::move(run_output_root),
          log_initialization) {}

Trainer::Trainer(
    ExperimentConfig config,
    std::vector<std::unique_ptr<BatchedRocketSimCollector>> collectors,
    std::filesystem::path run_output_root,
    bool log_initialization)
    : config_(std::move(config)),
      collectors_(std::move(collectors)),
      action_table_(config_.action_table),
      actor_(Actor(config_.model, config_.goal_critic, config_.es_lora)),
      actor_normalizer_(config_.model.observation_dim),
      actor_optimizer_(
          actor_policy_parameters(actor_),
          torch::optim::SGDOptions(config_.ppo.learning_rate).weight_decay(config_.ppo.weight_decay)),
      task_pool_(std::make_unique<WorkStealingThreadPool>()),
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
    throw std::invalid_argument("Trainer requires at least one collector.");
  }
  if (actor_->policy_lora()->out_features() != action_dim_for_collectors(collectors_)) {
    throw std::invalid_argument("model.action_dim must match the action table size.");
  }
  if (config_.model.observation_dim != collectors_.front()->obs_dim()) {
    throw std::invalid_argument("model.observation_dim must match obs builder output.");
  }
  total_agents_ = total_agents_for_collectors(collectors_);
  if (total_agents_ == 0) {
    throw std::invalid_argument("Trainer collectors must contain agents.");
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

  if (config_.goal_critic.lambda_Zg > 0.0F || config_.goal_critic.lambda_goal_actor > 0.0F) {
    gcrl_trainer_ = std::make_unique<GCRLTrainer>(config_.goal_critic, device_);
  }

  // Replay buffer
  replay_buffer_ = std::make_unique<ReplayBuffer>(
      config_.replay_buffer,
      static_cast<int>(total_agents_),
      config_.model.observation_dim,
      config_.model.action_dim,
      config_.goal_mapping.goal_dim,
      config_.crl.car_goal_dim,
      torch::Device(torch::kCPU));

  // Car-head GoalCritic (locomotion bootstrap, annealed during training)
  if (config_.crl.enabled) {
    car_goal_critic_ = GoalCritic(
        actor_->feature_dim(),
        config_.model.action_dim,
        config_.goal_critic.embedding_dim,
        config_.goal_critic.hidden_dim,
        config_.crl.car_goal_dim);
    car_goal_critic_->to(device_);
    std::vector<at::Tensor> car_params;
    for (auto& p : car_goal_critic_->parameters()) {
      if (p.requires_grad()) car_params.push_back(p);
    }
    car_goal_critic_optimizer_ = std::make_unique<torch::optim::Adam>(
        car_params,
        torch::optim::AdamOptions(static_cast<double>(config_.ppo.learning_rate)));
  }

  maybe_initialize_from_checkpoint();
  sanitize_actor_parameters();
  actor_snapshot_ = clone_actor(actor_, device_);
  actor_snapshot_->eval();

  // Clone actor replicas to each additional compute GPU for data-parallel updates.
  for (size_t i = 1; i < compute_devices_.size(); ++i) {
    auto replica = clone_actor(actor_, compute_devices_[i]);
    replica->train();
    compute_actors_.push_back(std::move(replica));
  }

  // Persistent collection actors: one per shard device for policy inference during rollout.
  for (size_t i = 0; i < shard_devices_.size(); ++i) {
    if (shard_devices_[i] == device_) {
      collection_actors_.push_back(actor_snapshot_);  // shared, don't clone
    } else {
      collection_actors_.push_back(clone_actor(actor_snapshot_, shard_devices_[i]));
      collection_actors_.back()->eval();
    }
  }

  shard_agent_offsets_.clear();
  std::int64_t agent_offset = 0;
  for (const auto& collector : collectors_) {
    if (!collector) {
      throw std::invalid_argument("Trainer collectors must be non-null.");
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

Trainer::~Trainer() {
  synchronize_cuda_if_needed(compute_devices_, "trainer shutdown");
}

std::int64_t Trainer::model_parameter_count() const {
  std::int64_t total = 0;
  for (const auto& param : actor_->parameters()) {
    total += param.numel();
  }
  return total;
}

// Load actor weights from archive, falling back to a key-by-key partial load
// if strict loading fails (e.g., old checkpoints missing goal_proj_).
static void load_actor_from_archive(
    Actor& actor,
    torch::serialize::InputArchive& archive,
    const torch::Device& device) {
  try {
    actor->load(archive);
  } catch (const std::exception& e) {
    std::cerr << "[checkpoint] Strict actor load failed: " << e.what()
              << "\nRetrying with partial load (goal_proj_ may be freshly initialized).\n";
    torch::NoGradGuard no_grad;
    for (auto& item : actor->named_parameters(/*recurse=*/true)) {
      try {
        torch::Tensor t;
        archive.read(item.key(), t, /*is_buffer=*/false);
        item.value().copy_(t.to(device));
      } catch (...) {
        std::cerr << "[checkpoint]   Skipping missing parameter: " << item.key() << "\n";
      }
    }
    for (auto& item : actor->named_buffers(/*recurse=*/true)) {
      try {
        torch::Tensor t;
        archive.read(item.key(), t, /*is_buffer=*/true);
        item.value().copy_(t.to(device));
      } catch (...) {}
    }
  }
}

int team_size_from_mode(const std::string& mode) {
  if (mode == "1v1" || mode == "1v1_random") return 1;
  if (mode == "2v2" || mode == "2v2_random") return 2;
  if (mode == "3v3" || mode == "3v3_random") return 3;
  return 1;
}

int agents_per_env_from_mode(const std::string& mode) {
  return 2 * team_size_from_mode(mode);
}

void Trainer::rebuild_collectors() {
  std::vector<std::pair<std::string, float>> alloc = {{std::to_string(config_.env.team_size) + "v" + std::to_string(config_.env.team_size), 1.0F}};

  // compute max team size
  int max_team_size = config_.env.team_size;

  // create obs builder with max team size so obs_dim is constant across modes
  auto obs_builder_cfg = config_.env;
  obs_builder_cfg.team_size = max_team_size;
  auto obs_builder = std::make_shared<PulsarObsBuilder>(obs_builder_cfg);
  auto action_parser = std::make_shared<DiscreteActionParser>(ControllerActionTable(config_.action_table));
  auto done_condition = std::make_shared<SimpleDoneCondition>(config_.env);
  const bool pin_host = device_.is_cuda();

  const int nominal_envs = config_.ppo.num_envs;
  constexpr int kReferenceAgentsPerEnv = 2;
  const int target_agent_budget = std::max(kReferenceAgentsPerEnv, nominal_envs * kReferenceAgentsPerEnv);

  // Treat curriculum mode_allocation as agent/sample share, not raw match-count
  // share. This keeps mixed 2v2/3v3 stages near the same update compute budget
  // as 1v1 while still giving each mode its requested learner-sample share.
  std::vector<std::pair<std::string, int>> mode_envs;
  int allocated_envs = 0;
  int allocated_agents = 0;
  for (const auto& [mode, frac] : alloc) {
    const int agents_per_env = agents_per_env_from_mode(mode);
    int envs = static_cast<int>(
        std::round(static_cast<double>(target_agent_budget) * static_cast<double>(frac) /
                   static_cast<double>(agents_per_env)));
    if (envs <= 0) envs = 1;
    mode_envs.emplace_back(mode, envs);
    allocated_envs += envs;
    allocated_agents += envs * agents_per_env;
  }
  const int requested_shards = std::max(1, std::min(config_.ppo.collection_shards, std::max(1, allocated_envs)));

  if (log_initialization_) {
    std::cout << "curriculum_agent_balanced_envs target_agents=" << target_agent_budget
              << " allocated_agents=" << allocated_agents
              << " allocated_envs=" << allocated_envs
              << " nominal_envs=" << nominal_envs
              << '\n';
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
          static_cast<float>(mode_envs[i].second) / static_cast<float>(std::max(1, allocated_envs))));
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
      collection_actors_.push_back(clone_actor(actor_snapshot_, shard_devices_[i]));
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

  std::cout << "rebuilt_collectors collectors=" << collectors_.size()
            << " modes=" << mode_envs.size()
            << " total_agents=" << total_agents_
            << " total_envs=" << allocated_envs
            << " nominal_envs=" << nominal_envs
            << " devices=" << join_device_list(shard_devices_)
            << " rollout_length=" << config_.ppo.rollout_length << '\n';
}

void Trainer::maybe_initialize_from_checkpoint() {
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
      validate_inference_checkpoint_metadata(metadata, config_);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "Refusing to auto-resume checkpoint " + base.string() +
          ": " + e.what() +
          " Use a fresh output directory for a new run, or set ppo.init_checkpoint explicitly if you intentionally want this checkpoint.");
    }
    if (log_initialization_ && metadata.config_hash != config_hash(config_)) {
      std::cerr << "warning: auto-resuming checkpoint " << base.string()
                << " with active config changes"
                << " checkpoint_config_hash=" << metadata.config_hash
                << " active_config_hash=" << config_hash(config_)
                << '\n';
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
    load_actor_from_archive(actor_, actor_archive, device_);
    sanitize_actor_parameters();
    actor_normalizer_.load(actor_archive);
    actor_->to(device_);
    actor_normalizer_.to(device_);
    if (std::filesystem::exists(base / "actor_optimizer.pt")) {
      torch::serialize::InputArchive optimizer_archive;
      optimizer_archive.load_from((base / "actor_optimizer.pt").string(), device_);
      try {
        actor_optimizer_.load(optimizer_archive);
      } catch (const std::exception& e) {
        std::cerr << "Warning: could not load actor optimizer state: " << e.what() << "\n";
      }
    }
    resumed_global_step_ = metadata.global_step;
    resumed_update_index_ = metadata.update_index;
  }

  if (log_initialization_) {
    std::cout << "initialized_from_checkpoint=" << base.string() << '\n';
  }
}

TrainerMetrics Trainer::update_actor(RolloutStorage& rollout, int /*update_index*/) {
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
  sanitize_actor_parameters();

  ++update_index_;
  metrics.replay_buffer_size = static_cast<int64_t>(replay_buffer_->size());

  // ---- CRL off-policy update ----
  // Each rollout triggers num_updates_per_rollout gradient steps on the replay buffer.
  // Ball head: symmetric InfoNCE on hindsight-relabeled future ball positions.
  // Car head: same loss on hindsight-relabeled future car positions, weighted by annealing schedule.
  if (!config_.crl.enabled || !replay_buffer_->ready() || !gcrl_trainer_) {
    metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
    sanitize_actor_parameters();
    return metrics;
  }

  actor_->train();

  const int n_updates = config_.crl.num_updates_per_rollout;
  const int B = config_.goal_critic.contrastive_batch_size;
  const float lambda_actor = config_.goal_critic.lambda_goal_actor;
  const float lambda_Zg = config_.goal_critic.lambda_Zg;

  double ball_critic_loss_sum = 0.0;
  double car_critic_loss_sum  = 0.0;
  double car_actor_loss_sum   = 0.0;
  double grad_norm_sum        = 0.0;
  std::int64_t grad_norm_steps = 0;

  const auto lopt = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  const auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

  const int64_t L = std::max(1, config_.model.sequence_length);

  for (int k = 0; k < n_updates; ++k) {
    auto batch = replay_buffer_->sample_crl_batch(B, static_cast<int>(L));

    const torch::Tensor actions_dev   = batch.actions.to(device_);
    const torch::Tensor ball_goals    = batch.future_ball_goals.to(device_);
    const torch::Tensor car_goals     = batch.future_car_goals.to(device_);
    // Car-head actor command: drive the car to the ball's position (3D). The car
    // head's *critic* still learns reachability from hindsight car positions
    // (car_goals); only the actor is aimed at the ball, so the controllable head
    // generates car->ball contact that the (pre-contact degenerate) ball head needs.
    const torch::Tensor ball_pos_goal = ball_goals.narrow(1, 0, 3);
    const torch::Tensor masks         = batch.masks.to(device_).to(torch::kBool);

    // Replay the recurrent context window through the SSM. The buffer stores
    // single transitions, but the Mamba encoder is sequential — feeding it
    // isolated observations (zero state) yields features the policy was never
    // executed on at collection time. Sampling a [B, L, obs] window and running
    // forward_sequence reconstructs collection-consistent features at step t.
    const torch::Tensor obs_seq = actor_normalizer_
        .normalize(batch.obs_seq.to(device_).reshape({B * L, config_.model.observation_dim}))
        .reshape({B, L, config_.model.observation_dim})
        .permute({1, 0, 2}).contiguous();  // [L, B, obs_dim]

    // Goal-conditioned policy: broadcast the (terminal) ball goal across the
    // window. The goal residual is applied after the recurrent pass, so it only
    // conditions the policy logits — features (encoded_seq) stay goal-agnostic,
    // preserving the φ(o,a) vs ψ(g) contrastive structure. ball_goals[i] and
    // car_goals[i] are the ball/car components of the SAME future state t', so
    // conditioning on the ball goal is coherent for both heads' actor losses.
    const torch::Tensor ball_goals_seq =
        ball_goals.unsqueeze(0).expand({L, B, config_.goal_critic.goal_dim}).contiguous();

    const auto seq_out = actor_->forward_sequence(
        obs_seq, ball_goals_seq, /*episode_starts=*/{}, /*compute_value=*/false);
    const torch::Tensor features = seq_out.features[L - 1];        // [B, feat_dim] goal-agnostic
    const torch::Tensor logits   = seq_out.policy_logits[L - 1];   // [B, action_dim] goal-conditioned

    actor_optimizer_.zero_grad();
    if (car_goal_critic_optimizer_) car_goal_critic_optimizer_->zero_grad();

    // Ball head (goal_dim=4): InfoNCE critic + Gumbel-Softmax actor
    torch::Tensor ball_critic_loss, ball_actor_loss, ball_score;
    gcrl_trainer_->compute_gcrl_losses(
        actor_->goal_critic(), features, actions_dev, logits, masks, ball_goals, ball_goals,
        /*compute_critic=*/true, /*compute_actor=*/(lambda_actor > 0.0F),
        ball_critic_loss, ball_actor_loss, ball_score);

    // Car head (car_goal_dim=3): InfoNCE critic + Gumbel-Softmax actor (annealed)
    const float w_car = config_.crl.car_head_initial_weight *
        std::max(0.0F, 1.0F - static_cast<float>(global_step_) / config_.crl.car_head_anneal_steps);

    torch::Tensor total_loss = ball_critic_loss;
    if (lambda_actor > 0.0F && ball_actor_loss.defined())
      total_loss = total_loss + lambda_actor * ball_actor_loss;

    if (car_goal_critic_ && w_car > 0.0F) {
      torch::Tensor car_critic_loss, car_actor_loss, car_score;
      gcrl_trainer_->compute_gcrl_losses(
          car_goal_critic_, features, actions_dev, logits, masks, car_goals, ball_pos_goal,
          /*compute_critic=*/true, /*compute_actor=*/(lambda_actor > 0.0F),
          car_critic_loss, car_actor_loss, car_score);
      total_loss = total_loss + w_car * car_critic_loss;
      if (lambda_actor > 0.0F && car_actor_loss.defined())
        total_loss = total_loss + w_car * lambda_actor * car_actor_loss;
      car_critic_loss_sum += car_critic_loss.item<double>();
      if (car_actor_loss.defined()) car_actor_loss_sum += car_actor_loss.item<double>();
    }

    if (torch::isfinite(total_loss).all().item<bool>()) {
      total_loss.backward();
      const float gn = clip_existing_gradients(*actor_, config_.ppo.max_grad_norm);
      actor_optimizer_.step();
      if (car_goal_critic_optimizer_) car_goal_critic_optimizer_->step();
      ball_critic_loss_sum += ball_critic_loss.item<double>();
      grad_norm_sum += static_cast<double>(gn);
      ++grad_norm_steps;
    } else {
      ++metrics.nonfinite_loss_skips;
    }

    ++global_step_;
    metrics.car_head_weight = static_cast<double>(
        config_.crl.car_head_initial_weight *
        std::max(0.0F, 1.0F - static_cast<float>(global_step_) / config_.crl.car_head_anneal_steps));
  }

  if (n_updates > 0) {
    const double denom = static_cast<double>(n_updates);
    metrics.policy_loss = ball_critic_loss_sum / denom;
    metrics.car_critic_loss = car_critic_loss_sum / denom;
    metrics.car_actor_loss  = car_actor_loss_sum / denom;
    metrics.grad_norm = grad_norm_steps > 0 ? grad_norm_sum / static_cast<double>(grad_norm_steps) : 0.0;
    metrics.grad_norm_valid_steps = grad_norm_steps;
    metrics.optimizer_steps = grad_norm_steps;
  }
  metrics.update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  sanitize_actor_parameters();
  return metrics;
}


void Trainer::sanitize_actor_parameters() {
  PULSAR_TRACE_SCOPE_CAT("trainer", "sanitize_actor_params");
  
  // Fast batched check: check if ALL parameters are finite using exactly 1 host-device synchronization
  bool actor_params_finite = true;
  std::vector<torch::Tensor> flat_params;
  for (auto& p : actor_->parameters()) {
    if (p.defined()) {
      flat_params.push_back(p.reshape({-1}));
    }
  }
  if (!flat_params.empty()) {
    actor_params_finite = torch::isfinite(torch::cat(flat_params)).all().item<bool>();
  }
  
  if (actor_params_finite) {
    return;
  }

  // Fallback: detailed in-place repair and optimizer state clearance
  bool changed = false;
  std::string first_param;
  for (auto& item : actor_->named_parameters(true)) {
    torch::Tensor param = item.value();
    if (param.defined() && !torch::isfinite(param).all().item<bool>()) {
      if (!changed) {
        first_param = item.key();
        changed = true;
      }
      std::cerr << "Warning: repairing non-finite actor parameter " << item.key() << '\n';
      torch::NoGradGuard no_grad;
      const torch::Tensor finite = torch::isfinite(param);
      const bool is_weight =
          item.key().size() >= std::string(".weight").size() &&
          item.key().compare(
              item.key().size() - std::string(".weight").size(),
              std::string(".weight").size(),
              ".weight") == 0;
      const bool default_to_one = item.key().find("norm") != std::string::npos && is_weight;
      const torch::Tensor replacement = default_to_one ? torch::ones_like(param) : torch::zeros_like(param);
      param.copy_(torch::where(finite, param, replacement));
      // Reset optimizer state for this parameter
      actor_optimizer_.state().erase(param.unsafeGetTensorImpl());
    }
  }
  if (changed) {
    std::cerr << "Successfully repaired non-finite actor parameters starting with " << first_param << "; cleared affected optimizer state.\n";
    if (compute_devices_.size() > 1) {
      sync_actor_to_replicas(actor_, compute_actors_);
    }
  }
}

Trainer::ESPopulationFitness Trainer::evaluate_es_population(
    const torch::Tensor& A_stack,
    const torch::Tensor& B_stack,
    int update_index,
    int wave_index) {
  const float sigma_used = config_.es_lora.sigma_ES;
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
  if (eval_modes.empty()) {
    const std::string mode = std::to_string(config_.env.team_size) + "v" +
        std::to_string(config_.env.team_size);
    eval_modes.emplace_back(mode, 1.0F);
  }

  struct EsModePlan {
    std::string mode;
    float raw_weight = 0.0F;
    int agents_per_env = 2;
    std::vector<std::int64_t> member_indices;
  };

  std::vector<EsModePlan> es_mode_plans;
  es_mode_plans.reserve(eval_modes.size());
  double agent_budget_score_sum = 0.0;
  for (const auto& [mode, raw_weight] : eval_modes) {
    const int agents_per_env = 2 * team_size_from_mode(mode);
    es_mode_plans.push_back({
        .mode = mode,
        .raw_weight = raw_weight,
        .agents_per_env = agents_per_env,
        .member_indices = {},
    });
    agent_budget_score_sum +=
        static_cast<double>(std::max(raw_weight, 0.0F)) /
        static_cast<double>(std::max(agents_per_env, 1));
  }
  if (agent_budget_score_sum <= 0.0) {
    agent_budget_score_sum = 1.0;
  }

  const bool allocate_antithetic_pairs = es_cfg.antithetic_sampling && (pop % 2) == 0;
  const int allocation_units = allocate_antithetic_pairs ? pop / 2 : pop;
  const int min_units_per_mode =
      allocation_units >= static_cast<int>(es_mode_plans.size()) ? 1 : 0;
  std::vector<int> units_per_mode(es_mode_plans.size(), 0);
  std::vector<double> target_units(es_mode_plans.size(), 0.0);
  int assigned_units = 0;
  for (std::size_t i = 0; i < es_mode_plans.size(); ++i) {
    const double score =
        static_cast<double>(std::max(es_mode_plans[i].raw_weight, 0.0F)) /
        static_cast<double>(std::max(es_mode_plans[i].agents_per_env, 1));
    target_units[i] = static_cast<double>(allocation_units) * score / agent_budget_score_sum;
    units_per_mode[i] = std::max(min_units_per_mode, static_cast<int>(std::floor(target_units[i])));
    assigned_units += units_per_mode[i];
  }
  while (assigned_units < allocation_units) {
    int best_idx = -1;
    double best_deficit = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < units_per_mode.size(); ++i) {
      const double deficit = target_units[i] - static_cast<double>(units_per_mode[i]);
      if (deficit > best_deficit) {
        best_deficit = deficit;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0) break;
    ++units_per_mode[static_cast<std::size_t>(best_idx)];
    ++assigned_units;
  }
  while (assigned_units > allocation_units) {
    int remove_idx = -1;
    double worst_surplus = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < units_per_mode.size(); ++i) {
      if (units_per_mode[i] <= min_units_per_mode) continue;
      const double surplus = static_cast<double>(units_per_mode[i]) - target_units[i];
      if (surplus > worst_surplus) {
        worst_surplus = surplus;
        remove_idx = static_cast<int>(i);
      }
    }
    if (remove_idx < 0) break;
    --units_per_mode[static_cast<std::size_t>(remove_idx)];
    --assigned_units;
  }

  int unit_cursor = 0;
  for (std::size_t mode_idx = 0; mode_idx < es_mode_plans.size(); ++mode_idx) {
    EsModePlan& plan = es_mode_plans[mode_idx];
    const int units = units_per_mode[mode_idx];
    plan.member_indices.reserve(static_cast<std::size_t>(allocate_antithetic_pairs ? units * 2 : units));
    for (int unit = 0; unit < units && unit_cursor < allocation_units; ++unit, ++unit_cursor) {
      if (allocate_antithetic_pairs) {
        plan.member_indices.push_back(unit_cursor);
        plan.member_indices.push_back(unit_cursor + allocation_units);
      } else {
        plan.member_indices.push_back(unit_cursor);
      }
    }
  }

  for (const EsModePlan& plan : es_mode_plans) {
    const std::string& mode = plan.mode;
    const int mode_pop = static_cast<int>(plan.member_indices.size());
    if (mode_pop <= 0) {
      continue;
    }
    ExperimentConfig mode_config = config_;
    mode_config.env.team_size = team_size_from_mode(mode);
    mode_config.env.spawn_opponents = true;

    const int team_size = mode_config.env.team_size;
    const int agents_per_env = team_size * 2;
    const int member_agents = eval_envs * agents_per_env;
    const torch::Tensor member_index_tensor = torch::from_blob(
        const_cast<std::int64_t*>(plan.member_indices.data()),
        {static_cast<long>(mode_pop)},
        torch::TensorOptions().dtype(torch::kInt64))
        .clone()
        .to(A_stack.device());
    const torch::Tensor A_mode = A_stack.index_select(0, member_index_tensor);
    const torch::Tensor B_mode = B_stack.index_select(0, member_index_tensor);

    result.agent_steps += static_cast<std::int64_t>(mode_pop) *
        static_cast<std::int64_t>(eval_envs) *
        static_cast<std::int64_t>(agents_per_env) *
        static_cast<std::int64_t>(es_cfg.eval_episodes_per_member) *
        static_cast<std::int64_t>(es_cfg.eval_rollout_length);

    struct EsShardResult {
      int member_start = 0;
      std::vector<float> reward_sum;
      std::vector<float> reward_count;
      std::vector<float> kl_sum;
      std::vector<float> kl_count;
      std::vector<int> episode_counts;
      std::vector<int> win_counts;
    };

    const int device_count = std::max<int>(
        1,
        static_cast<int>(compute_devices_.empty() ? 1 : compute_devices_.size()));
    const int requested_shards = std::max(1, std::min(
        mode_pop,
        es_cfg.eval_shards > 0 ? es_cfg.eval_shards : device_count));
    std::vector<EsShardSpec> shard_specs;
    shard_specs.reserve(static_cast<std::size_t>(requested_shards));
    for (int shard = 0; shard < requested_shards; ++shard) {
      const int base_members = mode_pop / requested_shards;
      const int extra_members = shard < (mode_pop % requested_shards) ? 1 : 0;
      const int member_count = base_members + extra_members;
      if (member_count <= 0) continue;
      const int member_start = shard * base_members + std::min(shard, mode_pop % requested_shards);
      const int total_workers = es_cfg.eval_workers;
      const int base_workers = total_workers > 0 ? total_workers / requested_shards : 0;
      const int extra_workers = total_workers > 0 && shard < (total_workers % requested_shards) ? 1 : 0;
      shard_specs.push_back({
          .member_start = member_start,
          .member_count = member_count,
          .worker_count = total_workers > 0 ? std::max(1, base_workers + extra_workers) : 0,
          .device = compute_devices_.empty()
              ? device_
              : compute_devices_[static_cast<std::size_t>(shard) % compute_devices_.size()],
      });
    }

    // Rebuild cached ES eval collectors when shard configuration changes (first call,
    // team_size change, or shard count change). Avoids recreating hundreds of
    // RocketSim arenas every 25 updates — the seed is fully controlled by reset_es_episode.
    {
      bool need_rebuild = (es_eval_cached_team_size_ != team_size) ||
          (es_eval_shard_specs_.size() != shard_specs.size());
      if (!need_rebuild) {
        for (std::size_t si = 0; si < shard_specs.size(); ++si) {
          if (es_eval_shard_specs_[si].member_count != shard_specs[si].member_count ||
              es_eval_shard_specs_[si].worker_count != shard_specs[si].worker_count) {
            need_rebuild = true;
            break;
          }
        }
      }
      if (need_rebuild) {
        es_eval_collectors_.clear();
        es_eval_shard_specs_ = shard_specs;
        es_eval_cached_team_size_ = team_size;
        for (std::size_t si = 0; si < shard_specs.size(); ++si) {
          ExperimentConfig shard_config = mode_config;
          shard_config.ppo.collection_workers = shard_specs[si].worker_count;
          const int shard_envs = shard_specs[si].member_count * eval_envs;
          es_eval_collectors_.push_back(make_es_eval_collector(
              shard_config, shard_envs, eval_envs, 0, 0, use_pinned_host_buffers_));
        }
      }
    }

    std::vector<std::future<EsShardResult>> shard_futures;
    shard_futures.reserve(shard_specs.size());
    const int physics_prefix_ticks =
        config_.env.half_tick_skip > 0 && config_.env.half_tick_skip < config_.env.tick_skip
            ? config_.env.half_tick_skip
            : 0;

    for (std::size_t shard_idx = 0; shard_idx < shard_specs.size(); ++shard_idx) {
      const EsShardSpec spec = shard_specs[shard_idx];
      BatchedRocketSimCollector* eval_collector = es_eval_collectors_[shard_idx].get();
      shard_futures.push_back(task_pool_->submit(
          [&, spec, shard_idx, eval_collector]() -> EsShardResult {
        EsShardResult shard_result;
        torch::NoGradGuard no_grad;
        shard_result.member_start = spec.member_start;
        shard_result.reward_sum.assign(static_cast<std::size_t>(spec.member_count), 0.0F);
        shard_result.reward_count.assign(static_cast<std::size_t>(spec.member_count), 0.0F);
        shard_result.kl_sum.assign(static_cast<std::size_t>(spec.member_count), 0.0F);
        shard_result.kl_count.assign(static_cast<std::size_t>(spec.member_count), 0.0F);
        shard_result.episode_counts.assign(static_cast<std::size_t>(spec.member_count), 0);
        shard_result.win_counts.assign(static_cast<std::size_t>(spec.member_count), 0);

        const int shard_envs = spec.member_count * eval_envs;

        Actor shard_actor = clone_actor(actor_, spec.device);
        shard_actor->eval();
        ObservationNormalizer shard_normalizer = actor_normalizer_.clone();
        shard_normalizer.to(spec.device);
        torch::Tensor A_shard = A_mode.narrow(0, spec.member_start, spec.member_count).to(spec.device);
        torch::Tensor B_shard = B_mode.narrow(0, spec.member_start, spec.member_count).to(spec.device);

#ifdef PULSAR_HAS_CUDA
        std::optional<c10::cuda::CUDAGuard> shard_device_guard;
        if (spec.device.is_cuda()) {
          shard_device_guard.emplace(spec.device);
        }
        std::optional<c10::cuda::CUDAStream> shard_stream;
        if (spec.device.is_cuda()) {
          if (shard_idx < shard_collection_streams_.size()) {
            shard_stream = shard_collection_streams_[shard_idx];
          } else {
            shard_stream = at::cuda::getStreamFromPool(false, spec.device.index());
          }
          c10::cuda::setCurrentCUDAStream(*shard_stream);
        }
#endif

        std::vector<std::uint8_t> controlled_host(static_cast<std::size_t>(shard_envs * agents_per_env), 0);
        for (int env_idx = 0; env_idx < shard_envs; ++env_idx) {
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
            .to(spec.device)
            .to(torch::kBool);
        const torch::Tensor controlled_float =
            controlled_mask.to(torch::kFloat32).view({spec.member_count, member_agents});
        const torch::Tensor controlled_count = controlled_float.sum(1);
        torch::Tensor reward_sum_tensor = torch::zeros({spec.member_count}, controlled_float.options());
        torch::Tensor reward_count_tensor = torch::zeros({spec.member_count}, controlled_float.options());
        torch::Tensor kl_sum_tensor = torch::zeros({spec.member_count}, controlled_float.options());
        torch::Tensor kl_count_tensor = torch::zeros({spec.member_count}, controlled_float.options());

        for (int ep = 0; ep < es_cfg.eval_episodes_per_member; ++ep) {
          eval_collector->reset_es_episode(
              update_index,
              wave_index * es_cfg.eval_episodes_per_member + ep,
              eval_envs);
          torch::Tensor recurrent_state = shard_actor->initial_recurrent_state(
              static_cast<int64_t>(shard_envs * agents_per_env),
              spec.device);

          for (int step = 0; step < es_cfg.eval_rollout_length; ++step) {
            torch::Tensor raw_obs_host = eval_collector->host_observations();
            torch::Tensor episode_starts_host = eval_collector->host_episode_starts();
            torch::Tensor action_masks_host = eval_collector->host_action_masks();
            std::future<void> physics_prefix_future;
            CollectorTimings step_timings{};
            if (physics_prefix_ticks > 0) {
              physics_prefix_future = std::async(std::launch::async, [&]() {
                eval_collector->step_physics_only(physics_prefix_ticks, &step_timings);
              });
            }

            torch::Tensor raw_obs = raw_obs_host.to(spec.device, use_pinned_host_buffers_);
            torch::Tensor episode_starts = episode_starts_host.to(spec.device, use_pinned_host_buffers_);
            torch::Tensor action_masks = action_masks_host.to(spec.device, use_pinned_host_buffers_).to(torch::kBool);
            torch::Tensor normalized_obs = shard_normalizer.normalize(raw_obs);

            const torch::Tensor goal_values = eval_collector->host_task_goal_positions()
                .to(spec.device, use_pinned_host_buffers_);
            ActorStepOutput output = shard_actor->forward_step_stateful(
                normalized_obs,
                recurrent_state,
                episode_starts,
                &recurrent_state,
                goal_values);
            torch::Tensor perturbed_logits = shard_actor->policy_eggroll_logits(
                output.features, A_shard, B_shard, sigma_used, goal_values);

            torch::Tensor base_actions = sample_masked_actions(
                output.policy_logits, action_masks, true, nullptr, config_.ppo.policy_temperature);
            torch::Tensor perturbed_actions = sample_masked_actions(
                perturbed_logits, action_masks, true, nullptr, config_.ppo.policy_temperature);
            torch::Tensor actions = torch::where(controlled_mask, perturbed_actions, base_actions);

            if ((step % es_cfg.kl_eval_stride) == 0) {
              const torch::Tensor base_masked = apply_action_mask_to_logits(output.policy_logits, action_masks);
              const torch::Tensor perturbed_masked = apply_action_mask_to_logits(perturbed_logits, action_masks);
              const float policy_temperature = std::max(config_.ppo.policy_temperature, 1.0e-6F);
              const torch::Tensor base_probs = torch::softmax(base_masked / policy_temperature, -1);
              const torch::Tensor perturbed_probs = torch::softmax(perturbed_masked / policy_temperature, -1);
              const torch::Tensor kl_values = (
                  perturbed_probs * (torch::log(perturbed_probs + 1.0e-8) - torch::log(base_probs + 1.0e-8)))
                  .sum(-1)
                  .view({spec.member_count, member_agents});
              kl_sum_tensor += (kl_values * controlled_float).sum(1);
              kl_count_tensor += controlled_count;
            }

            const torch::Tensor action_indices_cpu = actions.contiguous().to(torch::kCPU);
            if (physics_prefix_future.valid()) {
              physics_prefix_future.get();
            }
            eval_collector->step_after_physics_prefix(
                std::span<const std::int64_t>(
                    action_indices_cpu.data_ptr<std::int64_t>(),
                    static_cast<std::size_t>(action_indices_cpu.numel())),
                physics_prefix_ticks,
                &step_timings);

            const torch::Tensor rewards = eval_collector->host_rewards()
                .to(spec.device, use_pinned_host_buffers_)
                .view({spec.member_count, member_agents});
            reward_sum_tensor += (rewards * controlled_float).sum(1);
            reward_count_tensor += controlled_count;

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
              shard_result.episode_counts[static_cast<std::size_t>(member)] += 1;
              if (labels_ptr[i] == 0) {
                shard_result.win_counts[static_cast<std::size_t>(member)] += 1;
              }
            }
          }
        }
        torch::Tensor reward_sum_cpu = reward_sum_tensor.to(torch::kCPU);
        torch::Tensor reward_count_cpu = reward_count_tensor.to(torch::kCPU);
        torch::Tensor kl_sum_cpu = kl_sum_tensor.to(torch::kCPU);
        torch::Tensor kl_count_cpu = kl_count_tensor.to(torch::kCPU);
        const auto* reward_sum_ptr = reward_sum_cpu.data_ptr<float>();
        const auto* reward_count_ptr = reward_count_cpu.data_ptr<float>();
        const auto* kl_sum_ptr = kl_sum_cpu.data_ptr<float>();
        const auto* kl_count_ptr = kl_count_cpu.data_ptr<float>();
        for (int member = 0; member < spec.member_count; ++member) {
          shard_result.reward_sum[static_cast<std::size_t>(member)] = reward_sum_ptr[member];
          shard_result.reward_count[static_cast<std::size_t>(member)] = reward_count_ptr[member];
          shard_result.kl_sum[static_cast<std::size_t>(member)] = kl_sum_ptr[member];
          shard_result.kl_count[static_cast<std::size_t>(member)] = kl_count_ptr[member];
        }
        return shard_result;
      }));
    }

    std::vector<float> reward_sum(static_cast<std::size_t>(mode_pop), 0.0F);
    std::vector<float> reward_count(static_cast<std::size_t>(mode_pop), 0.0F);
    std::vector<float> kl_sum(static_cast<std::size_t>(mode_pop), 0.0F);
    std::vector<float> kl_count(static_cast<std::size_t>(mode_pop), 0.0F);
    std::vector<int> episode_counts(static_cast<std::size_t>(mode_pop), 0);
    std::vector<int> win_counts(static_cast<std::size_t>(mode_pop), 0);
    for (auto& future : shard_futures) {
      EsShardResult shard_result = future.get();
      for (std::size_t local = 0; local < shard_result.reward_sum.size(); ++local) {
        const std::size_t global = static_cast<std::size_t>(shard_result.member_start) + local;
        reward_sum[global] += shard_result.reward_sum[local];
        reward_count[global] += shard_result.reward_count[local];
        kl_sum[global] += shard_result.kl_sum[local];
        kl_count[global] += shard_result.kl_count[local];
        episode_counts[global] += shard_result.episode_counts[local];
        win_counts[global] += shard_result.win_counts[local];
      }
    }

    for (int i = 0; i < mode_pop; ++i) {
      const int denom = std::max(episode_counts[static_cast<std::size_t>(i)], 1);
      const float mode_winrate =
          static_cast<float>(win_counts[static_cast<std::size_t>(i)]) / static_cast<float>(denom);
      const float reward_mean = reward_sum[static_cast<std::size_t>(i)] /
          std::max(reward_count[static_cast<std::size_t>(i)], 1.0F);
      const float kl_mean = kl_sum[static_cast<std::size_t>(i)] /
          std::max(kl_count[static_cast<std::size_t>(i)], 1.0F);
      const std::size_t global = static_cast<std::size_t>(plan.member_indices[static_cast<std::size_t>(i)]);
      result.reward[global] = reward_mean;
      result.winrate[global] = mode_winrate;
      result.kl[global] = kl_mean;
    }
  }

  for (int i = 0; i < pop; ++i) {
    result.fitness[static_cast<std::size_t>(i)] =
        result.reward[static_cast<std::size_t>(i)]
        - es_cfg.beta_KL * result.kl[static_cast<std::size_t>(i)];
  }
  return result;
}

void Trainer::run_es_lora_update(int update_index, TrainerMetrics& metrics) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "es_update");
  const auto es_start = std::chrono::steady_clock::now();
  const auto& es_cfg = config_.es_lora;
  const float es_sigma_used = es_cfg.sigma_ES;
  const int pop = es_cfg.population_size;
  const int waves = es_cfg.virtual_population_waves;
  const int rank = es_cfg.rank;
  const int in_features = actor_->policy_lora()->in_features();
  const int out_features = actor_->policy_lora()->out_features();
  const int64_t effective_population = static_cast<int64_t>(pop) * static_cast<int64_t>(waves);
  const int device_count = std::max<int>(
      1,
      static_cast<int>(compute_devices_.empty() ? 1 : compute_devices_.size()));
  const int actual_eval_shards = std::max(1, std::min(
      pop,
      es_cfg.eval_shards > 0 ? es_cfg.eval_shards : device_count));

  const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  const auto es_eval_start = std::chrono::steady_clock::now();
  std::vector<torch::Tensor> A_waves;
  std::vector<torch::Tensor> B_waves;
  A_waves.reserve(static_cast<std::size_t>(waves));
  B_waves.reserve(static_cast<std::size_t>(waves));
  ESPopulationFitness population;
  population.fitness.reserve(static_cast<std::size_t>(effective_population));
  population.reward.reserve(static_cast<std::size_t>(effective_population));
  population.winrate.reserve(static_cast<std::size_t>(effective_population));
  population.kl.reserve(static_cast<std::size_t>(effective_population));

  for (int wave = 0; wave < waves; ++wave) {
    torch::Tensor A_stack;
    torch::Tensor B_stack;
    if (es_cfg.antithetic_sampling) {
      const int half_pop = pop / 2;
      torch::Tensor A_half = torch::randn({half_pop, rank, in_features}, tensor_options);
      torch::Tensor B_half = torch::randn({half_pop, out_features, rank}, tensor_options);
      A_stack = torch::cat({A_half, -A_half}, 0);
      B_stack = torch::cat({B_half, B_half}, 0);
    } else {
      A_stack = torch::randn({pop, rank, in_features}, tensor_options);
      B_stack = torch::randn({pop, out_features, rank}, tensor_options);
    }
    ESPopulationFitness wave_population = evaluate_es_population(A_stack, B_stack, update_index, wave);
    population.fitness.insert(population.fitness.end(), wave_population.fitness.begin(), wave_population.fitness.end());
    population.reward.insert(population.reward.end(), wave_population.reward.begin(), wave_population.reward.end());
    population.winrate.insert(population.winrate.end(), wave_population.winrate.begin(), wave_population.winrate.end());
    population.kl.insert(population.kl.end(), wave_population.kl.begin(), wave_population.kl.end());
    population.agent_steps += wave_population.agent_steps;
    A_waves.push_back(A_stack);
    B_waves.push_back(B_stack);
  }
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

  std::vector<float> normalized_f(total_members, 0.0F);
  if (es_cfg.rank_transform && total_members > 1 && sigma > 0.0F) {
    std::vector<std::size_t> order(total_members);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      return fitnesses[lhs] < fitnesses[rhs];
    });
    std::size_t begin = 0;
    while (begin < order.size()) {
      std::size_t end = begin + 1;
      while (end < order.size() && fitnesses[order[end]] == fitnesses[order[begin]]) {
        ++end;
      }
      const float average_rank = 0.5F * static_cast<float>(begin + end - 1);
      const float rank_value =
          2.0F * average_rank / static_cast<float>(total_members - 1) - 1.0F;
      for (std::size_t index = begin; index < end; ++index) {
        normalized_f[order[index]] = rank_value;
      }
      begin = end;
    }
    float rank_sigma = 0.0F;
    for (float value : normalized_f) {
      rank_sigma += value * value;
    }
    rank_sigma = std::sqrt(rank_sigma / static_cast<float>(total_members));
    if (rank_sigma > 0.0F) {
      for (float& value : normalized_f) {
        value /= rank_sigma;
      }
    }
  } else {
    for (uint64_t i = 0; i < total_members; ++i) {
      normalized_f[static_cast<std::size_t>(i)] = (fitnesses[static_cast<std::size_t>(i)] - mu) / (sigma + 1.0e-8F);
    }
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
    metrics.es_effective_population = static_cast<double>(effective_population);
    metrics.es_virtual_population_waves = static_cast<double>(waves);
    metrics.es_eval_shards = static_cast<double>(actual_eval_shards);
    metrics.es_agent_steps_per_second = metrics.es_eval_seconds > 0.0
        ? static_cast<double>(population.agent_steps) / metrics.es_eval_seconds
        : 0.0;
    metrics.es_policy_update_norm = 0.0;

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
  torch::Tensor delta_weight = torch::zeros({out_features, in_features}, tensor_options);
  const float perturb_scale = 1.0F / std::sqrt(static_cast<float>(rank));
  for (int wave = 0; wave < waves; ++wave) {
    const torch::Tensor weights = fitness_weights.narrow(0, static_cast<int64_t>(wave) * pop, pop);
    const torch::Tensor weighted_updates =
        torch::bmm(B_waves[static_cast<std::size_t>(wave)], A_waves[static_cast<std::size_t>(wave)]) *
        weights.view({pop, 1, 1});
    delta_weight += weighted_updates.sum(0);
  }
  delta_weight *= (es_cfg.eta_ES / es_sigma_used) * perturb_scale / static_cast<float>(total_members);

  double update_norm = static_cast<double>(delta_weight.norm().item<float>());
  double update_scale = 1.0;
  if (es_cfg.max_kl_mean > 0.0F && kl_mean > static_cast<double>(es_cfg.max_kl_mean)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_kl_mean) / std::max(kl_mean, 1.0e-12));
  }
  if (es_cfg.update_norm_clip && es_cfg.max_update_norm > 0.0F && update_norm > static_cast<double>(es_cfg.max_update_norm)) {
    update_scale = std::min(update_scale, static_cast<double>(es_cfg.max_update_norm) / std::max(update_norm, 1.0e-12));
  }
  if (update_scale < 1.0) {
    delta_weight.mul_(update_scale);
    update_norm *= update_scale;
  }

  const bool delta_weight_finite = torch::isfinite(delta_weight).all().item<bool>();
  if (!delta_weight_finite) {
    std::cerr << "Warning: skipping ES-LoRA update because delta_weight contains non-finite values!\n";
    update_norm = 0.0;
  } else {
    torch::NoGradGuard no_grad;
    actor_->apply_policy_eggroll_update(delta_weight);
    torch::Tensor policy_weight = actor_->policy_lora()->base->weight;
    actor_optimizer_.state().erase(policy_weight.unsafeGetTensorImpl());
    sanitize_actor_parameters();
  }

  metrics.es_fitness_mean = mu;
  metrics.es_fitness_std = sigma;
  metrics.es_fitness_best = static_cast<double>(best_fitness);
  metrics.es_update_norm = update_norm;
  metrics.es_reward_mean = reward_mean;
  metrics.es_winrate_mean = winrate_mean;
  metrics.es_kl_mean = kl_mean;
  metrics.es_effective_population = static_cast<double>(effective_population);
  metrics.es_virtual_population_waves = static_cast<double>(waves);
  metrics.es_eval_shards = static_cast<double>(actual_eval_shards);
  metrics.es_agent_steps_per_second = metrics.es_eval_seconds > 0.0
      ? static_cast<double>(population.agent_steps) / metrics.es_eval_seconds
      : 0.0;
  metrics.es_policy_update_norm = update_norm;

  auto lora_params = actor_->es_lora_parameters();
  metrics.es_lora_a_norm = static_cast<double>(lora_params[0].norm().item<float>());
  metrics.es_lora_b_norm = static_cast<double>(lora_params[1].norm().item<float>());

  metrics.es_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - es_start).count();
}

void Trainer::collect_rollout(
    RolloutStorage& dest,
    TrainerMetrics& metrics,
    std::int64_t* collected_agent_steps,
    Actor rollout_actor,
    ObservationNormalizer& normalizer) {
  PULSAR_TRACE_SCOPE_CAT("trainer", "collect_rollout");
  if (!rollout_actor) {
    throw std::invalid_argument("Trainer::collect_rollout requires a policy snapshot.");
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
  std::vector<Actor> rollout_actors;
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
    } else {
      // Clone per shard even when shard_device == device_: concurrent forward calls on
      // the same nn::Module from multiple threads are not safe in PyTorch.
      rollout_actors.push_back(clone_actor(rollout_actor, shard_device));
      rollout_actors.back()->eval();
    }
    // NOTE: Cloned actors and normalizers are required for thread-safe parallel execution.
    // Each sharded rollout worker runs asynchronously on its own thread and must operate on
    // disjoint model and normalizer instances to avoid data races.
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
      std::string mode;
    };

    // Launch one async task per shard that processes all rollout steps.
    // This reduces std::async launches from (shards × steps) to just shards.
    struct ShardResult {
      std::size_t shard;
      std::vector<PendingShardStep> steps;
      torch::Tensor next_recurrent_state;
      torch::Tensor sampled_value_sum;
      std::int64_t sampled_value_count = 0;
    };
    std::vector<std::future<ShardResult>> shard_futures;
    shard_futures.reserve(collectors_.size());

    for (std::size_t shard = 0; shard < collectors_.size(); ++shard) {
      BatchedRocketSimCollector* collector_ptr = collectors_[shard].get();
      Actor shard_actor = rollout_actors[shard];
      ObservationNormalizer* shard_normalizer = &shard_normalizers[shard];
      ObservationNormalizer* shard_normalizer_update = &shard_normalizer_updates[shard];
      torch::Tensor recurrent_state = rollout_recurrent_states[shard];
      const torch::Device shard_device = shard_devices_.empty() ? device_ : shard_devices_[shard];
      const std::size_t shard_idx = shard;

      shard_futures.push_back(task_pool_->submit(
          [&, collector_ptr, shard_actor, shard_normalizer, shard_normalizer_update, recurrent_state, shard_device, shard_idx]() mutable -> ShardResult {
        ShardResult result;
        result.shard = shard_idx;
        result.steps.reserve(config_.ppo.rollout_length);
        result.sampled_value_sum =
            torch::zeros({}, torch::TensorOptions().device(shard_device).dtype(torch::kFloat64));
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
            // Condition the rollout policy on the ball position (not the net):
            // the update conditions on the hindsight future ball goal, so the
            // deployed policy must be conditioned on a ball position too, and
            // commanding "reach the ball" is what drives contact during bootstrap.
            const torch::Tensor goal_values = collector_ptr->host_goal_positions()
                .to(shard_device, use_pinned_host_buffers_);
            output = shard_actor->forward_step_stateful(
                normalized_obs, recurrent_state, episode_starts, &recurrent_state, goal_values);
            actions = sample_masked_actions(
                output.policy_logits, action_masks, false, &action_log_probs, config_.ppo.policy_temperature);
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
          shard_step.mode = collector.mode();

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
          torch::Tensor task_goal_host = collector.host_task_goal_positions();

          if (replay_buffer_) {
            replay_buffer_->push_step(
                raw_obs_host,
                action_indices_cpu,
                goal_pos_host,
                collector.host_car_positions(),
                dones_host,
                action_masks_host,
                static_cast<int>(shard_step.agent_offset));
          }

          torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
          float gd_min = goal_norms.min().item<float>();
          float gd_mean = goal_norms.mean().item<float>();
          shard_step.total_goal_distance = static_cast<double>(gd_mean) * static_cast<double>(goal_pos_host.size(0));
          shard_step.min_goal_distance = static_cast<double>(gd_min);

          // Write tensor data directly to rollout storage.
          torch::Tensor sampled_value = output.value_win_logits.squeeze(-1);
          result.sampled_value_sum = result.sampled_value_sum + sampled_value.to(torch::kFloat64).sum();
          result.sampled_value_count += sampled_value.numel();
          std::unordered_map<std::string, torch::Tensor> all_values;
          all_values["extrinsic"] = sampled_value;
          std::unordered_map<std::string, torch::Tensor> all_rewards;
          all_rewards["extrinsic"] = extrinsic_rewards_host;
          all_rewards["gameplay"] = gameplay_r_host;
          all_rewards["mechanic"] = mechanic_r_host;

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
              task_goal_host,
              terminal_labels,
              collector.host_terminal_observations(),
              output.encoded);
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
    for (std::size_t fi = 0; fi < shard_futures.size(); ++fi) {
      ShardResult shard_result = shard_futures[fi].get();
      metrics.policy_forward_seconds += [&] { double s = 0; for (auto& st : shard_result.steps) s += st.policy_forward_seconds; return s; }();
      metrics.action_decode_seconds += [&] { double s = 0; for (auto& st : shard_result.steps) s += st.action_decode_seconds; return s; }();
      if (shard_result.next_recurrent_state.defined()) {
        rollout_recurrent_states[shard_result.shard] = shard_result.next_recurrent_state;
      }
      accumulated_sampled_value += shard_result.sampled_value_sum.to(torch::kCPU).item<double>();
      accumulated_value_count += shard_result.sampled_value_count;
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
        Actor& shard_actor = rollout_actors[shard];
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
        const torch::Tensor final_goal_values = collector.host_goal_positions()
            .to(shard_device, use_pinned_host_buffers_);
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
    torch::Tensor sampled_value_sum =
        torch::zeros({}, torch::TensorOptions().device(device_).dtype(torch::kFloat64));
    std::int64_t sampled_value_count = 0;
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
      const torch::Tensor goal_values = collector_->host_goal_positions()
          .to(device_, use_pinned_host_buffers_);
      output = rollout_actor->forward_step_stateful(
          normalized_obs,
          rollout_recurrent_states[0],
          episode_starts,
          &rollout_recurrent_states[0],
          goal_values);
      actions = sample_masked_actions(
          output.policy_logits, action_masks, false, &action_log_probs, config_.ppo.policy_temperature);
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
    torch::Tensor task_goal_host = collector_->host_task_goal_positions();
    torch::Tensor goal_norms = goal_pos_host.norm(2, 1);
    float gd_min = goal_norms.min().item<float>();
    float gd_mean = goal_norms.mean().item<float>();
    total_goal_distance += static_cast<double>(gd_mean) * static_cast<double>(goal_pos_host.size(0)) / static_cast<double>(total_agents_);
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
    sampled_value_sum = sampled_value_sum + sampled_value.to(torch::kFloat64).sum();
    sampled_value_count += sampled_value.numel();

    // ---- Off-policy buffer push ----
    if (replay_buffer_) {
      replay_buffer_->push_step(
          raw_obs_host,
          action_indices_cpu,
          goal_pos_host,                    // achieved ball pos after step (4D)
          collector_->host_car_positions(), // achieved car pos after step (3D)
          dones_host,
          action_masks_host);
    }

    // ---- Combined reward for RolloutStorage (no shaping — CRL is reward-free) ----
    std::unordered_map<std::string, torch::Tensor> all_rewards;
    all_rewards["extrinsic"] = extrinsic_rewards_host;
    all_rewards["gameplay"] = gameplay_r_host;
    all_rewards["mechanic"] = mechanic_r_host;

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
        task_goal_host,
        terminal_labels,
        terminal_obs_host,
        output.encoded);
    dest.set_mode_ids_slice(step, 0, collector_->mode_id());

    local_collected_steps += learner_step_count;
    }
  }
  accumulated_sampled_value += sampled_value_sum.to(torch::kCPU).item<double>();
  accumulated_value_count += sampled_value_count;
  {
    PULSAR_TRACE_SCOPE_CAT("trainer", "bootstrap_forward");
    torch::NoGradGuard no_grad;
    torch::Tensor final_raw_obs = collector_->host_observations().to(device_, use_pinned_host_buffers_);
    torch::Tensor final_normalized = normalizer.normalize(final_raw_obs);
    torch::Tensor final_starts = collector_->host_episode_starts().to(device_, use_pinned_host_buffers_);
    const torch::Tensor final_goal_values = collector_->host_goal_positions()
        .to(device_, use_pinned_host_buffers_);
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

  metrics.obs_build_seconds = collector_timings.obs_build_seconds;
  metrics.mask_build_seconds = collector_timings.mask_build_seconds;
  metrics.env_step_seconds = collector_timings.env_step_seconds;
  metrics.done_reset_seconds = collector_timings.done_reset_seconds;
  metrics.collection_agent_steps_per_second =
      local_collected_steps > 0 ? static_cast<double>(local_collected_steps) / collection_seconds : 0.0;

#ifdef PULSAR_HAS_CUDA
  // Flush all per-shard collection streams before returning. Collection enqueues
  // GPU work (policy forward, H2D/D2H copies) on these streams asynchronously, and
  // reads snapshot tensors / input buffers allocated on other streams without
  // record_stream(). Under overlap_collection_update the caller frees and reuses
  // those tensors (snapshot reclone, collection_actors_ reassignment) as soon as
  // this returns; if a shard stream still has pending reads, the caching allocator
  // can hand that memory to the concurrent update, corrupting it (manifesting as
  // intermittent SIGSEGVs in downstream physics). Synchronizing here makes
  // "collect_rollout returned" also mean "its GPU work finished", which the
  // CPU-level future.get() alone does not guarantee. Cost is negligible: collection
  // has already completed on the CPU side by this point.
  for (const auto& stream : shard_collection_streams_) {
    stream.synchronize();
  }
  c10::cuda::setCurrentCUDAStream(prev_collect_stream);
#endif
  *collected_agent_steps = local_collected_steps;
}

CheckpointMetadata Trainer::make_checkpoint_metadata(std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  nlohmann::json extra = nlohmann::json::object();
  extra["trainer_state_version"] = kTrainerStateVersion;
  if (!wandb_run_id.empty()) {
    extra["wandb_run_id"] = wandb_run_id;
  }
  return CheckpointMetadata{
      .schema_version = config_.schema_version,
      .obs_schema_version = config_.obs_schema_version,
      .config_hash = config_hash(config_),
      .action_table_hash = action_table_hash(config_.action_table),
      .architecture_name = "mamba2_goal",
      .device = config_.ppo.device,
      .global_step = global_step,
      .update_index = update_index,
      .critic_heads = {"extrinsic"},
      .extra = std::move(extra),
  };
}

void Trainer::save_checkpoint(const std::filesystem::path& directory, std::int64_t global_step, int update_index, const std::string& wandb_run_id) const {
  save_checkpoint_with_metadata(directory, make_checkpoint_metadata(global_step, update_index, wandb_run_id));
}

void Trainer::save_checkpoint_with_metadata(const std::filesystem::path& directory, const CheckpointMetadata& metadata) const {
  PULSAR_TRACE_SCOPE_CAT("trainer", "checkpoint_save");
  synchronize_cuda_if_needed(device_, "checkpoint save start");
  const std::filesystem::path staging = make_checkpoint_staging_directory(directory);
  remove_checkpoint_directory(staging);
  try {
    std::filesystem::create_directories(staging);
    save_experiment_config(config_, (staging / "config.json").string());
    save_checkpoint_metadata(metadata, (staging / "metadata.json").string());
    save_training_state(staging / "state.pt");
    commit_checkpoint_directory(staging, directory);
    synchronize_cuda_if_needed(device_, "checkpoint save end");
  } catch (...) {
    remove_checkpoint_directory(staging);
    throw;
  }
}

TrainerBenchmarkMetrics Trainer::benchmark(int updates) {
  const int bounded_updates = std::max(1, updates);
  TrainerBenchmarkMetrics result{};
  result.updates = bounded_updates;
  const bool previous_progress = benchmark_progress_;
  benchmark_progress_ = true;


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

    TrainerMetrics update_metrics = update_actor(rollout_, index + 1);
    result.agent_steps += collected_steps;
    result.update_seconds += update_metrics.update_seconds;
    result.forward_backward_seconds += update_metrics.forward_backward_seconds;
    result.optimizer_step_seconds += update_metrics.optimizer_step_seconds;
    result.policy_loss += update_metrics.policy_loss;
    result.grad_norm += update_metrics.grad_norm;
    std::cout << "bench_update_done update=" << (index + 1)
              << " update_seconds=" << update_metrics.update_seconds
              << " forward_backward_seconds=" << update_metrics.forward_backward_seconds
              << " optimizer_step_seconds=" << update_metrics.optimizer_step_seconds
              << " ppo_update_agent_steps_per_second="
              << (collected_steps > 0 ? static_cast<double>(collected_steps) / std::max(update_metrics.update_seconds, 1.0e-9) : 0.0)
              << '\n' << std::flush;

    synchronize_cuda_if_needed(device_, "benchmark snapshot clone");
    actor_snapshot_ = clone_actor(actor_, device_);
    actor_snapshot_->eval();
    // Sync persistent collection actors with new snapshot
    for (size_t i = 0; i < collection_actors_.size(); ++i) {
      if (collection_actors_[i] && shard_devices_[i] == device_) {
        collection_actors_[i] = actor_snapshot_;
      } else if (collection_actors_[i]) {
        collection_actors_[i] = clone_actor(actor_snapshot_, shard_devices_[i]);
        collection_actors_[i]->eval();
      }
    }
    const int update_index = index + 1;
    const int es_base_bench = config_.es_lora.es_interval;
    steps_since_last_es_ += 1;
    if (es_base_bench > 0 && steps_since_last_es_ >= es_base_bench) {
      TrainerMetrics es_metrics{};
      std::cout << "bench_es_update_start update=" << update_index << '/' << bounded_updates << '\n' << std::flush;
      run_es_lora_update(update_index, es_metrics);
      steps_since_last_es_ = 0;
      // Sync ES-LoRA weight changes to replica actors.
      if (compute_actors_.size() > 0) {
        sync_actor_to_replicas(actor_, compute_actors_);
      }
      synchronize_cuda_if_needed(device_, "benchmark ES snapshot clone");
      actor_snapshot_ = clone_actor(actor_, device_);
      actor_snapshot_->eval();
      for (size_t i = 0; i < collection_actors_.size(); ++i) {
        if (collection_actors_[i] && shard_devices_[i] == device_) {
          collection_actors_[i] = actor_snapshot_;
        } else if (collection_actors_[i]) {
          collection_actors_[i] = clone_actor(actor_snapshot_, shard_devices_[i]);
          collection_actors_[i]->eval();
        }
      }
      result.es_updates += 1;
      result.es_seconds += es_metrics.es_seconds;
      result.es_eval_seconds += es_metrics.es_eval_seconds;
      result.es_agent_steps_per_second += es_metrics.es_agent_steps_per_second;
      result.es_effective_population += es_metrics.es_effective_population;
      result.es_virtual_population_waves += es_metrics.es_virtual_population_waves;
      result.es_eval_shards += es_metrics.es_eval_shards;
      result.es_policy_update_norm += es_metrics.es_policy_update_norm;
      std::cout << "bench_es_update_done update=" << update_index
                << " es_seconds=" << es_metrics.es_seconds
                << " es_eval_seconds=" << es_metrics.es_eval_seconds
                << " es_agent_steps_per_second=" << es_metrics.es_agent_steps_per_second
                << " es_policy_update_norm=" << es_metrics.es_policy_update_norm
                << '\n' << std::flush;
    }

    rollout_.clear();
  }

  result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
  const double denom = static_cast<double>(bounded_updates);
  result.policy_loss /= denom;
  result.grad_norm /= denom;
  if (result.es_updates > 0) {
    const double es_denom = static_cast<double>(result.es_updates);
    result.es_agent_steps_per_second /= es_denom;
    result.es_effective_population /= es_denom;
    result.es_virtual_population_waves /= es_denom;
    result.es_eval_shards /= es_denom;
    result.es_policy_update_norm /= es_denom;
  }
  benchmark_progress_ = previous_progress;
  return result;
}

void Trainer::save_training_state(const std::filesystem::path& path) const {
  torch::NoGradGuard no_grad;
  torch::serialize::OutputArchive archive;
  actor_->save(archive);
  actor_normalizer_.save(archive);
  actor_optimizer_.save(archive);
  archive.save_to(path.string());
}

void Trainer::load_training_state(const std::filesystem::path& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), device_);
  load_actor_from_archive(actor_, archive, device_);
  sanitize_actor_parameters();
  actor_normalizer_.load(archive);
  try {
    actor_optimizer_.load(archive);
  } catch (const std::exception& e) {
    std::cerr << "Warning: could not load actor optimizer state: " << e.what() << "\n";
  }
  actor_->to(device_);
  actor_normalizer_.to(device_);
}

void Trainer::prune_old_checkpoints(const std::filesystem::path& checkpoint_dir) const {
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

void Trainer::init_or_load_anchor() {
  if (run_output_root_.empty() || config_.ppo.anchor_eval_interval <= 0) return;

  const std::filesystem::path anchor_dir = run_output_root_ / "anchor";
  const std::filesystem::path anchor_state = anchor_dir / "state.pt";

  if (std::filesystem::exists(anchor_state)) {
    // Resume case: load previously saved anchor
    anchor_actor_ = clone_actor(actor_, device_);
    {
      torch::serialize::InputArchive archive;
      archive.load_from(anchor_state.string(), device_);
      load_actor_from_archive(anchor_actor_, archive, device_);
      anchor_normalizer_ = ObservationNormalizer(config_.model.observation_dim);
      anchor_normalizer_.load(archive);
    }
    anchor_actor_->eval();
    anchor_actor_->to(device_);
    anchor_normalizer_.to(device_);
    std::cout << "anchor_loaded=" << anchor_state.string() << '\n';
  } else {
    // First run: snapshot current actor as the anchor
    anchor_actor_ = clone_actor(actor_, device_);
    anchor_actor_->eval();
    anchor_normalizer_ = actor_normalizer_.clone();
    anchor_normalizer_.to(device_);
    save_anchor();
    std::cout << "anchor_created=" << anchor_state.string() << '\n';
  }
}

void Trainer::save_anchor() const {
  if (!anchor_actor_ || run_output_root_.empty()) return;
  const std::filesystem::path anchor_dir = run_output_root_ / "anchor";
  std::filesystem::create_directories(anchor_dir);
  torch::serialize::OutputArchive archive;
  anchor_actor_->save(archive);
  anchor_normalizer_.save(archive);
  archive.save_to((anchor_dir / "state.pt").string());
}

Trainer::AnchorEvalResult Trainer::run_anchor_eval(int update_index) {
  if (!anchor_actor_) return {};
  const int eval_envs = config_.ppo.anchor_eval_envs;
  const int eval_steps = config_.ppo.anchor_eval_steps;
  if (eval_envs <= 0 || eval_steps <= 0) return {};

  const auto eval_start = std::chrono::steady_clock::now();

  // Determine team size from the primary collector mode
  const std::string eval_mode = collectors_.front()->mode();
  const int team_size = (eval_mode == "3v3") ? 3 : (eval_mode == "2v2") ? 2 : 1;
  const int agents_per_env = 2 * team_size;

  // Build a temporary eval collector: fixed team ordering, fresh seeds
  ExperimentConfig eval_config = config_;
  eval_config.ppo.num_envs = eval_envs;
  eval_config.ppo.collection_workers = std::min(config_.ppo.collection_workers, eval_envs);
  eval_config.env.team_size = team_size;
  eval_config.env.spawn_opponents = true;

  auto obs_builder_cfg = eval_config.env;
  obs_builder_cfg.team_size = 3;  // max; zero-pads unused slots

  std::vector<StateMutatorPtr> mutators;
  mutators.push_back(std::make_shared<FixedTeamSizeMutator>(eval_config.env));
  if (eval_config.env.mode.find("random") != std::string::npos) {
    mutators.push_back(std::make_shared<RandomStateMutator>(eval_config.env));
  } else {
    mutators.push_back(std::make_shared<KickoffMutator>(eval_config.env));
  }
  const auto reset_mutator = std::make_shared<MutatorSequence>(std::move(mutators));
  std::vector<TransitionEnginePtr> engines;
  engines.reserve(static_cast<std::size_t>(eval_envs));
  for (int env_idx = 0; env_idx < eval_envs; ++env_idx) {
    EnvConfig env_config = eval_config.env;
    env_config.seed += static_cast<std::uint64_t>(
        3'000'007 + update_index * 65'537 + env_idx);
    engines.push_back(std::make_shared<RocketSimTransitionEngine>(env_config, reset_mutator));
  }

  auto eval_collector = std::make_unique<BatchedRocketSimCollector>(
      eval_config,
      std::move(engines),
      std::make_shared<PulsarObsBuilder>(obs_builder_cfg),
      std::make_shared<DiscreteActionParser>(ControllerActionTable(config_.action_table)),
      std::make_shared<SimpleDoneCondition>(eval_config.env),
      use_pinned_host_buffers_);
  eval_collector->set_mode(eval_mode);
  eval_collector->set_randomize_obs_order(false);  // Blue = slots 0..team_size-1
  eval_collector->reset_all();

  // Clone both actors for eval on device_
  Actor current_actor = clone_actor(actor_snapshot_, device_);
  current_actor->eval();
  Actor anchor_actor = clone_actor(anchor_actor_, device_);
  anchor_actor->eval();

  ObservationNormalizer current_norm = actor_normalizer_.clone();
  current_norm.to(device_);
  ObservationNormalizer anchor_norm = anchor_normalizer_.clone();
  anchor_norm.to(device_);

  const int64_t blue_agents = static_cast<int64_t>(eval_envs * team_size);
  const int64_t orange_agents = static_cast<int64_t>(eval_envs * team_size);
  torch::Tensor blue_state = current_actor->initial_recurrent_state(blue_agents, device_);
  torch::Tensor orange_state = anchor_actor->initial_recurrent_state(orange_agents, device_);

  const int physics_prefix_ticks =
      config_.env.half_tick_skip > 0 && config_.env.half_tick_skip < config_.env.tick_skip
          ? config_.env.half_tick_skip : 0;

  int total_episodes = 0;
  int blue_wins = 0;
  int orange_wins = 0;

  const int obs_dim_val = eval_collector->obs_dim();
  const int action_dim_val = eval_collector->action_dim();
  const int goal_dim_val = config_.goal_mapping.goal_dim;

  for (int step = 0; step < eval_steps; ++step) {
    if (config_.model.sequence_length > 0 && step % config_.model.sequence_length == 0) {
      blue_state.zero_();
      orange_state.zero_();
    }

    torch::Tensor raw_obs_host = eval_collector->host_observations();
    torch::Tensor episode_starts_host = eval_collector->host_episode_starts();
    torch::Tensor action_masks_host = eval_collector->host_action_masks();
    torch::Tensor goal_values_host = eval_collector->host_task_goal_positions();

    std::future<void> physics_future;
    CollectorTimings step_timings{};
    if (physics_prefix_ticks > 0) {
      physics_future = std::async(std::launch::async, [&]() {
        eval_collector->step_physics_only(physics_prefix_ticks, &step_timings);
      });
    }

    const torch::Tensor raw_obs = raw_obs_host.to(device_, use_pinned_host_buffers_);
    const torch::Tensor episode_starts = episode_starts_host.to(device_, use_pinned_host_buffers_);
    const torch::Tensor action_masks = action_masks_host.to(device_, use_pinned_host_buffers_).to(torch::kBool);
    const torch::Tensor goal_values = goal_values_host.to(device_, use_pinned_host_buffers_);

    // Reshape to [eval_envs, agents_per_env, ...]
    const torch::Tensor obs_4d = raw_obs.view({eval_envs, agents_per_env, obs_dim_val});
    const torch::Tensor masks_4d = action_masks.view({eval_envs, agents_per_env, action_dim_val});
    const torch::Tensor starts_4d = episode_starts.view({eval_envs, agents_per_env});
    const torch::Tensor goals_4d = goal_values.view({eval_envs, agents_per_env, goal_dim_val});

    // Slice blue and orange
    const torch::Tensor blue_raw = obs_4d.slice(1, 0, team_size).reshape({blue_agents, obs_dim_val});
    const torch::Tensor orange_raw = obs_4d.slice(1, team_size, agents_per_env).reshape({orange_agents, obs_dim_val});
    const torch::Tensor blue_masks = masks_4d.slice(1, 0, team_size).reshape({blue_agents, action_dim_val});
    const torch::Tensor orange_masks = masks_4d.slice(1, team_size, agents_per_env).reshape({orange_agents, action_dim_val});
    const torch::Tensor blue_starts = starts_4d.slice(1, 0, team_size).reshape({blue_agents});
    const torch::Tensor orange_starts = starts_4d.slice(1, team_size, agents_per_env).reshape({orange_agents});
    const torch::Tensor blue_goals = goals_4d.slice(1, 0, team_size).reshape({blue_agents, goal_dim_val});
    const torch::Tensor orange_goals = goals_4d.slice(1, team_size, agents_per_env).reshape({orange_agents, goal_dim_val});

    torch::Tensor all_actions_cpu;
    {
      torch::NoGradGuard no_grad;
      const torch::Tensor blue_norm = current_norm.normalize(blue_raw);
      ActorStepOutput blue_out = current_actor->forward_step_stateful(
          blue_norm, blue_state, blue_starts, &blue_state, blue_goals);
      const torch::Tensor blue_actions = sample_masked_actions(
          blue_out.policy_logits, blue_masks, /*greedy=*/true, nullptr, 1.0F);

      const torch::Tensor orange_norm = anchor_norm.normalize(orange_raw);
      ActorStepOutput orange_out = anchor_actor->forward_step_stateful(
          orange_norm, orange_state, orange_starts, &orange_state, orange_goals);
      const torch::Tensor orange_actions = sample_masked_actions(
          orange_out.policy_logits, orange_masks, /*greedy=*/true, nullptr, 1.0F);

      // Interleave: [eval_envs, team_size] cat → [eval_envs, agents_per_env] → flatten
      const torch::Tensor combined = torch::cat(
          {blue_actions.view({eval_envs, team_size}),
           orange_actions.view({eval_envs, team_size})}, /*dim=*/1);
      all_actions_cpu = combined.reshape({static_cast<int64_t>(eval_envs * agents_per_env)})
                            .contiguous()
                            .to(torch::kCPU);
    }

    if (physics_future.valid()) physics_future.get();
    eval_collector->step_after_physics_prefix(
        std::span<const std::int64_t>(
            all_actions_cpu.data_ptr<std::int64_t>(),
            static_cast<std::size_t>(all_actions_cpu.numel())),
        physics_prefix_ticks,
        &step_timings);

    // Count outcomes
    const torch::Tensor dones_host = eval_collector->host_dones();
    const torch::Tensor labels_host = eval_collector->host_terminal_outcome_labels();
    const float* dones_ptr = dones_host.data_ptr<float>();
    const int64_t* labels_ptr = labels_host.data_ptr<int64_t>();

    for (int env_idx = 0; env_idx < eval_envs; ++env_idx) {
      const int base = env_idx * agents_per_env;
      bool env_done = false, env_scored = false, env_conceded = false;
      for (int a = 0; a < agents_per_env; ++a) {
        if (dones_ptr[base + a] > 0.5F) {
          env_done = true;
          if (labels_ptr[base + a] == 0) env_scored = true;    // Blue scored
          if (labels_ptr[base + a] == 1) env_conceded = true;  // Orange scored
        }
      }
      if (env_done) {
        ++total_episodes;
        if (env_scored) ++blue_wins;
        if (env_conceded) ++orange_wins;
      }
    }
  }

  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
  const int decisive = blue_wins + orange_wins;
  const double win_rate =
      decisive > 0 ? static_cast<double>(blue_wins) / static_cast<double>(decisive) : -1.0;

  return AnchorEvalResult{
      .win_rate = win_rate,
      .episodes = static_cast<int64_t>(total_episodes),
      .seconds = seconds,
  };
}

void Trainer::train(int updates, const std::string& checkpoint_dir, const std::string& config_path) {
  const bool train_forever = updates <= 0;
  std::cout << "train_start\n";
  std::filesystem::create_directories(checkpoint_dir);
  WandbLogger wandb(config_.wandb, checkpoint_dir, config_path, "train");
  std::int64_t global_step = resumed_global_step_;

  init_or_load_anchor();

  TrainerMetrics coll_metrics{};
  std::int64_t coll_steps = 0;
  collect_rollout(rollout_, coll_metrics, &coll_steps, actor_snapshot_, actor_normalizer_);
  global_step += coll_steps;
  std::future<void> checkpoint_future;
  const auto wait_for_checkpoint = [&checkpoint_future]() {
    if (checkpoint_future.valid()) {
      checkpoint_future.get();
    }
  };

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
      coll_metrics.scored_episode_rate = scored_rate;
    }

    // 2. Update the actor on the main thread so CUDA/Torch thread-local
    // resources do not churn every iteration. Optional overlap only
    // backgrounds collection for the next rollout.
    TrainerMetrics train_metrics{};
    std::future<std::int64_t> collect_future;
    if (overlap_collection_update) {
      collection_normalizer.emplace(actor_normalizer_.clone());
      collect_future = task_pool_->submit([&]() {
        std::int64_t steps = 0;
        collect_rollout(rollout_B_, next_coll_metrics, &steps, actor_snapshot_, *collection_normalizer);
        return steps;
      });
      wait_for_checkpoint();
      train_metrics = update_actor(rollout_, update_index);
      next_coll_steps = collect_future.get();
      actor_normalizer_ = collection_normalizer->clone();
    } else {
      wait_for_checkpoint();
      train_metrics = update_actor(rollout_, update_index);
    }

    // 3. ES-LoRA update (touches actor_, safe now that PPO update is done)
    const int es_base_main = config_.es_lora.es_interval;
    steps_since_last_es_ += 1;
    const bool es_fired_this_update =
        (es_base_main > 0) && (steps_since_last_es_ >= es_base_main);
    if (es_fired_this_update) {
      run_es_lora_update(update_index, coll_metrics);
      steps_since_last_es_ = 0;
      // Sync ES-LoRA weight changes to replica actors.
      if (compute_actors_.size() > 0) {
        sync_actor_to_replicas(actor_, compute_actors_);
      }
    }

    // 5. Clone snapshot for next iteration's collection after all work using
    // the previous snapshot has completed.
    Actor new_snapshot{nullptr};
    if (has_next) {
      synchronize_current_cuda_stream_if_needed(device_, "snapshot clone");
      new_snapshot = clone_actor(actor_, device_);
      new_snapshot->eval();
      // Sync persistent collection actors
      for (size_t i = 0; i < collection_actors_.size(); ++i) {
        if (collection_actors_[i]) {
          // Always clone per shard, even when shard_devices_[i] == device_:
          // sharing the same nn::Module across concurrent shard threads causes data races.
          collection_actors_[i] = clone_actor(new_snapshot, shard_devices_[i]);
          collection_actors_[i]->eval();
        }
      }
    }

    // 6. In the default memory-safe mode, collect after the update with the
    // fresh policy snapshot.
    if (has_next && !overlap_collection_update) {
      next_coll_metrics = TrainerMetrics{};
      next_coll_steps = 0;
      collect_rollout(rollout_B_, next_coll_metrics, &next_coll_steps, new_snapshot, actor_normalizer_);
    }

    global_step += next_coll_steps;

    coll_metrics.policy_loss = train_metrics.policy_loss;
    coll_metrics.grad_norm = train_metrics.grad_norm;
    coll_metrics.grad_norm_valid_steps = train_metrics.grad_norm_valid_steps;
    coll_metrics.optimizer_steps = train_metrics.optimizer_steps;
    coll_metrics.nonfinite_loss_skips = train_metrics.nonfinite_loss_skips;
    coll_metrics.nonfinite_grad_norm_skips = train_metrics.nonfinite_grad_norm_skips;
    coll_metrics.update_seconds = train_metrics.update_seconds;
    coll_metrics.goal_critic_loss = train_metrics.goal_critic_loss;
    coll_metrics.mean_goal_score = train_metrics.mean_goal_score;
    coll_metrics.mean_sampled_goal_distance = train_metrics.mean_sampled_goal_distance;

    // CRL off-policy metrics copies
    coll_metrics.replay_buffer_size = train_metrics.replay_buffer_size;
    coll_metrics.car_critic_loss = train_metrics.car_critic_loss;
    coll_metrics.car_actor_loss  = train_metrics.car_actor_loss;
    coll_metrics.car_head_weight = train_metrics.car_head_weight;

    coll_metrics.update_agent_steps_per_second =
        next_coll_steps > 0 ? static_cast<double>(next_coll_steps) / std::max(train_metrics.update_seconds, 1.0e-9) : 0.0;

    coll_metrics.overall_agent_steps_per_second =
        next_coll_steps > 0
            ? static_cast<double>(next_coll_steps) /
                  std::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count(), 1.0e-9)
            : 0.0;
    // Anchor eval: run current policy vs rolling anchor snapshot
    const int anchor_interval = config_.ppo.anchor_eval_interval;
    if (anchor_interval > 0 && update_index % anchor_interval == 0) {
      const AnchorEvalResult anchor_result = run_anchor_eval(update_index);
      coll_metrics.anchor_win_rate = anchor_result.win_rate;
      coll_metrics.anchor_eval_seconds = anchor_result.seconds;
      coll_metrics.anchor_eval_episodes = anchor_result.episodes;

      // Ratchet anchor forward when current policy has clearly surpassed it
      const float threshold = config_.ppo.anchor_update_threshold;
      if (anchor_result.win_rate >= 0.0 && anchor_result.win_rate >= static_cast<double>(threshold)) {
        anchor_actor_ = clone_actor(actor_snapshot_, device_);
        anchor_actor_->eval();
        anchor_normalizer_ = actor_normalizer_.clone();
        anchor_normalizer_.to(device_);
        save_anchor();
        coll_metrics.anchor_updated = true;
        std::cout << "anchor_updated update=" << update_index
                  << " win_rate=" << anchor_result.win_rate
                  << " threshold=" << threshold << '\n';
      }
    }

    if (update_index % 200 == 0) {
      trim_released_host_memory();
    }
    coll_metrics.process_rss_mb = current_process_rss_mb();
    coll_metrics.process_peak_rss_mb = current_process_peak_rss_mb();
    const CgroupMemoryStats cgroup_memory = current_cgroup_memory_stats();
    coll_metrics.cgroup_memory_current_mb = cgroup_memory.current_mb;
    coll_metrics.cgroup_memory_limit_mb = cgroup_memory.limit_mb;
    sample_cuda_memory_stats(coll_metrics, compute_devices_);

    append_metrics_line(checkpoint_dir, update_index, global_step, coll_metrics, es_fired_this_update);
    std::cout << "update=" << update_index
              << " global_step=" << global_step
              << " ball_critic_loss=" << coll_metrics.policy_loss
              << " car_critic_loss=" << coll_metrics.car_critic_loss
              << " car_head_weight=" << coll_metrics.car_head_weight
              << " grad_norm=" << coll_metrics.grad_norm
              << " optimizer_steps=" << coll_metrics.optimizer_steps
              << " nonfinite_loss_skips=" << coll_metrics.nonfinite_loss_skips
              << " replay_buf=" << coll_metrics.replay_buffer_size
              << " mechanic_reward=" << coll_metrics.mechanic_reward_mean
              << " rollout_steps=" << coll_metrics.rollout_steps
              << " completed_eps=" << coll_metrics.completed_episodes
              << " scored_eps=" << coll_metrics.scored_episodes
              << " conceded_eps=" << coll_metrics.conceded_episodes
              << " touch_rate=" << coll_metrics.touch_episode_rate
              << " mean_goal_dist=" << coll_metrics.mean_goal_distance
              << " goals=" << coll_metrics.goals_scored << "/" << coll_metrics.goals_conceded
              << " anchor_win_rate=" << coll_metrics.anchor_win_rate
              << " rss_mb=" << coll_metrics.process_rss_mb
              << " cuda_reserved_mb=" << coll_metrics.cuda_memory_reserved_mb
              << " cuda_ooms=" << coll_metrics.cuda_ooms
              << '\n';
    if (wandb.enabled()) {
      for (const auto& item : get_all_metrics(coll_metrics, update_index, global_step, es_fired_this_update)) {
        wandb.add_metric(item.section, item.key, item.value);
      }
      
      // Dynamic mode stats tables
      std::vector<std::string> columns = {"Mode", "Completed Episodes", "Touch Rate", "Multi-Touch Rate", "Scored Rate"};
      std::vector<std::vector<nlohmann::json>> table_data;
      for (const auto& [mode, count] : coll_metrics.mode_completed_episodes) {
        float touch_rate = 0.0F;
        float multi_touch_rate = 0.0F;
        float scored_rate = 0.0F;
        if (coll_metrics.mode_touch_rates.count(mode)) touch_rate = coll_metrics.mode_touch_rates.at(mode);
        if (coll_metrics.mode_multi_touch_rates.count(mode)) multi_touch_rate = coll_metrics.mode_multi_touch_rates.at(mode);
        if (coll_metrics.mode_scored_rates.count(mode)) scored_rate = coll_metrics.mode_scored_rates.at(mode);

        table_data.push_back({
          mode,
          static_cast<int>(count),
          touch_rate,
          multi_touch_rate,
          scored_rate
        });
      }
      if (!table_data.empty()) {
        wandb.add_table("mode_statistics", columns, table_data);
      }

      wandb.commit(global_step);
    }
    if (config_.ppo.checkpoint_interval > 0 && update_index % config_.ppo.checkpoint_interval == 0) {
      const std::filesystem::path checkpoint_path =
          std::filesystem::path(checkpoint_dir) / ("update_" + std::to_string(update_index));
      const CheckpointMetadata checkpoint_metadata =
          make_checkpoint_metadata(global_step, update_index, wandb.run_id());
      checkpoint_future = task_pool_->submit([this, checkpoint_path, checkpoint_metadata, checkpoint_dir]() {
        save_checkpoint_with_metadata(checkpoint_path, checkpoint_metadata);
        prune_old_checkpoints(checkpoint_dir);
      });
    }

    if (has_next) {
      std::swap(rollout_, rollout_B_);
      coll_metrics = std::move(next_coll_metrics);
      actor_snapshot_ = std::move(new_snapshot);
    }
    coll_steps = next_coll_steps;
  }
  wait_for_checkpoint();
  save_checkpoint(std::filesystem::path(checkpoint_dir) / "final", global_step, static_cast<int>(resumed_update_index_) + updates, wandb.run_id());
  wandb.finish();
}

}  // namespace pulsar

#endif
