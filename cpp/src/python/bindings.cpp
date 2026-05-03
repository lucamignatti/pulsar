#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"
#include "pulsar/training/ppo_math.hpp"

namespace py = pybind11;

namespace pulsar {

class PyPPOActor {
 public:
  PyPPOActor(std::string checkpoint_dir, std::string device)
      : checkpoint_dir_(std::move(checkpoint_dir)),
        device_(std::move(device)),
        config_(load_experiment_config(checkpoint_dir_ + "/config.json")),
        metadata_(load_checkpoint_metadata(checkpoint_dir_ + "/metadata.json")),
        model_(load_ppo_actor(checkpoint_dir_, device_)),
        normalizer_(config_.model.observation_dim),
        torch_device_(device_) {
    validate_inference_checkpoint_metadata(metadata_, config_);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_dir_ + "/model.pt", torch_device_);
    normalizer_.load(archive);
    normalizer_.to(torch_device_);
  }

  void reset(std::size_t batch_size) {
    // Reset all tracked agent states.
    agent_states_.clear();
  }

  std::vector<float> forward(const std::vector<float>& obs) {
    // Convenience: use agent ID 0 for single-step forward.
    const auto batch = forward_batch({obs}, {0});
    if (batch.empty()) {
      return {};
    }
    return batch.front();
  }

  std::vector<std::vector<float>> forward_batch(
      const std::vector<std::vector<float>>& obs_batch,
      const std::vector<std::int64_t>& agent_ids,
      const std::vector<float>& episode_starts = {}) {
    if (obs_batch.empty()) {
      return {};
    }
    const std::size_t batch_size = obs_batch.size();
    if (agent_ids.size() != batch_size) {
      throw std::runtime_error("agent_ids length must match batch size.");
    }
    if (!episode_starts.empty() && episode_starts.size() != batch_size) {
      throw std::runtime_error("episode_starts length must match batch size.");
    }

    // Gather per-agent RNN states into a batched ContinuumState.
    // For new agents, allocate a fresh initial state.
    std::vector<ContinuumState> per_agent_states(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::int64_t agent_id = agent_ids[i];
      auto it = agent_states_.find(agent_id);
      if (it == agent_states_.end()) {
        it = agent_states_.emplace(
            agent_id,
            model_->initial_state(1, torch_device_)).first;
      }
      per_agent_states[i] = clone_state(it->second);
    }

    // Stack the per-agent states into a single batched state.
    ContinuumState batched_state = stack_states(per_agent_states);

    // Flatten observations.
    std::vector<float> flat;
    flat.reserve(batch_size * static_cast<std::size_t>(config_.model.observation_dim));
    for (const auto& obs : obs_batch) {
      if (obs.size() != static_cast<std::size_t>(config_.model.observation_dim)) {
        throw std::runtime_error("Observation length does not match model.observation_dim.");
      }
      flat.insert(flat.end(), obs.begin(), obs.end());
    }

    torch::Tensor input = torch::from_blob(
                              flat.data(),
                              {static_cast<std::int64_t>(batch_size), config_.model.observation_dim},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .clone()
                              .to(torch_device_);
    torch::Tensor starts;
    if (!episode_starts.empty()) {
      starts = torch::tensor(episode_starts, torch::TensorOptions().dtype(torch::kFloat32)).to(torch_device_);
    } else {
      starts = torch::zeros({static_cast<std::int64_t>(batch_size)},
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));
    }

    torch::NoGradGuard no_grad;
    const torch::Tensor normalized = normalizer_.normalize(input);
    ActorStepOutput output = model_->forward_step(normalized, std::move(batched_state), starts);

    // Unstack the output state and write back per-agent.
    std::vector<ContinuumState> output_states = unstack_states(output.state, batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::int64_t agent_id = agent_ids[i];
      bool is_episode_start = !episode_starts.empty() && episode_starts[i] > 0.5F;
      if (is_episode_start) {
        agent_states_[agent_id] = model_->initial_state(1, torch_device_);
      } else {
        agent_states_[agent_id] = std::move(output_states[i]);
      }
    }

    const torch::Tensor logits = output.policy_logits.to(torch::kCPU).contiguous();
    std::vector<std::vector<float>> result(batch_size, std::vector<float>(config_.model.action_dim));
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::memcpy(
          result[i].data(),
          logits[i].data_ptr<float>(),
          static_cast<std::size_t>(config_.model.action_dim) * sizeof(float));
    }
    return result;
  }

 private:
  // Stack a vector of single-agent ContinuumStates into one batched state.
  static ContinuumState stack_states(const std::vector<ContinuumState>& states) {
    if (states.empty()) {
      return {};
    }
    std::vector<torch::Tensor> workspaces, stm_keys, stm_values, stm_strengths;
    std::vector<torch::Tensor> stm_write_indices, ltm_coeffs, timesteps;
    for (const auto& s : states) {
      workspaces.push_back(s.workspace);
      stm_keys.push_back(s.stm_keys);
      stm_values.push_back(s.stm_values);
      stm_strengths.push_back(s.stm_strengths);
      stm_write_indices.push_back(s.stm_write_index);
      ltm_coeffs.push_back(s.ltm_coeffs);
      timesteps.push_back(s.timestep);
    }
    return {
        torch::cat(workspaces, 0),
        torch::cat(stm_keys, 0),
        torch::cat(stm_values, 0),
        torch::cat(stm_strengths, 0),
        torch::cat(stm_write_indices, 0),
        torch::cat(ltm_coeffs, 0),
        torch::cat(timesteps, 0),
    };
  }

  // Unstack a batched ContinuumState into per-agent states.
  static std::vector<ContinuumState> unstack_states(const ContinuumState& batched, std::size_t count) {
    std::vector<ContinuumState> result(count);
    for (std::size_t i = 0; i < count; ++i) {
      std::int64_t idx = static_cast<std::int64_t>(i);
      result[i] = {
          batched.workspace[idx].unsqueeze(0),
          batched.stm_keys[idx].unsqueeze(0),
          batched.stm_values[idx].unsqueeze(0),
          batched.stm_strengths[idx].unsqueeze(0),
          batched.stm_write_index[idx].unsqueeze(0),
          batched.ltm_coeffs[idx].unsqueeze(0),
          batched.timestep[idx].unsqueeze(0),
      };
    }
    return result;
  }

  std::string checkpoint_dir_{};
  std::string device_{};
  ExperimentConfig config_{};
  CheckpointMetadata metadata_{};
  PPOActor model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::Device torch_device_;
  std::unordered_map<std::int64_t, ContinuumState> agent_states_{};
};

}  // namespace pulsar

PYBIND11_MODULE(pulsar_native, m) {
  py::class_<pulsar::PyPPOActor>(m, "PPOActor")
      .def("reset", &pulsar::PyPPOActor::reset, py::arg("batch_size"))
      .def("forward", &pulsar::PyPPOActor::forward, py::arg("obs"))
      .def(
          "forward_batch",
          &pulsar::PyPPOActor::forward_batch,
          py::arg("obs_batch"),
          py::arg("agent_ids"),
          py::arg("episode_starts") = std::vector<float>{});

  m.def(
      "load_ppo_actor",
      [](const std::string& checkpoint_dir, const std::string& device) {
        return pulsar::PyPPOActor(checkpoint_dir, device);
      },
      py::arg("checkpoint_dir"),
      py::arg("device") = "cpu");

  m.def(
      "load_checkpoint_metadata",
      [](const std::string& path) {
        const pulsar::CheckpointMetadata metadata = pulsar::load_checkpoint_metadata(path);
        py::dict result;
        result["schema_version"] = metadata.schema_version;
        result["obs_schema_version"] = metadata.obs_schema_version;
        result["config_hash"] = metadata.config_hash;
        result["action_table_hash"] = metadata.action_table_hash;
        result["architecture_name"] = metadata.architecture_name;
        result["device"] = metadata.device;
        result["global_step"] = metadata.global_step;
        result["update_index"] = metadata.update_index;
        return result;
      });
}
