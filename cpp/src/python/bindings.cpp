#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/model/normalizer.hpp"
#include "pulsar/model/ppo_actor.hpp"

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

  void reset(std::size_t max_batch_size) {
    max_batch_size_ = static_cast<std::int64_t>(max_batch_size);
    batched_state_ = model_->initial_state(max_batch_size_, torch_device_);
    agent_id_to_slot_.clear();
    slot_to_agent_id_.assign(static_cast<std::size_t>(max_batch_size_), -1);
  }

  void remove_agents(const std::vector<std::int64_t>& agent_ids) {
    for (std::int64_t agent_id : agent_ids) {
      auto it = agent_id_to_slot_.find(agent_id);
      if (it != agent_id_to_slot_.end()) {
        std::size_t slot = it->second;
        slot_to_agent_id_[slot] = -1;
        agent_id_to_slot_.erase(it);
      }
    }
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

    // Lazy-initialize the persistent batched state on first call.
    if (!batched_state_.workspace.defined()) {
      reset(std::max<std::size_t>(batch_size, 1));
    }

    // Assign slots to any previously unseen agent IDs.
    // Reuse freed slots when available.
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::int64_t agent_id = agent_ids[i];
      if (agent_id_to_slot_.find(agent_id) != agent_id_to_slot_.end()) {
        continue;  // Already assigned.
      }
      // Find a free slot.
      std::int64_t assigned_slot = -1;
      for (std::size_t s = 0; s < static_cast<std::size_t>(max_batch_size_); ++s) {
        if (slot_to_agent_id_[s] < 0) {
          assigned_slot = static_cast<std::int64_t>(s);
          break;
        }
      }
      if (assigned_slot < 0) {
        throw std::runtime_error(
            "No free slots in persistent batched state. "
            "Call reset() with a larger max_batch_size or remove_agents() to free slots.");
      }
      std::size_t slot = static_cast<std::size_t>(assigned_slot);
      slot_to_agent_id_[slot] = agent_id;
      agent_id_to_slot_[agent_id] = slot;

      // Initialize this agent's slice with a fresh state.
      ContinuumState init = model_->initial_state(1, torch_device_);
      copy_slice_in(batched_state_, slot, init);
    }

    // Build the observation tensor at max_batch_size_: zero-fill inactive slots.
    torch::Tensor input = torch::zeros(
        {max_batch_size_, config_.model.observation_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));
    torch::Tensor starts = torch::zeros(
        {max_batch_size_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));

    for (std::size_t i = 0; i < batch_size; ++i) {
      std::int64_t agent_id = agent_ids[i];
      std::size_t slot = agent_id_to_slot_[agent_id];
      const auto& obs = obs_batch[i];
      if (obs.size() != static_cast<std::size_t>(config_.model.observation_dim)) {
        throw std::runtime_error("Observation length does not match model.observation_dim.");
      }
      input[static_cast<std::int64_t>(slot)].copy_(
          torch::from_blob(
              const_cast<float*>(obs.data()),
              {config_.model.observation_dim},
              torch::TensorOptions().dtype(torch::kFloat32))
              .clone()
              .to(torch_device_));

      if (!episode_starts.empty() && episode_starts[i] > 0.5F) {
        starts[static_cast<std::int64_t>(slot)].fill_(1.0F);
      }
    }

    // For inactive slots, keep episode_starts=1 so their state resets harmlessly.
    for (std::size_t s = 0; s < static_cast<std::size_t>(max_batch_size_); ++s) {
      if (slot_to_agent_id_[s] < 0) {
        starts[static_cast<std::int64_t>(s)].fill_(1.0F);
      }
    }

    torch::NoGradGuard no_grad;
    const torch::Tensor normalized = normalizer_.normalize(input);
    ActorStepOutput output = model_->forward_step(
        normalized, std::move(batched_state_), starts);
    batched_state_ = std::move(output.state);

    // Extract logits only for the requested agents.
    const torch::Tensor logits = output.policy_logits.to(torch::kCPU).contiguous();
    std::vector<std::vector<float>> result(batch_size, std::vector<float>(config_.model.action_dim));
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::int64_t agent_id = agent_ids[i];
      std::size_t slot = agent_id_to_slot_[agent_id];
      std::memcpy(
          result[i].data(),
          logits[static_cast<std::int64_t>(slot)].data_ptr<float>(),
          static_cast<std::size_t>(config_.model.action_dim) * sizeof(float));
    }
    return result;
  }

 private:
  // Copy a single-agent state into one slice of the persistent batched state.
  static void copy_slice_in(ContinuumState& batched, std::size_t slot, const ContinuumState& single) {
    std::int64_t idx = static_cast<std::int64_t>(slot);
    batched.workspace[idx].copy_(single.workspace.squeeze(0));
    batched.stm_keys[idx].copy_(single.stm_keys.squeeze(0));
    batched.stm_values[idx].copy_(single.stm_values.squeeze(0));
    batched.stm_strengths[idx].copy_(single.stm_strengths.squeeze(0));
    batched.stm_write_index[idx].copy_(single.stm_write_index.squeeze(0));
    batched.ltm_coeffs[idx].copy_(single.ltm_coeffs.squeeze(0));
    batched.timestep[idx].copy_(single.timestep.squeeze(0));
  }

  std::string checkpoint_dir_{};
  std::string device_{};
  ExperimentConfig config_{};
  CheckpointMetadata metadata_{};
  PPOActor model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::Device torch_device_;

  // Persistent batched state: allocated once at max_batch_size_, slices updated
  // in-place by forward_step.  No per-call stack/unstack overhead.
  std::int64_t max_batch_size_ = 0;
  ContinuumState batched_state_{};
  std::unordered_map<std::int64_t, std::size_t> agent_id_to_slot_{};
  std::vector<std::int64_t> slot_to_agent_id_{};  // -1 for empty slots
};

}  // namespace pulsar

PYBIND11_MODULE(pulsar_native, m) {
  py::class_<pulsar::PyPPOActor>(m, "PPOActor")
      .def("reset", &pulsar::PyPPOActor::reset, py::arg("max_batch_size"))
      .def("remove_agents", &pulsar::PyPPOActor::remove_agents, py::arg("agent_ids"))
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
