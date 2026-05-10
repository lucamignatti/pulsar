#include <cstring>
#include <filesystem>
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
    const std::filesystem::path checkpoint_path(checkpoint_dir_);
    const std::filesystem::path state_path = checkpoint_path / "state.pt";
    const std::filesystem::path model_path = checkpoint_path / "model.pt";
    archive.load_from(
        std::filesystem::exists(state_path) ? state_path.string() : model_path.string(),
        torch_device_);
    normalizer_.load(archive);
    normalizer_.to(torch_device_);
  }

  std::vector<float> forward(const std::vector<float>& obs) {
    const auto batch = forward_batch({obs});
    if (batch.empty()) {
      return {};
    }
    return batch.front();
  }

  std::vector<std::vector<float>> forward_batch(
      const std::vector<std::vector<float>>& obs_batch,
      const std::vector<float>& episode_starts = {}) {
    if (obs_batch.empty()) {
      return {};
    }
    const std::size_t batch_size = obs_batch.size();

    torch::Tensor input = torch::zeros(
        {static_cast<std::int64_t>(batch_size), config_.model.observation_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));

    for (std::size_t i = 0; i < batch_size; ++i) {
      const auto& obs = obs_batch[i];
      if (obs.size() != static_cast<std::size_t>(config_.model.observation_dim)) {
        throw std::runtime_error("Observation length does not match model.observation_dim.");
      }
      input[static_cast<std::int64_t>(i)].copy_(
          torch::from_blob(
              const_cast<float*>(obs.data()),
              {config_.model.observation_dim},
              torch::TensorOptions().dtype(torch::kFloat32))
              .clone()
              .to(torch_device_));
    }

    torch::NoGradGuard no_grad;
    const torch::Tensor normalized = normalizer_.normalize(input);
    const torch::Tensor goal_values = torch::zeros(
        {static_cast<std::int64_t>(batch_size), config_.goal_critic.goal_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));
    ActorStepOutput output = model_->forward_step(normalized, goal_values);

    const torch::Tensor logits = output.policy_logits.to(torch::kCPU).contiguous();
    std::vector<std::vector<float>> result(batch_size, std::vector<float>(config_.model.action_dim));
    for (std::size_t i = 0; i < batch_size; ++i) {
      std::memcpy(
          result[i].data(),
          logits[static_cast<std::int64_t>(i)].data_ptr<float>(),
          static_cast<std::size_t>(config_.model.action_dim) * sizeof(float));
    }
    return result;
  }

 private:
  std::string checkpoint_dir_{};
  std::string device_{};
  ExperimentConfig config_{};
  CheckpointMetadata metadata_{};
  PPOActor model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::Device torch_device_;
};

}  // namespace pulsar

PYBIND11_MODULE(pulsar_native, m) {
  py::class_<pulsar::PyPPOActor>(m, "PPOActor")
      .def("forward", &pulsar::PyPPOActor::forward, py::arg("obs"))
      .def(
          "forward_batch",
          &pulsar::PyPPOActor::forward_batch,
          py::arg("obs_batch"),
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
