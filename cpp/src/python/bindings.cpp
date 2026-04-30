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
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/model/normalizer.hpp"

namespace py = pybind11;

namespace pulsar {

class PyLatentFutureActor {
 public:
  PyLatentFutureActor(std::string checkpoint_dir, std::string device)
      : checkpoint_dir_(std::move(checkpoint_dir)),
        device_(std::move(device)),
        config_(load_experiment_config(checkpoint_dir_ + "/config.json")),
        metadata_(load_checkpoint_metadata(checkpoint_dir_ + "/metadata.json")),
        model_(load_latent_future_actor(checkpoint_dir_, device_)),
        normalizer_(config_.model.observation_dim),
        torch_device_(device_) {
    validate_inference_checkpoint_metadata(metadata_, config_);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_dir_ + "/model.pt", torch_device_);
    normalizer_.load(archive);
    normalizer_.to(torch_device_);
  }

  void reset(std::size_t batch_size) {
    state_ = model_->initial_state(static_cast<std::int64_t>(batch_size), torch_device_);
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
    if (!state_.workspace.defined() || state_.workspace.size(0) != static_cast<std::int64_t>(batch_size)) {
      reset(batch_size);
    }

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
      if (episode_starts.size() != batch_size) {
        throw std::runtime_error("episode_starts length must match batch size.");
      }
      starts = torch::tensor(episode_starts, torch::TensorOptions().dtype(torch::kFloat32)).to(torch_device_);
    } else {
      starts = torch::zeros({static_cast<std::int64_t>(batch_size)},
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_));
    }

    torch::NoGradGuard no_grad;
    const torch::Tensor normalized = normalizer_.normalize(input);
    ActorStepOutput output = model_->forward_step(normalized, std::move(state_), starts);
    state_ = std::move(output.state);

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
  std::string checkpoint_dir_{};
  std::string device_{};
  ExperimentConfig config_{};
  CheckpointMetadata metadata_{};
  LatentFutureActor model_{nullptr};
  ObservationNormalizer normalizer_;
  torch::Device torch_device_;
  ContinuumState state_{};
};

}  // namespace pulsar

PYBIND11_MODULE(pulsar_native, m) {
  py::class_<pulsar::PyLatentFutureActor>(m, "LatentFutureActor")
      .def("reset", &pulsar::PyLatentFutureActor::reset, py::arg("batch_size"))
      .def("forward", &pulsar::PyLatentFutureActor::forward, py::arg("obs"))
      .def(
          "forward_batch",
          &pulsar::PyLatentFutureActor::forward_batch,
          py::arg("obs_batch"),
          py::arg("episode_starts") = std::vector<float>{});

  m.def(
      "load_latent_future_actor",
      [](const std::string& checkpoint_dir, const std::string& device) {
        return pulsar::PyLatentFutureActor(checkpoint_dir, device);
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
        result["future_evaluator_checkpoint"] = metadata.future_evaluator_checkpoint;
        result["future_evaluator_config_hash"] = metadata.future_evaluator_config_hash;
        result["future_evaluator_global_step"] = metadata.future_evaluator_global_step;
        result["future_evaluator_update_index"] = metadata.future_evaluator_update_index;
        result["future_evaluator_target_update_index"] = metadata.future_evaluator_target_update_index;
        return result;
      });
}
