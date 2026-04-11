#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsar/checkpoint/checkpoint.hpp"
#include "pulsar/config/config.hpp"
#include "pulsar/model/actor_critic.hpp"
#include "pulsar/model/normalizer.hpp"

namespace py = pybind11;

namespace pulsar {

class PySharedModel {
 public:
  PySharedModel(std::string checkpoint_dir, std::string device)
      : checkpoint_dir_(std::move(checkpoint_dir)),
        device_(std::move(device)),
        config_(load_experiment_config(checkpoint_dir_ + "/config.json")),
        metadata_(load_checkpoint_metadata(checkpoint_dir_ + "/metadata.json")),
        model_(load_shared_model(checkpoint_dir_, device_)),
        normalizer_(config_.model.observation_dim) {
    validate_checkpoint_metadata(metadata_, config_);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_dir_ + "/model.pt");
    normalizer_.load(archive);
  }

  std::vector<float> forward(const std::vector<float>& obs) {
    torch::Tensor input = torch::tensor(obs, torch::TensorOptions().dtype(torch::kFloat32))
                              .reshape({1, config_.model.observation_dim})
                              .to(torch::Device(device_));
    const torch::Tensor normalized = normalizer_.normalize(input);
    const PolicyOutput output = model_->forward(normalized);
    const torch::Tensor logits = output.logits.squeeze(0).to(torch::kCPU);

    std::vector<float> result(static_cast<std::size_t>(logits.size(0)));
    std::memcpy(result.data(), logits.data_ptr<float>(), result.size() * sizeof(float));
    return result;
  }

 private:
  std::string checkpoint_dir_{};
  std::string device_{};
  ExperimentConfig config_{};
  CheckpointMetadata metadata_{};
  SharedActorCritic model_{nullptr};
  ObservationNormalizer normalizer_;
};

}  // namespace pulsar

PYBIND11_MODULE(pulsar_native, m) {
  py::class_<pulsar::PySharedModel>(m, "SharedModel")
      .def("forward", &pulsar::PySharedModel::forward);

  m.def(
      "load_shared_model",
      [](const std::string& checkpoint_dir, const std::string& device) {
        return pulsar::PySharedModel(checkpoint_dir, device);
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
