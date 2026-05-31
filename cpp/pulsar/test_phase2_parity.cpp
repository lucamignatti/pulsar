// Phase 2 parity gate: models (Actor, Value, QuasimetricCritic).
//
// Loads reference weights from parity/ref/weights/{actor,value,critic}/*.npy,
// sets them on the C++ modules, runs the exact same inputs (parity/ref/model_fwd/),
// and checks that outputs match within tolerance.
//
// All arithmetic is done on CPU regardless of the default training device
// (parity is defined vs the CPU/Python reference; MPS precision may differ).
//
// Tolerances (matching sprint5 parity manifest):
//   actor_mean, actor_std, value: 1e-5
//   critic_q, infonce:            1e-5

#include <cstdio>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "pulsar/parity.hpp"
#include "pulsar/pulsar_models.hpp"

namespace par = pulsar::parity;
using namespace pulsar;

// ---- Load a npy file as a CPU torch::Tensor --------------------------------
static torch::Tensor npy_to_tensor(const std::string& path) {
  const par::NpyArray a = par::load_npy(path);
  // Build a float32 tensor from the double-precision NpyArray data
  std::vector<float> buf(a.data.begin(), a.data.end());
  // Shape
  std::vector<int64_t> shape(a.shape.begin(), a.shape.end());
  if (shape.empty()) shape = {1};
  return torch::from_blob(buf.data(), shape, torch::kFloat32).clone();
}

// ---- Load all named parameters for a module prefix -------------------------
// Iterates the module's named_parameters() and looks up
// "weights/<prefix>/<dot-separated-name>.npy" from the ref dir.
static void load_weights(torch::nn::Module& mod, const std::string& prefix,
                          const std::string& ref_dir) {
  for (const auto& item : mod.named_parameters()) {
    const std::string path = ref_dir + "/weights/" + prefix + "/" + item.key() + ".npy";
    torch::Tensor loaded = npy_to_tensor(path);
    torch::NoGradGuard no_grad;
    item.value().set_data(loaded.to(item.value().device()));
  }
}

// ---- Compare a tensor against a named reference ---------------------------
static int check_tensor(const std::string& ref_dir, const std::string& name,
                         const torch::Tensor& got, double tol = 1e-5) {
  const par::NpyArray ref = par::load_npy(ref_dir + "/" + name + ".npy");
  const torch::Tensor ref_t = [&] {
    std::vector<float> buf(ref.data.begin(), ref.data.end());
    std::vector<int64_t> shape(ref.shape.begin(), ref.shape.end());
    if (shape.empty()) shape = {1};
    return torch::from_blob(buf.data(), shape, torch::kFloat32).clone();
  }();

  const torch::Tensor got_cpu = got.detach().cpu().contiguous();
  const double diff = (got_cpu - ref_t).abs().max().item<double>();
  if (diff > tol) {
    std::printf("  FAIL %-32s max_abs_diff=%.2e (tol=%.2e)\n",
                name.c_str(), diff, tol);
    return 1;
  }
  std::printf("  ok   %-32s max_abs_diff=%.2e\n", name.c_str(), diff);
  return 0;
}

int main(int argc, char** argv) {
  const std::string ref_dir = argc > 1 ? argv[1] : "parity/ref";
  std::printf("[pulsar_phase2_parity] ref dir: %s\n\n", ref_dir.c_str());

  // Always run on CPU for parity (MPS may have slight numerical differences).
  const torch::Device dev = torch::kCPU;
  torch::manual_seed(0);  // not relied upon — weights come from ref

  int fails = 0;

  // Load reference inputs
  const torch::Tensor obs  = npy_to_tensor(ref_dir + "/model_fwd/obs.npy").to(dev);
  const torch::Tensor goal = npy_to_tensor(ref_dir + "/model_fwd/goal.npy").to(dev);
  const torch::Tensor act  = npy_to_tensor(ref_dir + "/model_fwd/act.npy").to(dev);

  std::printf("[inputs] obs=[%lldx%lld] goal=[%lldx%lld] act=[%lldx%lld]\n\n",
              (long long)obs.size(0),  (long long)obs.size(1),
              (long long)goal.size(0), (long long)goal.size(1),
              (long long)act.size(0),  (long long)act.size(1));

  // ---- PulsarActor -----------------------------------------------------------
  std::printf("--- PulsarActor ---\n");
  {
    PulsarActor actor;
    actor->to(dev);
    load_weights(*actor, "actor", ref_dir);
    actor->eval();

    torch::NoGradGuard no_grad;
    auto [mean, std, feat] = actor->forward(obs, goal);

    fails += check_tensor(ref_dir, "model_fwd/actor_mean", mean);
    fails += check_tensor(ref_dir, "model_fwd/actor_std",  std);
  }

  // ---- PulsarValue -----------------------------------------------------------
  std::printf("\n--- PulsarValue ---\n");
  {
    PulsarValue value;
    value->to(dev);
    load_weights(*value, "value", ref_dir);
    value->eval();

    torch::NoGradGuard no_grad;
    const torch::Tensor val = value->forward(obs, goal).flatten();

    fails += check_tensor(ref_dir, "model_fwd/value", val);
  }

  // ---- PulsarQuasimetricCritic + InfoNCE ------------------------------------
  std::printf("\n--- PulsarQuasimetricCritic ---\n");
  {
    PulsarQuasimetricCritic critic;
    critic->to(dev);
    load_weights(*critic, "critic", ref_dir);
    critic->eval();

    torch::NoGradGuard no_grad;
    const torch::Tensor q     = critic->forward(obs, act, goal);
    const torch::Tensor infonce = compute_pairwise_infonce_loss(critic, obs, act, goal);

    fails += check_tensor(ref_dir, "model_fwd/critic_q", q);
    fails += check_tensor(ref_dir, "model_fwd/infonce",  infonce.unsqueeze(0));
  }

  std::printf("\n[pulsar_phase2_parity] %s (%d failure(s))\n",
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
