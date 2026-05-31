// Phase 0 gate self-test: prove the parity harness round-trips into C++.
//
// Loads the reference dumps written by parity/dump_reference.py and checks that
// the .npy loader reads them with the manifest-declared shapes. This is the
// scaffold the later phase gates build on (they additionally compare against
// freshly computed C++ tensors). Exits non-zero on any failure.
#include <cstdio>
#include <string>

#include "pulsar/parity.hpp"

namespace par = pulsar::parity;

namespace {

int check_shape(const std::string& ref_dir, const std::string& name,
                std::initializer_list<std::int64_t> expect) {
  const par::NpyArray a = par::load_npy(ref_dir + "/" + name + ".npy");
  std::vector<std::int64_t> exp(expect);
  if (a.shape != exp) {
    std::printf("  FAIL %-28s shape mismatch\n", name.c_str());
    return 1;
  }
  std::printf("  ok   %-28s numel=%lld\n", name.c_str(),
              static_cast<long long>(a.numel()));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // ref dir defaults to <repo>/parity/ref; overridable as argv[1]
  const std::string ref_dir = argc > 1 ? argv[1] : "parity/ref";
  std::printf("[pulsar_parity_check] ref dir: %s\n", ref_dir.c_str());

  int fails = 0;
  try {
    fails += check_shape(ref_dir, "obs/midfield/blue", {47});
    fails += check_shape(ref_dir, "obs/dynamic/orange", {47});
    fails += check_shape(ref_dir, "model_fwd/actor_mean", {16, 8});
    fails += check_shape(ref_dir, "model_fwd/critic_q", {16});
    fails += check_shape(ref_dir, "weights/critic/phi.0.weight", {128, 55});
    fails += check_shape(ref_dir, "weights/value/base.0.weight", {128, 53});
    fails += check_shape(ref_dir, "grid/status_final", {12, 16});
  } catch (const std::exception& e) {
    std::printf("[pulsar_parity_check] EXCEPTION: %s\n", e.what());
    return 2;
  }

  if (fails) {
    std::printf("[pulsar_parity_check] %d FAILURES\n", fails);
    return 1;
  }
  std::printf("[pulsar_parity_check] all reference records load with expected shapes\n");
  return 0;
}
