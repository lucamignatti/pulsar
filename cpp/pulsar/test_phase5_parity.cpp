// Phase 5 parity gate: reachability grid curriculum.
//
// Drives the same synthetic sequence of record_kl / update_frontiers calls
// as parity/dump_reference.py::dump_grid() and checks:
//   - Final status array matches reference (exact)
//   - Final attempts array matches reference (exact)
//   - Final kl_running_avg matches reference (tol=1e-6)
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "pulsar/parity.hpp"
#include "pulsar/pulsar_curriculum.hpp"

namespace par = pulsar::parity;
using namespace pulsar;

static int check_array(const std::string& ref_dir, const std::string& name,
                        const double* got, std::size_t n, double tol) {
  const par::NpyArray ref = par::load_npy(ref_dir + "/" + name + ".npy");
  if (static_cast<std::size_t>(ref.numel()) != n) {
    std::printf("  FAIL %-32s numel mismatch\n", name.c_str());
    return 1;
  }
  double max_diff = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    max_diff = std::max(max_diff, std::abs(ref.data[i] - got[i]));
  }
  if (max_diff > tol) {
    std::printf("  FAIL %-32s max_abs_diff=%.2e (tol=%.2e)\n",
                name.c_str(), max_diff, tol);
    return 1;
  }
  std::printf("  ok   %-32s max_abs_diff=%.2e\n", name.c_str(), max_diff);
  return 0;
}

int main(int argc, char** argv) {
  const std::string ref_dir = argc > 1 ? argv[1] : "parity/ref";
  std::printf("[pulsar_phase5_parity] ref dir: %s\n\n", ref_dir.c_str());

  PulsarReachabilityGrid grid;
  std::mt19937 dummy_rng(42);  // only used for sample_start_state (not called here)

  // Load the exact KL values recorded by the Python reference.
  // trace cols: it, ci, cj, tier, unknown, frontier, mastered, kl
  const par::NpyArray trace  = par::load_npy(ref_dir + "/grid/trace.npy");
  const par::NpyArray kl_ref = par::load_npy(ref_dir + "/grid/kl_values.npy");
  const int n_rows = static_cast<int>(kl_ref.numel());
  std::printf("Replaying %d grid steps from parity trace...\n", n_rows);

  for (int row = 0; row < n_rows; ++row) {
    const int   ci = static_cast<int>(trace.data[row * 8 + 1]);
    const int   cj = static_cast<int>(trace.data[row * 8 + 2]);
    const float kl = static_cast<float>(kl_ref.data[row]);
    grid.record_kl(ci, cj, kl);
    grid.update_frontiers();
  }

  // Check final grid state
  std::printf("\n--- Grid state checks ---\n");
  int fails = 0;

  // status (int8 → double comparison via NpyArray)
  {
    const par::NpyArray ref_status = par::load_npy(ref_dir + "/grid/status_final.npy");
    double max_diff = 0.0;
    const auto& raw = grid.status_raw();
    for (int i = 0; i < PulsarReachabilityGrid::kXBins; ++i)
      for (int j = 0; j < PulsarReachabilityGrid::kYBins; ++j) {
        const double d = std::abs(ref_status.data[i * PulsarReachabilityGrid::kYBins + j] -
                                   static_cast<double>(static_cast<int8_t>(raw[i][j])));
        if (d > max_diff) max_diff = d;
      }
    if (max_diff > 0.0) {
      std::printf("  FAIL %-32s max_diff=%.0f\n", "grid/status_final", max_diff);
      ++fails;
    } else {
      std::printf("  ok   %-32s exact match\n", "grid/status_final");
    }
  }

  // attempts
  {
    std::vector<double> got(PulsarReachabilityGrid::kXBins * PulsarReachabilityGrid::kYBins);
    const auto& raw = grid.attempts_raw();
    for (int i = 0; i < PulsarReachabilityGrid::kXBins; ++i)
      for (int j = 0; j < PulsarReachabilityGrid::kYBins; ++j)
        got[i * PulsarReachabilityGrid::kYBins + j] = static_cast<double>(raw[i][j]);
    fails += check_array(ref_dir, "grid/attempts_final", got.data(), got.size(), 0.0);
  }

  // kl_running_avg (tolerance 1e-6 for float → double accumulation differences)
  {
    std::vector<double> got(PulsarReachabilityGrid::kXBins * PulsarReachabilityGrid::kYBins);
    const auto& raw = grid.kl_avg_raw();
    for (int i = 0; i < PulsarReachabilityGrid::kXBins; ++i)
      for (int j = 0; j < PulsarReachabilityGrid::kYBins; ++j)
        got[i * PulsarReachabilityGrid::kYBins + j] = raw[i][j];
    fails += check_array(ref_dir, "grid/kl_running_avg_final", got.data(), got.size(), 1e-6);
  }

  // Stats
  const auto [unk, front, mast] = grid.stats();
  const int ref_row = n_rows - 1;
  const int ref_unk   = static_cast<int>(trace.data[ref_row * 8 + 4]);
  const int ref_front = static_cast<int>(trace.data[ref_row * 8 + 5]);
  const int ref_mast  = static_cast<int>(trace.data[ref_row * 8 + 6]);

  std::printf("\n--- Grid stats (vs reference trace final row) ---\n");
  std::printf("  unknown:  C++=%d  ref=%d  %s\n", unk,   ref_unk,   unk == ref_unk ? "ok" : "FAIL");
  std::printf("  frontier: C++=%d  ref=%d  %s\n", front, ref_front, front == ref_front ? "ok" : "FAIL");
  std::printf("  mastered: C++=%d  ref=%d  %s\n", mast,  ref_mast,  mast == ref_mast ? "ok" : "FAIL");
  if (unk != ref_unk || front != ref_front || mast != ref_mast) ++fails;

  std::printf("\n[pulsar_phase5_parity] %s (%d failure(s))\n",
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
