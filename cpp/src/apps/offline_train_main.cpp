#include <filesystem>
#include <iostream>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/tracing/tracing.hpp"
#include "pulsar/training/offline_pretrainer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_lfpo_pretrain <config.json> <output_dir>\n";
    return 1;
  }

  try {
    pulsar::tracing::Session trace_session(std::filesystem::path(argv[2]) / "trace.perfetto.json", "pulsar_lfpo_pretrain");
    PULSAR_TRACE_SET_THREAD_NAME("main");
    const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);
    pulsar::OfflinePretrainer pretrainer(config);
    pretrainer.train(argv[2], argv[1]);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_lfpo_pretrain failed: " << exc.what() << '\n';
    return 1;
  }
}
