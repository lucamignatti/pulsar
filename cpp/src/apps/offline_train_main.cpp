#include <iostream>
#include <string>

#include "pulsar/config/config.hpp"
#include "pulsar/training/offline_pretrainer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pulsar_offline_train <config.json> <output_dir>\n";
    return 1;
  }

  try {
    const pulsar::ExperimentConfig config = pulsar::load_experiment_config(argv[1]);
    pulsar::OfflinePretrainer pretrainer(config);
    pretrainer.train(argv[2]);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "pulsar_offline_train failed: " << exc.what() << '\n';
    return 1;
  }
}
