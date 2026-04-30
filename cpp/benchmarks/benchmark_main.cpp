#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "pulsar/model/future_evaluator.hpp"
#include "pulsar/model/latent_future_actor.hpp"
#include "pulsar/training/lfpo_math.hpp"

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

pulsar::ModelConfig benchmark_model_config() {
  pulsar::ModelConfig config;
  config.observation_dim = 132;
  config.action_dim = 90;
  config.encoder_dim = 64;
  config.workspace_dim = 64;
  config.stm_slots = 8;
  config.stm_key_dim = 16;
  config.stm_value_dim = 16;
  config.ltm_slots = 8;
  config.ltm_dim = 16;
  config.controller_dim = 64;
  config.action_embedding_dim = 16;
  config.future_latent_dim = 32;
  config.future_horizon_count = 3;
  return config;
}

pulsar::FutureEvaluatorConfig benchmark_evaluator_config() {
  pulsar::FutureEvaluatorConfig config;
  config.horizons = {8, 32, 96};
  config.latent_dim = 32;
  config.model_dim = 64;
  config.layers = 1;
  config.heads = 4;
  config.feedforward_dim = 128;
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  const int updates = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  const int batch = argc > 2 ? std::max(1, std::atoi(argv[2])) : 32;
  const int candidates = 8;
  const int sequence = 8;

  const pulsar::ModelConfig model_config = benchmark_model_config();
  const pulsar::FutureEvaluatorConfig evaluator_config = benchmark_evaluator_config();
  pulsar::LatentFutureActor actor(model_config);
  pulsar::FutureEvaluator evaluator(evaluator_config, model_config.observation_dim);
  torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(3.0e-4));

  double policy_forward_seconds = 0.0;
  double evaluator_seconds = 0.0;
  double lfpo_seconds = 0.0;
  std::int64_t samples = 0;

  for (int update = 0; update < updates; ++update) {
    torch::Tensor obs = torch::randn({sequence, batch, model_config.observation_dim});
    torch::Tensor starts = torch::zeros({sequence, batch});
    auto state = actor->initial_state(batch, torch::kCPU);
    auto forward_start = std::chrono::steady_clock::now();
    const pulsar::ActorSequenceOutput output = actor->forward_sequence(obs, std::move(state), starts);
    policy_forward_seconds += seconds_since(forward_start);

    const torch::Tensor features = output.features.reshape({sequence * batch, actor->feature_dim()});
    const torch::Tensor candidate_actions =
        torch::randint(model_config.action_dim, {sequence * batch, candidates}, torch::TensorOptions().dtype(torch::kLong));
    const torch::Tensor feature_candidates =
        features.unsqueeze(1).expand({features.size(0), candidates, actor->feature_dim()}).reshape({-1, actor->feature_dim()});

    auto lfpo_start = std::chrono::steady_clock::now();
    const torch::Tensor predicted =
        actor->predict_future_latents(feature_candidates, candidate_actions.reshape({-1}))
            .reshape({features.size(0), candidates, model_config.future_horizon_count, model_config.future_latent_dim});
    auto evaluator_start = std::chrono::steady_clock::now();
    const torch::Tensor scores = pulsar::latent_action_scores(evaluator->classify_embeddings(predicted));
    evaluator_seconds += seconds_since(evaluator_start);
    const torch::Tensor advantages = pulsar::relative_candidate_advantages(scores);
    const torch::Tensor logits = output.policy_logits.reshape({sequence * batch, model_config.action_dim});
    const torch::Tensor masks = torch::ones(logits.sizes(), torch::TensorOptions().dtype(torch::kBool));
    const torch::Tensor log_probs =
        torch::log_softmax(pulsar::apply_action_mask_to_logits(logits, masks), -1).gather(1, candidate_actions);
    const torch::Tensor loss =
        pulsar::clipped_lfpo_policy_loss(log_probs, log_probs.detach(), advantages, 0.2F) -
        0.01F * pulsar::masked_action_entropy(logits, masks).mean();
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    lfpo_seconds += seconds_since(lfpo_start);
    samples += sequence * batch;
  }

  const double total_seconds = std::max(policy_forward_seconds + lfpo_seconds, 1.0e-9);
  std::cout << "collection_agent_steps_per_second=" << static_cast<double>(samples) / total_seconds << '\n';
  std::cout << "lfpo_update_agent_steps_per_second=" << static_cast<double>(samples) / std::max(lfpo_seconds, 1.0e-9) << '\n';
  std::cout << "offline_pretrain_samples_per_second=" << static_cast<double>(samples) / total_seconds << '\n';
  std::cout << "offline_pretrain_epoch_seconds=" << total_seconds << '\n';
  std::cout << "policy_forward_seconds=" << policy_forward_seconds << '\n';
  std::cout << "future_evaluator_seconds=" << evaluator_seconds << '\n';
  std::cout << "lfpo_forward_backward_seconds=" << lfpo_seconds << '\n';
  return EXIT_SUCCESS;
}
