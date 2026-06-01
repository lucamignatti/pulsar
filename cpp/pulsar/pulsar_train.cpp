// pulsar_train — Pulsar training training entry point.
//
// Wires together all Phase 1-5 components into a full training loop mirroring
// sprint5:529-817. Runs on MPS (Apple Silicon) > CUDA > CPU automatically.
//
// Usage:
//   pulsar_train [config.json] [output_dir] [max_updates]
//   pulsar_train                        # 300 updates, no output dir
//
// Config JSON keys (all optional, defaults match sprint5):
//   num_envs        (default 8)
//   num_steps       (default 256)
//   num_epochs      (default 4)
//   minibatch_size  (default 1024)
//   lr              (default 3e-4)
//   gamma           (default 0.99)
//   gae_lambda      (default 0.95)
//   tick_skip       (default 8)
//   max_steps       (default 200)
//   log_interval    (default 10)
//   collision_meshes_path (default "collision_meshes")
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>

#include <torch/torch.h>

#include "pulsar/pulsar_collector.hpp"
#include "pulsar/pulsar_curriculum.hpp"
#include "pulsar/pulsar_device.hpp"
#include "pulsar/pulsar_models.hpp"
#include "pulsar/pulsar_rollout_storage.hpp"
#include "pulsar/pulsar_update.hpp"
#include "pulsar/pulsar_wandb.hpp"

using namespace pulsar;

// Simple JSON field extraction (avoids pulling in nlohmann_json)
static int    json_int   (const std::string& s, const std::string& key, int    def) {
  const auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return def;
  const auto c = s.find(':', p); if (c == std::string::npos) return def;
  return std::stoi(s.substr(c + 1));
}
static double json_double(const std::string& s, const std::string& key, double def) {
  const auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return def;
  const auto c = s.find(':', p); if (c == std::string::npos) return def;
  return std::stod(s.substr(c + 1));
}
static std::string json_str(const std::string& s, const std::string& key,
                              const std::string& def) {
  const auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return def;
  const auto c = s.find('"', s.find(':', p) + 1);
  if (c == std::string::npos) return def;
  const auto e = s.find('"', c + 1);
  return e == std::string::npos ? def : s.substr(c + 1, e - c - 1);
}

int main(int argc, char** argv) {
  // ---- Parse args ----------------------------------------------------------
  std::string config_json = "{}";
  std::string output_dir  = "";
  int max_updates = 0;  // 0 = run indefinitely

  if (argc > 1) {
    const std::string arg1 = argv[1];
    if (!arg1.empty() && arg1[0] == '{') {
      config_json = arg1;  // inline JSON
    } else {
      std::ifstream f(arg1);
      if (f) { config_json = std::string((std::istreambuf_iterator<char>(f)),
                                          std::istreambuf_iterator<char>()); }
    }
  }
  if (argc > 2) output_dir  = argv[2];
  if (argc > 3) max_updates = std::atoi(argv[3]);

  // ---- Hyper-parameters (sprint5 defaults) ---------------------------------
  const int    num_envs    = json_int   (config_json, "num_envs",    8);
  const int    num_steps   = json_int   (config_json, "num_steps",   256);
  const int    num_epochs  = json_int   (config_json, "num_epochs",  4);
  const int    mb_size     = json_int   (config_json, "minibatch_size", num_envs * num_steps * 2 / 4);
  const float  lr          = static_cast<float>(json_double(config_json, "lr",         3e-4));
  const int    tick_skip   = json_int   (config_json, "tick_skip",   8);
  const int    max_steps   = json_int   (config_json, "max_steps",   200);
  const int    log_interval        = json_int(config_json, "log_interval",        10);
  const int    checkpoint_interval = json_int(config_json, "checkpoint_interval", 100);
  const std::string meshes = json_str(config_json, "collision_meshes_path", "collision_meshes");
  // num_workers: defaults to num_envs/2 if not specified.
  // Tune to physical core count minus 1 (e.g. 11 for a 12-core CPU).
  const int    num_workers  = json_int  (config_json, "num_workers",  std::max(1, num_envs / 2));
  const std::string wb_project = json_str(config_json, "wandb_project", "");
  const std::string wb_run     = json_str(config_json, "wandb_run",     "");
  const std::string wb_entity  = json_str(config_json, "wandb_entity",  "");

  const int N_agents = num_envs * 2;

  // ---- Device --------------------------------------------------------------
  const torch::Device dev = select_device();
  std::printf("[pulsar_train] device: %s\n", device_name(dev).c_str());
  std::printf("[pulsar_train] num_envs=%d  num_steps=%d  num_agents=%d  workers=%d  max_updates=%s\n",
              num_envs, num_steps, N_agents, num_workers,
              max_updates > 0 ? std::to_string(max_updates).c_str() : "unlimited");

  // ---- Curriculum ----------------------------------------------------------
  PulsarReachabilityGrid grid;

  // ---- Collector — passes the grid for per-reset curriculum sampling ------
  const bool pin_mem = (dev.type() == torch::kCUDA);  // pinning only helps CUDA
  PulsarCollector collector(num_envs, num_workers,
                          /*mutator=*/nullptr,
                          tick_skip, max_steps, pin_mem, meshes,
                          /*curriculum_grid=*/&grid);
  collector.initialize(/*base_seed=*/42);
  std::printf("[pulsar_train] collector initialized\n");

  // ---- WandB ---------------------------------------------------------------
  PulsarWandB wandb;
  if (!wb_project.empty()) {
    if (wandb.init(wb_project, wb_run, wb_entity)) {
      std::printf("[pulsar_train] wandb → project '%s'\n", wb_project.c_str());
    } else {
      std::printf("[pulsar_train] wandb unavailable (wandb not installed?)\n");
    }
  }

  // ---- Output dir ----------------------------------------------------------
  namespace fs = std::filesystem;
  if (!output_dir.empty()) fs::create_directories(output_dir);

  // ---- Models + update loop ------------------------------------------------
  torch::manual_seed(42);
  PulsarActor   actor;
  PulsarValue   value;
  PulsarQuasimetricCritic critic;
  actor->to(dev);  value->to(dev);  critic->to(dev);

  // ---- Resume from checkpoint if one exists --------------------------------
  int start_update = 1;
  if (!output_dir.empty()) {
    std::ifstream latest_f(output_dir + "/latest.txt");
    if (latest_f) {
      int latest_upd = 0;
      latest_f >> latest_upd;
      if (latest_upd > 0) {
        const std::string ckpt = output_dir + "/ckpt_" + std::to_string(latest_upd);
        torch::load(actor,  ckpt + "/actor.pt");
        torch::load(value,  ckpt + "/value.pt");
        torch::load(critic, ckpt + "/critic.pt");
        actor->to(dev);  value->to(dev);  critic->to(dev);
        start_update = latest_upd + 1;
        std::printf("[pulsar_train] resumed from update %d (%s)\n", latest_upd, ckpt.c_str());
      }
    }
  }

  PulsarUpdateLoop update_loop(actor, value, critic, lr);

  // ---- Rollout storage -----------------------------------------------------
  PulsarRolloutStorage storage(num_steps, N_agents, pin_mem);

  // ---- Goal tensors (fixed conditioning) -----------------------------------
  const auto agent_goals = PulsarUpdateLoop::build_agent_goals(N_agents, dev); // [N, 6]

  // ---- Training loop (sprint5:593-816) -------------------------------------
  std::vector<float> win_rate_history;
  const auto t_start = std::chrono::steady_clock::now();

  // Per-env episode stats
  std::vector<float> env_rewards(N_agents, 0.0f);  // cumulative episode reward
  bool curriculum_done = false;

  for (int update = start_update; max_updates <= 0 || update <= max_updates; ++update) {
    actor->eval();
    value->eval();
    critic->eval();

    // ---- Rollout -----------------------------------------------------------
    auto next_obs  = collector.current_obs().clone().to(dev);
    auto next_done = torch::zeros({N_agents}, dev);

    for (int step = 0; step < num_steps; ++step) {
      // Store current obs
      storage.set_obs(step, next_obs.cpu());

      torch::Tensor actions, logprobs, values_t;
      {
        torch::NoGradGuard ng;
        const auto [mean, std, _feat] = actor->forward(next_obs, agent_goals);
        // Gaussian sample
        const auto noise = torch::randn_like(mean);
        actions  = mean + std * noise;
        logprobs = gaussian_log_prob(actions, mean, std);
        values_t = value->forward(next_obs, agent_goals).flatten();
      }

      // Store in rollout
      storage.set_actions (step, actions.cpu());
      storage.set_logprobs(step, logprobs.cpu());
      storage.set_values  (step, values_t.cpu());

      // Step collector (actions go to CPU for env step)
      const auto actions_cpu = actions.cpu();
      collector.step(actions_cpu);

      const auto rew  = collector.last_rewards().to(dev);
      const auto done = collector.last_dones().to(dev);

      storage.set_rewards(step, rew.cpu());
      storage.set_dones  (step, done.cpu());

      // Store cell indices from curriculum mutator
      // (populated per-env; propagated through collector)
      storage.set_cells(step, collector.last_cells().cpu());

      // Track episode outcomes for win rate
      const float* rew_ptr  = rew.cpu().data_ptr<float>();
      const float* done_ptr = done.cpu().data_ptr<float>();
      for (int ai = 0; ai < N_agents; ai += 2) {  // blue agents are even
        if (done_ptr[ai] > 0.0f) {
          win_rate_history.push_back(rew_ptr[ai] > 0.0f ? 1.0f : 0.0f);
        }
      }

      next_obs  = collector.current_obs().clone().to(dev);
      next_done = collector.last_dones().clone().to(dev);
    }

    // ---- Update ------------------------------------------------------------
    actor->train();  value->train();  critic->train();

    const auto result = update_loop.update(
        storage, next_obs.cpu(), next_done.cpu(),
        num_epochs, mb_size, dev);

    // ---- Per-cell KLD signal → curriculum (sprint5:760-793) ----------------
    // Compute per-transition KLD = -(new_logprob - old_logprob) and group by cell.
    {
      const int B = num_steps * N_agents;
      const auto b_obs  = storage.obs.reshape({B, PulsarObsBuilder::kObsDim}).to(dev);
      const auto b_act  = storage.actions.reshape({B, PulsarContinuousActionParser::kActionDim}).to(dev);
      const auto b_logp = storage.logprobs.reshape({B}).to(dev);
      const auto b_cells= storage.cells.reshape({B, 2});  // [B, 2] int32

      // Rebuild minibatch agent goals (parity of flat index)
      const auto gs = PulsarUpdateLoop::goal_score_tensor(dev);
      const auto gd = PulsarUpdateLoop::goal_defend_tensor(dev);
      auto all_goals = torch::zeros({B, kGoalDim}, dev);
      const auto idx = torch::arange(B, dev);
      all_goals.index_put_({idx % 2 == 0}, gs.unsqueeze(0));
      all_goals.index_put_({idx % 2 == 1}, gd.unsqueeze(0));

      torch::Tensor kl_all;
      {
        actor->eval();
        torch::NoGradGuard ng;
        auto [am, astd, _f] = actor->forward(b_obs, all_goals);
        const auto new_logp = gaussian_log_prob(b_act, am, astd);
        kl_all = (-(new_logp - b_logp)).cpu();  // [B]
      }

      // Group by (ci, cj) and call record_kl
      const float* kl_ptr = kl_all.data_ptr<float>();
      const int32_t* cell_ptr = b_cells.data_ptr<int32_t>();

      std::unordered_map<uint32_t, std::pair<double, int>> cell_kl;
      for (int i = 0; i < B; ++i) {
        const int ci = cell_ptr[i * 2 + 0];
        const int cj = cell_ptr[i * 2 + 1];
        const uint32_t key = (static_cast<uint32_t>(ci) << 16) | static_cast<uint32_t>(cj);
        auto& entry = cell_kl[key];
        entry.first += static_cast<double>(kl_ptr[i]);
        entry.second += 1;
      }
      for (auto& [key, val] : cell_kl) {
        const int ci = static_cast<int>(key >> 16);
        const int cj = static_cast<int>(key & 0xFFFF);
        const float mean_kl = static_cast<float>(val.first / val.second);
        grid.record_kl(ci, cj, mean_kl);
      }
      grid.update_frontiers();

      // When the full field is mastered at tier 3, switch to standard kickoff.
      if (!curriculum_done && grid.is_complete()) {
        curriculum_done = true;
        std::printf("[pulsar_train] Curriculum complete! Switching to kickoff resets.\n");
        collector.switch_to_kickoff(static_cast<std::uint64_t>(update) * 100);
      }
    }

    // ---- Logging -----------------------------------------------------------
    if (update % log_interval == 0 || update == 1) {
      const auto [unk, front, mast] = grid.stats();
      const float wr = win_rate_history.size() >= 40 ?
          [&] { float s = 0; auto e = win_rate_history.end();
                for (auto it = e - 40; it != e; ++it) s += *it;
                return s / 40.0f; }() : 0.0f;
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
      const double fps = static_cast<double>(update * num_steps * N_agents * tick_skip) / elapsed;

      std::printf("  Upd %d | WR:%.2f | PG:%.4f VL:%.4f ENT:%.4f CL:%.4f "
                  "| Mast:%d Front:%d Unk:%d Tier:%d | %.0f ticks/s\n",
                  update, wr,
                  result.pg_loss, result.value_loss, result.entropy, result.infonce_loss,
                  mast, front, unk, grid.current_tier(), fps);

      wandb.log(update, {
          {"train/win_rate",        static_cast<double>(wr)},
          {"train/pg_loss",         static_cast<double>(result.pg_loss)},
          {"train/value_loss",      static_cast<double>(result.value_loss)},
          {"train/entropy",         static_cast<double>(result.entropy)},
          {"train/infonce_loss",    static_cast<double>(result.infonce_loss)},
          {"curriculum/mastered",   static_cast<double>(mast)},
          {"curriculum/frontier",   static_cast<double>(front)},
          {"curriculum/unknown",    static_cast<double>(unk)},
          {"curriculum/tier",       static_cast<double>(grid.current_tier())},
          {"perf/ticks_per_sec",    fps},
      });
    }

    // ---- Checkpoint ----------------------------------------------------------
    if (!output_dir.empty() && checkpoint_interval > 0 && update % checkpoint_interval == 0) {
      const std::string ckpt = output_dir + "/ckpt_" + std::to_string(update);
      fs::create_directories(ckpt);
      torch::save(actor,  ckpt + "/actor.pt");
      torch::save(value,  ckpt + "/value.pt");
      torch::save(critic, ckpt + "/critic.pt");
      std::ofstream(output_dir + "/latest.txt") << update << "\n";
      std::printf("[pulsar_train] checkpoint → %s\n", ckpt.c_str());
    }
  }

  std::printf("[pulsar_train] done\n");
  return 0;
}
