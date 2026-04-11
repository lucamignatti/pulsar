#pragma once

#include <vector>

#include "pulsar/config/config.hpp"
#include "pulsar/rl/interfaces.hpp"

namespace pulsar {

class FixedTeamSizeMutator final : public StateMutator {
 public:
  explicit FixedTeamSizeMutator(EnvConfig config);

  void apply(EnvState& state, std::uint64_t seed) const override;

 private:
  EnvConfig config_{};
};

class KickoffMutator final : public StateMutator {
 public:
  explicit KickoffMutator(EnvConfig config);

  void apply(EnvState& state, std::uint64_t seed) const override;

 private:
  EnvConfig config_{};
};

class MutatorSequence final : public StateMutator {
 public:
  explicit MutatorSequence(std::vector<StateMutatorPtr> mutators);

  void apply(EnvState& state, std::uint64_t seed) const override;

 private:
  std::vector<StateMutatorPtr> mutators_{};
};

}  // namespace pulsar

