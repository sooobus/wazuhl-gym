#include "llvm/Wazuhl/PolicyEvaluator.h"
#include "llvm/Wazuhl/Environment.h"
#include "llvm/Wazuhl/ReinforcementLearning.h"

namespace llvm {
namespace wazuhl {

void GuidedPolicyEvaluator::evaluate() {
  auto learner =
      rl::createLearner<rl::GuidedLearning>(OptimizationEnv, Interactor);
  learner.learn();
}

} // namespace wazuhl
} // namespace llvm
