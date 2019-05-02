#ifndef LLVM_WAZUHL_POLICYEVALUATOR_H
#define LLVM_WAZUHL_POLICYEVALUATOR_H

#include "llvm/Wazuhl/GymInteractor.h"
#include "llvm/Wazuhl/Environment.h"

namespace llvm {
namespace wazuhl {
  //class Environment;

  using State = typename Environment::State;
  using Action = typename Environment::Action;

  class PolicyEvaluator {
  public:
    PolicyEvaluator(Environment &env) : OptimizationEnv(env) {}
    void evaluate();
  private:
    Environment &OptimizationEnv;
  };

  class LearningPolicyEvaluator {
  public:
    LearningPolicyEvaluator(Environment &env) : OptimizationEnv(env) {}
    void evaluate();
  private:
    Environment &OptimizationEnv;
  };

  class GuidedPolicyEvaluator {
  public:
    GuidedPolicyEvaluator(Environment &env, GymInteractor<Action, State> &interactor) : OptimizationEnv(env), Interactor(interactor) {}
    void evaluate();
  private:
    Environment &OptimizationEnv;
    GymInteractor<Action, State> &Interactor;
};
}
}

#endif /* LLVM_WAZUHL_POLICYEVALUATOR_H */
