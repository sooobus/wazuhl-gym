#include "llvm/Wazuhl/Manager.h"
#include "llvm/Wazuhl/Environment.h"
#include "llvm/Wazuhl/FeatureCollector.h"
#include "llvm/Wazuhl/PolicyEvaluator.h"
#include "llvm/Wazuhl/GymInteractor.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"

#include <numeric>

using namespace llvm;
using namespace wazuhl;

cl::opt<bool> TrainingPhase("train-wazuhl", cl::desc("Enable Wazuhl training"),
                            cl::Hidden);

namespace {

void registerFeatureCollectors(Module &M, ModuleAnalysisManager &AM) {
  AM.registerPass([] { return ModuleFeatureCollector(); });

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  FAM.registerPass([] { return FunctionFeatureCollector(); });
}

template <class Evaluator> void evaluate(Environment &Env) {
  Evaluator OptimizationEvaluator{Env};
  OptimizationEvaluator.evaluate();
}

template <class Evaluator> void evaluate(Environment &Env, GymInteractor<typename Environment::Action, typename Environment::State> &Interactor) {
  Evaluator OptimizationEvaluator{Env, Interactor};
  OptimizationEvaluator.evaluate();
}

void train(Environment &Env) { evaluate<LearningPolicyEvaluator>(Env); }

void guided_train(Environment &Env, GymInteractor<typename Environment::Action, typename Environment::State> &Interactor) { evaluate<GuidedPolicyEvaluator>(Env, Interactor); }

void exploit(Environment &Env) { evaluate<PolicyEvaluator>(Env); }

} // namespace

namespace llvm {
namespace wazuhl {
PreservedAnalyses Manager::run(Module &IR, ModuleAnalysisManager &AM) {
  llvm::errs() << "Calling run with AlreadyRun = " << AlreadyRun;
  if (!AlreadyRun){
    registerFeatureCollectors(IR, AM);
    Environment OptimizationEnv{IR, AM};
    GymInteractor<typename Environment::Action, typename Environment::State> Interactor;

    if (DebugLogging)
      dbgs() << "Starting Wazuhl optimization process.\n";

    llvm::errs() << "Start training\n";
    if (this->Training || TrainingPhase) {
      guided_train(OptimizationEnv, Interactor);
    } //else {
    //exploit(OptimizationEnv);
    //}

    AlreadyRun = true;

  }

  return PreservedAnalyses::none();
}
} // namespace wazuhl
} // namespace llvm
