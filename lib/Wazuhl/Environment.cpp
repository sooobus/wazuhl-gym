#include "llvm/Wazuhl/Environment.h"

#include "llvm/Wazuhl/NormalizedTimer.h"

using namespace llvm;

namespace {
template <class PassT>
inline void runPass(PassT &Pass, Module &IR, ModuleAnalysisManager &AM,
                    PreservedAnalyses &PA) {
  PreservedAnalyses PassPA = Pass->run(IR, AM);

  // Update the analysis manager as each pass runs and potentially
  // invalidates analyses.
  AM.invalidate(IR, PassPA);

  // Finally, intersect the preserved analyses to compute the aggregate
  // preserved set for this pass manager.
  PA.intersect(std::move(PassPA));
}
} // namespace

namespace llvm {
namespace wazuhl {

Environment::Environment(Module &IR, ModuleAnalysisManager &AM)
    : IR(IR), AM(AM), Current(), PA(PreservedAnalyses::all()) {
  llvm::errs() << "Wazuhl has " << Action::getAllPossibleActions().size()
               << " actions to choose from\n";
  InitialIRState = AM.getResult<ModuleFeatureCollector>(IR);
  updateState();
}

Environment::State Environment::getState() { return Current; }

void Environment::updateState() {
  auto IRFeatures = AM.getResult<ModuleFeatureCollector>(IR);
  Current.setIRFeatures(IRFeatures.data());
  Current.setTime(NormalizedTimer::getTime());
}

void Environment::takeAction(const Action &A) {
  llvm::errs() << "Wazuhl is taking action '" << A.getName() << "' with index "
               << A.getIndex() << "\n";
  ++nTakenActions;
  // terminate actions taking if MaxAllowedActions already taken
  //if (nTakenActions == maxAllowedActions) {
  //  Terminated = true;
  //}
  if (A.getName() == "terminal"){
    llvm::errs() << "Wazuhl is making env terminated \n";
    Terminated = true;
  }
  else if (A.getName() == "empty"){
    llvm::errs() << "Wazuhl is not doing any passes\n";
  }
  else{
    auto Pass = A.takeAction();
    runPass(Pass, IR, AM, PA);
  }

  Current.recordAction(A);
  updateState();
}

bool Environment::isInTerminalState() {
  // Terminated is defined not really by the
  // state, but by the last taken action
  return Terminated;
}

double Environment::getReward() {
  // TODO: add checking the time threshold
  // and give a fine if the threshold was met

  // no reward till the end of the episode
  return 0.0;
}

PreservedAnalyses Environment::getPreservedAnalyses() { return PA; }

} // namespace wazuhl
} // namespace llvm
