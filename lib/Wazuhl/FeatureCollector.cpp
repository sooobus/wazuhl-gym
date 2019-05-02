#include "llvm/Wazuhl/FeatureCollector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace wazuhl;
using namespace wazuhl::config;

#define DEBUG_TYPE "wazuhl-feature-collector"

namespace {
class CollectorImpl : public InstVisitor<CollectorImpl> {
  friend class InstVisitor<CollectorImpl>;

public:
#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void visit##OPCODE(CLASS &) {                                                \
    CollectedFeatures.push_back(N);                                            \
    ++TotalInsts;                                                              \
  }
#include "llvm/IR/Instruction.def"

  CollectorImpl()
      : TotalInsts(0) {

  }

  std::vector<int> &&getCollectedFeatures() {
    return std::move(CollectedFeatures);
  }

private:
  std::vector<int> CollectedFeatures;
  //RawFeatureVector CollectedFeatures;
  double TotalInsts;
};
} // namespace

namespace llvm {
namespace wazuhl {
std::vector<int> FunctionFeatureCollector::run(Function &F,
                                               FunctionAnalysisManager &) {
  CollectorImpl Collector;
  Collector.visit(F);
  return Collector.getCollectedFeatures();
}

RawIRFeatures ModuleFeatureCollector::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  RawIRFeatures Result;

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (Function &F : M) {
    if (not F.empty()) {
      Result.addVectorForFunction(&F,
                                  FAM.getResult<FunctionFeatureCollector>(F));
    }
  }

  return Result;
}

AnalysisKey FunctionFeatureCollector::Key;
AnalysisKey ModuleFeatureCollector::Key;
} // namespace wazuhl
} // namespace llvm
