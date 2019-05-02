#ifndef LLVM_WAZUHL_FEATURECOLLECTOR_H
#define LLVM_WAZUHL_FEATURECOLLECTOR_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Wazuhl/Config.h"

#include <utility>

namespace llvm {
namespace wazuhl {


class RawIRFeatures {
public:
  RawIRFeatures() : Matrix(0){};

  void addVectorForFunction(Function *F, const std::vector<int> &V) {
    Matrix.insert(Matrix.end(), V.begin(), V.end());
  }

  std::vector<int> data(){
    return Matrix;
  }


private:
  std::vector<int> Matrix;

};

class FunctionFeatureCollector
    : public AnalysisInfoMixin<FunctionFeatureCollector> {
  friend AnalysisInfoMixin<FunctionFeatureCollector>;
  static AnalysisKey Key;

public:
  using Result = std::vector<int>;
  Result run(Function &F, FunctionAnalysisManager &AM);
};

class ModuleFeatureCollector
    : public AnalysisInfoMixin<ModuleFeatureCollector> {
  friend AnalysisInfoMixin<ModuleFeatureCollector>;
  static AnalysisKey Key;

public:
  using Result = RawIRFeatures;
  Result run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace wazuhl
} // namespace llvm

#endif /* LLVM_WAZUHL_FEATURECOLLECTOR_H */
