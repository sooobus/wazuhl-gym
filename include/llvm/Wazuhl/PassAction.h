#ifndef LLVM_WAZUHL_PASSACTION_H
#define LLVM_WAZUHL_PASSACTION_H

#include "llvm/IR/PassManagerInternal.h"
#include <string>
#include <vector>

namespace llvm {

class Module;
class PassInfo;

namespace wazuhl {

  using PassActionResult = detail::PassConcept<Module, AnalysisManager<Module>>;
  class PassAction;
  using PassActionList = std::vector<PassAction>;

  class PassAction {
  private:
    using PassConstructorT = std::function<PassActionResult *()>;

    const std::string Name;
    const PassConstructorT PassConstructor;
    const unsigned Index;
  public:
    PassAction(const std::string &name, PassConstructorT ctor, unsigned index) :
      Name(name), PassConstructor(ctor), Index(index) {}

    PassActionResult *takeAction() const {
      return PassConstructor();
    }

    const StringRef getName() const {
      return Name;
    }

    unsigned getIndex() const {
      return Index;
    }

    static PassActionList getAllPossibleActions();
    static PassAction getActionByIndex(unsigned Index);
    static const PassAction &getActionByName(const StringRef);
  };
}
}

#endif /* LLVM_WAZUHL_PASSACTION_H */
