#ifndef LLVM_WAZUHL_ACTION_H
#define LLVM_WAZUHL_ACTION_H

#include "llvm/IR/PassManagerInternal.h"
#include <string>
#include <vector>

namespace llvm {

class Module;
class PassInfo;

namespace wazuhl {

  using ActionResult = detail::PassConcept<Module, AnalysisManager<Module>>;
  class Action;
  using ActionList = std::vector<Action>;

  class Action {
  private:
    using PassConstructorT = std::function<ActionResult *()>;
    std::string Name;
    PassConstructorT PassConstructor;
  public:
    Action(const std::string &name, PassConstructorT ctor) :
      Name(name), PassConstructor(ctor) {}

    ActionResult *takeAction() const {
      return PassConstructor();
    }

    const StringRef getName() const {
      return Name;
    }

    static ActionList getAllPossibleActions();
    static const Action &getActionByName(const StringRef);
  };
}
}

#endif /* LLVM_WAZUHL_ACTION_H */
