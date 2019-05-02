#ifndef LLVM_WAZUHL_REINFORCEMENTLEARNING_H
#define LLVM_WAZUHL_REINFORCEMENTLEARNING_H

#include "llvm/Wazuhl/Random.h"


#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>



namespace llvm {
namespace wazuhl {
namespace rl {
template <class EnvironmentT, class QType, class PolicyT> class QLearning {
public:
  using Action = typename EnvironmentT::Action;
  using State = typename EnvironmentT::State;
  QLearning(EnvironmentT &Environment, QType &Q, const PolicyT &Policy,
            double alpha, double gamma)
      : Environment(Environment), Q(Q), Policy(Policy), alpha(alpha),
        gamma(gamma) {}

  void learn() {
    State S = Environment.getState();
    while (!Environment.isInTerminalState()) {
      //llvm::errs() << "COUNT " << ApprovedRecords.count_documents({}) << "\n";
      Action A = Policy.pick(S);
      Environment.takeAction(A);
      State newS = Environment.getState();
      auto R = Environment.getReward();
      Q(S, A) = Q(S, A) + alpha * (R + gamma * max(Q(newS)) - Q(S, A));
      S = newS;
    }
  }

private:
  EnvironmentT &Environment;
  QType &Q;
  const PolicyT &Policy;
  double alpha, gamma;
};

template <class EnvironmentT, class QType, class PolicyT> class NonLearning {
public:
  using Action = typename EnvironmentT::Action;
  using State = typename EnvironmentT::State;
  NonLearning(EnvironmentT &Environment, QType &Q, const PolicyT &Policy)
      : Environment(Environment), Q(Q), Policy(Policy) {}

  void learn() {
    State S;
    do {
      S = Environment.getState();
      Action A = Policy.pick(S);
      Environment.takeAction(A);
    } while (!Environment.isInTerminalState());
  }

private:
  EnvironmentT &Environment;
  QType &Q;
  const PolicyT &Policy;
};


template <class EnvironmentT, class InteractorT> class GuidedLearning {
public:
  using Action = typename EnvironmentT::Action;
  using State = typename EnvironmentT::State;
  GuidedLearning(EnvironmentT &Environment, InteractorT &Interactor)
      : Environment(Environment), Interactor(Interactor) {}

  void learn() {
    State S = Environment.getState();
    Interactor.sendState(S, false, -1);
    auto Done = Environment.isInTerminalState();
    while(!Done) {
      auto action_id = Interactor.getNextAction();
      Action A = action_id.first;
      int i = action_id.second;
      Environment.takeAction(A);
      S = Environment.getState();
      Done = Environment.isInTerminalState();
      llvm::errs() << "DONE " << Done << "\n";
      Interactor.sendState(S, Done, i);
    }
    llvm::errs() << "Learning finished\n";
  }

private:
  EnvironmentT &Environment;
  InteractorT &Interactor;
};


template <template <class...> class Learner, class EnvironmentT,
          class FunctionT, class PolicyT, class... Args>
Learner<EnvironmentT, FunctionT, PolicyT>
createLearner(EnvironmentT &Environment, FunctionT &ValueFunction,
              const PolicyT &Policy, Args... args) {
  return Learner<EnvironmentT, FunctionT, PolicyT>{Environment, ValueFunction,
                                                   Policy, args...};
}

template <template <class...> class Learner, class EnvironmentT, class InteractorT,
          class... Args>
Learner<EnvironmentT, InteractorT>
createLearner(EnvironmentT &Environment, InteractorT &Interactor,
              Args... args) {
  return Learner<EnvironmentT, InteractorT>{Environment, Interactor, args...};
}


namespace policies {
template <class Function> class Greedy {
public:
  using Action = typename Function::Action;
  using State = typename Function::State;

  Greedy(const Function &valueFuncion) : value(valueFuncion) {}
  Action pick(const State &s) const { return argmax(value(s)); }

private:
  const Function &value;
};

template <class Function> class Random {
public:
  using Action = typename Function::Action;
  using State = typename Function::State;

  Action pick(const State &s) const {
    static const auto &allActions = Action::getAllPossibleActions();
    return random::pickOutOf(allActions);
  }
};

template <class Function>
class EpsilonGreedy : private Greedy<Function>, private Random<Function> {
public:
  using Action = typename Function::Action;
  using State = typename Function::State;

  EpsilonGreedy(double epsilon, const Function &valueFuncion)
      : Greedy<Function>(valueFuncion), Random<Function>(), epsilon(epsilon) {}
  Action pick(const State &s) const {
    if (random::flipACoin(epsilon))
      return Random<Function>::pick(s);
    return Greedy<Function>::pick(s);
  }

private:
  double epsilon;
};
} // namespace policies
} // namespace rl
} // namespace wazuhl
} // namespace llvm

#endif /* LLVM_WAZUHL_REINFORCEMENTLEARNING_H */
