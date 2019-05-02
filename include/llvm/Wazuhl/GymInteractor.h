#ifndef LLVM_GYMINTERACTOR_H
#define LLVM_GYMINTERACTOR_H

#include "llvm/Support/raw_ostream.h"

#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>



using bsoncxx::builder::stream::close_array;
using bsoncxx::builder::stream::close_document;
using bsoncxx::builder::stream::document;
using bsoncxx::builder::stream::finalize;
using bsoncxx::builder::stream::open_array;
using bsoncxx::builder::stream::open_document;
using bsoncxx::document::view;
using mongocxx::pipeline;

namespace {
//using ExperienceUnit = llvm::wazuhl::ExperienceReplay::ExperienceUnit;
//using StateType = llvm::wazuhl::ExperienceReplay::State;
//using ResultType = llvm::wazuhl::ExperienceReplay::Result;
//using RecalledExperience = llvm::wazuhl::ExperienceReplay::RecalledExperience;
using MongifiedExperience = bsoncxx::document::value;



template <typename Type, class Doc, class IterableT>
void addArray(Doc &Destination, const llvm::StringRef ArrayName,
              const IterableT &List) {
  auto array = Destination << ArrayName << open_array;
  for (auto element : List) {
    array = array << (Type)element;
  }
  array << close_array;
}

/*
template <class IterableT>
void fillStateArray(IterableT &List, const llvm::StringRef ArrayName,
                    view &DocumentView) {
  auto element = DocumentView[ArrayName.str()];
  bsoncxx::array::view array = element.get_array();
  unsigned i = 0;
  for (auto value : array) {
    List[i++] = value.get_double();
  }
}

template <class IterableT>
void fillContextArray(IterableT &List, const llvm::StringRef ArrayName,
                      view &DocumentView) {
  auto element = DocumentView[ArrayName.str()];
  bsoncxx::array::view array = element.get_array();
  for (auto value : array) {
    List.push_back(static_cast<unsigned>(value.get_int32()));
  }
}
 */
} // anonymous namespace



namespace llvm {
namespace wazuhl {

template<class Action, class State>
class GymInteractor {
public:
  GymInteractor(){
    ExternalMemory = MongoClient["wazuhl"];
    createCollection(PossibleActions, "possible_actions");
    createCollection(ActionsQueue, "actions");
    createCollection(StatesQueue, "states");
    sendPossibleActions();
  }
  ~GymInteractor() {}

  std::pair<Action, int> getNextAction(){
    std::string next_action;
    int i = -1;
    auto maybe_result =
        ActionsQueue.find_one({});
    while(!maybe_result){
      maybe_result =
          ActionsQueue.find_one({});
    }
    if(maybe_result) {
      auto doc = maybe_result->view();
      i = doc["index"].get_int32();
      auto view = doc["action"].get_utf8().value;
      next_action = view.to_string();
    }
    ActionsQueue.delete_one(document{} << "index" << i << finalize);
    return {Action::getActionByName(next_action), i};
  }

  void sendState(State S, bool Done, int id){
    auto to_upload = document{};
    to_upload << "index" << id;
    addArray<int>(to_upload, "state", S);
    to_upload << "done" << Done;
    StatesQueue.insert_one(to_upload.view());
  }

private:
  void createCollection(mongocxx::collection &collection,
                                       const std::string &name){
    if (!ExternalMemory.has_collection(name)) {
      ExternalMemory.create_collection(name);
    }
    collection = ExternalMemory.collection(name);
  }

  void sendPossibleActions(){
    auto to_upload = document{};
    std::vector<std::string> Names;
    for(auto& A : Action::getAllPossibleActions()){
      Names.push_back(A.getName());
    }
    addArray<std::string>(to_upload, "actions", Names);
    PossibleActions.insert_one(to_upload.view());
  }

  mongocxx::instance instance{};
  mongocxx::client MongoClient{mongocxx::uri{"mongodb://mongo:27017"}};
  mongocxx::database ExternalMemory;
  mongocxx::collection ActionsQueue;
  mongocxx::collection PossibleActions;
  mongocxx::collection StatesQueue;

};

}
}

#endif //LLVM_GYMINTERACTOR_H
