find_package(Caffe REQUIRED CONFIG)

find_package(libmongocxx REQUIRED)
find_package(libbsoncxx REQUIRED)
#find_package(PkgConfig)
#pkg_check_modules(LIBBSONXX REQUIRED libbsoncxx)
#pkg_check_modules(LIBMONGOCXX REQUIRED libmongocxx)

get_target_property(Caffe_INCLUDE_DIRS caffe INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(Caffe_DEFINITIONS caffe INTERFACE_COMPILE_DEFINITIONS)

include_directories(
  SYSTEM
  ${LLVM_MAIN_SRC_DIR}/lib/Passes
  ${Caffe_INCLUDE_DIRS}
  ${LIBMONGOCXX_INCLUDE_DIRS}
  ${LIBBSONCXX_INCLUDE_DIRS}
  )

link_directories(
  ${LIBMONGOCXX_LIBRARIES}
  ${LIBBSONCXX_LIBRARIES}
  )

add_definitions(
  -DGOOGLE_PROTOBUF_NO_RTTI
  -DBOOST_NO_EXCEPTIONS
  -DBOOST_NO_RTTI
  )

add_llvm_library(LLVMWazuhl
  Config.cpp
  Environment.cpp
  FeatureCollector.cpp
  Manager.cpp
  NormalizedTimer.cpp
  PassAction.cpp
  PolicyEvaluator.cpp
  Random.cpp

  LINK_LIBS
  ${Caffe_LIBRARIES}
  ${LIBMONGOCXX_LIBRARIES}
  ${LIBBSONCXX_LIBRARIES}
  )

set(MINIBATCH_SIZE 32)
set(EXPERIENCE_SIZE 3000)
set(NUMBER_OF_FEATURES 1025)
set(NUMBER_OF_ACTIONS 107)
set(HIDDEN_SIZE_1 256)
set(HIDDEN_SIZE_2 128)
set(ACTION_HIDDEN_SIZE 32)
set(CONTEXT_SIZE 30)
math(EXPR CONTEXT_ALPHABET_SIZE "${NUMBER_OF_ACTIONS}+1")
set(CONTEXT_EMBEDDING_SIZE 16)
set(CONTEXT_LSTM_SIZE 32)

configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Wazuhl/Config.h.in
  ${LLVM_INCLUDE_DIR}/llvm/Wazuhl/Config.h
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/model.prototxt.in
  ${CMAKE_CURRENT_BINARY_DIR}/model.prototxt
  )

set_property(TARGET LLVMWazuhl APPEND_STRING PROPERTY
  COMPILE_FLAGS " -fexceptions")

target_compile_definitions(LLVMWazuhl
  PRIVATE ${Caffe_DEFINITIONS})

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/model.prototxt
  solver.prototxt
  DESTINATION wazuhl
  )
