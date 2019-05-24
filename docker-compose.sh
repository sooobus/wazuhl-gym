#!/usr/bin/env bash

wget https://github.com/mongodb/mongo-c-driver/releases/download/1.13.0/mongo-c-driver-1.13.0.tar.gz
tar xzf mongo-c-driver-1.13.0.tar.gz
rm mongo-c-driver-1.13.0.tar.gz
git clone https://github.com/mongodb/mongo-cxx-driver.git --branch releases/stable --depth 1
git clone https://github.com/BVLC/caffe.git
git clone https://github.com/SavchenkoValeriy/wazuhl-llvm-test-suite.git suites/llvm-test-suite
cp CMakeLists_for_mb.txt suites/llvm-test-suite/MicroBenchmarks/CMakeLists.txt
git clone https://github.com/SavchenkoValeriy/wazuhl-clang tools/clang
docker-compose build