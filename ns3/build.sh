#!/bin/sh
# Compile the ns-3 scorer outside the ns-3 tree, against an existing optimised ns-3.48 build.
#   NS3_BUILD: build directory of an optimised ns-3.48 with the lte module
set -e
cd "$(dirname "$0")"
: "${NS3_BUILD:?set NS3_BUILD to an optimised ns-3.48 build directory}"
mkdir -p bin
c++ -DNS3_BUILD_PROFILE_OPTIMIZED -stdlib=libc++ -O3 -DNDEBUG -std=c++23 -Wall \
    -I"$NS3_BUILD/include" uav-score.cc -o bin/uav-score \
    -Wl,-rpath,"$NS3_BUILD/lib" "$NS3_BUILD"/lib/libns3.48-*-optimized.dylib
