#!/bin/sh
# Compile the ns-3 scorer outside the ns-3 tree, against an existing ns-3.48 build.
#   NS3_BUILD: build directory of an optimised or release ns-3.48 with the lte module
# macOS (clang, libc++, .dylib) and Linux (gcc 12 or later, .so) are both handled; the build
# profile is read from the library names (libns3.48-core-optimized.* or libns3.48-core.*).
set -e
cd "$(dirname "$0")"
: "${NS3_BUILD:?set NS3_BUILD to an optimised or release ns-3.48 build directory}"
mkdir -p bin
for d in lib lib64; do [ -d "$NS3_BUILD/$d" ] && LIB="$NS3_BUILD/$d"; done
: "${LIB:?no lib or lib64 directory in $NS3_BUILD}"
case "$(uname)" in
    Darwin) EXT=dylib; STD=-stdlib=libc++ ;;
    *)      EXT=so;    STD= ;;
esac
if ls "$LIB"/libns3.48-core-optimized.$EXT >/dev/null 2>&1; then
    PROFILE=OPTIMIZED; LIBS=$(ls "$LIB"/libns3.48-*-optimized.$EXT)
elif ls "$LIB"/libns3.48-core.$EXT >/dev/null 2>&1; then
    PROFILE=RELEASE; LIBS=$(ls "$LIB"/libns3.48-*.$EXT)
else
    echo "no optimised or release ns-3.48 libraries in $LIB" >&2; exit 1
fi
# shellcheck disable=SC2086
${CXX:-c++} -DNS3_BUILD_PROFILE_$PROFILE $STD -O3 -DNDEBUG -std=c++23 -Wall \
    -I"$NS3_BUILD/include" uav-score.cc -o bin/uav-score \
    -Wl,-rpath,"$LIB" $LIBS
