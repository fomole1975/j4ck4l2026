#!/bin/bash
set -eu
export NCPU="$(nproc)" PROFILE=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -yqq --no-install-recommends clang lld llvm libclang-rt-dev >/dev/null
variant=avx2
if grep -qiw avx512bw /proc/cpuinfo; then variant=avx512; fi
target="shopping-sessions-$variant"
build() {
    local name="$1" flags="$2"
    CXX=clang++ cmake -S /source -B "/lab/compile-$name" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_CXX_FLAGS=$flags" -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld >/dev/null
    cmake --build "/lab/compile-$name" --target "$target" --parallel "$NCPU"
}
build base ""
build thin "-flto=thin"
build train "-flto=thin -fprofile-instr-generate"
LLVM_PROFILE_FILE=/lab/branches.profraw NCPU=1 DATA_DIR=/lab/dev OUTPUT_DIR=/lab/profile-output \
  "/lab/compile-train/$target"
llvm-profdata merge /lab/branches.profraw -o /lab/branches.profdata
build pgo "-flto=thin -fprofile-instr-use=/lab/branches.profdata"

for tag in base thin pgo; do
    DATA_DIR=/lab/dev OUTPUT_DIR="/lab/check-$tag" "/lab/compile-$tag/$target"
done
for pair in "base plain" "thin plain" "pgo plain" "pgo snappy" \
            "pgo snappy" "pgo plain" "thin plain" "base plain"; do
    read -r tag codec <<< "$pair"
    echo "compiler-study $tag $codec on $variant"
    time DATA_DIR=/lab/final OUTPUT_DIR=/lab/study-final SHOPPING_CODEC="$codec" "/lab/compile-$tag/$target"
    du -sb /lab/study-final
done
