#!/bin/sh
set -eu
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "Skipping x86 sort kernels on $(uname -m)"; exit 0 ;;
esac

case "$(uname -s)" in
    Linux) flags=$(sed -n '/^flags[[:space:]]*:/ { p; q; }' /proc/cpuinfo) ;;
    Darwin) flags=$(sysctl -n machdep.cpu.leaf7_features | tr '[:upper:]' '[:lower:]') ;;
    *) echo "CPU feature detection is unavailable on $(uname -s)"; exit 0 ;;
esac

cd "$(dirname "$0")/../.."
binary=$(mktemp)
trap 'rm -f "$binary"' EXIT
for isa in avx2 avx512f; do
    case " $flags " in *" $isa "*) ;; *) continue ;; esac
    echo "Testing $isa sort kernels"
    "${CXX:-clang++}" -std=c++20 -O3 -I src "-m$isa" tests/sort-kernels.cpp -o "$binary"
    "$binary"
done
