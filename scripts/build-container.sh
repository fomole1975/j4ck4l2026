#!/bin/sh
# Called only in the Docker build. The runtime image contains native binaries.
set -eu
export CXX=g++
prefix=/opt/pipeline
configure() {
    name=$1
    shift
    cmake -S /build -B "/tmp/$name" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" "$@"
}

profile_targets=
if [ "$1" = profile ]; then
    # On native Linux, /proc/cpuinfo exposes only OS-enabled instruction sets.
    # Train each supported fast path so older scoring CPUs can use a profile too.
    features=$(sed -n 's/^flags[[:space:]]*: //p' /proc/cpuinfo | sed -n '1p')
    has() { case " $features " in *" $1 "*) return 0;; *) return 1;; esac; }
    if has avx2 && has bmi1 && has bmi2 && has popcnt && has pclmulqdq; then
        profile_targets=shopping-sessions-avx2
        if has avx512f && has avx512bw; then
            profile_targets="$profile_targets shopping-sessions-avx512"
            if has avx512_vbmi2; then profile_targets="$profile_targets shopping-sessions-vbmi2"; fi
        fi
    fi
fi

configure standard
if [ -z "$profile_targets" ]; then
    cmake --build /tmp/standard --parallel 4
    cmake --install /tmp/standard --strip
else
    # Compile the dispatcher and the other instruction sets without instrumentation.
    for target in shopping-sessions shopping-sessions-generic shopping-sessions-avx2 shopping-sessions-avx512 shopping-sessions-vbmi2; do
        case " $profile_targets " in *" $target "*) continue;; esac
        cmake --build /tmp/standard --target "$target" --parallel 4
        cmake --install /tmp/standard --component "$target" --strip
    done

    for target in $profile_targets; do
        dir="opt-$target"
        profile="/tmp/profile-$target"
        configure "$dir" "-DCMAKE_CXX_FLAGS=-flto=auto -fno-semantic-interposition -fno-plt -fomit-frame-pointer -fprofile-generate=$profile"
        cmake --build "/tmp/$dir" --target "$target" --parallel 4
        NCPU=1 DATA_DIR=/training OUTPUT_DIR=/tmp/train-output "/tmp/$dir/$target"
        configure "$dir" "-DCMAKE_CXX_FLAGS=-flto=auto -fno-semantic-interposition -fno-plt -fomit-frame-pointer -fprofile-use=$profile -fprofile-correction"
        cmake --build "/tmp/$dir" --target "$target" --parallel 4
        cmake --install "/tmp/$dir" --component "$target" --strip
        echo "Installed GCC profile-guided binary $target"
    done
fi
cmake --build /tmp/standard --target shopping-writer --parallel 4
cmake --install /tmp/standard --component writer --strip
mkdir -p "$prefix/lib"
arrow=$(python3 -c 'import pyarrow;print(pyarrow.get_library_dirs()[0])')
cp "$arrow"/libarrow.so.* "$arrow"/libparquet.so.* "$prefix/lib/"
for library in libstdc++.so.6 libgcc_s.so.1; do
    cp -L "$(g++ -print-file-name="$library")" "$prefix/lib/$library"
done
