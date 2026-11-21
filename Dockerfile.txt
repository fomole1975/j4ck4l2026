FROM gcc:15-bookworm@sha256:9ca91b05c7b07d2979f16413e8b2cd6ec8a7c80ffca4121ccab0aeba33f90460 AS toolchain
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv cmake ninja-build && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/tools && /opt/tools/bin/pip install --no-cache-dir --no-deps pyarrow==23.0.1
ENV PATH=/opt/tools/bin:$PATH

FROM rust:1.88.0-slim-bookworm@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 AS training-data
ADD --checksum=sha256:13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip /tmp/generator.zip
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/* \
 && unzip -q /tmp/generator.zip -d /tmp \
 && cargo build --offline --release --manifest-path /tmp/generator/Cargo.toml \
 && /tmp/generator/target/release/generate --seed 1 --events 1000000 --clicks 250000 --threads 2 --output /training

FROM toolchain AS source
WORKDIR /build
COPY CMakeLists.txt .
COPY src/ src/
COPY scripts/build-container.sh /build/compile.sh

FROM source AS built-amd64-amd64
RUN --mount=type=bind,from=training-data,source=/training,target=/training sh compile.sh profile

# Native ARM and cross-builds use the normal release compiler without training
# under an emulator. The amd64 release includes all four ISA variants.
FROM source AS untrained
RUN sh compile.sh release
FROM untrained AS built-arm64-arm64
FROM untrained AS built-arm64-amd64
FROM untrained AS built-amd64-arm64

FROM built-${BUILDARCH}-${TARGETARCH} AS installed
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libjemalloc2 && rm -rf /var/lib/apt/lists/*
COPY --from=installed /opt/pipeline /opt/pipeline
ENV LD_LIBRARY_PATH=/opt/pipeline/lib
RUN ldd /opt/pipeline/bin/shopping-sessions* > /tmp/libraries && ! grep -q 'not found' /tmp/libraries && rm /tmp/libraries
ENV LD_PRELOAD=libjemalloc.so.2 MALLOC_CONF=thp:always,metadata_thp:always,narenas:4
USER 1000:1000
CMD ["/opt/pipeline/bin/shopping-sessions"]
