# Shopping Sessions Pipeline

An ultra-high-performance C++23 batch analytics engine designed to process tens of millions of e-commerce events, ad clicks, and identity links into daily user and campaign attribution Parquet tables (`user_daily` and `campaign_daily`).

Built with zero-copy memory-mapped parsing, custom AVX2/AVX-512 SIMD sorting networks, L1-cache-resident session aggregation, Profile-Guided Optimization (PGO), and a standalone single-syscall Parquet writer.

---

## Architecture & Key Optimizations

### 1. Zero-Copy Memory-Mapped I/O (`src/input.cpp`)
- Maps input CSV shards directly into virtual memory via `mmap` with kernel readahead (`MADV_SEQUENTIAL | MADV_WILLNEED`) and Transparent Huge Page hints (`MADV_HUGEPAGE`) to eliminate minor page-fault storms on multi-gigabyte datasets.
- Splits physical files along newline boundaries into cache-friendly chunks for lock-free parallel ingestion.

### 2. Overlapped Parallel Parsing & Identity Resolution (`src/engine.cpp`)
- **Single-Barrier Ingestion:** Resolves user identity graphs (`identities.csv`) concurrently with parsing `events` and `clicks` in a single thread-pool dispatch, eliminating intermediate synchronization barriers.
- **Packed ASCII Keys:** Packs short alphanumeric identifiers (`user_id`, `event_id`, `click_id`) into 64-bit integer representations for single-cycle hashing and comparison, falling back to string interning only when necessary.
- **Shard-Local Deduplication:** Deduplicates events by earliest ingest timestamp across partitioned literal and general candidate tables using software prefetching (`__builtin_prefetch`) to hide L3 cache latency.

### 3. Custom SIMD Sorting Kernels (`src/sessions.hpp`, `src/time_sort32.hpp`)
- Uses hand-tuned bitonic/Bose-Nelson SIMD sorting networks (`sort_small`, `time_sort32::sort`, and AVX-512 `sort_clicks`) to sort per-user event timestamps and click offsets in registers before falling back to scalar sorts for large histories.
- Multi-ISA dispatch selects the optimal binary variant (`generic`, `avx2`, `avx512`, `vbmi2`) at runtime.

### 4. L1-Cache-Resident Sessionization & Attribution (`src/engine.cpp`)
- **Inline Unlinked Histories:** Sessionizes unlinked users (`person == absent`) directly inside `group_by_raw_id` without materializing intermediate span buffers.
- **Compact `BucketSummary`:** Separates `{minimum, maximum}` timestamp bounds (8 bytes per bin) from auxiliary counts and session indices so the active 30-minute session gap bucket table (`2,387` bins $\approx 19\text{ KB}$) fits entirely within a 32 KB L1 data cache.
- Performs exact multi-day boundary stitching, last-touch / assisted campaign attribution, and top-category tie-breaking in a single pass per user.

### 5. Single-Syscall Native Parquet Writer (`src/run_parquet.cpp`, `src/output.cpp`)
- Serializes Thrift Compact Protocol headers, dictionary pages, RLE definition levels, and plain/dictionary-encoded column chunks into a pre-reserved in-memory buffer, flushing each output partition (`part-*.parquet`) to disk in a **single POSIX `write()` syscall**.
- Dynamically falls back to an Arrow-backed writer (`dlopen`) only if page limits or non-default codecs (`SHOPPING_CODEC=snappy`) are requested.

### 6. Multi-ISA PGO Container Build (`Dockerfile`, `CMakeLists.txt`, `scripts/build-container.sh`)
- Compiles with GCC 15 using `-O3 -flto=auto -fno-semantic-interposition -fno-plt -fomit-frame-pointer` and `-Wa,-mbranches-within-32B-boundaries`.
- Runs an automated Profile-Guided Optimization (PGO) training pass during `linux/amd64` image builds using a deterministic Rust-generated synthetic workload.
- Ships in a minimal `debian:bookworm-slim` runtime image with `libjemalloc2` (`MALLOC_CONF=thp:always,metadata_thp:always,narenas:4`).

---

## Repository Layout

```text
├── CMakeLists.txt               # Multi-ISA C++23 build configuration
├── Dockerfile                   # Multi-stage PGO & minimal runtime container build
├── scripts/
│   └── build-container.sh       # PGO instrumentation, training run, and final build script
├── src/
│   ├── engine.cpp               # Parallel ingestion, deduplication, sessionization & attribution
│   ├── input.cpp                # mmap file reader with madvise huge-page/readahead hints
│   ├── output.cpp               # Partitioning & writer dispatch
│   ├── run_parquet.cpp          # Native Thrift/Parquet serializer & single-syscall writer
│   ├── sessions.hpp             # AVX2/AVX-512 event & click sorting networks
│   └── time_sort32.hpp          # 32-bit timestamp offset SIMD sorting kernels
└── tests/
    └── sort-kernels.cpp         # Exhaustive bounds-checked SIMD kernel test suite
```

---

## Building & Running

### Build the Docker Image
*(Note: If your Dockerfile is named `Dockerfile.txt`, pass `-f Dockerfile.txt` or rename it to `Dockerfile`.)*

```bash
docker build --platform linux/amd64 -t shopping-sessions:latest .
```

### Run the Pipeline
Mount your input directory (`DATA_DIR`) and output directory (`OUTPUT_DIR`):

```bash
docker run --rm --platform linux/amd64 \
  -v /path/to/input:/data:ro \
  -v /path/to/output:/output \
  -e DATA_DIR=/data \
  -e OUTPUT_DIR=/output \
  shopping-sessions:latest
```

#### Environment Variables
| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | *(Required)* | Directory containing input `events`, `clicks`, `identities`, and dimension files. |
| `OUTPUT_DIR` | *(Required)* | Destination directory where `user_daily/` and `campaign_daily/` Parquet partitions are written. |
| `NCPU` | Host CPU count | Number of worker threads to spawn. |
| `SHOPPING_CODEC` | `plain` | Parquet compression codec (`plain` for native fast path, or `snappy` via Arrow fallback). |

---

## Testing & Verification

### 1. SIMD Sort Kernel Suite
Verify `sort_small`, `time_sort32::sort`, and AVX-512 `sort_clicks` across all vector widths and boundary alignments:

```bash
docker build --platform linux/amd64 --target toolchain -t shopping-toolchain .
docker run --rm -e CXX=g++ -v "$PWD:/src:ro" -w /src shopping-toolchain sh .github/scripts/test-sort-kernels.sh
```

### 2. End-to-End Smoke Test
Extract the deterministic 1M-event / 250k-click dataset from the `training-data` build stage and validate output schemas:

```bash
docker build --platform linux/amd64 --target training-data -t shopping-training .
docker create --name tmp-train shopping-training
docker cp tmp-train:/training ./training-data
docker rm tmp-train

mkdir -p ./training-out
docker run --rm --platform linux/amd64 \
  -v "$PWD/training-data:/data:ro" \
  -v "$PWD/training-out:/output" \
  -e DATA_DIR=/data \
  -e OUTPUT_DIR=/output \
  shopping-sessions:latest
```