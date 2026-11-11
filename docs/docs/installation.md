# Installation

Platform requirements and source installation for RaBitQ Library **0.8.3**.
For the first Python example, see [Quick Start](quick_start.md).

## Requirements

| Platform | CPU baseline |
| --- | --- |
| Linux x86-64 | AVX2 and FMA |
| Linux ARM64 (AArch64) | NEON |
| Windows x86-64 | AVX2 and FMA |
| macOS 14+ ARM64 (Apple Silicon) | NEON |

Source builds require a C++17 compiler and CMake 3.20 or newer. Full ANN builds
also require OpenMP; core/FastScan can disable it with `RABITQ_USE_OPENMP=OFF`.
Windows uses MSVC (Visual Studio 2026 with the Desktop development with C++
workload and CMake 4.2+ for its generator). Apple Silicon uses AppleClang and
an external OpenMP runtime such as Homebrew `libomp`.

Linux ARM64 source builds run on native AArch64 CI.

<details>
<summary>CPU dispatch details</summary>

On x86-64, optional AVX-512 kernels require AVX2, FMA, and AVX-512F/BW/DQ;
MSVC builds also require AVX-512VL/CD. Popcount-specific kernels additionally
require AVX-512 VPOPCNTDQ. Detection checks CPU features and OS support for
the required register state. Missing AVX-512 features select the AVX2 backend.

On ARM64, the build excludes x86 kernels and uses NEON and portable scalar
implementations. Standard FastScan requires AVX2/FMA, a supported AVX-512
backend, or NEON; scalar fallbacks do not remove that requirement. A pure-core
build can use scalar kernels without the FastScan CPU baseline.

</details>

## Get the source

For C++ source builds, obtain the 0.8.3 checkout once. With Git
installed, run:

```bash
git clone --branch v0.8.3 --depth 1 https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
```

Run the source-build commands below from this repository root. If you already
have the matching checkout, use it directly.


