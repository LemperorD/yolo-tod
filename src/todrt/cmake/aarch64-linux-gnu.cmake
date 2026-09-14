# aarch64-linux-gnu.cmake —— 在 x86 构建机上交叉编译到 Jetson（Orin/Xavier）
#
# 用法：
#   cmake -S src/todrt -B build/aarch64 \
#         -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
#         -DTODRT_WITH_TENSORRT=ON \
#         -DTensorRT_ROOT=/opt/trt-aarch64 \
#         -DCUDAToolkit_ROOT=/opt/cuda-aarch64
#
# 前置条件（两条路任选其一）：
#   A) NVIDIA 官方交叉编译容器（推荐，头/库都齐）：
#        docker run --rm -it -v $PWD:/work nvcr.io/nvidia/tensorrt:<tag>-cross-aarch64
#      容器内 CMAKE_TOOLCHAIN_FILE 通常已配置好，直接用即可。
#   B) 手工准备：从 JetPack 里取出 aarch64 的 CUDA + TensorRT 头/库（含 libnvonnxparser.so）
#      以及一个 aarch64 的 gcc 工具链（Ubuntu: apt install g++-aarch64-linux-gnu）。
#
# 链接共享库时如果报 "file truncated"（aarch64 链接 32 位偏移限制），
# 加上 -Wl,--no-keep-memory 或改用 lld，或分多个 so 拆分。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 工具链前缀：按需改成 aarch64-none-linux-gnu- / aarch64-linux-gnu-
set(TODRT_CROSS_PREFIX "aarch64-linux-gnu-" CACHE STRING "交叉工具链前缀")
set(CMAKE_C_COMPILER   "${TODRT_CROSS_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${TODRT_CROSS_PREFIX}g++")

# Jetson Orin（Ampere, sm_87）；Xavier 用 72，Orin NX/Nano 也是 87
set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "目标 GPU 架构")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 目标机 sysroot（JetPack 的根文件系统）。若使用交叉编译容器可留空。
if(DEFINED ENV{TODRT_SYSROOT})
  set(CMAKE_SYSROOT "$ENV{TODRT_SYSROOT}")
endif()
