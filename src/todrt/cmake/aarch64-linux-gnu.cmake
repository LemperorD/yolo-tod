# aarch64-linux-gnu.cmake —— 在 x86_64 构建机上交叉编译 todrt 到 aarch64
#
# 适用目标：Jetson Orin/Xavier（TensorRT + DLA）、RK3588（RKNN NPU）、
#           通用 ARM 笔记本/开发板（CPU 后路）。
#
# 用法：
#   # RK3588（NPU）
#   cmake -S src/todrt -B build/rk3588 \
#         -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
#         -DTODRT_WITH_RKNN=ON -DTODRT_RKNN_ROOT=/opt/rknn-aarch64
#
#   # Jetson Orin（TensorRT）
#   cmake -S src/todrt -B build/orin \
#         -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
#         -DTODRT_WITH_TENSORRT=ON -DTensorRT_ROOT=/opt/trt-aarch64 \
#         -DCUDAToolkit_ROOT=/opt/cuda-aarch64
#
#   # 只想要一份能在 ARM 上跑的 CPU 版（不需要任何厂商 SDK）
#   cmake -S src/todrt -B build/arm64 \
#         -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
#         -DTODRT_WITH_TENSORRT=OFF
#
# 前置条件（任选其一）：
#   A) NVIDIA 官方交叉编译容器（TensorRT 路线最省事，头/库都齐）：
#        docker run --rm -it -v $PWD:/work nvcr.io/nvidia/tensorrt:<tag>-cross-aarch64
#   B) 手工准备：
#        - aarch64 工具链：  sudo apt install g++-aarch64-linux-gnu
#        - 目标机 sysroot：  从目标机 rsync /lib /usr/lib /usr/include（或用 JetPack 根文件系统）
#        - 厂商 SDK：        从 rknpu2 取 librknnrt.so；从 JetPack 取 TensorRT 头/库
#
# 注意：
#   * **产物不可混用**：交叉编译出来的可执行文件与 .so 只能在 aarch64 上运行；
#     NPU/GPU 相关的库（librknnrt.so / libnvinfer.so）必须用**目标架构**的那一份，
#     不能用 x86 的（find_library 会按 CMAKE_FIND_ROOT_PATH 只搜 sysroot）。
#   * 链接共享库时报 "file truncated" 时加 -Wl,--no-keep-memory 或改用 lld。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 工具链前缀：Debian/Ubuntu 是 aarch64-linux-gnu-；Yocto/Linaro 可能是 aarch64-none-linux-gnu-
set(TODRT_CROSS_PREFIX "aarch64-linux-gnu-" CACHE STRING "交叉工具链前缀")
set(CMAKE_C_COMPILER   "${TODRT_CROSS_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${TODRT_CROSS_PREFIX}g++")
set(CMAKE_AR           "${TODRT_CROSS_PREFIX}ar"      CACHE FILEPATH "")
set(CMAKE_RANLIB       "${TODRT_CROSS_PREFIX}ranlib"  CACHE FILEPATH "")
set(CMAKE_STRIP        "${TODRT_CROSS_PREFIX}strip"   CACHE FILEPATH "")

# 目标 GPU 架构（仅 TensorRT/CUDA 路线用得到）：
#   Orin / Orin NX / Orin Nano = 87，Xavier = 72，Jetson Nano = 53
set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "目标 GPU 架构（Jetson Orin=87, Xavier=72）")

# 只在 sysroot 里找库/头（避免误链到构建机的 x86 库 —— 这是交叉编译最常见的坑）
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 目标机 sysroot（JetPack / 板子根文件系统）。交叉编译容器里通常已配好，可留空。
if(DEFINED ENV{TODRT_SYSROOT})
  set(CMAKE_SYSROOT "$ENV{TODRT_SYSROOT}")
  message(STATUS "todrt: 使用 sysroot = $ENV{TODRT_SYSROOT}")
elseif(EXISTS "/usr/aarch64-linux-gnu")
  set(CMAKE_SYSROOT "/usr/aarch64-linux-gnu")
  message(STATUS "todrt: 自动使用 sysroot = /usr/aarch64-linux-gnu")
endif()

# 让 CMake 的 find_* 也在 sysroot 下的多架构目录里搜（Debian 布局）
list(APPEND CMAKE_FIND_ROOT_PATH
  ${CMAKE_SYSROOT}
  ${CMAKE_SYSROOT}/usr
  ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
