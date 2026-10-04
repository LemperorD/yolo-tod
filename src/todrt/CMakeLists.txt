# todrt —— yolo-tod 的 C++ 推理运行时
#
# **目标平台：Linux，同时支持 x86_64（含 AMD）与 aarch64。** 本文件不含任何
# Windows / MSVC 假设；这两个架构是硬要求，Windows 只用于跑不依赖本库的 Python 部分。
#
# 架构相关开关在 §"架构与指令集" 里统一处理（-march/-mcpu、OpenVINO 的 lib 子目录、
# 交叉编译的 sysroot 等），下面的后端探测都只关心"库在不在"。
#
# 后端（四选多，各自独立开关）：
#   TODRT_WITH_TENSORRT  —— NVIDIA GPU / Jetson DLA      （libnvinfer + libnvonnxparser）
#   TODRT_WITH_RKNN      —— Rockchip RK3588 NPU           （librknnrt.so，aarch64）
#   TODRT_WITH_ORT       —— 通用 CPU 后路（AMD x86 等）   （libonnxruntime）
#   TODRT_WITH_OPENVINO  —— x86 CPU / Intel iGPU/NPU      （OpenVINO Runtime）
#
# 典型用法：
#   # x86_64（含 AMD）笔记本：CPU 后路
#   cmake -S src/todrt -B build/x86 -DTODRT_NATIVE=ON \
#         -DTODRT_WITH_ORT=ON -DTODRT_WITH_OPENVINO=ON
#
#   # aarch64 开发板/笔记本（RK3588）：NPU
#   cmake -S src/todrt -B build/rk3588 -DTODRT_NATIVE=ON -DTODRT_WITH_RKNN=ON
#
#   # Jetson Orin（TensorRT 随 JetPack 装在 /usr）
#   cmake -S src/todrt -B build/orin -DTODRT_WITH_TENSORRT=ON
#
#   # 只想跑装配自检（不需要任何 SDK）
#   cmake -S src/todrt -B build/cpu && cmake --build build/cpu -j && ./build/cpu/todrt_smoke
cmake_minimum_required(VERSION 3.16)
project(todrt LANGUAGES CXX VERSION 0.1.0
        DESCRIPTION "yolo-tod C++ inference runtime (TensorRT / RKNN / ONNX Runtime / OpenVINO)")

# ---------------------------------------------------------------- 选项

option(TODRT_WITH_TENSORRT "编译 TensorRT 后端（NVIDIA GPU / Jetson DLA）" ON)
option(TODRT_WITH_RKNN "编译 RKNN 后端（Rockchip RK3588 NPU，aarch64）" OFF)
option(TODRT_WITH_ORT "编译 ONNX Runtime 后端（通用 CPU 后路）" OFF)
option(TODRT_WITH_OPENVINO "编译 OpenVINO 后端（x86 CPU / Intel iGPU）" OFF)
option(TODRT_BUILD_CLI "构建 todrt_cli 命令行工具" ON)
option(TODRT_BUILD_SMOKE "构建 CPU 自检程序（不需要任何后端）" ON)
option(TODRT_BUILD_SHARED "构建共享库 libtodrt.so（否则静态库）" OFF)
option(TODRT_WARNINGS_AS_ERRORS "把编译警告当错误" OFF)
option(TODRT_INSTALL "生成安装目标（install / 供其它项目 find_package）" ON)
option(TODRT_FETCH_DEPS "缺失依赖时尝试 FetchContent 自动拉取（仅 ONNX Runtime）" OFF)

# 架构与指令集
option(TODRT_NATIVE "针对本机 CPU 优化（-march=native / -mcpu=native）" OFF)
set(TODRT_ARCH "" CACHE STRING "强制目标架构（x86_64 / aarch64 / armv7）；空 = 按 CMAKE_SYSTEM_PROCESSOR 自动探测")

# SDK 搜索路径（自定义安装 / 交叉编译时指定）
set(TODRT_RKNN_ROOT "" CACHE PATH "librknnrt.so 所在目录（含 rknn_api.h 的父目录）")
set(TODRT_ORT_ROOT "" CACHE PATH "ONNX Runtime 解包目录（含 include/ 与 lib/）")
set(TODRT_OPENVINO_ROOT "" CACHE PATH "OpenVINO Runtime 安装目录")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- 架构与指令集
#
# x86_64 与 aarch64 都要能跑，这里把"架构差异"集中成两个变量：
#   TODRT_ARCH      规范化后的架构名
#   TODRT_ARCH_SUBDIRS  该架构下第三方库常见的 lib/ 子目录名（供 find_library 用）
#
# 注意：默认**不加** -march=native。原因很实际 —— NPU/GPU 板子与开发机通常不是
# 同一颗 CPU，native 编出来的 so 拷贝过去会直接 illegal instruction。
# 需要极限性能时用 -DTODRT_NATIVE=ON 显式打开（仅在本机运行才用）。
if(TODRT_ARCH STREQUAL "")
  set(TODRT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
string(TOLOWER "${TODRT_ARCH}" TODRT_ARCH)

if(TODRT_ARCH MATCHES "^(x86_64|amd64|x64)$")
  set(TODRT_ARCH "x86_64")
  set(TODRT_ARCH_SUBDIRS lib lib64 lib/x86_64-linux-gnu)
  set(TODRT_OV_SUBDIRS lib/intel64 lib)
  set(TODRT_ARCH_DESC "x86_64（含 AMD）")
elseif(TODRT_ARCH MATCHES "^(aarch64|arm64)$")
  set(TODRT_ARCH "aarch64")
  set(TODRT_ARCH_SUBDIRS lib lib/aarch64-linux-gnu lib64)
  set(TODRT_OV_SUBDIRS lib/aarch64 lib)
  set(TODRT_ARCH_DESC "aarch64（Jetson / RK3588 / ARM 笔记本）")
elseif(TODRT_ARCH MATCHES "^(armv7l|armv7|armhf|arm)$")
  set(TODRT_ARCH "armv7")
  set(TODRT_ARCH_SUBDIRS lib lib/arm-linux-gnueabihf)
  set(TODRT_OV_SUBDIRS lib/armv7l lib)
  set(TODRT_ARCH_DESC "armv7")
else()
  # 未知架构不报错：仍可编译纯 CPU 部分（只是找不到厂商 SDK 的子目录提示）
  set(TODRT_ARCH_SUBDIRS lib lib64)
  set(TODRT_OV_SUBDIRS lib)
  set(TODRT_ARCH_DESC "${TODRT_ARCH}（未识别的架构，仅保证 CPU 部分可用）")
  message(WARNING "未识别的目标架构 '${TODRT_ARCH}'：第三方 SDK 的子目录探测可能失败。"
                  "可用 -DTODRT_ARCH=x86_64|aarch64 显式指定。")
endif()

# 交叉编译时 CMAKE_SYSTEM_PROCESSOR 来自 toolchain file；两者不一致要提醒
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL TODRT_ARCH)
  message(STATUS "todrt: 交叉编译中，目标架构按 toolchain 的 ${CMAKE_SYSTEM_PROCESSOR} 处理")
  set(TODRT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ---------------------------------------------------------------- 公共编译设置

add_library(todrt_compile_options INTERFACE)
target_compile_features(todrt_compile_options INTERFACE cxx_std_17)
target_include_directories(todrt_compile_options INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(todrt_compile_options INTERFACE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(todrt_compile_options INTERFACE $<$<CONFIG:Release>:-O3>)
  target_compile_options(todrt_compile_options INTERFACE $<$<CONFIG:RelWithDebInfo>:-O2 -g>)
  target_compile_options(todrt_compile_options INTERFACE $<$<CONFIG:Debug>:-O0 -g>)

  if(TODRT_NATIVE)
    if(TODRT_ARCH STREQUAL "x86_64")
      target_compile_options(todrt_compile_options INTERFACE -march=native -mtune=native)
    else()
      target_compile_options(todrt_compile_options INTERFACE -mcpu=native -mtune=native)
    endif()
    message(STATUS "todrt: TODRT_NATIVE=ON —— 产物绑定本机 CPU，不要拷贝到别的机器")
  elseif(TODRT_ARCH STREQUAL "x86_64")
    # x86_64 基线：SSE2 保底 + AVX/AVX2/FMA/F16C（2011 年后的 CPU 普遍支持）
    target_compile_options(todrt_compile_options INTERFACE -mavx -mavx2 -mfma -mf16c)
    message(STATUS "todrt: x86_64 指令集基线 = AVX/AVX2/FMA/F16C"
                   "（需要更保守可自行 -DCMAKE_CXX_FLAGS=-msse4.2）")
  elseif(TODRT_ARCH STREQUAL "aarch64")
    # aarch64 基线：ARMv8-A + NEON（aarch64 上 NEON 是强制的）。
    # 分两步设置 -march，避免覆盖：armv8-a 是底线，若工具链支持再抬到 +fp16。
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv8.2-a+fp16" TODRT_HAS_FP16)
    if(TODRT_HAS_FP16)
      target_compile_options(todrt_compile_options INTERFACE -march=armv8.2-a+fp16)
      message(STATUS "todrt: aarch64 指令集基线 = armv8.2-a+fp16（NEON + 半精度）")
    else()
      target_compile_options(todrt_compile_options INTERFACE -march=armv8-a)
      message(STATUS "todrt: aarch64 指令集基线 = armv8-a（工具链不支持 +fp16）")
    endif()
  endif()

  # 共享库导出符号时使用默认可见性（避免下游项目开了 -fvisibility=hidden 后漏符号）
  set(CMAKE_CXX_VISIBILITY_PRESET default)

  # 源码注释是 UTF-8 中文；显式声明输入输出编码，避免 GBK locale 的发行版环境踩坑
  target_compile_options(todrt_compile_options INTERFACE -finput-charset=UTF-8)
  target_compile_options(todrt_compile_options INTERFACE -fexec-charset=UTF-8)
else()
  # 非 GCC/Clang：忽略上面的架构调优与编码选项。
  # 本库的目标平台是 Linux（x86_64 / aarch64）；这里保留一个最小分支，只为在
  # 完全没有 GCC 的机器上仍能构建 CPU 部分并跑 todrt_smoke（纯逻辑自检）。
  # ⚠️ MSVC 在代码页 936 的环境下会把 UTF-8 中文注释后的代码错切，必须显式 /utf-8，
  #    否则报出的是"非成员函数上不允许修饰符"这类看似无关的错。
  if(MSVC)
    target_compile_options(todrt_compile_options INTERFACE /utf-8 /bigobj)
    message(STATUS "todrt: 检测到 MSVC —— 仅用于本机 CPU 自检，不参与实机部署")
  else()
    message(STATUS "todrt: 编译器 ${CMAKE_CXX_COMPILER_ID} 非 GCC/Clang —— 跳过架构调优与编码选项"
                   "（本库的目标平台是 Linux；这里只是允许在缺少这些选项时也能配置）")
  endif()
endif()

# ---------------------------------------------------------------- 无后端的核心库

# 框架/机制类实现：不含"静态初始化自注册"代码，放进静态库是安全的。
set(TODRT_CORE_SOURCES
  src/core.cpp
  src/json.cpp
  src/config_io.cpp
  src/factory.cpp
  src/decode.cpp
)

# ★ 含"静态初始化自注册"的模块（变体配方 / 前后处理）。
#
# 为什么必须直接编进可执行目标、而不能放进静态库：
#   静态库只把"被引用到的符号"所在的目标文件拉进来。这些 TU 的注册发生在静态
#   初始化中，没有任何外部符号引用它们，于是整个目标文件被丢弃，表现为
#   "运行时一个变体/策略都没注册"——而且不报错，非常难查。
#   回归验证：把 ${TODRT_REGISTRY_SOURCES} 从可执行目标里去掉，todrt_smoke 会
#   显示"已注册变体 0 个"。
file(GLOB TODRT_MODEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp)
set(TODRT_REGISTRY_SOURCES
  src/preprocess.cpp
  src/nms.cpp
  src/backend/backend_support.cpp
  ${TODRT_MODEL_SOURCES}
)
list(LENGTH TODRT_MODEL_SOURCES TODRT_MODEL_COUNT)
if(TODRT_MODEL_COUNT EQUAL 0)
  message(WARNING "没有找到任何变体配方（src/models/*.cpp）——工厂将是空的。")
endif()

add_library(todrt_core STATIC ${TODRT_CORE_SOURCES})
target_link_libraries(todrt_core PUBLIC todrt_compile_options)
set_target_properties(todrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------- 后端探测

# 先声明"有哪些后端"，再逐个探测；engine_stub.cpp 只在全都没有时参与编译。
set(TODRT_HAVE_TENSORRT 0)
set(TODRT_HAVE_RKNN 0)
set(TODRT_HAVE_ORT 0)
set(TODRT_HAVE_OPENVINO 0)

set(TODRT_HAVE_ANY_BACKEND 0)
set(TODRT_BACKEND_SOURCES src/backend/backend_dispatch.cpp)
set(TODRT_BACKEND_REGISTRY_SOURCES src/backend/simple_detector.cpp)
set(TODRT_BACKEND_LIBS "")
set(TODRT_BACKEND_INCLUDES "")

# ---------------- TensorRT ----------------
if(TODRT_WITH_TENSORRT)
  # TensorRT 的安装布局在各版本/平台上不一致，按可靠性依次尝试：
  #   1) 标准 CMake 包（TensorRT 9+ 提供 TensorRTConfig.cmake）
  #   2) 手工指定 TensorRT_ROOT
  #   3) 发行版默认路径（含架构相关子目录，见 TODRT_ARCH_SUBDIRS）
  find_package(TensorRT QUIET COMPONENTS nvinfer nvonnxparser)
  if(NOT TensorRT_FOUND)
    set(_trt_hints
      ${TensorRT_ROOT} $ENV{TensorRT_ROOT} $ENV{TRT_ROOT}
      /usr /usr/local
      /usr/src/tensorrt
    )
    find_path(TODRT_TRT_INCLUDE_DIR NvInfer.h
      HINTS ${_trt_hints}
      PATH_SUFFIXES include include/${TODRT_ARCH}-linux-gnu)
    find_library(TODRT_NVINFER_LIB nvinfer
      HINTS ${_trt_hints} PATH_SUFFIXES ${TODRT_ARCH_SUBDIRS})
    find_library(TODRT_NVONNXPARSER_LIB nvonnxparser
      HINTS ${_trt_hints} PATH_SUFFIXES ${TODRT_ARCH_SUBDIRS})
    if(TODRT_TRT_INCLUDE_DIR AND TODRT_NVINFER_LIB AND TODRT_NVONNXPARSER_LIB)
      set(TensorRT_FOUND ON)
      add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TODRT_NVINFER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TODRT_TRT_INCLUDE_DIR}")
      add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TODRT_NVONNXPARSER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TODRT_TRT_INCLUDE_DIR}")
    endif()
  endif()

  if(TensorRT_FOUND)
    set(TODRT_HAVE_TENSORRT 1)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(WARNING "找到 TensorRT 但没找到 CUDA Toolkit；请设 CUDAToolkit_ROOT 或 CUDA_HOME")
    endif()
    # 与其它后端同构：一个后端 = 一个 engine_<x>.cpp（内部同时含 builder 与 detector）
    list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_trt.cpp)
    list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_trt.cpp)
    list(APPEND TODRT_BACKEND_LIBS TensorRT::nvinfer TensorRT::nvonnxparser)
    if(CUDAToolkit_FOUND)
      list(APPEND TODRT_BACKEND_LIBS CUDA::cudart)
    else()
      find_library(TODRT_CUDART cudart HINTS /usr/local/cuda/lib64 /usr/local/cuda/lib)
      if(TODRT_CUDART)
        list(APPEND TODRT_BACKEND_LIBS ${TODRT_CUDART})
      endif()
    endif()
    message(STATUS "todrt: TensorRT 后端 ON（include=${TODRT_TRT_INCLUDE_DIR}）")
  else()
    message(WARNING "未找到 TensorRT（NvInfer.h/nvinfer）→ 该后端关闭。")
  endif()
endif()

# ---------------- RKNN（Rockchip NPU，仅 aarch64）----------------
if(TODRT_WITH_RKNN)
  if(NOT TODRT_ARCH STREQUAL "aarch64")
    # RKNN 只有 aarch64 的运行库；在 x86 上明确关闭，而不是留一个必然失败的探测
    message(WARNING "RKNN 后端只在 aarch64 上可用（当前架构 ${TODRT_ARCH}）→ 已关闭。"
                    " 交叉编译到 RK3588 请用 src/todrt/cmake/aarch64-linux-gnu.cmake")
  else()
    set(_rknn_hints ${TODRT_RKNN_ROOT} $ENV{RKNN_ROOT} /usr /usr/local /opt/rknn)
    find_path(TODRT_RKNN_INCLUDE_DIR rknn_api.h
      HINTS ${_rknn_hints} PATH_SUFFIXES include include/rknn)
    # Linux 上是 librknnrt.so；Android 上是 librknn_api.so（本库按 Linux 走）
    find_library(TODRT_RKNN_LIB NAMES rknnrt rknn_api
      HINTS ${_rknn_hints} PATH_SUFFIXES ${TODRT_ARCH_SUBDIRS})
    if(TODRT_RKNN_INCLUDE_DIR AND TODRT_RKNN_LIB)
      set(TODRT_HAVE_RKNN 1)
      list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_rknn.cpp)
      list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_rknn.cpp)
      list(APPEND TODRT_BACKEND_LIBS ${TODRT_RKNN_LIB})
      list(APPEND TODRT_BACKEND_INCLUDES ${TODRT_RKNN_INCLUDE_DIR})
      message(STATUS "todrt: RKNN 后端 ON（include=${TODRT_RKNN_INCLUDE_DIR}, lib=${TODRT_RKNN_LIB}）")
      message(STATUS "todrt:   运行期需要 libOpenCL.so 与 /dev/rknpu* 权限（用户需在 video/render 组）")
    else()
      message(WARNING
        "未找到 RKNN（rknn_api.h / librknnrt.so）→ 该后端关闭。\n"
        "  RK3588 上：从 rknpu2 仓库取 runtime/Linux/librknn_api/ 下的 aarch64 版本，\n"
        "  把 librknnrt.so 放到 /usr/lib，rknn_api.h 放到 /usr/include，或用 -DTODRT_RKNN_ROOT=...")
    endif()
  endif()
endif()

# ---------------- ONNX Runtime（x86_64 与 aarch64 都能用）----------------
if(TODRT_WITH_ORT)
  find_package(onnxruntime QUIET)
  if(NOT onnxruntime_FOUND)
    set(_ort_hints ${TODRT_ORT_ROOT} $ENV{ORT_ROOT} $ENV{ONNXRUNTIME_ROOT}
      /usr /usr/local /opt/onnxruntime)
    find_path(TODRT_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
      HINTS ${_ort_hints} PATH_SUFFIXES include include/onnxruntime include/core/session)
    # 官方预编译包直接解包就是 lib/libonnxruntime.so；发行版会放进架构子目录
    find_library(TODRT_ORT_LIB NAMES onnxruntime
      HINTS ${_ort_hints} PATH_SUFFIXES lib ${TODRT_ARCH_SUBDIRS})
    if(TODRT_ORT_INCLUDE_DIR AND TODRT_ORT_LIB)
      set(onnxruntime_FOUND ON)
    endif()
  endif()

  if(onnxruntime_FOUND)
    set(TODRT_HAVE_ORT 1)
    list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_ort.cpp)
    list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_ort.cpp)
    if(NOT TARGET onnxruntime::onnxruntime AND TODRT_ORT_LIB)
      add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${TODRT_ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TODRT_ORT_INCLUDE_DIR}")
    endif()
    list(APPEND TODRT_BACKEND_LIBS onnxruntime::onnxruntime)
    message(STATUS "todrt: ONNX Runtime 后端 ON（include=${TODRT_ORT_INCLUDE_DIR}, lib=${TODRT_ORT_LIB}）")
    message(STATUS "todrt:   装的是 onnxruntime（CPU）还是 onnxruntime-gpu 决定 CUDA EP 能否用")
  else()
    set(_ort_pkg "onnxruntime-linux-x64-<版本>")
    if(TODRT_ARCH STREQUAL "aarch64")
      set(_ort_pkg "onnxruntime-linux-aarch64-<版本>")
    endif()
    message(WARNING
      "未找到 ONNX Runtime（onnxruntime_cxx_api.h / libonnxruntime.so）→ 该后端关闭。\n"
      "  上手最快的办法是下载官方预编译包（注意选对本机架构）：\n"
      "    https://github.com/microsoft/onnxruntime/releases  → ${_ort_pkg}\n"
      "    cmake ... -DTODRT_ORT_ROOT=/path/to/${_ort_pkg}")
  endif()
endif()

# ---------------- OpenVINO ----------------
if(TODRT_WITH_OPENVINO)
  find_package(OpenVINO QUIET)
  if(OpenVINO_FOUND)
    set(TODRT_HAVE_OPENVINO 1)
    list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_openvino.cpp)
    list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_openvino.cpp)
    list(APPEND TODRT_BACKEND_LIBS openvino::runtime)
    message(STATUS "todrt: OpenVINO 后端 ON（${OpenVINO_VERSION}）")
    message(STATUS "todrt:   CPU 设备始终可用；GPU/NPU 需对应插件随 OpenVINO 一起安装")
  elseif(TODRT_OPENVINO_ROOT)
    # 手动指定根目录（解包版 Runtime 没有 CMake config 时）。
    # 注意 lib 子目录与架构相关：x86 是 lib/intel64，aarch64 是 lib/aarch64。
    find_path(TODRT_OV_INCLUDE_DIR openvino/openvino.hpp
      HINTS ${TODRT_OPENVINO_ROOT} PATH_SUFFIXES include runtime/include)
    find_library(TODRT_OV_LIB NAMES openvino
      HINTS ${TODRT_OPENVINO_ROOT}
      PATH_SUFFIXES ${TODRT_OV_SUBDIRS} runtime/${TODRT_OV_SUBDIRS})
    if(TODRT_OV_INCLUDE_DIR AND TODRT_OV_LIB)
      set(TODRT_HAVE_OPENVINO 1)
      list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_openvino.cpp)
      list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_openvino.cpp)
      list(APPEND TODRT_BACKEND_LIBS ${TODRT_OV_LIB})
      list(APPEND TODRT_BACKEND_INCLUDES ${TODRT_OV_INCLUDE_DIR})
      message(STATUS "todrt: OpenVINO 后端 ON（手动路径 include=${TODRT_OV_INCLUDE_DIR}, lib=${TODRT_OV_LIB}）")
    endif()
  endif()
  if(NOT TODRT_HAVE_OPENVINO)
    message(WARNING
      "未找到 OpenVINO → 该后端关闭。\n"
      "  安装：pip install openvino（会带 Runtime 与 CMake config），或下载 archives 解包后\n"
      "        cmake ... -DTODRT_OPENVINO_ROOT=/path/to/openvino\n"
      "  架构提示：x86 的库在 lib/intel64，aarch64 在 lib/aarch64。")
  endif()
endif()

# 一个后端都没有时挂 stub
if(NOT TODRT_HAVE_TENSORRT AND NOT TODRT_HAVE_RKNN AND NOT TODRT_HAVE_ORT
   AND NOT TODRT_HAVE_OPENVINO)
  list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_stub.cpp)
  list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_stub.cpp)
  message(WARNING
    "没有启用任何推理后端 → 只能构建 CPU 部分（工厂/配置/前处理/解码/NMS/自检）。\n"
    "  实机部署请按上面的提示打开至少一个后端。")
endif()

# 安装目录变量（CMAKE_INSTALL_BINDIR 等在下面会被提前用到，必须先 include）
include(GNUInstallDirs)

# ---------------------------------------------------------------- 主库与可执行

# 共享库 / 静态库由 TODRT_BUILD_SHARED 决定。
# 注意：静态版 libtodrt.a **仍要求调用方把 ${TODRT_REGISTRY_SOURCES} 编进自己的
# 可执行目标**（自注册代码不能被链接器丢掉，见上面的说明）；共享库没有这个问题，
# 因为 .so 里的静态初始化一定会执行。所以对外分发推荐 TODRT_BUILD_SHARED=ON。
if(TODRT_BUILD_SHARED)
  add_library(todrt SHARED ${TODRT_BACKEND_SOURCES})
  set_target_properties(todrt PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  set(TODRT_LIB_KIND "SHARED")
else()
  add_library(todrt STATIC ${TODRT_BACKEND_SOURCES})
  set(TODRT_LIB_KIND "STATIC")
endif()

target_link_libraries(todrt PUBLIC todrt_core)
target_include_directories(todrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(TODRT_BACKEND_INCLUDES)
  target_include_directories(todrt PRIVATE ${TODRT_BACKEND_INCLUDES})
endif()
if(TODRT_BACKEND_LIBS)
  target_link_libraries(todrt PUBLIC ${TODRT_BACKEND_LIBS})
endif()
set_target_properties(todrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 后端开关编译给所有 TU（engine_stub / 分派器 / 各后端都靠这些宏）
if(TODRT_HAVE_TENSORRT OR TODRT_HAVE_RKNN OR TODRT_HAVE_ORT OR TODRT_HAVE_OPENVINO)
  set(TODRT_HAVE_ANY_BACKEND 1)
else()
  set(TODRT_HAVE_ANY_BACKEND 0)
endif()
target_compile_definitions(todrt PUBLIC
  TODRT_HAVE_TENSORRT=${TODRT_HAVE_TENSORRT}
  TODRT_HAVE_RKNN=${TODRT_HAVE_RKNN}
  TODRT_HAVE_ORT=${TODRT_HAVE_ORT}
  TODRT_HAVE_OPENVINO=${TODRT_HAVE_OPENVINO}
  TODRT_HAVE_ANY_BACKEND=${TODRT_HAVE_ANY_BACKEND})

# 后端开关宏要同时给可执行目标（它们自己也会引用 TODRT_HAVE_*）
set(TODRT_BACKEND_DEFS
  TODRT_HAVE_TENSORRT=${TODRT_HAVE_TENSORRT}
  TODRT_HAVE_RKNN=${TODRT_HAVE_RKNN}
  TODRT_HAVE_ORT=${TODRT_HAVE_ORT}
  TODRT_HAVE_OPENVINO=${TODRT_HAVE_OPENVINO}
  TODRT_HAVE_ANY_BACKEND=${TODRT_HAVE_ANY_BACKEND})

if(TODRT_BUILD_SMOKE)
  add_executable(todrt_smoke tests/cpp_smoke.cpp ${TODRT_REGISTRY_SOURCES}
                 ${TODRT_BACKEND_REGISTRY_SOURCES})
  target_compile_definitions(todrt_smoke PRIVATE ${TODRT_BACKEND_DEFS})
  if(TODRT_BACKEND_INCLUDES)
    target_include_directories(todrt_smoke PRIVATE ${TODRT_BACKEND_INCLUDES})
  endif()
  target_link_libraries(todrt_smoke PRIVATE todrt)
  enable_testing()
  add_test(NAME todrt_smoke COMMAND todrt_smoke)
endif()

if(TODRT_BUILD_CLI)
  add_executable(todrt_cli apps/todrt_cli.cpp ${TODRT_REGISTRY_SOURCES}
                 ${TODRT_BACKEND_REGISTRY_SOURCES})
  target_compile_definitions(todrt_cli PRIVATE ${TODRT_BACKEND_DEFS})
  if(TODRT_BACKEND_INCLUDES)
    target_include_directories(todrt_cli PRIVATE ${TODRT_BACKEND_INCLUDES})
  endif()
  target_link_libraries(todrt_cli PRIVATE todrt)
  install(TARGETS todrt_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# ---------------------------------------------------------------- 安装与导出
#
# 装出去之后，其它项目可以：
#   find_package(todrt REQUIRED)
#   target_link_libraries(app PRIVATE todrt::todrt)
# 或 pkg-config:  pkg-config --cflags --libs todrt
if(TODRT_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS todrt todrt_core todrt_compile_options
    EXPORT todrtTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # 已启用的后端要写进 config，好让下游知道"这个 so 里有什么"
  install(EXPORT todrtTargets
    FILE todrtTargets.cmake
    NAMESPACE todrt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/todrt)

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/todrtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/todrtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/todrt)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/todrtConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/todrtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/todrtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/todrt)

  # 通过 todrt_core 把后端开关宏传递出去（下游 include 头文件时需要一致）
  set(TODRT_INSTALLED_BACKENDS
    "TensorRT:${TODRT_HAVE_TENSORRT} RKNN:${TODRT_HAVE_RKNN} ORT:${TODRT_HAVE_ORT} OpenVINO:${TODRT_HAVE_OPENVINO}")

  # pkg-config（给不用 CMake 的下游）
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/todrt.pc.in
                 ${CMAKE_CURRENT_BINARY_DIR}/todrt.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/todrt.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

message(STATUS "todrt: ──────────────────────────────────────────────")
message(STATUS "todrt: 平台    = ${CMAKE_SYSTEM_NAME}/${TODRT_ARCH}  ${TODRT_ARCH_DESC}")
message(STATUS "todrt: 编译器  = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  "  构建类型 = ${CMAKE_BUILD_TYPE}  库类型 = ${TODRT_LIB_KIND}")
if(CMAKE_CROSSCOMPILING)
  message(STATUS "todrt: 交叉编译 = 是（sysroot=${CMAKE_SYSROOT}）")
endif()
message(STATUS "todrt: 后端    = TensorRT:${TODRT_HAVE_TENSORRT}"
  "  RKNN:${TODRT_HAVE_RKNN}  ORT:${TODRT_HAVE_ORT}  OpenVINO:${TODRT_HAVE_OPENVINO}")
if(NOT TODRT_HAVE_ANY_BACKEND)
  message(STATUS "todrt: 提示    = 没有任何后端，只能跑工厂/配置/解码/NMS 与自检")
endif()
message(STATUS "todrt: ──────────────────────────────────────────────")
