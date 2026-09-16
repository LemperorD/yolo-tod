# todrt —— yolo-tod 的 C++ 推理运行时
#
# 目标平台是 **Linux**（Jetson Orin / RK3588 / x86 含 AMD / Intel）。本文件不含任何
# Windows 专有逻辑；在 Windows 上能编的只有"无后端"的 CPU 部分
# （工厂 + 配置 + 解码 + NMS + 自检），仅用于开发期逻辑验证。
#
# 后端（四选多，各自独立开关）：
#   TODRT_WITH_TENSORRT  —— NVIDIA GPU / Jetson DLA      （libnvinfer + libnvonnxparser）
#   TODRT_WITH_RKNN      —— Rockchip RK3588 NPU           （librknnrt.so，Linux）
#   TODRT_WITH_ORT       —— 通用 CPU 后路（AMD x86 等）   （libonnxruntime）
#   TODRT_WITH_OPENVINO  —— x86 CPU / Intel iGPU/NPU      （OpenVINO Runtime）
#
# 典型用法：
#   # 开发机（无任何 SDK）只验证工厂/配置/解码
#   cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=OFF
#   cmake --build build/todrt -j && ./build/todrt/todrt_smoke
#
#   # Jetson Orin（TensorRT 随 JetPack 装在 /usr）
#   cmake -S src/todrt -B build/orin -DTODRT_WITH_TENSORRT=ON
#
#   # RK3588（librknnrt.so 在 /usr/lib）
#   cmake -S src/todrt -B build/rk3588 -DTODRT_WITH_RKNN=ON
#
#   # AMD x86 主机（CPU 后路）：ONNX Runtime 或 OpenVINO 任选其一/两者
#   cmake -S src/todrt -B build/x86 -DTODRT_WITH_ORT=ON -DTODRT_WITH_OPENVINO=ON
cmake_minimum_required(VERSION 3.16)
project(todrt LANGUAGES CXX VERSION 0.1.0)

# ---------------------------------------------------------------- 选项

option(TODRT_WITH_TENSORRT "编译 TensorRT 后端（NVIDIA GPU / Jetson DLA）" ON)
option(TODRT_WITH_RKNN "编译 RKNN 后端（Rockchip RK3588 NPU）" OFF)
option(TODRT_WITH_ORT "编译 ONNX Runtime 后端（通用 CPU 后路）" OFF)
option(TODRT_WITH_OPENVINO "编译 OpenVINO 后端（x86 CPU / Intel iGPU）" OFF)
option(TODRT_BUILD_CLI "构建 todrt_cli 命令行工具" ON)
option(TODRT_BUILD_SMOKE "构建 CPU 自检程序（不需要任何后端）" ON)
option(TODRT_BUILD_SHARED "构建共享库（否则静态库）" OFF)
option(TODRT_WARNINGS_AS_ERRORS "把编译警告当错误" OFF)

# SDK 搜索路径（交叉编译/自定义安装时指定）
set(TODRT_RKNN_ROOT "" CACHE PATH "librknnrt.so 所在目录（含 rknn_api.h 的父目录）")
set(TODRT_ORT_ROOT "" CACHE PATH "ONNX Runtime 解包目录（含 include/ 与 lib/）")
set(TODRT_OPENVINO_ROOT "" CACHE PATH "OpenVINO Runtime 安装目录")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- 公共编译设置

set(TODRT_CXX_STANDARD 17)

add_library(todrt_compile_options INTERFACE)
target_compile_features(todrt_compile_options INTERFACE cxx_std_17)
target_include_directories(todrt_compile_options INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(todrt_compile_options INTERFACE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(todrt_compile_options INTERFACE $<$<CONFIG:Release>:-O3>)
  target_compile_options(todrt_compile_options INTERFACE $<$<CONFIG:RelWithDebInfo>:-O2 -g>)
endif()

# 源码里的注释是 UTF-8 中文。MSVC 在代码页 936 的机器上会把它们按 GBK 解释，
# 进而把注释后的代码错切（"非成员函数上不允许修饰符"这类离奇报错就是这么来的）。
# 目标平台是 Linux（gcc/clang 默认 UTF-8），此处只为在 Windows 上做 CPU 自检。
if(MSVC)
  target_compile_options(todrt_compile_options INTERFACE /utf-8 /bigobj)
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

set(TODRT_BACKEND_SOURCES src/backend/backend_dispatch.cpp)
set(TODRT_BACKEND_REGISTRY_SOURCES src/backend/simple_detector.cpp)
set(TODRT_BACKEND_LIBS "")
set(TODRT_BACKEND_INCLUDES "")

# ---------------- TensorRT ----------------
if(TODRT_WITH_TENSORRT)
  # TensorRT 的安装布局在各版本/平台上不一致，按可靠性依次尝试：
  #   1) 标准 CMake 包（TensorRT 9+ 提供 TensorRTConfig.cmake）
  #   2) 手工指定 TensorRT_ROOT
  #   3) JetPack 默认路径（aarch64：/usr/include/aarch64-linux-gnu）
  find_package(TensorRT QUIET COMPONENTS nvinfer nvonnxparser)
  if(NOT TensorRT_FOUND)
    set(_trt_hints
      ${TensorRT_ROOT} $ENV{TensorRT_ROOT} $ENV{TRT_ROOT}
      /usr /usr/local
      /usr/include/aarch64-linux-gnu
      /usr/src/tensorrt
    )
    find_path(TODRT_TRT_INCLUDE_DIR NvInfer.h
      HINTS ${_trt_hints} PATH_SUFFIXES include include/aarch64-linux-gnu)
    find_library(TODRT_NVINFER_LIB nvinfer
      HINTS ${_trt_hints} PATH_SUFFIXES lib lib/aarch64-linux-gnu)
    find_library(TODRT_NVONNXPARSER_LIB nvonnxparser
      HINTS ${_trt_hints} PATH_SUFFIXES lib lib/aarch64-linux-gnu)
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
    list(APPEND TODRT_BACKEND_SOURCES
      src/backend/engine_trt.cpp
      src/backend/builder_trt.cpp)
    list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/builder_trt.cpp)
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

# ---------------- RKNN（Rockchip NPU）----------------
if(TODRT_WITH_RKNN)
  set(_rknn_hints ${TODRT_RKNN_ROOT} $ENV{RKNN_ROOT} /usr /usr/local /opt/rknn)
  find_path(TODRT_RKNN_INCLUDE_DIR rknn_api.h
    HINTS ${_rknn_hints} PATH_SUFFIXES include include/rknn)
  # Linux 上是 librknnrt.so；Android 上是 librknn_api.so（本库按 Linux 走）
  find_library(TODRT_RKNN_LIB NAMES rknnrt rknn_api
    HINTS ${_rknn_hints} PATH_SUFFIXES lib lib/aarch64-linux-gnu)
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

# ---------------- ONNX Runtime ----------------
if(TODRT_WITH_ORT)
  find_package(onnxruntime QUIET)
  if(NOT onnxruntime_FOUND)
    set(_ort_hints ${TODRT_ORT_ROOT} $ENV{ORT_ROOT} $ENV{ONNXRUNTIME_ROOT}
      /usr /usr/local /opt/onnxruntime)
    find_path(TODRT_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
      HINTS ${_ort_hints} PATH_SUFFIXES include include/onnxruntime include/core/session)
    find_library(TODRT_ORT_LIB NAMES onnxruntime
      HINTS ${_ort_hints} PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)
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
    message(WARNING
      "未找到 ONNX Runtime（onnxruntime_cxx_api.h / libonnxruntime.so）→ 该后端关闭。\n"
      "  上手最快的办法是下载官方预编译包（含头文件与 .so）：\n"
      "    https://github.com/microsoft/onnxruntime/releases  → 解包后\n"
      "    cmake ... -DTODRT_ORT_ROOT=/path/to/onnxruntime-linux-x64-1.18.0")
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
    # 手动指定根目录（解包版 Runtime 没有 CMake config 时）
    find_path(TODRT_OV_INCLUDE_DIR openvino/openvino.hpp
      HINTS ${TODRT_OPENVINO_ROOT} PATH_SUFFIXES include runtime/include)
    find_library(TODRT_OV_LIB NAMES openvino
      HINTS ${TODRT_OPENVINO_ROOT} PATH_SUFFIXES lib lib/intel64 runtime/lib/intel64)
    if(TODRT_OV_INCLUDE_DIR AND TODRT_OV_LIB)
      set(TODRT_HAVE_OPENVINO 1)
      list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_openvino.cpp)
      list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_openvino.cpp)
      list(APPEND TODRT_BACKEND_LIBS ${TODRT_OV_LIB})
      list(APPEND TODRT_BACKEND_INCLUDES ${TODRT_OV_INCLUDE_DIR})
      message(STATUS "todrt: OpenVINO 后端 ON（手动路径 include=${TODRT_OV_INCLUDE_DIR}）")
    endif()
  endif()
  if(NOT TODRT_HAVE_OPENVINO)
    message(WARNING
      "未找到 OpenVINO → 该后端关闭。\n"
      "  安装：pip install openvino（会带 Runtime 与 CMake config），或下载 archives 解包后\n"
      "        cmake ... -DTODRT_OPENVINO_ROOT=/path/to/openvino")
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

# ---------------------------------------------------------------- 主库与可执行

add_library(todrt ${TODRT_BACKEND_SOURCES})
target_link_libraries(todrt PUBLIC todrt_core)
target_include_directories(todrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(TODRT_BACKEND_INCLUDES)
  target_include_directories(todrt PRIVATE ${TODRT_BACKEND_INCLUDES})
endif()
if(TODRT_BACKEND_LIBS)
  target_link_libraries(todrt PUBLIC ${TODRT_BACKEND_LIBS})
endif()
set_target_properties(todrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 后端开关编译给所有 TU（engine_stub / 分派器 / 各后端都靠这些宏）
target_compile_definitions(todrt PUBLIC
  TODRT_HAVE_TENSORRT=${TODRT_HAVE_TENSORRT}
  TODRT_HAVE_RKNN=${TODRT_HAVE_RKNN}
  TODRT_HAVE_ORT=${TODRT_HAVE_ORT}
  TODRT_HAVE_OPENVINO=${TODRT_HAVE_OPENVINO})
if(TODRT_HAVE_TENSORRT OR TODRT_HAVE_RKNN OR TODRT_HAVE_ORT OR TODRT_HAVE_OPENVINO)
  target_compile_definitions(todrt PUBLIC TODRT_HAVE_ANY_BACKEND=1)
else()
  target_compile_definitions(todrt PUBLIC TODRT_HAVE_ANY_BACKEND=0)
endif()

# 后端开关宏要同时给可执行目标（它们自己也会引用 TODRT_HAVE_*）
set(TODRT_BACKEND_DEFS
  TODRT_HAVE_TENSORRT=${TODRT_HAVE_TENSORRT}
  TODRT_HAVE_RKNN=${TODRT_HAVE_RKNN}
  TODRT_HAVE_ORT=${TODRT_HAVE_ORT}
  TODRT_HAVE_OPENVINO=${TODRT_HAVE_OPENVINO})

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
endif()

# ---------------------------------------------------------------- 安装

include(GNUInstallDirs)
install(TARGETS todrt todrt_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "todrt: 平台=${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} 构建类型=${CMAKE_BUILD_TYPE}")
message(STATUS "todrt: 后端 = TensorRT:${TODRT_HAVE_TENSORRT} RKNN:${TODRT_HAVE_RKNN} ORT:${TODRT_HAVE_ORT} OpenVINO:${TODRT_HAVE_OPENVINO}")
