cmake_minimum_required(VERSION 3.16)
project(todrt LANGUAGES CXX VERSION 0.1.0
        DESCRIPTION "yolo-tod C++ inference runtime (TensorRT / RKNN / ONNX Runtime / OpenVINO)")

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

# native 指令集仅适合本机运行，跨机器部署默认关闭。
option(TODRT_NATIVE "针对本机 CPU 优化（-march=native / -mcpu=native）" OFF)
set(TODRT_ARCH "" CACHE STRING "强制目标架构（x86_64 / aarch64 / armv7）；空 = 按 CMAKE_SYSTEM_PROCESSOR 自动探测")

set(TODRT_RKNN_ROOT "" CACHE PATH "librknnrt.so 所在目录（含 rknn_api.h 的父目录）")
set(TODRT_ORT_ROOT "" CACHE PATH "ONNX Runtime 解包目录（含 include/ 与 lib/）")
set(TODRT_OPENVINO_ROOT "" CACHE PATH "OpenVINO Runtime 安装目录")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

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

  set(TODRT_ARCH_SUBDIRS lib lib64)
  set(TODRT_OV_SUBDIRS lib)
  set(TODRT_ARCH_DESC "${TODRT_ARCH}（未识别的架构，仅保证 CPU 部分可用）")
  message(WARNING "未识别的目标架构 '${TODRT_ARCH}'：第三方 SDK 的子目录探测可能失败。"
                  "可用 -DTODRT_ARCH=x86_64|aarch64 显式指定。")
endif()

if(CMAKE_CROSSCOMPILING AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL TODRT_ARCH)
  message(STATUS "todrt: 交叉编译中，目标架构按 toolchain 的 ${CMAKE_SYSTEM_PROCESSOR} 处理")
  set(TODRT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

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

    target_compile_options(todrt_compile_options INTERFACE -mavx -mavx2 -mfma -mf16c)
    message(STATUS "todrt: x86_64 指令集基线 = AVX/AVX2/FMA/F16C"
                   "（需要更保守可自行 -DCMAKE_CXX_FLAGS=-msse4.2）")
  elseif(TODRT_ARCH STREQUAL "aarch64")

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

  set(CMAKE_CXX_VISIBILITY_PRESET default)

  target_compile_options(todrt_compile_options INTERFACE -finput-charset=UTF-8)
  target_compile_options(todrt_compile_options INTERFACE -fexec-charset=UTF-8)
else()

  if(MSVC)
    target_compile_options(todrt_compile_options INTERFACE /utf-8 /bigobj)
    message(STATUS "todrt: 检测到 MSVC —— 仅用于本机 CPU 自检，不参与实机部署")
  else()
    message(STATUS "todrt: 编译器 ${CMAKE_CXX_COMPILER_ID} 非 GCC/Clang —— 跳过架构调优与编码选项"
                   "（本库的目标平台是 Linux；这里只是允许在缺少这些选项时也能配置）")
  endif()
endif()

set(TODRT_CORE_SOURCES
  src/core.cpp
  src/json.cpp
  src/config_io.cpp
  src/factory.cpp
  src/decode.cpp
)

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

set(TODRT_HAVE_TENSORRT 0)
set(TODRT_HAVE_RKNN 0)
set(TODRT_HAVE_ORT 0)
set(TODRT_HAVE_OPENVINO 0)

set(TODRT_HAVE_ANY_BACKEND 0)
set(TODRT_BACKEND_SOURCES src/backend/backend_dispatch.cpp)
set(TODRT_BACKEND_REGISTRY_SOURCES src/backend/simple_detector.cpp)
set(TODRT_BACKEND_LIBS "")
set(TODRT_BACKEND_INCLUDES "")

if(TODRT_WITH_TENSORRT)

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

if(TODRT_WITH_RKNN)
  if(NOT TODRT_ARCH STREQUAL "aarch64")

    message(WARNING "RKNN 后端只在 aarch64 上可用（当前架构 ${TODRT_ARCH}）→ 已关闭。"
                    " 交叉编译到 RK3588 请用 src/todrt/cmake/aarch64-linux-gnu.cmake")
  else()
    set(_rknn_hints ${TODRT_RKNN_ROOT} $ENV{RKNN_ROOT} /usr /usr/local /opt/rknn)
    find_path(TODRT_RKNN_INCLUDE_DIR rknn_api.h
      HINTS ${_rknn_hints} PATH_SUFFIXES include include/rknn)

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

if(TODRT_WITH_ORT)
  find_package(onnxruntime QUIET)
  if(NOT onnxruntime_FOUND)
    set(_ort_hints ${TODRT_ORT_ROOT} $ENV{ORT_ROOT} $ENV{ONNXRUNTIME_ROOT}
      /usr /usr/local /opt/onnxruntime)
    find_path(TODRT_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
      HINTS ${_ort_hints} PATH_SUFFIXES include include/onnxruntime include/core/session)

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

if(NOT TODRT_HAVE_TENSORRT AND NOT TODRT_HAVE_RKNN AND NOT TODRT_HAVE_ORT
   AND NOT TODRT_HAVE_OPENVINO)
  list(APPEND TODRT_BACKEND_SOURCES src/backend/engine_stub.cpp)
  list(APPEND TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_stub.cpp)
  message(WARNING
    "没有启用任何推理后端 → 只能构建 CPU 部分（工厂/配置/前处理/解码/NMS/自检）。\n"
    "  实机部署请按上面的提示打开至少一个后端。")
endif()

include(GNUInstallDirs)

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

set(TODRT_BACKEND_DEFS
  TODRT_HAVE_TENSORRT=${TODRT_HAVE_TENSORRT}
  TODRT_HAVE_RKNN=${TODRT_HAVE_RKNN}
  TODRT_HAVE_ORT=${TODRT_HAVE_ORT}
  TODRT_HAVE_OPENVINO=${TODRT_HAVE_OPENVINO}
  TODRT_HAVE_ANY_BACKEND=${TODRT_HAVE_ANY_BACKEND})

if(TODRT_BUILD_SMOKE)
  # 将自注册单元直接链接进可执行文件，避免静态库裁剪掉注册器。
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

if(TODRT_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS todrt todrt_core todrt_compile_options
    EXPORT todrtTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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

  set(TODRT_INSTALLED_BACKENDS
    "TensorRT:${TODRT_HAVE_TENSORRT} RKNN:${TODRT_HAVE_RKNN} ORT:${TODRT_HAVE_ORT} OpenVINO:${TODRT_HAVE_OPENVINO}")

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
