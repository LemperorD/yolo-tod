# todrt —— yolo-tod 的 C++ 推理运行时（TensorRT / Jetson Orin / dGPU）
#
# 目标平台是 **Linux**（JetPack 6.x aarch64 或 x86_64 + dGPU）。这里不写任何
# Windows 专有逻辑；在 Windows 上能编的只有"无 TensorRT 的 CPU 部分"
# （工厂 + 配置 + 解码 + NMS + 自检），仅用于开发期逻辑验证。
#
# 三种典型用法：
#
#   1) 开发机（无 GPU）只验证工厂/配置/解码：
#        cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=OFF
#        cmake --build build/todrt -j && ./build/todrt/todrt_smoke
#
#   2) 实机（Jetson Orin，TensorRT 随 JetPack 安装在 /usr）：
#        cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=ON
#        cmake --build build/todrt -j$(nproc)
#
#   3) x86 构建机交叉编译到 Jetson：
#        cmake -S src/todrt -B build/aarch64 \
#              -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
#              -DTODRT_WITH_TENSORRT=ON -DTensorRT_ROOT=<aarch64 TRT 解包目录>
cmake_minimum_required(VERSION 3.16)
project(todrt LANGUAGES CXX VERSION 0.1.0)

# ---------------------------------------------------------------- 选项

option(TODRT_WITH_TENSORRT "编译 TensorRT 后端（实机部署需要）" ON)
option(TODRT_BUILD_CLI "构建 todrt_cli 命令行工具" ON)
option(TODRT_BUILD_SMOKE "构建 CPU 自检程序（不需要 GPU/TensorRT）" ON)
option(TODRT_BUILD_SHARED "构建共享库（否则静态库）" OFF)
option(TODRT_WARNINGS_AS_ERRORS "把编译警告当错误" OFF)

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
if(TODRT_WARNINGS_AS_ERRORS)
  target_compile_options(todrt_compile_options INTERFACE -Werror)
endif()

# ---------------------------------------------------------------- 核心库（不依赖 TRT）

# 框架/机制类实现：不含"静态初始化自注册"代码，放进静态库是安全的。
set(TODRT_CORE_SOURCES
  src/core.cpp
  src/json.cpp
  src/config_io.cpp
  src/factory.cpp
  src/decode.cpp
)

# ★ 含"静态初始化自注册"的模块（变体配方 / 前后处理 / 后端构建器）。
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
  ${TODRT_MODEL_SOURCES}
)
list(LENGTH TODRT_MODEL_SOURCES TODRT_MODEL_COUNT)
if(TODRT_MODEL_COUNT EQUAL 0)
  message(WARNING "没有找到任何变体配方（src/models/*.cpp）——工厂将是空的。")
endif()

add_library(todrt_core STATIC ${TODRT_CORE_SOURCES})
target_link_libraries(todrt_core PUBLIC todrt_compile_options)
set_target_properties(todrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------- 主库（按需挂 TRT）

# 未启用 TensorRT 时：stub 提供 CreateDeviceDetector / backend_available，
# 并且它自己注册一个"不可用"的构建器（这样工厂清单与自检依然完整）。
set(TODRT_BACKEND_SOURCES src/backend/engine_stub.cpp)
set(TODRT_BACKEND_REGISTRY_SOURCES src/backend/engine_stub.cpp)
set(TODRT_TRT_FOUND OFF)

if(TODRT_WITH_TENSORRT)
  # TensorRT 的安装布局在各版本/平台上不一致，按可靠性依次尝试：
  #   1) 标准 CMake 包（TensorRT 9+ 提供 TensorRTConfig.cmake）
  #   2) 手工指定 TensorRT_ROOT / TensorRT_INCLUDE_DIR
  #   3) JetPack 默认路径（aarch64 上常见：/usr/include/aarch64-linux-gnu）
  find_package(TensorRT QUIET COMPONENTS nvinfer nvonnxparser)
  if(NOT TensorRT_FOUND)
    set(_trt_hints
      ${TensorRT_ROOT} $ENV{TensorRT_ROOT} $ENV{TRT_ROOT}
      /usr /usr/local
      /usr/include/aarch64-linux-gnu
      /usr/src/tensorrt
    )
    find_path(TODRT_TRT_INCLUDE_DIR NvInfer.h
      HINTS ${_trt_hints}
      PATH_SUFFIXES include include/aarch64-linux-gnu)
    find_library(TODRT_NVINFER_LIB nvinfer HINTS ${_trt_hints} PATH_SUFFIXES lib lib/aarch64-linux-gnu)
    find_library(TODRT_NVONNXPARSER_LIB nvonnxparser HINTS ${_trt_hints}
      PATH_SUFFIXES lib lib/aarch64-linux-gnu)
    if(TODRT_TRT_INCLUDE_DIR AND TODRT_NVINFER_LIB AND TODRT_NVONNXPARSER_LIB)
      set(TODRT_TRT_FOUND ON)
      add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TODRT_NVINFER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TODRT_TRT_INCLUDE_DIR}")
      add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
      set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TODRT_NVONNXPARSER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TODRT_TRT_INCLUDE_DIR}")
    endif()
  else()
    set(TODRT_TRT_FOUND ON)
  endif()

  if(TODRT_TRT_FOUND)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(WARNING "找到 TensorRT 但没找到 CUDA Toolkit；请设置 CUDAToolkit_ROOT 或 CUDA_HOME")
    endif()
    set(TODRT_BACKEND_SOURCES
      src/backend/engine_trt.cpp
      src/backend/builder_trt.cpp)
    set(TODRT_BACKEND_REGISTRY_SOURCES src/backend/builder_trt.cpp)
    message(STATUS "todrt: TensorRT 后端已启用（include=${TODRT_TRT_INCLUDE_DIR}）")
    if(EXISTS "${TODRT_TRT_INCLUDE_DIR}/NvInferVersion.h")
      file(STRINGS "${TODRT_TRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_lines
           REGEX "#define NV_TENSORRT_(MAJOR|MINOR|PATCH)")
      foreach(_l ${_trt_ver_lines})
        message(STATUS "todrt:   ${_l}")
      endforeach()
    endif()
  else()
    message(WARNING
      "未找到 TensorRT（NvInfer.h / nvinfer）。将只构建 CPU 部分（工厂/配置/解码/NMS/自检）。\n"
      "  实机部署请安装 TensorRT 或指定 -DTensorRT_ROOT=<路径>。")
  endif()
endif()

add_library(todrt ${TODRT_BACKEND_SOURCES})
target_link_libraries(todrt PUBLIC todrt_core)
target_include_directories(todrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(todrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TODRT_TRT_FOUND)
  target_compile_definitions(todrt PUBLIC TODRT_HAVE_TENSORRT=1)
  target_link_libraries(todrt PUBLIC TensorRT::nvinfer TensorRT::nvonnxparser)
  if(CUDAToolkit_FOUND)
    target_link_libraries(todrt PUBLIC CUDA::cudart)
  else()
    find_library(TODRT_CUDART cudart HINTS /usr/local/cuda/lib64 /usr/local/cuda/lib)
    if(TODRT_CUDART)
      target_link_libraries(todrt PUBLIC ${TODRT_CUDART})
    endif()
  endif()
else()
  target_compile_definitions(todrt PUBLIC TODRT_HAVE_TENSORRT=0)
endif()

# ---------------------------------------------------------------- 自检与工具

if(TODRT_BUILD_SMOKE)
  add_executable(todrt_smoke tests/cpp_smoke.cpp ${TODRT_REGISTRY_SOURCES}
                 ${TODRT_BACKEND_REGISTRY_SOURCES})
  target_link_libraries(todrt_smoke PRIVATE todrt)
  enable_testing()
  add_test(NAME todrt_smoke COMMAND todrt_smoke)
endif()

if(TODRT_BUILD_CLI)
  add_executable(todrt_cli apps/todrt_cli.cpp ${TODRT_REGISTRY_SOURCES}
                 ${TODRT_BACKEND_REGISTRY_SOURCES})
  target_link_libraries(todrt_cli PRIVATE todrt)
endif()

# ---------------------------------------------------------------- 安装

include(GNUInstallDirs)
install(TARGETS todrt todrt_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "todrt: 平台=${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} 构建类型=${CMAKE_BUILD_TYPE}")
