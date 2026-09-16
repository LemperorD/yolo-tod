// backend_dispatch.cpp —— 工厂 → 后端的路由（唯一需要"知道所有后端"的地方）
//
// 设计上这是**唯一**一处按名字分派到具体后端的地方：
//   Detector::Create() → CreateDeviceDetector() → （本文件）→ CreateXxxDetector()
//
// 每个后端自己的工厂函数都在各自的 .cpp 里（受对应宏保护），所以：
//   * 只编了 TensorRT 的机器上，选 ort 会得到"该后端未编译进来"的明确提示；
//   * 新增后端只需要在本文件加一个分支 + 一个 .cpp 文件。
#include <string>

#include "todrt/factory.hpp"

#if TODRT_HAVE_TENSORRT
#include "todrt/backend/trt_factory.hpp"
#endif
#if TODRT_HAVE_RKNN
#include "todrt/backend/rknn_factory.hpp"
#endif
#if TODRT_HAVE_ORT
#include "todrt/backend/ort_factory.hpp"
#endif
#if TODRT_HAVE_OPENVINO
#include "todrt/backend/openvino_factory.hpp"
#endif

namespace todrt {
namespace {

/// 归一化 builder 名，便于 "yolov8-trt" / "trt" / "rknn_npu" 这类写法都能对上。
std::string canon(const std::string& s) {
  std::string k;
  k.reserve(s.size());
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') {
      k.push_back(static_cast<char>(c - 'A' + 'a'));
    } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      k.push_back(c);
    }
  }
  return k;
}

/// 该后端在当前构建里是否存在。用于给出"没编进来"而不是"不认识"的准确提示。
struct BackendAvailability {
  const char* name;
  bool compiled;
  const char* build_flag;
};

BackendAvailability LookupBackend(const std::string& b) {
  if (b == "yolov8trt" || b == "trt" || b == "tensorrt" || b == "yolov8") {
    return {"TensorRT", TODRT_HAVE_TENSORRT != 0, "-DTODRT_WITH_TENSORRT=ON"};
  }
  if (b == "rknn" || b == "rknntr" || b == "rknnnpu") {
    return {"RKNN", TODRT_HAVE_RKNN != 0, "-DTODRT_WITH_RKNN=ON"};
  }
  if (b == "ort" || b == "onnxruntime") {
    return {"ONNX Runtime", TODRT_HAVE_ORT != 0, "-DTODRT_WITH_ORT=ON"};
  }
  if (b == "openvino" || b == "ov") {
    return {"OpenVINO", TODRT_HAVE_OPENVINO != 0, "-DTODRT_WITH_OPENVINO=ON"};
  }
  return {"(未知)", false, nullptr};
}

}  // namespace

std::unique_ptr<Detector> CreateDeviceDetector(const std::string& model_name,
                                               const DetectorOptions& opts,
                                               const ModelRecipe& recipe,
                                               const std::string& builder_name,
                                               const std::string& preproc_name,
                                               const std::string& postproc_name) {
  const std::string b = canon(builder_name);
  const BackendAvailability avail = LookupBackend(b);

  if (avail.build_flag == nullptr) {
    std::string known;
#if TODRT_HAVE_TENSORRT
    known += "yolov8-trt ";
#endif
#if TODRT_HAVE_RKNN
    known += "rknn ";
#endif
#if TODRT_HAVE_ORT
    known += "ort ";
#endif
#if TODRT_HAVE_OPENVINO
    known += "openvino ";
#endif
    throw TritError("未知的引擎构建器：" + builder_name + "。当前构建可用：" + known +
                    "（新增后端见 src/todrt/README.md）");
  }
  if (!avail.compiled) {
    throw TritError(std::string("后端 ") + avail.name + " 没有编译进当前构建（builder=" +
                    builder_name + "）。\n  启用它：cmake " + avail.build_flag +
                    " ... 并确保对应 SDK 可被找到（见 src/todrt/README.md 的构建矩阵）。");
  }

#if TODRT_HAVE_TENSORRT
  if (b == "yolov8trt" || b == "trt" || b == "tensorrt" || b == "yolov8") {
    return CreateTrtDetector(model_name, opts, recipe, builder_name, preproc_name, postproc_name);
  }
#endif
#if TODRT_HAVE_RKNN
  if (b == "rknn" || b == "rknntr" || b == "rknnnpu") {
    return CreateRknnDetector(model_name, opts, recipe, builder_name, preproc_name, postproc_name);
  }
#endif
#if TODRT_HAVE_ORT
  if (b == "ort" || b == "onnxruntime") {
    return CreateOrtDetector(model_name, opts, recipe, builder_name, preproc_name, postproc_name);
  }
#endif
#if TODRT_HAVE_OPENVINO
  if (b == "openvino" || b == "ov") {
    return CreateOpenVinoDetector(model_name, opts, recipe, builder_name, preproc_name,
                                  postproc_name);
  }
#endif

  // 理论上到不了这里（上面已经逐个判断过）
  throw TritError("内部错误：后端 " + std::string(avail.name) + " 已声明编译但无分派分支。");
}

bool backend_available(std::string* reason) {
  std::string list;
#if TODRT_HAVE_TENSORRT
  list += "TensorRT ";
#endif
#if TODRT_HAVE_RKNN
  list += "RKNN ";
#endif
#if TODRT_HAVE_ORT
  list += "ONNX Runtime ";
#endif
#if TODRT_HAVE_OPENVINO
  list += "OpenVINO ";
#endif
  if (reason) {
    *reason = list.empty() ? std::string("未编译任何后端（仅工厂/配置/前后处理可用）")
                           : ("已编译后端：" + list);
  }
  return !list.empty();
}

}  // namespace todrt
