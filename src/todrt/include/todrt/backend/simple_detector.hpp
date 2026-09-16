// simple_detector.hpp —— 后端共用的 Detector 外壳
//
// 四个后端（TensorRT / RKNN / ONNX Runtime / OpenVINO）的 Detector 子类做的事完全一样：
//   前处理(pipeline) → 引擎跑一次 → 后处理(NMS) → 坐标反变换
// 差别只在"引擎跑一次"这一段。所以这里把它抽成一个薄外壳：
// 后端只需要提供一个 IEngineRunner 实现。
//
// 这样做的收益很直接：新增一个后端 = 一个 .cpp（实现 runner + 工厂函数），
// 不需要再抄一遍 Detector 的生命周期/计时/错误处理。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/modules.hpp"

namespace todrt {

/// 一台"能跑一次的引擎"。四个后端各实现一份。
///
/// 约定：
///   * Run() 收到的是**主机侧**输入缓冲（dtype/layout 见 input_spec()），
///     输出必须写进调用方给的 outputs 缓冲（主机侧，float32）；
///   * 输出张量的 dtype 由 output_types() 声明；if 声明为 kF32 则 outputs[k]
///     就是 float*，且长度 = 输出元素数；
///   * 所有内存管理、设备拷贝、计时都在实现内部完成。
class IEngineRunner {
 public:
  virtual ~IEngineRunner() = default;
  virtual const std::string& name() const = 0;

  /// 引擎的输入契约（决定前处理该怎么产出数据）。
  virtual EngineInputSpec input_spec() const = 0;
  /// 引擎的输出形状与类型（按输出序号）。
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  virtual std::vector<DataType> output_types() const = 0;

  /// 跑一次。失败抛 TritError（里面要说清是哪一步、为什么）。
  /// @param input        主机侧输入缓冲指针
  /// @param input_bytes  输入字节数
  /// @param batch        本次 batch
  /// @param outputs      主机侧输出缓冲（每个输出的 float 缓冲首地址）
  /// @param output_elems 每个输出的元素个数（float 计）
  virtual void Run(const void* input, size_t input_bytes, int batch,
                   const std::vector<float*>& outputs,
                   const std::vector<size_t>& output_elems) = 0;

  /// 引擎侧最近一次推理耗时（ms）；无原生计时则返回 0。
  virtual double last_infer_ms() const = 0;
  /// 人类可读的引擎信息（写进 Describe()）。
  virtual std::string engine_info() const = 0;
};

/// 通用 Detector：把 IEngineRunner 包成完整的推理流水线。
class RunnerDetector : public Detector {
 public:
  RunnerDetector(std::string model, DetectorOptions opts,
                 std::unique_ptr<IEngineRunner> runner,
                 std::unique_ptr<IPreprocessor> pre, std::unique_ptr<IPostprocessor> post)
      : runner_(std::move(runner)), pre_(std::move(pre)), post_(std::move(post)) {
    set_model_name(std::move(model));
    mutable_options() = std::move(opts);
  }

  void input_shape(int* w, int* h) const override {
    *w = pre_ ? pre_->out_width() : 0;
    *h = pre_ ? pre_->out_height() : 0;
  }

  std::vector<std::vector<int64_t>> output_shapes() const override {
    return runner_ ? runner_->output_shapes() : std::vector<std::vector<int64_t>>{};
  }

  std::string Describe() const override;

  std::vector<TensorView> last_outputs() const override { return last_; }

 protected:
  std::vector<Detection> RunBatch(const PreprocessResult& pp, double* preprocess_ms,
                                  double* infer_ms, double* postprocess_ms) override;

 private:
  std::unique_ptr<IEngineRunner> runner_;
  std::unique_ptr<IPreprocessor> pre_;
  std::unique_ptr<IPostprocessor> post_;
  std::vector<std::vector<float>> host_outputs_;
  std::vector<TensorView> last_;
};

}  // namespace todrt
