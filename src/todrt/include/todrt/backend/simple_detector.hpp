// simple_detector.hpp —— 后端共用的 Detector 外壳
//
// 四个后端（TensorRT / RKNN / ONNX Runtime / OpenVINO）的 Detector 做的事完全一样：
//   前处理 → 引擎跑一次 → 解码 + NMS → 坐标反变换
// 差别只在"引擎跑一次"这一段。所以这里把它抽成两层：
//
//   IEngineRunner     ← 后端只实现"跑一次"
//   AsyncDetectorBase ← 异步骨架：Submit/TryGet/Wait 的队列与线程、Bench 计时
//   RunnerDetector    ← 把 IEngineRunner 包成完整流水线（继承上面那个）
//
// ⚠️ 经验教训：AsyncDetectorBase 原本是 factory.cpp 里的**私有**实现，于是
// "只有在工厂的 CreateDeviceDetector 路径上才拿得到它"。TensorRT 因为要暴露
// DLA 诊断，自己手写了一份 Detector 子类，结果随 Detector 接口演进腐化到编不过
// （详见 engine_trt.cpp 的 TrtRunner 注释）。现在它是公开基类，四个后端一律走
// 同一条路径 —— 接口变化只需要改这一处。
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/modules.hpp"

namespace todrt {

/// 一台"能跑一次的引擎"。四个后端各实现一份。
///
/// 约定：
///   * Run() 收到的是**主机侧**输入缓冲（dtype/layout 见 input_spec()），
///     输出写进调用方给的 outputs 缓冲（主机侧，float32）；
///   * output_types() 声明引擎真实输出类型。通用外壳只接受 kF32 ——
///     若声明其它类型，外壳会明确报错而不是把 half/量化值当 float 解成乱框；
///     后端应在 Run 内完成反量化/类型转换（RKNN 用 want_float，TRT 用 half_to_float）；
///   * 内存管理、设备拷贝、计时都在实现内部完成。
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

/// 异步 Detector 骨架。子类只需实现 `RunBatch()`（一次推理 + 解码），
/// 队列/线程/计时/输入校验都在这里统一处理。
class AsyncDetectorBase : public Detector {
 public:
  explicit AsyncDetectorBase(DetectorOptions opts) { mutable_options() = std::move(opts); }
  ~AsyncDetectorBase() override { StopWorker(); }

  const std::string& model_name() const override { return model_name_; }
  const DetectorOptions& options() const override { return prototype_options(); }

  std::vector<std::vector<Detection>> Run(const std::vector<ImageView>& images) override {
    std::vector<std::vector<Detection>> out;
    if (images.empty()) return out;
    if (!pre_) throw TritError("Detector 未配置前处理器");
    out.reserve(images.size());
    // 同步路径直接串行执行，不经过队列（避免与异步请求抢缓冲）。
    for (const ImageView& img : images) {
      if (!img.valid()) throw TritError("Run: 输入图像无效");
      const std::vector<ImageView> one{img};
      PreprocessResult pp = pre_->Run(one);
      double a = 0, b = 0, c = 0;
      out.push_back(RunBatch(pp, &a, &b, &c));
    }
    return out;
  }

  uint64_t Submit(const ImageView& image) override {
    if (!image.valid()) throw TritError("Submit: 输入图像无效（空指针或尺寸为 0）");
    if (!pre_) throw TritError("Detector 未配置前处理器");
    EnsureWorker();
    auto job = std::make_shared<Job>();
    {
      std::lock_guard<std::mutex> lk(mu_);
      job->id = ++counter_;
    }
    // 拷贝像素：调用方的缓冲（视频帧/相机 buf）可以立刻被复用
    const size_t row = image.row_bytes();
    job->pixels.resize(row * static_cast<size_t>(image.height));
    std::memcpy(job->pixels.data(), image.data, job->pixels.size());
    job->view = image;
    job->view.data = job->pixels.data();
    job->view.step = row;
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(job);
    }
    cv_.notify_one();
    return job->id;
  }

  bool TryGet(InferenceResult* out) override { return Pop(out, 0, true); }
  bool Wait(InferenceResult* out, int timeout_ms) override { return Pop(out, timeout_ms, false); }

  size_t pending() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.size() + (in_flight_ ? 1 : 0);
  }

  Detector::BenchResult Bench(const ImageView& image, int warmup, int iters) override {
    if (!image.valid()) throw TritError("Bench: 输入图像无效");
    if (!pre_) throw TritError("Detector 未配置前处理器");
    BenchResult r;
    r.device = to_string(options().device);
    r.precision = to_string(options().precision);
    if (iters <= 0) return r;

    // 预热：TensorRT/GPU 首次执行会有明显额外开销，不计入统计。
    const std::vector<ImageView> one{image};
    PreprocessResult pp = pre_->Run(one);
    for (int i = 0; i < std::max(0, warmup); ++i) {
      double a = 0, b = 0, c = 0;
      RunBatch(pp, &a, &b, &c);
    }
    double sum_inf = 0, sum_all = 0;
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      double a = 0, b = 0, c = 0;
      RunBatch(pp, &a, &b, &c);
      const auto t1 = std::chrono::steady_clock::now();
      sum_inf += b;
      sum_all += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    r.iters = iters;
    r.preprocess_ms = 0.0;  // pp 复用，前处理不计入（见 run/bench 的说明）
    r.infer_ms = sum_inf / iters;
    r.postprocess_ms = std::max(0.0, (sum_all - sum_inf) / iters);
    r.end_to_end_ms = sum_all / iters;
    r.fps = r.end_to_end_ms > 0 ? 1000.0 / r.end_to_end_ms : 0.0;
    return r;
  }

 protected:
  /// 子类构造时注入前处理器（流水线需要它做前处理与坐标反变换）。
  void set_preprocessor(std::unique_ptr<IPreprocessor> p) { pre_ = std::move(p); }
  const IPreprocessor* preprocessor() const { return pre_.get(); }

  // 基类 Detector 已提供 protected 的 set_model_name() / mutable_options() /
  // prototype_options()，子类直接用；这里不再重复定义（避免同名遮蔽）。

 private:
  struct Job {
    uint64_t id = 0;
    ImageView view;
    std::vector<uint8_t> pixels;
    std::vector<Detection> dets;
    double pre_ms = 0, inf_ms = 0, post_ms = 0;
  };

  void EnsureWorker() {
    if (worker_.joinable()) return;
    stop_ = false;
    worker_ = std::thread([this] { WorkerLoop(); });
  }

  void StopWorker() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void WorkerLoop() {
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        job = queue_.front();
        queue_.pop_front();
        in_flight_ = true;
      }
      try {
        const std::vector<ImageView> batch{job->view};
        PreprocessResult pp = pre_->Run(batch);
        job->dets = RunBatch(pp, &job->pre_ms, &job->inf_ms, &job->post_ms);
      } catch (const std::exception& e) {
        log_error(std::string("异步推理失败：") + e.what());
        // 失败也投递（空结果），避免调用方永久阻塞
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        finished_.push_back(job);
        in_flight_ = false;
      }
      cv_done_.notify_all();
    }
  }

  bool Pop(InferenceResult* out, int timeout_ms, bool nonblocking) {
    std::unique_lock<std::mutex> lk(mu_);
    if (finished_.empty()) {
      if (nonblocking) return false;
      if (timeout_ms < 0) {
        cv_done_.wait(lk, [this] { return !finished_.empty() || stop_; });
      } else {
        cv_done_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                          [this] { return !finished_.empty() || stop_; });
      }
      if (finished_.empty()) return false;
    }
    std::shared_ptr<Job> job = finished_.front();
    finished_.pop_front();
    if (out) {
      out->request_id = job->id;
      out->detections = std::move(job->dets);
      out->preprocess_ms = job->pre_ms;
      out->infer_ms = job->inf_ms;
      out->postprocess_ms = job->post_ms;
    }
    return true;
  }

  std::unique_ptr<IPreprocessor> pre_;
  std::thread worker_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_done_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::deque<std::shared_ptr<Job>> finished_;
  uint64_t counter_ = 0;
  bool stop_ = false;
  bool in_flight_ = false;
};

/// 通用 Detector：把 IEngineRunner 包成完整流水线。四个后端都走这条路径。
class RunnerDetector : public AsyncDetectorBase {
 public:
  RunnerDetector(std::string model, DetectorOptions opts,
                 std::unique_ptr<IEngineRunner> runner,
                 std::unique_ptr<IPreprocessor> pre, std::unique_ptr<IPostprocessor> post)
      : AsyncDetectorBase(std::move(opts)), runner_(std::move(runner)), post_(std::move(post)) {
    set_model_name(std::move(model));
    set_preprocessor(std::move(pre));
  }

  void input_shape(int* w, int* h) const override {
    const IPreprocessor* p = preprocessor();
    *w = p ? p->out_width() : 0;
    *h = p ? p->out_height() : 0;
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
  std::unique_ptr<IPostprocessor> post_;
  std::vector<std::vector<float>> host_outputs_;
  std::vector<TensorView> last_;
};

}  // namespace todrt
