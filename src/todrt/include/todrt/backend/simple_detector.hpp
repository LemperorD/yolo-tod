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

// 输入与输出缓冲均在主机侧；后端须把输出转换为 float32 后交给通用解码器。
class IEngineRunner {
 public:
  virtual ~IEngineRunner() = default;
  virtual const std::string& name() const = 0;

  virtual EngineInputSpec input_spec() const = 0;
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  virtual std::vector<DataType> output_types() const = 0;

  virtual void Run(const void* input, size_t input_bytes, int batch,
                   const std::vector<float*>& outputs,
                   const std::vector<size_t>& output_elems) = 0;

  virtual double last_infer_ms() const = 0;
  virtual std::string engine_info() const = 0;
};

// 统一管理队列与工作线程，后端只实现一次推理所需的 RunBatch。
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
    // 异步提交立即拷贝像素，允许调用方随后复用相机帧缓冲。
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
    r.preprocess_ms = 0.0;
    r.infer_ms = sum_inf / iters;
    r.postprocess_ms = std::max(0.0, (sum_all - sum_inf) / iters);
    r.end_to_end_ms = sum_all / iters;
    r.fps = r.end_to_end_ms > 0 ? 1000.0 / r.end_to_end_ms : 0.0;
    return r;
  }

 protected:
  void set_preprocessor(std::unique_ptr<IPreprocessor> p) { pre_ = std::move(p); }
  const IPreprocessor* preprocessor() const { return pre_.get(); }

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

}
