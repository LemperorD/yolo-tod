// factory.cpp —— 工厂注册表、Detector 装配、异步骨架、PlanAssembly 自检
//
// 阅读顺序建议：
//   1. register_*_factory  —— 自注册表怎么被填满
//   2. Detector::Create    —— 名字怎么变成对象（model × builder × preproc × postproc）
//   3. AsyncDetectorBase   —— 异步骨架（队列/线程/计时），子类只需实现 RunBatch()
//   4. PlanAssembly        —— 无 GPU 也能跑的装配自检
#include "todrt/factory.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <thread>

namespace todrt {

// ------------------------------------------------------------------ 自注册表填充

namespace {

std::mutex g_factory_mutex;

std::unordered_map<std::string, ModelRecipeFn>& model_factories() {
  static std::unordered_map<std::string, ModelRecipeFn> m;
  return m;
}
std::unordered_map<std::string, BuilderFn>& builder_factories() {
  static std::unordered_map<std::string, BuilderFn> m;
  return m;
}
std::unordered_map<std::string, PreprocFn>& preproc_factories() {
  static std::unordered_map<std::string, PreprocFn> m;
  return m;
}
std::unordered_map<std::string, PostprocFn>& postproc_factories() {
  static std::unordered_map<std::string, PostprocFn> m;
  return m;
}

}  // namespace

void register_entry(const RegistryEntry& entry, std::vector<std::string> factories) {
  (void)factories;
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  auto& reg = detail::registry();
  auto& order = detail::registry_order();

  auto it = reg.find(entry.name);
  if (it != reg.end() && it->second.family != entry.family) {
    // 同名跨族 = 配置写错的前兆，直接报错而不是悄悄覆盖。
    std::ostringstream oss;
    oss << "注册名冲突：" << entry.name << " 已作为 " << it->second.family << " 注册（"
        << it->second.file << ":" << it->second.line << "），现在又作为 " << entry.family
        << " 注册（" << entry.file << ":" << entry.line << "）。";
    throw TritError(oss.str());
  }
  if (it == reg.end()) order.push_back(entry.name);
  reg[entry.name] = entry;

  for (const auto& a : entry.aliases) {
    if (a.empty()) continue;
    RegistryEntry alias = entry;
    alias.name = a;
    alias.canonical = false;
    if (!reg.count(a)) order.push_back(a);
    reg[a] = alias;
  }
}

void register_model_recipe(const std::string& name, ModelRecipeFn fn) {
  if (!fn) throw TritError("register_model_recipe: 空配方函数 " + name);
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  model_factories()[name] = fn;
  // 元数据：若 models/<x>.cpp 已显式登记过同名条目，则以显式元数据为准
  if (!registry_has(name)) {
    RegistryEntry e;
    e.name = name;
    e.family = "model";
    e.source = "builtin";
    e.file = "<recipe>";
    register_entry(e, {"model"});
  }
}

void register_model_metadata(const std::string& name, const std::vector<std::string>& aliases,
                             const std::string& source, const std::string& license,
                             const std::string& hardware, const std::string& cost,
                             const char* file, int line) {
  RegistryEntry e;
  e.name = name;
  e.family = "model";
  e.source = source;
  e.license = license;
  e.hardware = hardware;
  e.cost = cost;
  e.aliases = aliases;
  e.file = file ? file : "<meta>";
  e.line = line;
  register_entry(e, {"model"});

  // 让别名也能直接当工厂键用（用户可能写 "spae-yolov8"）。
  // 注意：这里的 recipe 一定已经注册——同一 TU 内静态初始化自上而下，
  // TOD_RT_DEFINE_RECIPE 在 TOD_RT_META 之前。跨 TU 时按名查找，取不到就跳过。
  ModelRecipeFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    auto it = model_factories().find(name);
    if (it != model_factories().end()) fn = it->second;
  }
  if (!fn) return;
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  for (const auto& a : aliases) {
    if (!a.empty()) model_factories()[a] = fn;
  }
}

void register_builder_factory(const std::string& name, BuilderFn fn) {
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  builder_factories()[name] = fn;
}

void register_preproc_factory(const std::string& name, PreprocFn fn) {
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  preproc_factories()[name] = fn;
}

void register_postproc_factory(const std::string& name, PostprocFn fn) {
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  postproc_factories()[name] = fn;
}

bool has_model(const std::string& name) {
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    if (model_factories().count(name) != 0) return true;
  }
  // 退回注册表：TOD_RT_META 里登记的别名（含连字符写法）也应能解析到规范配方
  return registry_has(name) && registry_get(name).family == "model";
}

namespace {
/// 把「别名 / 规范名」归一化成配方函数。注册表里的别名条目 Name 仍是别名本身，
/// 所以这里用 RegistryEntry::aliases 反查一次规范名。
ModelRecipeFn find_recipe(const std::string& name) {
  std::lock_guard<std::mutex> lk(g_factory_mutex);
  auto& m = model_factories();
  auto it = m.find(name);
  if (it != m.end()) return it->second;
  if (!registry_has(name)) return nullptr;
  const RegistryEntry& e = registry_get(name);
  if (e.family != "model") return nullptr;
  // 该条目本身就是规范名？
  it = m.find(e.name);
  if (it != m.end()) return it->second;
  // 反查：谁的 aliases 里含这个名字
  for (const auto& kv : m) {
    if (!registry_has(kv.first)) continue;
    const RegistryEntry& cand = registry_get(kv.first);
    for (const auto& a : cand.aliases) {
      if (a == name) return kv.second;
    }
    if (cand.name == e.name) return kv.second;
  }
  return nullptr;
}
}  // namespace

std::vector<std::string> model_names() {
  // 只返回"规范名"（canonical）：别名会被注册表标成 canonical=false，
  // 这样 CLI 列表不会出现一堆同义项，而别名依然能当工厂键使用。
  std::vector<std::string> all;
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    all.reserve(model_factories().size());
    for (const auto& kv : model_factories()) all.push_back(kv.first);
  }
  std::vector<std::string> out;
  out.reserve(all.size());
  for (const auto& n : all) {
    if (registry_has(n) && !registry_get(n).canonical) continue;
    out.push_back(n);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

ModelRecipe model_recipe(const std::string& name) {
  ModelRecipeFn fn = find_recipe(name);
  if (!fn) {
    std::ostringstream oss;
    oss << "没有名为 " << name << " 的变体配方。已注册：";
    for (const auto& n : model_names()) oss << n << " ";
    oss << "（新增变体见 src/todrt/src/models/ 与 README 模板）";
    throw TritError(oss.str());
  }
  return fn();
}

std::unique_ptr<IPreprocessor> make_preprocessor(const std::string& name) {
  PreprocFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    auto it = preproc_factories().find(name);
    if (it != preproc_factories().end()) fn = it->second;
  }
  if (!fn) throw TritError("未注册的前处理策略：" + name);
  return fn();
}

std::unique_ptr<IPostprocessor> make_postprocessor(const std::string& name) {
  PostprocFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    auto it = postproc_factories().find(name);
    if (it != postproc_factories().end()) fn = it->second;
  }
  if (!fn) throw TritError("未注册的后处理策略：" + name);
  return fn();
}

std::unique_ptr<IModelBuilder> make_builder(const std::string& name) {
  BuilderFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_factory_mutex);
    auto it = builder_factories().find(name);
    if (it != builder_factories().end()) fn = it->second;
  }
  if (!fn) throw TritError("未注册的引擎构建器：" + name);
  return fn();
}

// ------------------------------------------------------------------ 名字解析

namespace {

/// 把 "auto" 解析成具体实现名：优先变体配方，其次全局默认。
std::string resolve(const std::string& requested, const std::string& from_recipe,
                    const std::string& fallback) {
  if (!requested.empty() && requested != "auto") return requested;
  if (!from_recipe.empty() && from_recipe != "auto") return from_recipe;
  return fallback;
}

std::string join_ints(const std::vector<int>& v) {
  std::ostringstream oss;
  for (size_t i = 0; i < v.size(); ++i) oss << (i ? "," : "") << v[i];
  return oss.str();
}

/// 把变体配方的默认值补进 options（DeployConfig 里的结构信息优先级最高）。
void merge_recipe_defaults(DetectorOptions& o, const ModelRecipe& r) {
  PostprocessOptions& pp = o.postproc_opts;

  if (pp.decode.level_channels.empty()) pp.decode.level_channels = o.deploy.level_channels;
  if (pp.decode.strides == std::vector<int>{4, 8, 16, 32} && !r.decode.strides.empty()) {
    pp.decode.strides = r.decode.strides;
  }
  if (pp.decode.layout == OutputLayout::kAnchorMajorDfl &&
      r.layout != OutputLayout::kAnchorMajorDfl) {
    pp.decode.layout = r.layout;
  }

  // 纯部署偏好：只有用户没改过默认值时才套用配方
  if (pp.decode.conf_threshold == 0.25f) pp.decode.conf_threshold = r.decode.conf_threshold;
  if (pp.iou_threshold == 0.45f) pp.iou_threshold = r.postproc_opts.iou_threshold;
  if (pp.max_det == 300) pp.max_det = r.postproc_opts.max_det;
  if (pp.decode.max_det == 300) pp.decode.max_det = pp.max_det;
  if (r.postproc_opts.class_agnostic) pp.class_agnostic = true;
  if (r.postproc_opts.nms == NmsKind::kSoft) pp.nms = NmsKind::kSoft;

  // 结构字段最终与 deploy 对齐（配置永远压过代码里的默认）
  pp.decode.num_classes = o.deploy.num_classes;
  pp.decode.reg_max = o.deploy.reg_max;
  if (o.deploy.layout != OutputLayout::kAnchorMajorDfl) pp.decode.layout = o.deploy.layout;
  if (!o.deploy.strides.empty() && o.deploy.strides != std::vector<int>{4, 8, 16, 32}) {
    pp.decode.strides = o.deploy.strides;
  }
  if (!o.deploy.level_channels.empty()) pp.decode.level_channels = o.deploy.level_channels;

  if (o.preproc_opts.input_width == 640 && o.preproc_opts.input_height == 640) {
    o.preproc_opts.input_width = o.deploy.input[0];
    o.preproc_opts.input_height = o.deploy.input[1];
  }
  pp.decode.input_width = o.preproc_opts.input_width;
  pp.decode.input_height = o.preproc_opts.input_height;
}

}  // namespace

// ------------------------------------------------------------------ 异步骨架

namespace {

/// 异步 Detector 基类：Submit/TryGet/Wait 的队列、线程、计时都在这，
/// 子类（如 TrtDetector）只实现 RunBatch()。
class AsyncDetectorBase : public Detector {
 public:
  explicit AsyncDetectorBase(DetectorOptions opts) { mutable_options() = std::move(opts); }

  ~AsyncDetectorBase() override { StopWorker(); }

  const std::string& model_name() const override { return model_; }
  const DetectorOptions& options() const override { return prototype_options(); }

  uint64_t Submit(const ImageView& image) override {
    if (!image.valid()) throw TritError("Submit: 输入图像无效（空指针或尺寸为 0）");
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

  std::vector<std::vector<Detection>> Run(const std::vector<ImageView>& images) override {
    std::vector<std::vector<Detection>> out;
    if (images.empty()) return out;
    if (!pre_) throw TritError("Detector 未配置前处理器");
    out.reserve(images.size());
    // 同步路径直接串行执行，不经过队列（避免与异步请求抢缓冲）。
    for (const ImageView& img : images) {
      if (!img.valid()) throw TritError("Run: 输入图像无效");
      const std::vector<ImageView> one_in{img};
      PreprocessResult pp = pre_->Run(one_in);
      double a = 0, b = 0, c = 0;
      out.push_back(RunBatch(pp, &a, &b, &c));
    }
    return out;
  }

  Detector::BenchResult Bench(const ImageView& image, int warmup, int iters) override {
    if (!image.valid()) throw TritError("Bench: 输入图像无效");
    if (!pre_) throw TritError("Detector 未配置前处理器");
    BenchResult r;
    r.device = to_string(options().device);
    r.precision = to_string(options().precision);
    if (iters <= 0) return r;

    // 预热：TensorRT/GPU 首次执行会有明显额外开销，不计入统计。
    PreprocessResult pp = pre_->Run(std::vector<ImageView>{image});
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
    r.infer_ms = sum_inf / iters;
    r.postprocess_ms = std::max(0.0, (sum_all - sum_inf) / iters);
    r.end_to_end_ms = sum_all / iters;
    r.fps = r.end_to_end_ms > 0 ? 1000.0 / r.end_to_end_ms : 0.0;
    return r;
  }

 protected:
  void set_model_name(std::string n) { model_ = std::move(n); }
  void set_preprocessor(std::unique_ptr<IPreprocessor> p) { pre_ = std::move(p); }

  std::unique_ptr<IPreprocessor> pre_;

 private:
  struct Job {
    uint64_t id = 0;
    ImageView view;
    std::vector<uint8_t> pixels;
    std::vector<Detection> dets;
    double pre_ms = 0, inf_ms = 0, post_ms = 0;
    bool done = false;
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
        job->done = true;
      } catch (const std::exception& e) {
        log_error(std::string("异步推理失败：") + e.what());
        job->done = true;  // 失败也投递（空结果），避免调用方永久阻塞
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

  std::string model_;
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

}  // namespace

// ------------------------------------------------------------------ 装配

std::unique_ptr<Detector> Detector::Create(const std::string& model_name_in,
                                           const DetectorOptions& opts_in) {
  const std::string model_name = model_name_in.empty() ? opts_in.model : model_name_in;
  if (model_name.empty()) throw TritError("Detector::Create: 未指定模型名");
  if (!has_model(model_name)) {
    std::ostringstream oss;
    oss << "未知模型 " << model_name << "；已注册变体：";
    for (const auto& n : model_names()) oss << n << " ";
    throw TritError(oss.str());
  }

  ModelRecipe recipe = model_recipe(model_name);

  DetectorOptions opts = opts_in;
  if (!recipe.hardware_preset.empty()) opts.ApplyPreset(recipe.hardware_preset);
  merge_recipe_defaults(opts, recipe);
  if (recipe.tune) recipe.tune(opts);
  opts.model = model_name;

  const std::string builder_name = resolve(opts.builder, recipe.builder, "auto");
  const std::string preproc_name = resolve(opts.preproc, recipe.preproc, "detect_letterbox");
  const std::string postproc_name = resolve(opts.postproc, recipe.postproc, "detect_nms");

  if (builder_name == "auto") {
    throw TritError(
        "没有可用的引擎构建器：请在 DetectorOptions.builder 指定，"
        "或给注册的变体配方填 ModelRecipe::builder。");
  }

  log_info("装配 Detector: model=" + model_name + " builder=" + builder_name +
           " preproc=" + preproc_name + " postproc=" + postproc_name +
           " device=" + to_string(opts.device) + " precision=" + to_string(opts.precision));

  return CreateDeviceDetector(model_name, opts, recipe, builder_name, preproc_name, postproc_name);
}

std::unique_ptr<Detector> Detector::CreateFromFile(const std::string& config_path) {
  DetectorOptions opts = DetectorOptions::FromFile(config_path);
  return Create(opts.model, opts);
}

// ------------------------------------------------------------------ 装配自检

AssemblyReport PlanAssembly(const std::string& model_name, const DetectorOptions& opts_in) {
  AssemblyReport rep;
  rep.model = model_name;
  rep.registry_missing = check_registry();

  ModelRecipe recipe;
  try {
    recipe = model_recipe(model_name);
  } catch (const std::exception& e) {
    rep.warnings.push_back(std::string("找不到变体配方：") + e.what());
    return rep;
  }

  DetectorOptions opts = opts_in;
  opts.model = model_name;
  if (!recipe.hardware_preset.empty()) opts.ApplyPreset(recipe.hardware_preset);
  merge_recipe_defaults(opts, recipe);
  if (recipe.tune) recipe.tune(opts);

  rep.builder = resolve(opts.builder, recipe.builder, "auto");
  rep.preproc = resolve(opts.preproc, recipe.preproc, "detect_letterbox");
  rep.postproc = resolve(opts.postproc, recipe.postproc, "detect_nms");
  rep.device = to_string(opts.device);
  rep.precision = to_string(opts.precision);
  rep.layout = to_string(opts.deploy.layout);
  rep.strides = join_ints(opts.deploy.strides);

  // 子工厂存在性检查（不需要 GPU）
  try {
    (void)make_preprocessor(rep.preproc);
  } catch (const std::exception& e) {
    rep.warnings.push_back(std::string("前处理不可用：") + e.what());
  }
  try {
    (void)make_postprocessor(rep.postproc);
  } catch (const std::exception& e) {
    rep.warnings.push_back(std::string("后处理不可用：") + e.what());
  }
  try {
    std::unique_ptr<IModelBuilder> builder = make_builder(rep.builder);
    std::string reason;
    rep.builder_available = builder->Available(&reason);
    rep.builder_unavailable_reason = reason;
  } catch (const std::exception& e) {
    rep.builder_available = false;
    rep.builder_unavailable_reason = e.what();
    rep.warnings.push_back(std::string("构建器不可用：") + e.what());
  }

  // 硬件/精度一致性
  if (opts.device == Device::kDla && opts.precision == Precision::kFP32) {
    rep.warnings.push_back("DLA 不支持 FP32，请改用 FP16 或 INT8。");
  }
  if (opts.device == Device::kDla && opts.dynamic_shape) {
    rep.warnings.push_back("DLA 上动态 shape 支持有限，建议固定输入尺寸（dynamic_shape=false）。");
  }
  if (opts.deploy.layout == OutputLayout::kFeatureMajorDfl &&
      opts.postproc_opts.decode.level_channels.empty()) {
    rep.warnings.push_back(
        "layout=feature-major-dfl 必须提供 level_channels（部署配置里的 features）。");
  }

  // anchor 数自检：解码器会拿这个数字与引擎输出对拍，先在这里算给用户看
  {
    const std::vector<Grid> grids = make_grids(opts.postproc_opts.decode.strides,
                                               opts.preproc_opts.input_width,
                                               opts.preproc_opts.input_height);
    int64_t total = 0;
    std::ostringstream oss;
    oss << "输入 " << opts.preproc_opts.input_width << "×" << opts.preproc_opts.input_height
        << " strides={" << rep.strides << "} → anchor=";
    for (size_t i = 0; i < grids.size(); ++i) {
      total += grids[i].anchors();
      oss << (i ? " + " : "") << grids[i].w << "×" << grids[i].h;
    }
    oss << " = " << total << "（解码时会与引擎输出形状对拍）";
    log_info(oss.str());
  }
  return rep;
}

std::string AssemblyReport::ToText() const {
  std::ostringstream oss;
  oss << "装配自检（未加载引擎）\n";
  oss << "  模型      : " << model << "\n";
  oss << "  构建器    : " << builder
      << (builder_available ? "  [可用: " + builder_unavailable_reason + "]"
                            : "  [不可用: " + builder_unavailable_reason + "]")
      << "\n";
  oss << "  前处理    : " << preproc << "\n";
  oss << "  后处理    : " << postproc << "\n";
  oss << "  设备/精度 : " << device << " / " << precision << "\n";
  oss << "  输出布局  : " << layout << "  strides=" << strides << "\n";
  if (!registry_missing.empty()) {
    oss << "  注册表缺失:\n";
    for (const auto& m : registry_missing) oss << "    - " << m << "\n";
  }
  if (!warnings.empty()) {
    oss << "  警告:\n";
    for (const auto& w : warnings) oss << "    ! " << w << "\n";
  }
  return oss.str();
}

std::string AssemblyReport::ToJson() const {
  auto esc = [](const std::string& s) {
    std::string o;
    for (char c : s) {
      if (c == '"' || c == '\\') o.push_back('\\');
      o.push_back(c);
    }
    return o;
  };
  std::ostringstream oss;
  oss << "{\n  \"model\": \"" << esc(model) << "\",\n"
      << "  \"builder\": \"" << esc(builder) << "\",\n"
      << "  \"builder_available\": " << (builder_available ? "true" : "false") << ",\n"
      << "  \"builder_unavailable_reason\": \"" << esc(builder_unavailable_reason) << "\",\n"
      << "  \"preproc\": \"" << esc(preproc) << "\",\n"
      << "  \"postproc\": \"" << esc(postproc) << "\",\n"
      << "  \"device\": \"" << esc(device) << "\",\n"
      << "  \"precision\": \"" << esc(precision) << "\",\n"
      << "  \"layout\": \"" << esc(layout) << "\",\n"
      << "  \"strides\": [" << strides << "],\n";
  auto arr = [&](const char* key, const std::vector<std::string>& v, bool last) {
    oss << "  \"" << key << "\": [";
    for (size_t i = 0; i < v.size(); ++i) oss << (i ? ", " : "") << "\"" << esc(v[i]) << "\"";
    oss << "]" << (last ? "\n" : ",\n");
  };
  arr("warnings", warnings, false);
  arr("registry_missing", registry_missing, true);
  oss << "}\n";
  return oss.str();
}

}  // namespace todrt
