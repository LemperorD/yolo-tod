// cpp_smoke.cpp —— 推理端架构自检（**不需要 GPU / TensorRT / OpenCV**）
//
// 定位与训练侧的 tests/smoke.py 一致：把"能被纯 CPU 验证的部分"全部验证掉，
// 剩下的（引擎构建、DLA 归属、真实延迟）留给实机。
//
// 覆盖：
//   1. 工厂自注册：模型/builder/preproc/postproc 按名字可取、别名可用、依赖完整
//   2. 配置：JSON 解析、DeployConfig 结构字段、schema 校验、硬件预设、JSON 往返
//   3. 前处理：letterbox 几何（scale/pad）、BGR→RGB、padding 值、灰度、stretch
//   4. 解码：anchor 网格、DFL 期望、anchor-major / feature-major 对拍、错误配置报错
//   5. NMS：硬 NMS / soft-NMS / class-agnostic / max_det
//   6. 端到端接线：前处理 → 后处理 → 坐标反变换（用合成张量，无需引擎）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/json.hpp"

namespace {

int g_checks = 0;
int g_failed = 0;

void Section(const std::string& s) { std::printf("\n== %s ==\n", s.c_str()); }

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (ok) {
    std::printf("  [ok]   %s\n", what.c_str());
  } else {
    ++g_failed;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

void CheckNear(float got, float want, float tol, const std::string& what) {
  const bool ok = std::fabs(got - want) <= tol;
  ++g_checks;
  if (ok) {
    std::printf("  [ok]   %s (=%.4f)\n", what.c_str(), got);
  } else {
    ++g_failed;
    std::printf("  [FAIL] %s: 期望 %.4f，实际 %.4f\n", what.c_str(), want, got);
  }
}

/// 期望"这里必须抛错"——静默出错是本项目最想避免的失败模式。
void CheckThrows(const std::function<void()>& fn, const std::string& what) {
  ++g_checks;
  try {
    fn();
    ++g_failed;
    std::printf("  [FAIL] %s：本应抛错但成功了\n", what.c_str());
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.size() > 64) msg = msg.substr(0, 64) + "...";
    std::printf("  [ok]   %s（如预期报错：%s）\n", what.c_str(), msg.c_str());
  }
}

using namespace todrt;

// ------------------------------------------------------------------ 1. 工厂

void TestFactory() {
  Section("1. 工厂自注册与按名字装配");

  const std::vector<std::string> models = model_names();
  Check(!models.empty(),
        "至少注册了一个变体（实际 " + std::to_string(models.size()) + " 个）");
  Check(has_model("SPAE_YOLOv8n"), "SPAE_YOLOv8n 已注册");
  Check(has_model("spae-yolov8"), "别名 spae-yolov8 可用");
  Check(has_model("YOLOv8n_baseline"), "YOLOv8n_baseline 已注册");
  Check(has_model("yolov8n"), "别名 yolov8n 可用");

  const std::vector<std::string> missing = check_registry();
  Check(missing.empty(), "注册表依赖完整（requires 都能解析）");
  for (const auto& m : missing) std::printf("        缺失：%s\n", m.c_str());

  Section("1b. 子工厂");
  Check(make_preprocessor("detect_letterbox") != nullptr, "前处理工厂可用（detect_letterbox）");
  Check(make_postprocessor("detect_nms") != nullptr, "后处理工厂可用（detect_nms）");
  CheckThrows([] { (void)make_preprocessor("不存在的策略"); }, "未知前处理报错");
  CheckThrows([] { (void)model_recipe("不存在的变体"); }, "未知变体报错");

  Section("1c. 装配自检（PlanAssembly，不需要 GPU）");
  DetectorOptions o;
  o.model = "SPAE_YOLOv8n";
  const AssemblyReport rep = PlanAssembly("SPAE_YOLOv8n", o);
  Check(rep.model == "SPAE_YOLOv8n", "报告里模型名正确");
  Check(rep.builder == "yolov8-trt",
        "builder 由配方解析为 yolov8-trt（实际 " + rep.builder + "）");
  Check(rep.preproc == "detect_letterbox", "preproc 由配方解析");
  Check(rep.postproc == "detect_nms", "postproc 由配方解析");
  Check(rep.strides == "4,8,16,32", "strides 来自部署配置/配方（实际 " + rep.strides + "）");
  Check(rep.layout == "anchor-major-dfl", "layout 解析正确（实际 " + rep.layout + "）");
  std::printf("--- PlanAssembly 输出 ---\n%s", rep.ToText().c_str());
  std::printf("--- 元数据（SPAE_YOLOv8n）---\n%s\n", registry_get("SPAE_YOLOv8n").source.c_str());
}

// ------------------------------------------------------------------ 2. 配置

void TestConfig() {
  Section("2. 部署配置解析与校验");

  const std::string cfg = R"({
    // 允许注释与尾随逗号（工程配置文件手写时很常见）
    "schema": "todrt.deploy/v1",
    "model": "SPAE_YOLOv8n",
    "model_id": "SPAE-YOLOv8n",
    "dataset": "visdrone2019-det",
    "nc": 10,
    "reg_max": 16,
    "layout": "anchor-major-dfl",
    "strides": [4, 8, 16, 32],
    "input": [640, 640],
    "pad_multiple": 32,
    "engine": { "onnx": "spae.onnx", "serialize_out": "spae.engine" },
    "hardware": { "device": "dla", "precision": "fp16", "dla_core": 0 },
    "runtime": { "cuda_graphs": true, "workspace_mb": 2048 },
    "postprocess": { "conf": 0.2, "iou": 0.5, "max_det": 500, "nms": "hard" },
    "preprocess": { "mode": "letterbox" },
  })";

  DetectorOptions o = DetectorOptions::FromJson(json::parse(cfg));
  Check(o.model == "SPAE_YOLOv8n", "model 解析正确（实际 " + o.model + "）");
  Check(o.deploy.num_classes == 10, "nc 解析正确");
  Check(o.deploy.reg_max == 16, "reg_max 解析正确");
  Check(o.deploy.strides.size() == 4, "strides 长度正确");
  Check(o.deploy.layout == OutputLayout::kAnchorMajorDfl, "layout 解析正确");
  Check(o.device == Device::kDla, "device 解析正确");
  Check(o.precision == Precision::kFP16, "precision 解析正确");
  Check(o.cuda_graphs, "runtime.cuda_graphs 解析正确（实际 " + std::string(o.cuda_graphs ? "true" : "false") + "）");
  Check(o.workspace_mb == 2048, "runtime.workspace_mb 覆盖默认值");
  CheckNear(o.postproc_opts.decode.conf_threshold, 0.2f, 1e-6f, "postprocess.conf 生效");
  Check(o.postproc_opts.decode.input_width == 640, "input 尺寸注入解码器");
  Check(o.deploy.build.onnx_path == "spae.onnx", "engine.onnx 解析正确");
  Check(o.deploy.build.engine_path.empty(), "engine_path 未设置时为空");
  Check(o.postproc_opts.decode.layout == OutputLayout::kAnchorMajorDfl,
        "postprocess 未覆盖时沿用 deploy.layout");

  Section("2b. 硬件预设");
  DetectorOptions p = DetectorOptions::FromJson(json::parse(cfg));
  p.ApplyPreset("dgp");
  Check(p.device == Device::kGpu, "preset=dgp → device=gpu");
  Check(p.cuda_graphs, "preset=dgp → CUDA Graph 开启");
  DetectorOptions q;
  q.ApplyPreset("orin");
  Check(q.device == Device::kDla && q.precision == Precision::kFP16, "preset=orin → DLA + FP16");
  Check(!q.dynamic_shape, "preset=orin → 固定 shape（DLA 更稳）");

  Section("2c. 必须报错的配置（静默出错是最坏情况）");
  CheckThrows(
      [] {
        DeployConfig d;
        d.layout = OutputLayout::kFeatureMajorDfl;
        d.Validate();
      },
      "feature-major 缺 level_channels 报错");
  CheckThrows(
      [] {
        DeployConfig d;
        d.reg_max = 15;  // 非 2 的幂
        d.Validate();
      },
      "reg_max 非 2 的幂报错");
  CheckThrows(
      [] {
        DeployConfig d;
        d.num_classes = 0;
        d.Validate();
      },
      "nc=0 报错");
  CheckThrows([] { (void)json::parse("{ \"a\": }"); }, "JSON 语法错误报错");
  CheckThrows([] { (void)parse_precision("fp8ish"); }, "未知精度报错");

  Section("2c-2. JSON 布尔值（曾经踩过的坑：Value(bool) 只写 num_ 不写 bool_）");
  {
    const json::Value jb = json::parse(R"({"t": true, "f": false, "n": 1})");
    Check(jb["t"].is_bool() && jb["t"].as_bool(false), "true 解析为 true");
    Check(jb["f"].is_bool() && !jb["f"].as_bool(true), "false 解析为 false");
    Check(jb["n"].as_bool(false), "数字 1 兼容为 true（配置容错）");
    Check(json::Value(true).as_bool(false), "直接构造 Value(true) 也是 true");
    const std::string rt = R"({"runtime": {"cuda_graphs": true, "verbose": false}})";
    const DetectorOptions jo = DetectorOptions::FromJson(json::parse(rt));
    Check(jo.cuda_graphs, "runtime.cuda_graphs=true 能一路传到 DetectorOptions");
    Check(!jo.verbose, "runtime.verbose=false 能一路传到 DetectorOptions");
  }

  Section("2d. JSON 往返一致");
  const std::string dumped = o.ToJson().dump(true);
  DetectorOptions back = DetectorOptions::FromJson(json::parse(dumped));
  Check(back.deploy.num_classes == o.deploy.num_classes, "往返后 nc 一致");
  Check(back.postproc_opts.decode.conf_threshold == o.postproc_opts.decode.conf_threshold,
        "往返后 conf 一致");
  Check(back.preproc_opts.input_width == o.preproc_opts.input_width, "往返后输入尺寸一致");
  Check(back.deploy.layout == o.deploy.layout, "往返后 layout 一致");
}

// ------------------------------------------------------------------ 3. 前处理

void TestPreprocess() {
  Section("3. 前处理（letterbox 几何 + 像素映射）");

  const int W = 1280, H = 720;
  std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3);
  for (size_t i = 0; i < px.size(); i += 3) {
    px[i] = 200;      // B
    px[i + 1] = 100;  // G
    px[i + 2] = 50;   // R
  }
  ImageView img;
  img.data = px.data();
  img.width = W;
  img.height = H;
  img.channels = 3;
  img.bgr = true;

  auto pre = make_detect_preprocessor(PreprocessOptions::ForInput(640, 640));
  const PreprocessResult r = pre->Run({img});

  // Ultralytics 语义：r = min(640/1280, 640/720) = 0.5 → 640×360，pad_y = 140
  CheckNear(r.scale[0], 0.5f, 1e-6f, "scale = min(640/1280, 640/720) = 0.5");
  CheckNear(r.pad_x[0], 0.f, 0.5f, "pad_x = 0（宽已占满）");
  CheckNear(r.pad_y[0], 140.f, 0.5f, "pad_y = (640-360)/2 = 140");
  Check(r.batch == 1 && r.width == 640 && r.height == 640, "输出张量形状 1×3×640×640");
  Check(r.tensor.size() == static_cast<size_t>(3) * 640 * 640, "张量元素数正确");

  const size_t plane = 640u * 640u;
  CheckNear(r.tensor[0], 114.f / 255.f, 1e-5f, "顶部 padding 值为 114/255");
  const size_t cy = 320, cx = 320;
  CheckNear(r.tensor[0 * plane + cy * 640 + cx], 50.f / 255.f, 2e-3f, "BGR→RGB：R = 50/255");
  CheckNear(r.tensor[1 * plane + cy * 640 + cx], 100.f / 255.f, 2e-3f, "G = 100/255");
  CheckNear(r.tensor[2 * plane + cy * 640 + cx], 200.f / 255.f, 2e-3f, "B = 200/255");

  Section("3b. 坐标反变换与越界裁剪");
  const BBox back =
      inv_transform(BBox{0.f, 140.f, 640.f, 640.f}, r.scale[0], r.pad_x[0], r.pad_y[0]);
  CheckNear(back.x1, 0.f, 0.5f, "反变换 x1 = 0");
  CheckNear(back.y1, 0.f, 0.5f, "反变换 y1 = 0");
  CheckNear(back.x2, 1280.f, 0.5f, "反变换 x2 = 原图宽");
  CheckNear(back.y2, 1000.f, 1.0f, "网络下边界超出原图（应被裁剪）");

  const auto clipped = pre->ToSourceCoords({{Detection{back, 0.5f, 0}}}, r);
  CheckNear(clipped[0][0].box.y2, 720.f, 0.5f, "ToSourceCoords 裁剪到原图高度");

  Section("3c. stretch 模式与灰度图");
  PreprocessOptions so = PreprocessOptions::ForInput(320, 320);
  so.mode = ResizeMode::kStretch;
  auto pre_s = make_detect_preprocessor(so);
  const PreprocessResult rs = pre_s->Run({img});
  CheckNear(rs.scale[0], 0.25f, 1e-6f, "stretch：scale = 320/1280");
  CheckNear(rs.pad_x[0], 0.f, 0.01f, "stretch：无 padding");

  std::vector<uint8_t> gray(static_cast<size_t>(W) * H, 128);
  ImageView gi;
  gi.data = gray.data();
  gi.width = W;
  gi.height = H;
  gi.channels = 1;
  const PreprocessResult rg = pre->Run({gi});
  CheckNear(rg.tensor[0 * plane + 320 * 640 + 320], 128.f / 255.f, 2e-3f, "灰度图 → 三通道复制");

  Section("3d. 无效输入必须报错");
  ImageView bad;
  CheckThrows([&] { (void)pre->Run({bad}); }, "空图像报错");
  CheckThrows([&] { (void)pre->Run({}); }, "空批次报错");
}

// ------------------------------------------------------------------ 4. 解码

/// 合成 anchor-major 输出：DFL 用大 logit 近似 one-hot，使距离期望 = bin。
std::vector<float> MakeAnchorMajorOutput(const DecodeOptions& opt, int W, int H, float conf,
                                         int class_id, int bin) {
  const std::vector<Grid> grids = make_grids(opt.strides, W, H);
  int64_t anchors = 0;
  for (const auto& g : grids) anchors += g.anchors();
  const int64_t no = 4 * opt.reg_max + opt.num_classes;
  std::vector<float> data(static_cast<size_t>(no) * static_cast<size_t>(anchors), 0.f);

  auto at = [&](int64_t ch, int64_t a) -> float& {
    return data[static_cast<size_t>(ch * anchors + a)];
  };
  int64_t a = 0;
  for (const auto& g : grids) {
    for (int y = 0; y < g.h; ++y) {
      for (int x = 0; x < g.w; ++x, ++a) {
        // 只让第一个 anchor（P2 层 (0,0)）高分，便于精确断言
        const bool target = (a == 0);
        const float cls = target ? conf : 0.01f;
        for (int c = 0; c < opt.num_classes; ++c) {
          at(4 * opt.reg_max + c, a) = (c == class_id) ? cls : 0.01f;
        }
        for (int grp = 0; grp < 4; ++grp) {
          for (int i = 0; i < opt.reg_max; ++i) {
            at(grp * opt.reg_max + i, a) = (i == bin) ? 12.f : -12.f;
          }
        }
      }
    }
  }
  return data;
}

TensorView ViewOf(const std::vector<float>& data, const std::vector<int64_t>& shape) {
  TensorView tv;
  tv.data = data.data();
  tv.dtype = DataType::kF32;
  tv.shape = shape;
  return tv;
}

void TestDecode() {
  Section("4. 解码（anchor 网格 / DFL / anchor-major）");

  DecodeOptions opt;
  opt.num_classes = 10;
  opt.reg_max = 16;
  opt.strides = {4, 8, 16, 32};
  opt.conf_threshold = 0.25f;
  opt.input_width = 640;
  opt.input_height = 640;

  const std::vector<Grid> grids = make_grids(opt.strides, 640, 640);
  Check(grids.size() == 4, "4 个特征层（P2–P5）");
  int64_t total = 0;
  for (const auto& g : grids) total += g.anchors();
  Check(total == 34000, "anchor 总数 = 34000（160²+80²+40²+20²）");
  Check(grids[0].w == 160 && grids[1].w == 80 && grids[2].w == 40 && grids[3].w == 20,
        "各层网格尺寸正确");

  Section("4b. DFL 期望");
  float dist[16];
  for (int i = 0; i < 16; ++i) dist[i] = (i == 5) ? 12.f : -12.f;
  CheckNear(dfl_expectation(dist, 16), 5.f, 1e-3f, "one-hot 分布 → 期望 = bin 索引");
  for (int i = 0; i < 16; ++i) dist[i] = 1.f;
  CheckNear(dfl_expectation(dist, 16), 7.5f, 1e-3f, "均匀分布 → 期望 = (n-1)/2");
  CheckNear(sigmoid(0.f), 0.5f, 1e-6f, "sigmoid(0) = 0.5");
  Check(sigmoid(100.f) > 0.999f && sigmoid(-100.f) < 0.001f, "sigmoid 极值稳定（不溢出）");

  Section("4c. anchor-major 解码（含 ultralytics 的 stride 缩放语义）");
  {
    const std::vector<float> data = MakeAnchorMajorOutput(opt, 640, 640, 0.9f, 3, 5);
    const TensorView tv =
        ViewOf(data, {1, 4 * opt.reg_max + opt.num_classes, 34000});
    const std::vector<Detection> dets = decode_predictions(tv, opt);
    Check(dets.size() == 1, "只解出高分的那 1 个目标（实际 " + std::to_string(dets.size()) + "）");
    if (!dets.empty()) {
      Check(dets[0].class_id == 3, "类别解码正确（3）");
      CheckNear(dets[0].score, 0.9f, 1e-5f, "分数解码正确");
      // anchor(0,0) 中心 = (0.5*4, 0.5*4) = (2,2)；DFL 距离 5 个**格子** → 5*stride=20 像素
      // （ultralytics: dfl 输出的是格数，乘 stride 才是像素）
      CheckNear(dets[0].box.x1, 2.f - 5.f * 4.f, 0.05f, "x1 = cx - l*stride");
      CheckNear(dets[0].box.y1, 2.f - 5.f * 4.f, 0.05f, "y1 = cy - t*stride");
      CheckNear(dets[0].box.x2, 2.f + 5.f * 4.f, 0.05f, "x2 = cx + r*stride");
      CheckNear(dets[0].box.y2, 2.f + 5.f * 4.f, 0.05f, "y2 = cy + b*stride");
    }
  }

  Section("4d. feature-major 与 anchor-major 对拍（同一份语义数据两种排布）");
  {
    const std::vector<float> am = MakeAnchorMajorOutput(opt, 640, 640, 0.9f, 3, 5);
    const int64_t no = 4 * opt.reg_max + opt.num_classes;
    const std::vector<int> level_ch(grids.size(), static_cast<int>(no));
    int64_t total_ch = 0;
    for (int c : level_ch) total_ch += c;

    std::vector<float> fm(static_cast<size_t>(total_ch) * 34000, 0.f);
    int64_t ch_base = 0, a_base = 0;
    for (const auto& g : grids) {
      for (int64_t ai = 0; ai < g.anchors(); ++ai) {
        const int64_t a_src = a_base + ai;
        for (int64_t c = 0; c < no; ++c) {
          fm[static_cast<size_t>((ch_base + c) * 34000 + a_src)] =
              am[static_cast<size_t>(c * 34000 + a_src)];
        }
      }
      ch_base += no;
      a_base += g.anchors();
    }

    DecodeOptions fopt = opt;
    fopt.layout = OutputLayout::kFeatureMajorDfl;
    fopt.level_channels = level_ch;

    const std::vector<Detection> d1 =
        decode_predictions(ViewOf(fm, {1, total_ch, 34000}), fopt);
    const std::vector<Detection> d2 = decode_predictions(ViewOf(am, {1, no, 34000}), opt);

    Check(d1.size() == d2.size(),
          "两种排布解出的目标数一致（" + std::to_string(d1.size()) + " vs " +
              std::to_string(d2.size()) + "）");
    if (!d1.empty() && !d2.empty()) {
      CheckNear(d1[0].box.x1, d2[0].box.x1, 1e-3f, "feature-major 与 anchor-major 的 x1 一致");
      CheckNear(d1[0].box.y2, d2[0].box.y2, 1e-3f, "两排布的 y2 一致");
      Check(d1[0].class_id == d2[0].class_id, "两排布类别一致");
    }
  }

  Section("4e. 转置排布 [B,A,C]");
  {
    const std::vector<float> am = MakeAnchorMajorOutput(opt, 640, 640, 0.9f, 3, 5);
    const int64_t no = 4 * opt.reg_max + opt.num_classes;
    std::vector<float> tr(static_cast<size_t>(34000) * static_cast<size_t>(no), 0.f);
    for (int64_t a = 0; a < 34000; ++a) {
      for (int64_t c = 0; c < no; ++c) {
        tr[static_cast<size_t>(a * no + c)] = am[static_cast<size_t>(c * 34000 + a)];
      }
    }
    DecodeOptions topt = opt;
    topt.layout = OutputLayout::kAnchorMajorDflTransposed;
    const std::vector<Detection> d = decode_predictions(ViewOf(tr, {1, 34000, no}), topt);
    Check(d.size() == 1, "转置排布解出 1 个目标");
    if (!d.empty()) CheckNear(d[0].box.x1, 2.f - 20.f, 0.05f, "转置排布 x1 与正排一致");
  }

  Section("4f. 错误配置必须报错（而不是给出错框）");
  {
    const std::vector<float> data = MakeAnchorMajorOutput(opt, 640, 640, 0.9f, 0, 5);
    const TensorView tv = ViewOf(data, {1, 4 * opt.reg_max + opt.num_classes, 34000});

    DecodeOptions bad = opt;
    bad.strides = {8, 16, 32};
    CheckThrows([&] { (void)decode_predictions(tv, bad); },
                "strides 与 anchor 数不匹配时报错");

    DecodeOptions bad2 = opt;
    bad2.num_classes = 80;
    CheckThrows([&] { (void)decode_predictions(tv, bad2); }, "nc 与输出通道不匹配时报错");

    CheckThrows([&] { (void)decode_predictions(ViewOf(data, {1, 84, 8400}), opt); },
                "输出形状与 strides 不符时报错");
  }

  Section("4g. plugin-nms（EfficientNMS）输出格式");
  {
    const float rows[12] = {10.f, 20.f, 30.f, 40.f, 0.9f, 1.f, 11.f, 21.f, 31.f, 41.f, 0.1f, 2.f};
    DecodeOptions p = opt;
    p.layout = OutputLayout::kPluginNms;
    TensorView tv;
    tv.data = rows;
    tv.dtype = DataType::kF32;
    tv.shape = {2, 6};
    const std::vector<Detection> real = decode_predictions(tv, p);
    Check(real.size() == 1, "plugin-nms：低于 conf 的被过滤（剩 1 条）");
    if (!real.empty()) CheckNear(real[0].box.x2, 30.f, 1e-6f, "plugin-nms 坐标搬运正确");
  }
}

// ------------------------------------------------------------------ 5. NMS

void TestNms() {
  Section("5. NMS");

  Detection a;
  a.box = {0, 0, 100, 100};
  a.score = 0.9f;
  a.class_id = 0;
  Detection b;
  b.box = {5, 5, 105, 105};
  b.score = 0.8f;
  b.class_id = 0;
  Detection c;
  c.box = {500, 500, 600, 600};
  c.score = 0.7f;
  c.class_id = 0;
  Detection d;
  d.box = {5, 5, 105, 105};
  d.score = 0.6f;
  d.class_id = 1;
  std::vector<Detection> ds = {a, b, c, d};

  CheckNear(iou(a.box, b.box), 95.f * 95.f / (10000.f + 10000.f - 9025.f), 1e-4f, "IoU 计算");
  CheckNear(iou(a.box, c.box), 0.f, 1e-6f, "无重叠 IoU = 0");

  std::vector<int> keep;
  nms_hard(ds, 0.5f, /*class_agnostic=*/false, keep);
  Check(keep.size() == 3, "按类别抑制：保留 a/c/d（实际 " + std::to_string(keep.size()) + "）");
  nms_hard(ds, 0.5f, /*class_agnostic=*/true, keep);
  Check(keep.size() == 2, "class-agnostic：d 也被抑制（实际 " + std::to_string(keep.size()) + "）");

  std::vector<Detection> soft = ds;
  nms_soft(soft, 0.05f, 0.5f, 0.001f, false, keep);
  Check(!keep.empty(), "soft-NMS 至少保留最高分框");
  Check(soft[1].score < 0.8f,
        "soft-NMS 衰减了重叠框分数（" + std::to_string(soft[1].score) + " < 0.8）");

  Section("5b. 后处理器（conf / max_det / batch）");
  PostprocessOptions po;
  po.decode.num_classes = 10;
  po.decode.reg_max = 16;
  po.decode.strides = {4, 8, 16, 32};
  po.decode.input_width = 640;
  po.decode.input_height = 640;
  po.decode.conf_threshold = 0.5f;
  po.iou_threshold = 0.45f;
  po.max_det = 1;
  auto post = make_nms_postprocessor(po);

  const std::vector<float> data = MakeAnchorMajorOutput(po.decode, 640, 640, 0.9f, 3, 5);
  PreprocessResult pre;
  pre.batch = 1;
  pre.width = 640;
  pre.height = 640;
  pre.scale = {0.5f};
  pre.pad_x = {0.f};
  pre.pad_y = {140.f};
  pre.src_w = {1280};
  pre.src_h = {720};

  const auto res = post->Run({ViewOf(data, {1, 4 * po.decode.reg_max + po.decode.num_classes, 34000})}, pre);
  Check(res.size() == 1, "后处理返回 batch=1 的结果");
  Check(res[0].size() <= 1, "max_det=1 生效（实际 " + std::to_string(res[0].size()) + "）");
}

// ------------------------------------------------------------------ 6. 端到端

void TestPipeline() {
  Section("6. 端到端接线（合成张量，无需引擎）");

  const int W = 1280, H = 720;
  std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 100);
  ImageView img;
  img.data = px.data();
  img.width = W;
  img.height = H;
  img.channels = 3;

  PostprocessOptions po;
  po.decode.num_classes = 10;
  po.decode.reg_max = 16;
  po.decode.strides = {4, 8, 16, 32};
  po.decode.input_width = 640;
  po.decode.input_height = 640;
  po.decode.conf_threshold = 0.25f;
  po.max_det = 300;

  auto pre = make_detect_preprocessor(PreprocessOptions::ForInput(640, 640));
  const PreprocessResult pp = pre->Run({img});
  CheckNear(pp.scale[0], 0.5f, 1e-6f, "端到端：letterbox scale 正确");
  CheckNear(pp.pad_y[0], 140.f, 0.5f, "端到端：pad_y 正确");

  auto post = make_nms_postprocessor(po);
  const std::vector<float> data = MakeAnchorMajorOutput(po.decode, 640, 640, 0.9f, 3, 4);
  auto dets = post->Run({ViewOf(data, {1, 4 * po.decode.reg_max + po.decode.num_classes, 34000})}, pp);
  auto src = pre->ToSourceCoords(std::move(dets), pp);
  Check(!src.empty() && !src[0].empty(), "端到端：解出目标");
  if (!src.empty() && !src[0].empty()) {
    const Detection& d = src[0][0];
    Check(d.box.x1 >= 0.f && d.box.y1 >= 0.f && d.box.x2 <= W && d.box.y2 <= H,
          "端到端：坐标回到原图并被裁剪");
  }

  Section("6b. 注册表总表（供文档使用）");
  const std::string catalog = registry_catalog("");
  Check(catalog.find("SPAE_YOLOv8n") != std::string::npos, "catalog 里含 SPAE_YOLOv8n");
  Check(catalog.find("model") != std::string::npos, "catalog 按工厂族分组");
}

}  // namespace

int main() {
  std::printf("todrt 推理端架构自检（不需要 GPU / TensorRT）\n");
  set_log_sink([](detail::LogLevel lv, const std::string& msg) {
    if (lv == detail::LogLevel::kInfo || lv == detail::LogLevel::kWarn ||
        lv == detail::LogLevel::kError) {
      std::printf("      | %s\n", msg.c_str());
    }
  });

  try {
    TestFactory();
    TestConfig();
    TestPreprocess();
    TestDecode();
    TestNms();
    TestPipeline();
  } catch (const std::exception& e) {
    std::printf("\n[未捕获异常] %s\n", e.what());
    return 1;
  }

  std::printf("\n============================================\n");
  std::printf("检查项：%d，失败：%d\n", g_checks, g_failed);
  if (g_failed == 0) {
    std::printf("全部通过 ✅（引擎构建 / DLA 归属 / 真实延迟请在目标机上验证）\n");
  } else {
    std::printf("存在失败项 ❌\n");
  }
  return g_failed == 0 ? 0 : 1;
}
