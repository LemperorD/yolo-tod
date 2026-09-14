// preprocess.cpp —— 前处理策略：BGR/RGB HWC uint8 → NCHW float
//
// 为什么自己写而不用 OpenCV：实机上（Jetson + V4L2/GStreamer）常常希望少一层依赖，
// 而且这里必须精确复现 Ultralytics 的 letterbox 参数（缩放系数与 padding 取整），
// 否则框会系统性偏移几个像素 —— 对小目标检测来说这是致命的。
#include <algorithm>
#include <cmath>
#include <cstring>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/modules.hpp"

namespace todrt {
namespace {

inline float pixel_at(const ImageView& img, int x, int y, int c) {
  const uint8_t* row = img.data + static_cast<size_t>(y) * img.row_bytes();
  const uint8_t v = row[static_cast<size_t>(x) * static_cast<size_t>(img.channels) +
                        static_cast<size_t>(c)];
  return static_cast<float>(v);
}

/// 双线性采样（通道 c），坐标在**原图**像素坐标系内。
inline float bilinear(const ImageView& img, float sx, float sy, int c, bool clamp_edges) {
  const int x0 = static_cast<int>(std::floor(sx));
  const int y0 = static_cast<int>(std::floor(sy));
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);

  auto fetch = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || x >= img.width || y >= img.height) {
      if (!clamp_edges) return 0.f;
      x = std::min(std::max(x, 0), img.width - 1);
      y = std::min(std::max(y, 0), img.height - 1);
    }
    return pixel_at(img, x, y, c);
  };

  const float v00 = fetch(x0, y0);
  const float v10 = fetch(x0 + 1, y0);
  const float v01 = fetch(x0, y0 + 1);
  const float v11 = fetch(x0 + 1, y0 + 1);
  const float top = v00 + (v10 - v00) * fx;
  const float bot = v01 + (v11 - v01) * fx;
  return top + (bot - top) * fy;
}

/// 单张图的几何规划。
struct Plan {
  float scale = 1.f;
  int pad_x = 0;
  int pad_y = 0;
  int new_w = 0;
  int new_h = 0;
};

Plan plan_one(const ImageView& img, const PreprocessOptions& o) {
  Plan p;
  if (o.mode == ResizeMode::kStretch) {
    p.scale = static_cast<float>(o.input_width) / static_cast<float>(img.width);
    p.new_w = o.input_width;
    p.new_h = o.input_height;
    return p;
  }
  float r = std::min(static_cast<float>(o.input_width) / static_cast<float>(img.width),
                     static_cast<float>(o.input_height) / static_cast<float>(img.height));
  if (o.mode == ResizeMode::kIntegerScale) {
    // 只允许整数倍：避免小目标被非整数倍重采样"抹平"
    r = std::max(1.f, std::floor(r));
  }
  p.scale = r;
  p.new_w = static_cast<int>(std::round(static_cast<float>(img.width) * r));
  p.new_h = static_cast<int>(std::round(static_cast<float>(img.height) * r));
  // 与 Ultralytics 一致：先算总 padding 再各自整除 2
  p.pad_x = (o.input_width - p.new_w) / 2;
  p.pad_y = (o.input_height - p.new_h) / 2;
  return p;
}

/// 检测用前处理器：letterbox / stretch / integer-scale 三合一（由 Options 选择）。
class DetectPreprocessor : public IPreprocessor {
 public:
  DetectPreprocessor() : DetectPreprocessor(PreprocessOptions{}) {}
  explicit DetectPreprocessor(PreprocessOptions o) : opt_(o) {
    if (opt_.input_width <= 0 || opt_.input_height <= 0) {
      throw TritError("前处理输入尺寸非法");
    }
    if (opt_.pad_multiple > 1 &&
        (opt_.input_width % opt_.pad_multiple || opt_.input_height % opt_.pad_multiple)) {
      log_warn("输入尺寸 " + std::to_string(opt_.input_width) + "×" +
               std::to_string(opt_.input_height) + " 不是 pad_multiple=" +
               std::to_string(opt_.pad_multiple) + " 的整数倍；请确认与导出时一致。");
    }
  }

  const std::string& name() const override { return name_; }
  const PreprocessOptions& options() const override { return opt_; }
  int out_width() const override { return opt_.input_width; }
  int out_height() const override { return opt_.input_height; }

  PreprocessResult Run(const std::vector<ImageView>& images) override {
    if (images.empty()) throw TritError("前处理收到空批次");
    for (const auto& im : images) {
      if (!im.valid()) throw TritError("前处理收到无效图像（空指针或尺寸为 0）");
    }

    PreprocessResult out;
    out.width = opt_.input_width;
    out.height = opt_.input_height;
    out.batch = static_cast<int>(images.size());
    out.channels = 3;
    out.tensor.assign(static_cast<size_t>(out.batch) * out.sample_stride(), 0.f);
    out.scale.resize(images.size());
    out.pad_x.resize(images.size());
    out.pad_y.resize(images.size());
    out.src_w.resize(images.size());
    out.src_h.resize(images.size());

    const size_t plane = static_cast<size_t>(out.height) * static_cast<size_t>(out.width);

    for (size_t n = 0; n < images.size(); ++n) {
      const ImageView& img = images[n];
      const Plan p = plan_one(img, opt_);

      out.scale[n] = p.scale;
      out.pad_x[n] = static_cast<float>(p.pad_x);
      out.pad_y[n] = static_cast<float>(p.pad_y);
      out.src_w[n] = img.width;
      out.src_h[n] = img.height;

      float* dst = out.tensor.data() + n * out.sample_stride();

      // 1) 先按填充策略铺满整幅
      switch (opt_.pad_value) {
        case PadValue::kZero:
          std::fill(dst, dst + out.sample_stride(), opt_.norm_bias);  // 0 * scale + bias
          break;
        case PadValue::kGray114: {
          const float v = 114.f * opt_.norm_scale + opt_.norm_bias;
          std::fill(dst, dst + out.sample_stride(), v);
          break;
        }
        case PadValue::kEdge: {
          // 边缘复制：填充区取原图最近边缘像素
          for (int y = 0; y < out.height; ++y) {
            const int sy = std::min(std::max(y - p.pad_y, 0), img.height - 1);
            for (int x = 0; x < out.width; ++x) {
              const int sx = std::min(std::max(x - p.pad_x, 0), img.width - 1);
              for (int c = 0; c < 3; ++c) {
                dst[c * plane + static_cast<size_t>(y) * out.width + x] =
                    pixel_at(img, sx, sy, src_channel(c, img)) * opt_.norm_scale +
                    opt_.norm_bias;
              }
            }
          }
          break;
        }
      }

      // 2) 再把缩放后的图像覆盖到 padded 区域
      const bool clamp_edges = p.scale >= 1.f;
      for (int y = 0; y < p.new_h; ++y) {
        const int dy = y + p.pad_y;
        if (dy < 0 || dy >= out.height) continue;
        const float sy = (static_cast<float>(y) + 0.5f) / p.scale - 0.5f;
        for (int x = 0; x < p.new_w; ++x) {
          const int dx = x + p.pad_x;
          if (dx < 0 || dx >= out.width) continue;
          const float sx = (static_cast<float>(x) + 0.5f) / p.scale - 0.5f;
          for (int c = 0; c < 3; ++c) {
            const float v = bilinear(img, sx, sy, src_channel(c, img), clamp_edges);
            dst[c * plane + static_cast<size_t>(dy) * out.width + dx] =
                v * opt_.norm_scale + opt_.norm_bias;
          }
        }
      }
    }
    return out;
  }

 private:
  /// 目标通道 c（0=R,1=G,2=B）对应的源图通道索引（灰度图复用同一通道）。
  int src_channel(int c, const ImageView& img) const {
    if (img.channels == 1) return 0;
    if (img.channels == 4) return c;  // BGRA/RGBA：丢弃 alpha
    // 3 通道：BGR 输入要反序才得到 RGB；to_rgb=false 时保持源顺序
    if (!opt_.to_rgb) return c;
    return img.bgr ? (2 - c) : c;
  }

  std::string name_ = "detect-letterbox";
  PreprocessOptions opt_;
};

}  // namespace

// ------------------------------------------------------------------ 接口默认实现

BBox inv_transform(const BBox& b, float scale, float pad_x, float pad_y) {
  const float s = scale > 0.f ? scale : 1.f;
  BBox o;
  o.x1 = (b.x1 - pad_x) / s;
  o.y1 = (b.y1 - pad_y) / s;
  o.x2 = (b.x2 - pad_x) / s;
  o.y2 = (b.y2 - pad_y) / s;
  return o;
}

std::vector<std::vector<Detection>> IPreprocessor::ToSourceCoords(
    std::vector<std::vector<Detection>> dets, const PreprocessResult& pre) const {
  for (size_t n = 0; n < dets.size(); ++n) {
    if (n >= pre.scale.size()) break;
    const float s = pre.scale[n];
    const float px = pre.pad_x[n];
    const float py = pre.pad_y[n];
    const float max_x = static_cast<float>(pre.src_w[n]);
    const float max_y = static_cast<float>(pre.src_h[n]);
    for (auto& d : dets[n]) {
      d.box = inv_transform(d.box, s, px, py);
      d.box.x1 = std::min(std::max(d.box.x1, 0.f), max_x);
      d.box.y1 = std::min(std::max(d.box.y1, 0.f), max_y);
      d.box.x2 = std::min(std::max(d.box.x2, 0.f), max_x);
      d.box.y2 = std::min(std::max(d.box.y2, 0.f), max_y);
    }
    std::sort(dets[n].begin(), dets[n].end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
  }
  return dets;
}

// ------------------------------------------------------------------ 注册与构造

namespace {
std::unique_ptr<IPreprocessor> FactoryDetectPreproc() {
  return std::unique_ptr<IPreprocessor>(new DetectPreprocessor());
}
}  // namespace

TOD_RT_REGISTER_PREPROC(detect_letterbox, FactoryDetectPreproc)
TOD_RT_REGISTER_PREPROC(letterbox, FactoryDetectPreproc)
TOD_RT_REGISTER_PREPROC(detect, FactoryDetectPreproc)

/// 供后端按部署配置构造带参数的前处理器。
std::unique_ptr<IPreprocessor> make_detect_preprocessor(const PreprocessOptions& o) {
  return std::unique_ptr<IPreprocessor>(new DetectPreprocessor(o));
}

}  // namespace todrt
