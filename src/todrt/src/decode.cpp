// decode.cpp —— YOLOv8 系列检测头输出的 CPU 解码（DFL / sigmoid / anchor 网格）
//
// 这一层的定位是「唯一真相」：GPU 端插件只是加速，语义必须与这里一致；
// 与 Python 实现对拍、以及本库的单元测试，都以这里的输出为基准。
//
// 支持四种导出排布（见 OutputLayout），其中两种是本项目实际会遇到的：
//   * kAnchorMajorDfl          —— ultralytics 官方 YOLOv8 Detect 导出，[B, 4*reg_max+nc, A]
//   * kFeatureMajorDfl         —— Efficient_UAVDet 独立实现，[B, Σ(4*reg_max+nc), A]
//   * kAnchorMajorDflTransposed—— 转置版，[B, A, 4*reg_max+nc]
//   * kPluginNms               —— EfficientNMS 等插件已完成解码，[N, 6] 或 [N, 7]
#include <algorithm>
#include <cmath>
#include <cstring>

#include "todrt/modules.hpp"

namespace todrt {
namespace {

/// 半精度 → 单精度。
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // 次正规数
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/// 把任意受支持 dtype 的 blob 读成 float。
class Reader {
 public:
  explicit Reader(const TensorView& v) {
    switch (v.dtype) {
      case DataType::kF32: f32_ = static_cast<const float*>(v.data); break;
      case DataType::kF16: f16_ = static_cast<const uint16_t*>(v.data); break;
      case DataType::kI8: i8_ = static_cast<const int8_t*>(v.data); break;
      case DataType::kU8: u8_ = static_cast<const uint8_t*>(v.data); break;
      case DataType::kI32: i32_ = static_cast<const int32_t*>(v.data); break;
    }
  }

  inline float at(int64_t i) const {
    if (f32_) return f32_[i];
    if (f16_) return half_to_float(f16_[i]);
    if (i8_) return static_cast<float>(i8_[i]);
    if (u8_) return static_cast<float>(u8_[i]);
    if (i32_) return static_cast<float>(i32_[i]);
    throw TritError("解码器不支持的数据类型");
  }

 private:
  const float* f32_ = nullptr;
  const uint16_t* f16_ = nullptr;
  const int8_t* i8_ = nullptr;
  const uint8_t* u8_ = nullptr;
  const int32_t* i32_ = nullptr;
};

/// softmax 后求期望（Kahan 无关紧要，分布长度最多 64）。
inline float expectation(const float* d, int n) {
  float m = d[0];
  for (int i = 1; i < n; ++i) m = std::max(m, d[i]);
  float w = 0.f;
  float t = 0.f;
  for (int i = 0; i < n; ++i) {
    const float e = std::exp(d[i] - m);
    t += e;
    w += e * static_cast<float>(i);
  }
  return t > 0.f ? w / t : 0.f;
}

/// 校验 anchor 数与 strides 推算是否一致——这是最常见的配置错误，必须响亮报错。
void check_anchors(const TensorView& out, const std::vector<Grid>& grids, int64_t anchors,
                   const char* layout_name) {
  int64_t expected = 0;
  for (const auto& g : grids) expected += g.anchors();
  if (expected != anchors) {
    std::string detail;
    for (const auto& g : grids) {
      detail += std::to_string(g.w) + "×" + std::to_string(g.h) + "(s=" +
                std::to_string(g.stride) + ") ";
    }
    throw TritError(
        std::string("输出布局 ") + layout_name + " 的 anchor 数与 strides 推算不一致：引擎输出 " +
        std::to_string(anchors) + "，按 strides 推算 " + std::to_string(expected) + " [" + detail +
        "]。请检查部署配置里的 input 尺寸与 strides（P2–P5 应为 4,8,16,32）；" +
        "本张量形状 rank=" + std::to_string(out.rank()));
  }
}

/// anchor-major（含转置形式）：每个 anchor 的 4*reg_max 个 DFL 值与 nc 个分数连续。
struct AnchorMajorDecoder {
  static void Decode(const TensorView& out, const DecodeOptions& opt,
                     const std::vector<Grid>& grids, std::vector<Detection>& dets) {
    const int reg_max = opt.reg_max;
    const int nc = opt.num_classes;
    const int64_t no = static_cast<int64_t>(4 * reg_max + nc);

    int64_t channels = 0;
    int64_t anchors = 0;
    int64_t ch_stride = 1;    // 同一 anchor 内相邻通道的步长
    int64_t anch_stride = 0;  // 相邻 anchor 的步长
    const char* layout_name = "anchor-major-dfl";
    if (opt.layout == OutputLayout::kAnchorMajorDflTransposed) {
      if (out.rank() != 3) {
        throw TritError("anchor-major-dfl-transposed 期望 3 维输出 [B,A,C]，实际 rank=" +
                        std::to_string(out.rank()));
      }
      anchors = out.dim(1);
      channels = out.dim(2);
      ch_stride = 1;
      anch_stride = channels;
      layout_name = "anchor-major-dfl-transposed";
    } else {
      if (out.rank() != 3) {
        throw TritError("anchor-major-dfl 期望 3 维输出 [B,C,A]，实际 rank=" +
                        std::to_string(out.rank()));
      }
      channels = out.dim(1);
      anchors = out.dim(2);
      ch_stride = anchors;
      anch_stride = 1;
    }
    if (channels != no) {
      throw TritError("输出通道数 " + std::to_string(channels) + " 与 4*reg_max+nc=" +
                      std::to_string(no) + " 不符（检查部署配置的 nc/reg_max）");
    }
    check_anchors(out, grids, anchors, layout_name);
    if (reg_max > 64) throw TritError("reg_max 过大（>64）");

    const Reader r(out);
    float dist[4 * 64];

    int64_t offset = 0;
    for (const auto& g : grids) {
      for (int y = 0; y < g.h; ++y) {
        for (int x = 0; x < g.w; ++x) {
          const int64_t a = offset++;
          const int64_t base = a * anch_stride;
          for (int i = 0; i < reg_max; ++i) {
            dist[i] = r.at(base + static_cast<int64_t>(i) * ch_stride);
            dist[reg_max + i] = r.at(base + static_cast<int64_t>(reg_max + i) * ch_stride);
            dist[2 * reg_max + i] = r.at(base + static_cast<int64_t>(2 * reg_max + i) * ch_stride);
            dist[3 * reg_max + i] = r.at(base + static_cast<int64_t>(3 * reg_max + i) * ch_stride);
          }
          const DflResult d = dfl_decode(dist, reg_max);

          int best = 0;
          float best_score = -1.f;
          for (int c = 0; c < nc; ++c) {
            const float s =
                r.at(base + static_cast<int64_t>(4 * reg_max + c) * ch_stride);
            if (s > best_score) {
              best_score = s;
              best = c;
            }
          }
          if (best_score < opt.conf_threshold) continue;

          const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(g.stride);
          const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(g.stride);
          Detection det;
          det.class_id = best;
          det.score = best_score;
          det.box_score = best_score;
          det.box.x1 = cx - d.l * static_cast<float>(g.stride);
          det.box.y1 = cy - d.t * static_cast<float>(g.stride);
          det.box.x2 = cx + d.r * static_cast<float>(g.stride);
          det.box.y2 = cy + d.b * static_cast<float>(g.stride);
          dets.push_back(det);
        }
      }
    }
  }
};

/// feature-major：[B, Σ(4*reg_max+nc), A]，每层的 DFL 值全在前、分数在后，层间拼接。
/// SPAE-YOLOv8 的 Efficient_UAVDet 独立实现即此排布。
struct FeatureMajorDecoder {
  static void Decode(const TensorView& out, const DecodeOptions& opt,
                     const std::vector<Grid>& grids, std::vector<Detection>& dets) {
    const int reg_max = opt.reg_max;
    const int nc = opt.num_classes;
    if (out.rank() != 3) {
      throw TritError("feature-major-dfl 期望 3 维输出 [B,C,A]，实际 rank=" +
                      std::to_string(out.rank()));
    }
    const int64_t total_ch = out.dim(1);
    const int64_t anchors = out.dim(2);

    std::vector<int> level_ch = opt.level_channels;
    if (level_ch.empty()) level_ch.assign(grids.size(), 4 * reg_max + nc);
    if (level_ch.size() != grids.size()) {
      throw TritError("level_channels 长度（" + std::to_string(level_ch.size()) +
                      "）与 strides 长度（" + std::to_string(grids.size()) + "）不一致");
    }
    int64_t sum_ch = 0;
    for (int c : level_ch) sum_ch += c;
    if (sum_ch != total_ch) {
      throw TritError("level_channels 之和 " + std::to_string(sum_ch) + " 与输出通道 " +
                      std::to_string(total_ch) + " 不符（feature-major 需要精确的每层通道数）");
    }
    check_anchors(out, grids, anchors, "feature-major-dfl");
    if (reg_max > 64) throw TritError("reg_max 过大（>64）");

    const Reader r(out);
    float dist[4 * 64];

    int64_t ch_base = 0;
    int64_t a_base = 0;
    for (size_t li = 0; li < grids.size(); ++li) {
      const Grid& g = grids[li];
      const int ch = level_ch[li];
      const int cls_off = ch - nc;  // 分类分数在该层通道内的起始偏移
      for (int y = 0; y < g.h; ++y) {
        for (int x = 0; x < g.w; ++x) {
          const int64_t a = a_base + static_cast<int64_t>(y) * g.w + x;
          for (int i = 0; i < reg_max; ++i) {
            dist[i] = r.at((ch_base + i) * anchors + a);
            dist[reg_max + i] = r.at((ch_base + reg_max + i) * anchors + a);
            dist[2 * reg_max + i] = r.at((ch_base + 2 * reg_max + i) * anchors + a);
            dist[3 * reg_max + i] = r.at((ch_base + 3 * reg_max + i) * anchors + a);
          }
          const DflResult d = dfl_decode(dist, reg_max);

          int best = 0;
          float best_score = -1.f;
          for (int c = 0; c < nc; ++c) {
            const float s = r.at((ch_base + cls_off + c) * anchors + a);
            if (s > best_score) {
              best_score = s;
              best = c;
            }
          }
          if (best_score < opt.conf_threshold) continue;

          const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(g.stride);
          const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(g.stride);
          Detection det;
          det.class_id = best;
          det.score = best_score;
          det.box_score = best_score;
          det.box.x1 = cx - d.l * static_cast<float>(g.stride);
          det.box.y1 = cy - d.t * static_cast<float>(g.stride);
          det.box.x2 = cx + d.r * static_cast<float>(g.stride);
          det.box.y2 = cy + d.b * static_cast<float>(g.stride);
          dets.push_back(det);
        }
      }
      ch_base += ch;
      a_base += g.anchors();
    }
  }
};

/// 插件输出：EfficientNMS_TRT 等已完成解码，这里只做搬运与阈值过滤。
struct PluginDecoder {
  static void Decode(const TensorView& out, const DecodeOptions& opt,
                     std::vector<Detection>& dets) {
    if (out.rank() != 2) {
      throw TritError("plugin-nms 期望 2 维输出 [N,6] 或 [N,7]，实际 rank=" +
                      std::to_string(out.rank()));
    }
    const int64_t cols = out.dim(1);
    const int64_t rows = out.dim(0);
    if (cols != 6 && cols != 7) {
      throw TritError("plugin-nms 输出列数必须是 6 或 7，实际 " + std::to_string(cols));
    }
    const int base = (cols == 7) ? 1 : 0;  // 7 列时第 0 列是 batch index
    const Reader r(out);
    for (int64_t i = 0; i < rows; ++i) {
      Detection d;
      d.box.x1 = r.at(i * cols + base + 0);
      d.box.y1 = r.at(i * cols + base + 1);
      d.box.x2 = r.at(i * cols + base + 2);
      d.box.y2 = r.at(i * cols + base + 3);
      d.score = r.at(i * cols + base + 4);
      d.class_id = static_cast<int>(r.at(i * cols + base + 5));
      d.box_score = d.score;
      if (d.score >= opt.conf_threshold) dets.push_back(d);
    }
  }
};

}  // namespace

// ------------------------------------------------------------------ 基础算子

float sigmoid(float x) {
  if (x >= 0.f) {
    const float z = std::exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = std::exp(x);
  return z / (1.f + z);
}

float dfl_expectation(const float* dist, int reg_max) { return expectation(dist, reg_max); }

DflResult dfl_decode(const float* dist, int reg_max) {
  DflResult r;
  r.l = expectation(dist, reg_max);
  r.t = expectation(dist + reg_max, reg_max);
  r.r = expectation(dist + 2 * reg_max, reg_max);
  r.b = expectation(dist + 3 * reg_max, reg_max);
  return r;
}

std::vector<Grid> make_grids(const std::vector<int>& strides, int input_w, int input_h) {
  std::vector<Grid> grids;
  grids.reserve(strides.size());
  for (int s : strides) {
    if (s <= 0) throw TritError("stride 必须为正数");
    Grid g;
    g.stride = s;
    g.w = input_w / s;
    g.h = input_h / s;
    if (g.w <= 0 || g.h <= 0) {
      throw TritError("输入尺寸 " + std::to_string(input_w) + "×" + std::to_string(input_h) +
                      " 对 stride=" + std::to_string(s) + " 太小");
    }
    grids.push_back(g);
  }
  return grids;
}

std::vector<Detection> decode_predictions(const TensorView& output, const DecodeOptions& opt) {
  if (!output.valid()) throw TritError("decode_predictions: 输出张量无效");
  std::vector<Detection> dets;

  if (opt.layout == OutputLayout::kPluginNms) {
    PluginDecoder::Decode(output, opt, dets);
  } else {
    // 只有 batch=1 时 anchor 网格与 dim(2)/dim(1) 的对应关系才唯一。
    // 多 batch：请按 batch 维切片后逐帧解码（服务里通常也是逐帧后处理）。
    if (output.rank() >= 1 && output.dim(0) != 1) {
      throw TritError("解码要求 batch=1（当前 batch=" + std::to_string(output.dim(0)) +
                      "）；多 batch 请按 batch 维切片后逐帧解码");
    }
    const std::vector<Grid> grids = make_grids(opt.strides, opt.input_width, opt.input_height);
    if (opt.layout == OutputLayout::kFeatureMajorDfl) {
      FeatureMajorDecoder::Decode(output, opt, grids, dets);
    } else {
      AnchorMajorDecoder::Decode(output, opt, grids, dets);
    }
  }

  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (opt.max_det > 0 && static_cast<int>(dets.size()) > opt.max_det) {
    dets.resize(static_cast<size_t>(opt.max_det));
  }
  return dets;
}

}  // namespace todrt
