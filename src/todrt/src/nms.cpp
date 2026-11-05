#include <algorithm>
#include <cmath>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/modules.hpp"

namespace todrt {

float iou(const BBox& a, const BBox& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float iw = x2 - x1;
  const float ih = y2 - y1;
  if (iw <= 0.f || ih <= 0.f) return 0.f;
  const float inter = iw * ih;
  const float uni = a.area() + b.area() - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

namespace {

inline bool same_group(const Detection& a, const Detection& b, bool class_agnostic) {
  return class_agnostic || a.class_id == b.class_id;
}

}

void nms_hard(const std::vector<Detection>& dets, float iou_threshold, bool class_agnostic,
              std::vector<int>& keep) {
  keep.clear();
  const size_t n = dets.size();
  if (n == 0) return;

  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  const bool sorted = std::is_sorted(order.begin(), order.end(), [&](size_t a, size_t b) {
    return dets[a].score >= dets[b].score;
  });
  if (!sorted) {
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return dets[a].score > dets[b].score; });
  }

  std::vector<char> removed(n, 0);
  for (size_t oi = 0; oi < n; ++oi) {
    const size_t i = order[oi];
    if (removed[i]) continue;
    keep.push_back(static_cast<int>(i));
    for (size_t oj = oi + 1; oj < n; ++oj) {
      const size_t j = order[oj];
      if (removed[j]) continue;
      if (!same_group(dets[i], dets[j], class_agnostic)) continue;
      if (iou(dets[i].box, dets[j].box) > iou_threshold) removed[j] = 1;
    }
  }
}

void nms_soft(std::vector<Detection>& dets, float iou_threshold, float sigma, float score_floor,
              bool class_agnostic, std::vector<int>& keep) {
  keep.clear();
  const size_t n = dets.size();
  if (n == 0) return;

  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return dets[a].score > dets[b].score; });

  std::vector<float> decay(n, 1.f);
  std::vector<char> removed(n, 0);
  const float denom = 2.f * sigma * sigma;

  for (size_t oi = 0; oi < n; ++oi) {
    const size_t i = order[oi];
    if (removed[i]) continue;
    const float si = dets[i].score * decay[i];
    if (si < score_floor) {
      removed[i] = 1;
      continue;
    }
    keep.push_back(static_cast<int>(i));
    for (size_t oj = oi + 1; oj < n; ++oj) {
      const size_t j = order[oj];
      if (removed[j]) continue;
      if (!same_group(dets[i], dets[j], class_agnostic)) continue;
      const float ov = iou(dets[i].box, dets[j].box);
      if (ov > iou_threshold) {
        decay[j] *= std::exp(-(ov * ov) / denom);
        // Soft-NMS 原地衰减分数，最终结果应使用衰减后的置信度。
        dets[j].score *= std::exp(-(ov * ov) / denom);
      }
    }
  }
}

namespace {

class NmsPostprocessor : public IPostprocessor {
 public:
  NmsPostprocessor() = default;
  explicit NmsPostprocessor(PostprocessOptions o) : opt_(std::move(o)) {
    if (opt_.decode.input_width <= 0 || opt_.decode.input_height <= 0) {
      throw TritError("后处理需要 decode.input_width/height（由部署配置的 input 注入）");
    }
  }

  const std::string& name() const override { return name_; }
  const PostprocessOptions& options() const override { return opt_; }

  std::vector<std::vector<Detection>> Run(const std::vector<TensorView>& outputs,
                                         const PreprocessResult& pre) override {
    if (outputs.empty()) throw TritError("后处理收到空的输出张量列表");
    const int batch = pre.batch > 0 ? pre.batch : 1;
    std::vector<std::vector<Detection>> result(static_cast<size_t>(batch));

    const TensorView& main = outputs[0];
    if (!main.valid()) throw TritError("主输出张量无效");

    if (batch == 1) {
      result[0] = decode_predictions(main, opt_.decode);
      Apply(result[0]);
    } else {
      result = DecodeBatched(main, batch);
    }
    return result;
  }

 private:
  void Apply(std::vector<Detection>& dets) {
    std::vector<int> keep;
    if (opt_.nms == NmsKind::kSoft) {
      nms_soft(dets, opt_.iou_threshold, opt_.sigma, opt_.soft_score_threshold,
               opt_.class_agnostic, keep);
    } else {
      nms_hard(dets, opt_.iou_threshold, opt_.class_agnostic, keep);
    }
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int i : keep) out.push_back(dets[i]);
    std::sort(out.begin(), out.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (opt_.max_det > 0 && static_cast<int>(out.size()) > opt_.max_det) {
      out.resize(static_cast<size_t>(opt_.max_det));
    }
    dets.swap(out);
  }

  std::vector<std::vector<Detection>> DecodeBatched(const TensorView& main, int batch) {
    std::vector<std::vector<Detection>> out(static_cast<size_t>(batch));
    if (opt_.decode.layout == OutputLayout::kPluginNms) {
      return {decode_predictions(main, opt_.decode)};
    }
    const DataType dt = main.dtype;
    const size_t elem = dtype_size(dt);
    const int64_t per_batch = main.numel() / batch;
    for (int b = 0; b < batch; ++b) {
      TensorView slice = main;
      slice.data = static_cast<const uint8_t*>(main.data) +
                   static_cast<size_t>(b) * static_cast<size_t>(per_batch) * elem;
      slice.shape[0] = 1;
      out[static_cast<size_t>(b)] = decode_predictions(slice, opt_.decode);
      Apply(out[static_cast<size_t>(b)]);
    }
    return out;
  }

  std::string name_ = "detect-nms";
  PostprocessOptions opt_;
};

std::unique_ptr<IPostprocessor> FactoryNmsPostproc() {
  return std::unique_ptr<IPostprocessor>(new NmsPostprocessor());
}

}

TOD_RT_REGISTER_POSTPROC(detect_nms, FactoryNmsPostproc)
TOD_RT_REGISTER_POSTPROC(nms, FactoryNmsPostproc)
TOD_RT_REGISTER_POSTPROC(detect, FactoryNmsPostproc)

std::unique_ptr<IPostprocessor> make_nms_postprocessor(const PostprocessOptions& o) {
  return std::unique_ptr<IPostprocessor>(new NmsPostprocessor(o));
}

}
