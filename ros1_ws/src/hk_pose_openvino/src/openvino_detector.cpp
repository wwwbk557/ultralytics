#include "hk_pose_openvino/openvino_detector.h"

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace hk_pose_openvino {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void setError(char* error, std::size_t error_len, const std::string& message) {
  if (error != nullptr && error_len > 0) {
    std::snprintf(error, error_len, "%s", message.c_str());
  }
}

float boxIou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float inter_w = std::max(0.0f, x2 - x1);
  const float inter_h = std::max(0.0f, y2 - y1);
  const float inter = inter_w * inter_h;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl lives only in this translation unit, which is compiled with
// _GLIBCXX_USE_CXX11_ABI=0 to match the official OpenVINO binaries.  It never
// exposes std::string across the boundary.
// ---------------------------------------------------------------------------
struct OpenVinoDetector::Impl {
  bool init(const DetectorOptions& options, char* error, std::size_t error_len);
  bool detect(const cv::Mat& bgr, std::vector<Detection>* detections,
              DetectorTimings* timings, char* error, std::size_t error_len);

  void letterbox(const cv::Mat& src, cv::Mat* dst, float* letterbox_scale,
                 float* letterbox_pad_x, float* letterbox_pad_y) const;
  void decode(const float* output, std::vector<Detection>* detections) const;
  void decodeRaw18(const float* output, std::vector<Detection>* detections) const;
  void decodeV5(const float* output, std::vector<Detection>* detections) const;
  bool buildRawAnchors(char* error, std::size_t error_len);
  void nmsPerClass(std::vector<Detection>* detections,
                   float iou_threshold) const;
  void nmsGlobal(std::vector<Detection>* detections,
                 float iou_threshold) const;

  ov::Core core;
  ov::CompiledModel compiled;
  ov::InferRequest request;
  ov::Tensor input_tensor;

  std::string model_path;
  std::string device = "CPU";
  float conf_threshold = 0.90f;
  float quality_threshold = 0.05f;
  float nms_threshold = 0.45f;
  int num_threads = 0;
  int my_color = -1;

  int input_width = 640;
  int input_height = 480;
  int channels = 23;
  int anchors = 6300;
  int num_classes = 14;
  int num_colors = 0;
  int kpt_count = 4;
  bool raw_output = false;
  bool raw_channel_first = false;
  std::vector<float> raw_anchor_x;
  std::vector<float> raw_anchor_y;
  std::vector<float> raw_anchor_stride;
  enum class Layout { kAux14, kColourKind, kV5Armor };
  Layout layout = Layout::kAux14;

  // Reused per-frame buffers.
  cv::Mat letterboxed;
  cv::Mat rgb8;
  cv::Mat rgb_float;
  std::vector<cv::Mat> rgb_planes;

  // Per-frame letterbox parameters, needed to map corners back.
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
  int src_width = 0;
  int src_height = 0;
};

bool OpenVinoDetector::Impl::init(const DetectorOptions& options, char* error,
                                  std::size_t error_len) {
  if (options.model_path == nullptr || options.model_path[0] == '\0') {
    setError(error, error_len, "model_path is empty");
    return false;
  }
  model_path = options.model_path;
  if (options.device != nullptr && options.device[0] != '\0') {
    device = options.device;
  }
  conf_threshold = std::clamp(options.conf_threshold, 0.0f, 1.0f);
  quality_threshold = std::clamp(options.quality_threshold, 0.0f, 1.0f);
  nms_threshold = std::clamp(options.nms_threshold, 0.0f, 1.0f);
  num_threads = std::max(0, options.num_threads);
  my_color = options.my_color;
  raw_output = options.raw_output;
  input_width = options.input_width;
  input_height = options.input_height;

  std::shared_ptr<ov::Model> model;
  try {
    model = core.read_model(model_path);
  } catch (const std::exception& e) {
    setError(error, error_len,
             "failed to read model '" + model_path + "': " + e.what());
    return false;
  }

  try {
    if (model->inputs().size() != 1) {
      setError(error, error_len, "model must have exactly one input");
      return false;
    }
    const auto in_shape = model->input(0).get_partial_shape();
    if (in_shape.rank().is_dynamic() || in_shape.rank().get_length() != 4) {
      setError(error, error_len, "model input must be 4-D NCHW");
      return false;
    }
    if (in_shape[1].is_dynamic() || in_shape[1].get_length() != 3) {
      setError(error, error_len, "model input channel must be 3");
      return false;
    }

    // Static 480x640 export: trust the graph. Dynamic export: reshape to the
    // configured size.
    bool need_reshape = in_shape[0].is_dynamic() || in_shape[2].is_dynamic() ||
                        in_shape[3].is_dynamic();
    if (!in_shape[2].is_dynamic()) {
      input_height = static_cast<int>(in_shape[2].get_length());
    }
    if (!in_shape[3].is_dynamic()) {
      input_width = static_cast<int>(in_shape[3].get_length());
    }
    if (input_width <= 0 || input_height <= 0) {
      setError(error, error_len, "invalid model input size");
      return false;
    }
    if (!in_shape[0].is_dynamic() && in_shape[0].get_length() != 1) {
      need_reshape = true;
    }
    if (need_reshape) {
      model->reshape(ov::PartialShape{1, 3, input_height, input_width});
    }

    const auto out_shape = model->output(0).get_partial_shape();
    if (out_shape.rank().is_dynamic() || out_shape.rank().get_length() != 3 ||
        out_shape[1].is_dynamic() || out_shape[2].is_dynamic()) {
      setError(error, error_len,
               "model output must have a static [1, anchors, channels] shape");
      return false;
    }
    const int out_dim1 = static_cast<int>(out_shape[1].get_length());
    const int out_dim2 = static_cast<int>(out_shape[2].get_length());
    raw_channel_first = (out_dim1 == 18 && out_dim2 != 18);
    raw_output = raw_output || raw_channel_first;
    if (raw_channel_first) {
      channels = out_dim1;
      anchors = out_dim2;
    } else {
      anchors = out_dim1;
      channels = out_dim2;
    }
    if (raw_output && channels != 18) {
      setError(error, error_len, "raw output must have 18 channels");
      return false;
    }
    if (model->output(0).get_element_type() != ov::element::f32) {
      setError(error, error_len, "model output must be f32");
      return false;
    }

    if (channels == 23) {
      layout = Layout::kAux14;
      num_classes = channels - 9;  // 8 corners + 1 quality + N classes
    } else if (channels == 18) {
      layout = Layout::kColourKind;
      num_classes = 14;  // colour(2) x kind(7)
    } else if (channels == 22) {
      // 0526 / Infantry-v5: [4 keypoints(8), confidence(1),
      // colour scores(4), armour class scores(9)].
      layout = Layout::kV5Armor;
      num_classes = 9;
      num_colors = 4;
    } else {
      std::ostringstream oss;
      oss << "unsupported output channels: " << channels
          << " (expected 23 aux-14, 18 colour x kind, or 22 V5 armour)";
      setError(error, error_len, oss.str());
      return false;
    }
    if (num_classes <= 0) {
      setError(error, error_len,
               "invalid class count derived from the output shape");
      return false;
    }
    if (raw_output && !buildRawAnchors(error, error_len)) {
      return false;
    }

    // CPU: THROUGHPUT is noticeably faster on the target laptop (all cores).
    // GPU: leave the plugin default; on the 2024.6 iGPU it beats an explicit
    // LATENCY hint.  AUTO/MULTI also keep the plugin default.
    // The 0526 model takes FP16 input; convert to f32 in the graph so the
    // rest of the pipeline always fills a plain float tensor.
    if (model->input(0).get_element_type() == ov::element::f16) {
      ov::preprocess::PrePostProcessor ppp(model);
      ppp.input().tensor().set_element_type(ov::element::f32);
      model = ppp.build();
    }

    const bool is_cpu =
        (device == "CPU") || (device.compare(0, 3, "CPU") == 0);
    ov::AnyMap properties;
    if (is_cpu) {
      properties.emplace(ov::hint::performance_mode(
          ov::hint::PerformanceMode::THROUGHPUT));
    }
    if (num_threads > 0) {
      properties.emplace(ov::inference_num_threads(num_threads));
    }
    compiled = core.compile_model(model, device, properties);

    request = compiled.create_infer_request();
    input_tensor = request.get_input_tensor();
    if (input_tensor.get_element_type() != ov::element::f32) {
      setError(error, error_len, "model input must be f32");
      return false;
    }

    // Warm up so the first real frame is not penalised by plugin setup.
    float* warm = input_tensor.data<float>();
    std::fill(warm, warm + input_tensor.get_size(), 0.0f);
    for (int i = 0; i < 2; ++i) {
      request.infer();
    }

    // Allocate reusable preprocessing buffers.
    letterboxed = cv::Mat(input_height, input_width, CV_8UC3,
                          cv::Scalar(114, 114, 114));
    rgb8 = cv::Mat(input_height, input_width, CV_8UC3);
    rgb_planes.resize(3);
  } catch (const std::exception& e) {
    setError(error, error_len,
             std::string("OpenVINO initialisation failed: ") + e.what());
    return false;
  }
  return true;
}

void OpenVinoDetector::Impl::letterbox(const cv::Mat& src, cv::Mat* dst,
                                       float* letterbox_scale,
                                       float* letterbox_pad_x,
                                       float* letterbox_pad_y) const {
  const int src_h = src.rows;
  const int src_w = src.cols;
  const float r = std::min(input_height / static_cast<float>(src_h),
                           input_width / static_cast<float>(src_w));
  const int new_w = std::min(
      input_width, static_cast<int>(std::lround(src_w * r)));
  const int new_h = std::min(
      input_height, static_cast<int>(std::lround(src_h * r)));
  const float dw = (input_width - new_w) / 2.0f;
  const float dh = (input_height - new_h) / 2.0f;
  const int top = static_cast<int>(std::lround(dh - 0.1f));
  const int left = static_cast<int>(std::lround(dw - 0.1f));

  if (dst->rows != input_height || dst->cols != input_width ||
      dst->type() != CV_8UC3) {
    *dst = cv::Mat(input_height, input_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  } else {
    dst->setTo(cv::Scalar(114, 114, 114));
  }

  cv::Mat resized;
  if (new_w != src_w || new_h != src_h) {
    cv::resize(src, resized, cv::Size(new_w, new_h), 0.0, 0.0,
               cv::INTER_LINEAR);
  } else {
    resized = src;
  }
  resized.copyTo((*dst)(cv::Rect(left, top, new_w, new_h)));

  *letterbox_scale = r;
  *letterbox_pad_x = static_cast<float>(left);
  *letterbox_pad_y = static_cast<float>(top);
}

bool OpenVinoDetector::Impl::detect(const cv::Mat& bgr,
                                    std::vector<Detection>* detections,
                                    DetectorTimings* timings, char* error,
                                    std::size_t error_len) {
  detections->clear();
  *timings = DetectorTimings();
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    setError(error, error_len, "input image is empty or not CV_8UC3");
    return false;
  }

  src_width = bgr.cols;
  src_height = bgr.rows;

  try {
    const auto t0 = Clock::now();
    letterbox(bgr, &letterboxed, &scale, &pad_x, &pad_y);
    cv::cvtColor(letterboxed, rgb8, cv::COLOR_BGR2RGB);
    rgb8.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);
    cv::split(rgb_float, rgb_planes);

    float* input = input_tensor.data<float>();
    const size_t plane =
        static_cast<size_t>(input_width) * static_cast<size_t>(input_height);
    for (int c = 0; c < 3; ++c) {
      std::memcpy(input + c * plane, rgb_planes[c].ptr<float>(),
                  plane * sizeof(float));
    }
    timings->preprocess_ms = elapsedMs(t0);

    const auto t1 = Clock::now();
    request.infer();
    timings->infer_ms = elapsedMs(t1);

    const auto t2 = Clock::now();
    const float* output = request.get_output_tensor().data<float>();
    decode(output, detections);
    timings->postprocess_ms = elapsedMs(t2);
  } catch (const std::exception& e) {
    setError(error, error_len, std::string("inference failed: ") + e.what());
    return false;
  }
  return true;
}

bool OpenVinoDetector::Impl::buildRawAnchors(char* error,
                                             std::size_t error_len) {
  raw_anchor_x.clear();
  raw_anchor_y.clear();
  raw_anchor_stride.clear();
  raw_anchor_x.reserve(static_cast<std::size_t>(anchors));
  raw_anchor_y.reserve(static_cast<std::size_t>(anchors));
  raw_anchor_stride.reserve(static_cast<std::size_t>(anchors));
  const int strides[3] = {8, 16, 32};
  int total = 0;
  for (int s : strides) {
    const int gw = input_width / s;
    const int gh = input_height / s;
    if (gw <= 0 || gh <= 0) {
      setError(error, error_len, "invalid raw anchor grid");
      return false;
    }
    for (int y = 0; y < gh; ++y) {
      for (int x = 0; x < gw; ++x) {
        raw_anchor_x.push_back(static_cast<float>(x) + 0.5f);
        raw_anchor_y.push_back(static_cast<float>(y) + 0.5f);
        raw_anchor_stride.push_back(static_cast<float>(s));
      }
    }
    total += gw * gh;
  }
  if (total != anchors) {
    std::ostringstream oss;
    oss << "raw anchor mismatch: " << total << " vs " << anchors;
    setError(error, error_len, oss.str());
    return false;
  }
  return true;
}

void OpenVinoDetector::Impl::decodeRaw18(
    const float* output, std::vector<Detection>* detections) const {
  detections->clear();
  detections->reserve(32);

  const float inv_scale = 1.0f / scale;
  const float max_x = static_cast<float>(src_width - 1);
  const float max_y = static_cast<float>(src_height - 1);
  const int a = anchors;
  auto at = [&](int c, int idx) {
    return raw_channel_first ? output[static_cast<std::size_t>(c) * a + idx]
                             : output[static_cast<std::size_t>(idx) * 18 + c];
  };

  for (int i = 0; i < a; ++i) {
    const float q = at(8, i);
    const float quality = 1.0f / (1.0f + std::exp(-q));
    if (quality < quality_threshold) {
      continue;
    }

    const float c0 = at(9, i);
    const float c1 = at(10, i);
    const float cmax = std::max(c0, c1);
    const float e0 = std::exp(c0 - cmax);
    const float e1 = std::exp(c1 - cmax);
    const float csum = e0 + e1;
    const int colour = (c0 >= c1) ? 0 : 1;
    const float color_p = ((colour == 0) ? e0 : e1) / csum;

    float kbest = at(11, i);
    int kind = 0;
    for (int k = 1; k < 7; ++k) {
      const float v = at(11 + k, i);
      if (v > kbest) {
        kbest = v;
        kind = k;
      }
    }
    float ksum = 0.0f;
    for (int k = 0; k < 7; ++k) {
      ksum += std::exp(at(11 + k, i) - kbest);
    }
    const float kind_p = 1.0f / ksum;

    const float class_prob = std::max(0.0f, color_p) *
                             std::max(0.0f, kind_p);
    // Soft score for NMS ordering: objectness x P(colour, role | object).
    // The quality_threshold gate above only decides membership; it must not
    // collapse the objectness ordering of the surviving anchors.
    const float score = quality * class_prob;
    if (score < conf_threshold) {
      continue;
    }

    Detection det;
    det.class_id = colour * 7 + kind;
    det.color_id = colour;
    det.score = score;
    det.quality = quality;

    const float ax = raw_anchor_x[static_cast<std::size_t>(i)];
    const float ay = raw_anchor_y[static_cast<std::size_t>(i)];
    const float st = raw_anchor_stride[static_cast<std::size_t>(i)];
    float min_x = max_x;
    float min_y = max_y;
    float max_corner_x = 0.0f;
    float max_corner_y = 0.0f;
    for (int k = 0; k < kpt_count; ++k) {
      const float rx = at(2 * k, i);
      const float ry = at(2 * k + 1, i);
      const float x =
          std::clamp(((rx + ax) * st - pad_x) * inv_scale, 0.0f, max_x);
      const float y =
          std::clamp(((ry + ay) * st - pad_y) * inv_scale, 0.0f, max_y);
      det.corners[k] = cv::Point2f(x, y);
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_corner_x = std::max(max_corner_x, x);
      max_corner_y = std::max(max_corner_y, y);
    }
    det.bbox =
        cv::Rect2f(min_x, min_y, max_corner_x - min_x, max_corner_y - min_y);
    if (det.bbox.width < 1.0f || det.bbox.height < 1.0f) {
      continue;
    }
    detections->push_back(det);
  }

  nmsPerClass(detections, nms_threshold);
}

void OpenVinoDetector::Impl::decode(const float* output,
                                    std::vector<Detection>* detections) const {
  if (raw_output) {
    decodeRaw18(output, detections);
    return;
  }
  if (layout == Layout::kV5Armor) {
    decodeV5(output, detections);
    return;
  }
  detections->clear();
  detections->reserve(32);

  const float inv_scale = 1.0f / scale;
  const float max_x = static_cast<float>(src_width - 1);
  const float max_y = static_cast<float>(src_height - 1);

  for (int i = 0; i < anchors; ++i) {
    const float* row = output + static_cast<size_t>(i) * channels;
    const float quality = row[8];
    if (quality < quality_threshold) {
      continue;
    }

    int class_id = -1;
    float class_prob = 0.0f;
    if (layout == Layout::kAux14) {
      class_id = 0;
      class_prob = row[9];
      for (int c = 1; c < num_classes; ++c) {
        const float p = row[9 + c];
        if (p > class_prob) {
          class_prob = p;
          class_id = c;
        }
      }
    } else {
      // [corners(8), quality(1), colour(2), kind(7)]
      const int colour = row[9] >= row[10] ? 0 : 1;
      int kind = 0;
      float kind_prob = row[11];
      for (int k = 1; k < 7; ++k) {
        const float p = row[11 + k];
        if (p > kind_prob) {
          kind_prob = p;
          kind = k;
        }
      }
      class_id = colour * 7 + kind;
      class_prob = std::max(0.0f, row[9 + colour]) * std::max(0.0f, kind_prob);
    }

    const float score = quality * class_prob;
    if (score < conf_threshold) {
      continue;
    }

    Detection det;
    det.class_id = class_id;
    det.score = score;
    det.quality = quality;

    float min_x = max_x;
    float min_y = max_y;
    float max_corner_x = 0.0f;
    float max_corner_y = 0.0f;
    for (int k = 0; k < kpt_count; ++k) {
      const float x =
          std::clamp((row[2 * k] - pad_x) * inv_scale, 0.0f, max_x);
      const float y =
          std::clamp((row[2 * k + 1] - pad_y) * inv_scale, 0.0f, max_y);
      det.corners[k] = cv::Point2f(x, y);
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_corner_x = std::max(max_corner_x, x);
      max_corner_y = std::max(max_corner_y, y);
    }
    det.bbox =
        cv::Rect2f(min_x, min_y, max_corner_x - min_x, max_corner_y - min_y);
    if (det.bbox.width < 1.0f || det.bbox.height < 1.0f) {
      continue;
    }
    detections->push_back(det);
  }

  nmsPerClass(detections, nms_threshold);
}

void OpenVinoDetector::Impl::decodeV5(
    const float* output, std::vector<Detection>* detections) const {
  detections->clear();
  detections->reserve(32);

  const float inv_scale = 1.0f / scale;
  const float max_x = static_cast<float>(src_width - 1);
  const float max_y = static_cast<float>(src_height - 1);

  for (int i = 0; i < anchors; ++i) {
    const float* row = output + static_cast<size_t>(i) * channels;

    // V5 contract: [kpt0x,kpt0y,...,kpt3x,kpt3y, conf, colour0..3, cls0..8]
    const float conf = 1.0f / (1.0f + std::exp(-row[8]));
    if (conf < conf_threshold) {
      continue;
    }

    int color_id = 0;
    float best_color = row[9];
    for (int c = 1; c < 4; ++c) {
      if (row[9 + c] > best_color) {
        best_color = row[9 + c];
        color_id = c;
      }
    }
    if (color_id == 3) {
      continue;  // purple / invalid armor
    }
    // Keep the repo's own-colour filter when my_color is 0 (blue) or 1 (red).
    if (color_id == 0 && my_color == 1) {
      continue;
    }
    if (color_id == 1 && my_color == 0) {
      continue;
    }

    int class_id = 0;
    float best_class = row[13];
    for (int c = 1; c < 9; ++c) {
      if (row[13 + c] > best_class) {
        best_class = row[13 + c];
        class_id = c;
      }
    }

    Detection det;
    det.class_id = class_id;
    det.color_id = color_id;
    det.score = conf;
    det.quality = conf;

    float min_x = max_x;
    float min_y = max_y;
    float max_corner_x = 0.0f;
    float max_corner_y = 0.0f;
    bool valid = true;
    for (int k = 0; k < 4; ++k) {
      const float x = (row[2 * k] - pad_x) * inv_scale;
      const float y = (row[2 * k + 1] - pad_y) * inv_scale;
      if (x < 0.0f || x > max_x || y < 0.0f || y > max_y) {
        valid = false;
        break;
      }
      det.corners[k] = cv::Point2f(x, y);
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_corner_x = std::max(max_corner_x, x);
      max_corner_y = std::max(max_corner_y, y);
    }
    if (!valid) {
      continue;
    }
    det.bbox =
        cv::Rect2f(min_x, min_y, max_corner_x - min_x, max_corner_y - min_y);
    if (det.bbox.width < 1.0f || det.bbox.height < 1.0f) {
      continue;
    }
    detections->push_back(det);
  }

  nmsGlobal(detections, nms_threshold);
}

void OpenVinoDetector::Impl::nmsGlobal(
    std::vector<Detection>* detections, float iou_threshold) const {
  if (detections->empty()) {
    return;
  }
  std::vector<Detection> candidates = std::move(*detections);
  detections->clear();
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Detection& a, const Detection& b) {
                     return a.score > b.score;
                   });
  std::vector<char> suppressed(candidates.size(), 0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    detections->push_back(candidates[i]);
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] &&
          boxIou(candidates[i].bbox, candidates[j].bbox) > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
}

void OpenVinoDetector::Impl::nmsPerClass(
    std::vector<Detection>* detections, float iou_threshold) const {
  if (detections->empty()) {
    return;
  }
  std::vector<Detection> candidates = std::move(*detections);
  detections->clear();

  std::vector<std::vector<int>> by_class(num_classes);
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const int cls = candidates[i].class_id;
    if (cls >= 0 && cls < num_classes) {
      by_class[cls].push_back(i);
    }
  }

  for (auto& indices : by_class) {
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
      return candidates[a].score > candidates[b].score;
    });
    std::vector<char> suppressed(indices.size(), 0);
    for (size_t i = 0; i < indices.size(); ++i) {
      if (suppressed[i]) {
        continue;
      }
      detections->push_back(candidates[indices[i]]);
      for (size_t j = i + 1; j < indices.size(); ++j) {
        if (!suppressed[j] &&
            boxIou(candidates[indices[i]].bbox, candidates[indices[j]].bbox) >
                iou_threshold) {
          suppressed[j] = 1;
        }
      }
    }
  }

  std::stable_sort(detections->begin(), detections->end(),
                   [](const Detection& a, const Detection& b) {
                     return a.score > b.score;
                   });
}

// ---------------------------------------------------------------------------
// ABI-neutral facade (compiled in both ABIs, all work delegated to Impl).
// ---------------------------------------------------------------------------
OpenVinoDetector::OpenVinoDetector() : impl_(new Impl()) {}
OpenVinoDetector::~OpenVinoDetector() = default;

bool OpenVinoDetector::init(const DetectorOptions& options, char* error,
                            std::size_t error_len) {
  return impl_->init(options, error, error_len);
}

bool OpenVinoDetector::detect(const cv::Mat& bgr,
                              std::vector<Detection>* detections,
                              DetectorTimings* timings, char* error,
                              std::size_t error_len) {
  return impl_->detect(bgr, detections, timings, error, error_len);
}

int OpenVinoDetector::inputWidth() const { return impl_->input_width; }
int OpenVinoDetector::inputHeight() const { return impl_->input_height; }
int OpenVinoDetector::numClasses() const { return impl_->num_classes; }
int OpenVinoDetector::numColors() const { return impl_->num_colors; }
int OpenVinoDetector::outputChannels() const { return impl_->channels; }
int OpenVinoDetector::numAnchors() const { return impl_->anchors; }

void OpenVinoDetector::setConfThreshold(float value) {
  impl_->conf_threshold = std::clamp(value, 0.0f, 1.0f);
}

void OpenVinoDetector::setQualityThreshold(float value) {
  impl_->quality_threshold = std::clamp(value, 0.0f, 1.0f);
}

void OpenVinoDetector::setNmsThreshold(float value) {
  impl_->nms_threshold = std::clamp(value, 0.0f, 1.0f);
}

float OpenVinoDetector::confThreshold() const {
  return impl_->conf_threshold;
}

float OpenVinoDetector::qualityThreshold() const {
  return impl_->quality_threshold;
}

float OpenVinoDetector::nmsThreshold() const {
  return impl_->nms_threshold;
}

}  // namespace hk_pose_openvino
