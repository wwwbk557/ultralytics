#ifndef HK_POSE_OPENVINO_OPENVINO_DETECTOR_H_
#define HK_POSE_OPENVINO_OPENVINO_DETECTOR_H_

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace hk_pose_openvino {

/// One detected object: four corner points (in source-image pixels) + class.
struct Detection {
  int class_id = -1;
  int color_id = -1;  ///< only used by the 0526/V5 armor layout (0..3)
  float score = 0.0f;
  float quality = 0.0f;
  cv::Point2f corners[4];
  cv::Rect2f bbox;  ///< axis-aligned hull of the four corners, used for NMS
};

/**
 * @brief Tuning knobs.
 *
 * Strings are `const char*` on purpose: this struct crosses a translation-unit
 * boundary where the OpenVINO side is built with `_GLIBCXX_USE_CXX11_ABI=0`
 * (official OpenVINO Linux builds) while the ROS side uses ABI=1.  The detector
 * copies the strings during init().
 */
struct DetectorOptions {
  const char* model_path = nullptr;
  const char* device = "CPU";
  int input_width = 640;   ///< only used when the ONNX input shape is dynamic
  int input_height = 480;  ///< only used when the ONNX input shape is dynamic
  float conf_threshold = 0.20f;
  float quality_threshold = 0.05f;
  float nms_threshold = 0.45f;
  int num_threads = 0;  ///< 0 -> OpenVINO default
  int my_color = -1;    ///< V5 armor only: -1=keep all, 0=blue, 1=red
  bool raw_output = false;  ///< 18ch output is raw [1, anchors, 18]
};

struct DetectorTimings {
  double preprocess_ms = 0.0;
  double infer_ms = 0.0;
  double postprocess_ms = 0.0;

  double totalMs() const { return preprocess_ms + infer_ms + postprocess_ms; }
};

/**
 * @brief OpenVINO wrapper for the 1027 PicoPoseAuxBox and 0526 V5 exports.
 *
 * Supported output layouts:
 *   row-major [1, anchors, C]:
 *     23 channels: corners(8) + quality(1) + aux-14 class scores(14)
 *     18 channels: corners(8) + quality(1) + colour(2) + kind(7)
 *     22 channels: V5 armor: corners(8) + confidence(1) +
 *                  colour scores(4) + armour class scores(9)
 *   raw channel-first [1, 18, anchors]:
 *     kpt(8) + quality logit(1) + colour logits(2) + kind logits(7)
 *
 * Raw 18 output is decoded by the detector: keypoints use
 * (raw + anchor) * stride, quality uses sigmoid, colour/kind use softmax.
 *
 * The implementation is a pimpl so this header stays free of OpenVINO headers:
 * only openvino_detector.cpp is compiled with the old libstdc++ ABI.
 */
class OpenVinoDetector {
 public:
  OpenVinoDetector();
  ~OpenVinoDetector();
  OpenVinoDetector(const OpenVinoDetector&) = delete;
  OpenVinoDetector& operator=(const OpenVinoDetector&) = delete;

  bool init(const DetectorOptions& options, char* error, std::size_t error_len);

  /**
   * @param bgr        source image, CV_8UC3 BGR
   * @param detections output, sorted by score (class-wise NMS already applied)
   * @param timings    preprocess / OpenVINO infer / decode+NMS durations
   */
  bool detect(const cv::Mat& bgr, std::vector<Detection>* detections,
              DetectorTimings* timings, char* error, std::size_t error_len);

  void setConfThreshold(float value);
  void setQualityThreshold(float value);
  void setNmsThreshold(float value);
  float confThreshold() const;
  float qualityThreshold() const;
  float nmsThreshold() const;

  int inputWidth() const;
  int inputHeight() const;
  int numClasses() const;
  int numColors() const;
  int outputChannels() const;
  int numAnchors() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hk_pose_openvino

#endif  // HK_POSE_OPENVINO_OPENVINO_DETECTOR_H_
