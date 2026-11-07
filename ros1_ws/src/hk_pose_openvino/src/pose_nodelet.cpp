#include "hk_pose_openvino/pose_nodelet.h"

#include <cv_bridge/cv_bridge.h>
#include <pluginlib/class_list_macros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace hk_pose_openvino {
namespace {

cv::Scalar classColor(int class_id) {
  // BGR. 0-6 blue side, 7-13 red side (index == colour * 7 + kind).
  static const cv::Scalar kPalette[14] = {
      {255, 160, 0}, {255, 128, 0}, {255, 96, 0}, {255, 64, 0},
      {255, 32, 0},  {255, 208, 0}, {0, 215, 255},
      {0, 0, 255},   {0, 32, 255},  {0, 64, 255}, {0, 96, 255},
      {0, 128, 255}, {0, 160, 255}, {0, 200, 255}};
  if (class_id >= 0 && class_id < 14) {
    return kPalette[class_id];
  }
  return cv::Scalar(0, 255, 0);
}

}  // namespace

void PoseNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();

  const std::string package_path = ros::package::getPath("hk_pose_openvino");
  std::string default_model = package_path + "/models/epoch100_rm_23.onnx";
  std::string model_path;
  pnh.param<std::string>("model_path", model_path, default_model);
  if (model_path.empty()) {
    model_path = default_model;
  }

  std::string device;
  pnh.param<std::string>("device", device, "CPU");
  if (device.empty()) {
    device = "CPU";
  }

  DetectorOptions options;
  options.model_path = model_path.c_str();
  options.device = device.c_str();
  pnh.param("input_width", options.input_width, 640);
  pnh.param("input_height", options.input_height, 480);

  double conf_threshold = 0.90;
  double quality_threshold = 0.05;
  double nms_threshold = 0.45;
  pnh.param("conf_threshold", conf_threshold, 0.90);
  pnh.param("quality_threshold", quality_threshold, 0.05);
  pnh.param("nms_threshold", nms_threshold, 0.45);
  options.conf_threshold =
      static_cast<float>(std::clamp(conf_threshold, 0.0, 1.0));
  options.quality_threshold =
      static_cast<float>(std::clamp(quality_threshold, 0.0, 1.0));
  options.nms_threshold =
      static_cast<float>(std::clamp(nms_threshold, 0.0, 1.0));
  pnh.param("num_threads", options.num_threads, 0);
  pnh.param("my_color", options.my_color, -1);
  pnh.param("raw_output", options.raw_output, false);

  detector_.reset(new OpenVinoDetector());
  char error[512] = {0};
  if (!detector_->init(options, error, sizeof(error))) {
    NODELET_FATAL_STREAM("hk_pose_openvino: " << error);
    throw std::runtime_error(error);
  }

  if (detector_->numColors() == 4) {
    // 0526 / Infantry-v5 armor: 9 armor classes x 4 colors.
    static const char* const kArmor9[9] = {
        "sentry", "hero", "engineer", "infantry3", "infantry4",
        "infantry5", "outpost", "base_bottom", "base_top"};
    class_names_.assign(kArmor9, kArmor9 + 9);
    color_names_ = {"blue", "red", "white", "purple"};
  } else {
    static const char* const kDefaultClassNames[14] = {
        "blue_1",     "blue_2", "blue_3", "blue_4", "blue_5",
        "blue_guard", "blue_outpost",
        "red_1",      "red_2",  "red_3",  "red_4",  "red_5",
        "red_guard",  "red_outpost"};
    class_names_.assign(kDefaultClassNames, kDefaultClassNames + 14);
  }
  std::vector<std::string> configured_names;
  if (pnh.getParam("class_names", configured_names) &&
      static_cast<int>(configured_names.size()) == detector_->numClasses()) {
    class_names_ = configured_names;
  } else if (!configured_names.empty()) {
    NODELET_WARN("class_names has %zu entries but the model has %d classes; "
                 "using the model defaults",
                 configured_names.size(), detector_->numClasses());
  }
  while (static_cast<int>(class_names_.size()) < detector_->numClasses()) {
    class_names_.push_back("class_" + std::to_string(class_names_.size()));
  }

  std::string image_topic;
  std::string output_topic;
  int queue_size = 1;
  pnh.param<std::string>("image_topic", image_topic, "/hk_camera/image_raw/compressed");
  pnh.param<std::string>("output_topic", output_topic,
                         "/hk_pose_openvino/image_result");
  pnh.param("queue_size", queue_size, 1);
  queue_size = std::max(1, queue_size);

  pnh.param("draw_labels", draw_labels_, true);
  pnh.param("draw_points", draw_points_, true);
  pnh.param("line_thickness", line_thickness_, 0);
  pnh.param("font_scale", font_scale_, 0.0);
  pnh.param("log_interval", log_interval_, 30);

  reconfigure_server_.reset(
      new dynamic_reconfigure::Server<hk_pose_openvino::PoseConfigConfig>(
          reconfigure_mutex_, pnh));
  {
    dynamic_reconfigure::Server<hk_pose_openvino::PoseConfigConfig>::CallbackType
        callback = boost::bind(&PoseNodelet::reconfigureCallback, this, _1, _2);
    reconfigure_server_->setCallback(callback);
  }

  image_transport::ImageTransport it(pnh);
  publisher_ = it.advertise(output_topic, 1);
  // Direct CompressedImage subscription: image_topic is the full topic name
  // (e.g. /hk_camera/image_raw/compressed); cv_bridge decodes in-process.
  subscriber_ = pnh.subscribe(image_topic, queue_size,
                              &PoseNodelet::compressedCallback, this);

  NODELET_INFO("model   : %s", model_path.c_str());
  NODELET_INFO("device  : %s", device.c_str());
  NODELET_INFO("input   : %dx%d (WxH)", detector_->inputWidth(),
               detector_->inputHeight());
  const char* layout_name =
      detector_->numColors() == 4
          ? "22 (V5 armour 4pt/4colour/9cls)"
          : (detector_->outputChannels() == 23 ? "23 (aux-14)"
                                               : "18 (colour x kind)");
  NODELET_INFO("output  : %d anchors x %d channels (%d classes, layout %s)",
               detector_->numAnchors(), detector_->outputChannels(),
               detector_->numClasses(), layout_name);
  NODELET_INFO("topics  : %s -> %s (compressed)", image_topic.c_str(),
               output_topic.c_str());
  NODELET_INFO("thresholds: conf=%.2f quality=%.2f nms=%.2f my_color=%d",
               options.conf_threshold, options.quality_threshold,
               options.nms_threshold, options.my_color);
}

void PoseNodelet::reconfigureCallback(
    hk_pose_openvino::PoseConfigConfig& config, uint32_t level) {
  // The initial callback runs with level 0xFFFFFFFF from the private node
  // handle, so parameters already loaded above must not be overwritten by the
  // generated defaults.  Only apply thresholds on later (real) changes.
  if (level == 0xFFFFFFFF) {
    draw_labels_ = config.draw_labels;
    draw_points_ = config.draw_points;
    return;
  }
  detector_->setConfThreshold(static_cast<float>(config.conf_threshold));
  detector_->setQualityThreshold(
      static_cast<float>(config.quality_threshold));
  detector_->setNmsThreshold(static_cast<float>(config.nms_threshold));
  draw_labels_ = config.draw_labels;
  draw_points_ = config.draw_points;
  NODELET_INFO("reconfigured: conf=%.3f quality=%.3f nms=%.3f",
               config.conf_threshold, config.quality_threshold,
               config.nms_threshold);
}

void PoseNodelet::compressedCallback(
    const sensor_msgs::CompressedImageConstPtr& msg) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    ++dropped_frames_;
    return;
  }

  cv::Mat bgr;
  try {
    bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (const cv_bridge::Exception& e) {
    NODELET_ERROR_THROTTLE(5.0, "cv_bridge failed (format=%s): %s",
                           msg->format.c_str(), e.what());
    return;
  }
  if (bgr.empty()) {
    return;
  }

  std::vector<Detection> detections;
  DetectorTimings timings;
  char error[512] = {0};
  if (!detector_->detect(bgr, &detections, &timings, error, sizeof(error))) {
    NODELET_ERROR_THROTTLE(5.0, "detect failed: %s", error);
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (last_frame_time_ != std::chrono::steady_clock::time_point{}) {
    const double dt =
        std::chrono::duration<double>(now - last_frame_time_).count();
    if (dt > 1e-6) {
      const double instant = 1.0 / dt;
      fps_ = (fps_ <= 0.0) ? instant : 0.9 * fps_ + 0.1 * instant;
    }
  }
  last_frame_time_ = now;

  bgr.copyTo(vis_);
  draw(&vis_, detections, timings);

  auto out_msg = cv_bridge::CvImage(msg->header,
                                    sensor_msgs::image_encodings::BGR8, vis_)
                     .toImageMsg();
  publisher_.publish(out_msg);
  ++frame_count_;

  if (log_interval_ > 0 &&
      frame_count_ % static_cast<unsigned long>(log_interval_) == 0) {
    NODELET_INFO(
        "frame %lu: detect %.2f ms (pre %.2f / infer %.2f / post %.2f) | "
        "%zu obj | %.1f fps | dropped %lu",
        frame_count_, timings.totalMs(), timings.preprocess_ms,
        timings.infer_ms, timings.postprocess_ms, detections.size(), fps_,
        dropped_frames_);
  }
}

void PoseNodelet::draw(cv::Mat* image,
                       const std::vector<Detection>& detections,
                       const DetectorTimings& timings) {
  cv::Mat& img = *image;
  const int auto_thickness =
      std::max(1, static_cast<int>(std::lround(img.cols / 900.0)));
  const int thickness = line_thickness_ > 0 ? line_thickness_ : auto_thickness;
  const double font_scale =
      font_scale_ > 0.0 ? font_scale_ : std::max(0.45, img.cols / 2000.0);

  for (const Detection& det : detections) {
    const cv::Scalar color = classColor(det.class_id);

    // The network emits the four corners in a consistent order, but sorting
    // by angle around the centroid guarantees a simple, non-self-intersecting
    // quadrilateral even if a frame's point order is unusual.
    std::vector<cv::Point2f> corners(det.corners, det.corners + 4);
    cv::Point2f center(0.0f, 0.0f);
    for (const cv::Point2f& p : corners) {
      center += p;
    }
    center *= 0.25f;
    std::sort(corners.begin(), corners.end(),
              [&center](const cv::Point2f& a, const cv::Point2f& b) {
                return std::atan2(a.y - center.y, a.x - center.x) <
                       std::atan2(b.y - center.y, b.x - center.x);
              });

    std::vector<cv::Point> points;
    points.reserve(4);
    for (const cv::Point2f& p : corners) {
      points.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    // The four predicted points, connected into a quadrilateral.
    cv::polylines(img, points, /*isClosed=*/true, color, thickness,
                  cv::LINE_AA);
    if (draw_points_) {
      for (const cv::Point& p : points) {
        cv::circle(img, p, thickness * 2, cv::Scalar(0, 255, 0),
                   cv::FILLED, cv::LINE_AA);
      }
    }

    if (!draw_labels_) {
      continue;
    }
    const char* name = (det.class_id >= 0 &&
                        det.class_id < static_cast<int>(class_names_.size()))
                           ? class_names_[det.class_id].c_str()
                           : "?";
    char label[160];
    if (det.color_id >= 0 &&
        det.color_id < static_cast<int>(color_names_.size())) {
      std::snprintf(label, sizeof(label), "%s_%s %.2f",
                    color_names_[det.color_id].c_str(), name, det.score);
    } else {
      std::snprintf(label, sizeof(label), "%s %.2f", name, det.score);
    }

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    int x = static_cast<int>(det.bbox.x);
    int y = static_cast<int>(det.bbox.y) - 6;
    if (y - text_size.height - 4 < 0) {
      y = static_cast<int>(det.bbox.y + det.bbox.height) + text_size.height + 8;
    }
    x = std::clamp(x, 0, std::max(0, img.cols - text_size.width - 10));
    const int min_y = text_size.height + 6;
    const int max_y = std::max(min_y, img.rows - baseline - 2);
    y = std::clamp(y, min_y, max_y);

    cv::rectangle(img,
                  cv::Rect(x - 3, y - text_size.height - 3,
                           text_size.width + 6,
                           text_size.height + baseline + 6),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv::LINE_AA);
  }

  char hud[320];
  if (img.cols >= 1000) {
    std::snprintf(hud, sizeof(hud),
                  "detect %.2f ms (pre %.2f | infer %.2f | post %.2f)  "
                  "proc %.1f fps  %zu obj",
                  timings.totalMs(), timings.preprocess_ms,
                  timings.infer_ms, timings.postprocess_ms, fps_,
                  detections.size());
  } else {
    std::snprintf(hud, sizeof(hud), "detect %.2f ms | proc %.1f fps | %zu obj",
                  timings.totalMs(), fps_, detections.size());
  }

  int baseline = 0;
  const cv::Size hud_size =
      cv::getTextSize(hud, cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.1,
                      thickness + 1, &baseline);
  const int hud_x = 8;
  const int hud_y = 10 + hud_size.height;
  cv::rectangle(img,
                cv::Rect(hud_x - 4, 6, hud_size.width + 14,
                         hud_size.height + baseline + 12),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(img, hud, cv::Point(hud_x, hud_y), cv::FONT_HERSHEY_SIMPLEX,
              font_scale * 1.1, cv::Scalar(0, 255, 255), thickness + 1,
              cv::LINE_AA);
}

}  // namespace hk_pose_openvino

PLUGINLIB_EXPORT_CLASS(hk_pose_openvino::PoseNodelet, nodelet::Nodelet)
