#ifndef HK_POSE_OPENVINO_POSE_NODELET_H_
#define HK_POSE_OPENVINO_POSE_NODELET_H_

#include <nodelet/nodelet.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CompressedImage.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hk_pose_openvino/PoseConfigConfig.h"
#include "hk_pose_openvino/openvino_detector.h"

namespace hk_pose_openvino {

/**
 * @brief Nodelet glue: image in -> OpenVINO -> annotated image out.
 *
 * The callback uses try_lock so that a slow frame is dropped instead of
 * queueing; combined with queue_size=1 this keeps latency low under load.
 */
class PoseNodelet : public nodelet::Nodelet {
 public:
  void onInit() override;

 private:
  void compressedCallback(const sensor_msgs::CompressedImageConstPtr& msg);
  void draw(cv::Mat* image, const std::vector<Detection>& detections,
            const DetectorTimings& timings);
  void reconfigureCallback(hk_pose_openvino::PoseConfigConfig& config, uint32_t level);

  std::unique_ptr<OpenVinoDetector> detector_;
  std::vector<std::string> class_names_;
  std::vector<std::string> color_names_;

  ros::Subscriber subscriber_;
  image_transport::Publisher publisher_;

  std::unique_ptr<dynamic_reconfigure::Server<hk_pose_openvino::PoseConfigConfig>>
      reconfigure_server_;
  boost::recursive_mutex reconfigure_mutex_;

  std::mutex mutex_;
  cv::Mat vis_;

  bool draw_labels_ = true;
  bool draw_points_ = true;
  int line_thickness_ = 0;
  double font_scale_ = 0.0;
  int log_interval_ = 30;

  double fps_ = 0.0;
  std::chrono::steady_clock::time_point last_frame_time_{};
  unsigned long frame_count_ = 0;
  unsigned long dropped_frames_ = 0;
};

}  // namespace hk_pose_openvino

#endif  // HK_POSE_OPENVINO_POSE_NODELET_H_
