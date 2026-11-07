#!/usr/bin/env python3
"""Publish one image (or a directory of images) as sensor_msgs/CompressedImage.

Used by pose_openvino_test.launch to exercise the nodelet without a camera.
The topic is the full compressed topic name, matching the nodelet's direct
CompressedImage subscription.
"""
from __future__ import print_function

import glob
import os
import sys

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


class TestImagePublisher(object):
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/hk_camera_cs/image_raw/compressed")
        self.fps = max(0.1, float(rospy.get_param("~fps", 10.0)))
        self.loop = bool(rospy.get_param("~loop", True))
        self.frame_id = rospy.get_param("~frame_id", "test_camera")

        source = rospy.get_param("~image", "")
        if not source:
            raise SystemExit("~image is empty")

        if os.path.isdir(source):
            paths = sorted(glob.glob(os.path.join(source, "*")))
        else:
            paths = [source]

        self.images = []
        for path in paths:
            if not os.path.isfile(path):
                continue
            image = cv2.imread(path)
            if image is None:
                rospy.logwarn("cannot read %s", path)
                continue
            self.images.append((os.path.basename(path), image))

        if not self.images:
            raise SystemExit("no readable image found at %s" % source)

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(self.topic, CompressedImage, queue_size=1)
        self.index = 0
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.fps), self.tick)

        rospy.loginfo("test_image_publisher: %d image(s) -> %s at %.1f Hz",
                      len(self.images), self.topic, self.fps)
        for name, image in self.images:
            rospy.loginfo("  %s (%dx%d)", name, image.shape[1], image.shape[0])

    def tick(self, _event):
        if not self.loop and self.index >= len(self.images):
            return
        name, image = self.images[self.index % len(self.images)]
        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(image, dst_format="jpg")
        except Exception as exc:  # pragma: no cover - defensive
            rospy.logerr("cv2_to_compressed_imgmsg failed for %s: %s", name, exc)
            return
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)
        self.index += 1


if __name__ == "__main__":
    rospy.init_node("test_image_publisher")
    try:
        TestImagePublisher()
        rospy.spin()
    except KeyboardInterrupt:
        sys.exit(0)
