#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
*
*        Krishi coBot (KC) Theme (eYRC 2025-26)
*        Task 1B: Bad Fruit Detection & TF Publisher (with ArUco-based TF)
*
*****************************************************************************************
"""

TEAM_ID = 5  # Replace with your actual team ID

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

SHOW_IMAGE = True

class FruitsTF(Node):
    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to color and depth images
        self.create_subscription(Image, '/camera/color/image_raw', self.color_cb, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_cb, 10)

        # Timer to process images
        self.create_timer(0.2, self.process_image)

        if SHOW_IMAGE:
            cv2.namedWindow('bad_fruit_detection', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF node started with ArUco-based TF.")

    # ---------------- Callbacks ----------------
    def color_cb(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            self.get_logger().error("CV Bridge color conversion failed")

    def depth_cb(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except:
            self.get_logger().error("CV Bridge depth conversion failed")

    # ---------------- Bad Fruit Detection ----------------
    def bad_fruit_detection(self, rgb_image):
        """Detect bad fruits (greyish-white) and return center, distance, width, id."""
        bad_fruits = []
        if rgb_image is None or self.depth_image is None:
            return bad_fruits

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 150])
        upper = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphology
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w/2), int(y + h/2)
            depth_val = float(self.depth_image[cy, cx]) / 1000.0  # meters
            if depth_val == 0 or depth_val > 2:
                continue
            fruit_info = {
                'center': (cx, cy),
                'distance': depth_val,
                'width': w,
                'id': fruit_id
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1

            # Draw
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(rgb_image, f"bad_fruit", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.circle(rgb_image, (cx, cy), 5, (255,0,0), -1)
        return bad_fruits

    # ---------------- Process Image & Publish TF ----------------
    def process_image(self):
        if self.cv_image is None or self.depth_image is None:
            return

        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724
        centerCamY = 361.978
        focalX = 915.3
        focalY = 914.03

        bad_fruits = self.bad_fruit_detection(self.cv_image)

        for fruit in bad_fruits:
            cx, cy = fruit['center']
            z = fruit['distance']
            x = z * (cx - centerCamX) / focalX
            y = z * (cy - centerCamY) / focalY

            # Publish TF relative to camera_link
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = "camera_link"
            t_cam.child_frame_id = f"cam_{fruit['id']}"
            t_cam.transform.translation.x = x
            t_cam.transform.translation.y = y
            t_cam.transform.translation.z = z
            t_cam.transform.rotation.x = 0.0
            t_cam.transform.rotation.y = 0.0
            t_cam.transform.rotation.z = 0.0
            t_cam.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_cam)

            # Lookup transform from base_link → camera_link
            try:
                trans = self.tf_buffer.lookup_transform('base_link', t_cam.child_frame_id, rclpy.time.Time())
                # Publish TF from base_link → bad fruit
                t_base = TransformStamped()
                t_base.header.stamp = self.get_clock().now().to_msg()
                t_base.header.frame_id = 'base_link'
                t_base.child_frame_id = f"{TEAM_ID}_bad_fruit_{fruit['id']}"
                t_base.transform.translation.x = trans.transform.translation.x
                t_base.transform.translation.y = trans.transform.translation.y
                t_base.transform.translation.z = trans.transform.translation.z
                t_base.transform.rotation = trans.transform.rotation
                self.tf_broadcaster.sendTransform(t_base)
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().warn("Transform lookup failed for fruit {}".format(fruit['id']))
                continue

        if SHOW_IMAGE:
            cv2.imshow('bad_fruit_detection', self.cv_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

