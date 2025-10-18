#!/usr/bin/env python3
"""
Task1B: Bad Fruit Detection & TF Publisher (Best of Both Worlds)
---------------------------------------------------------------
- Accurate fruit positions using contour centroid + median depth
- RViz axes oriented exactly as desired
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import cv2
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from shapely.geometry import Point, Polygon
from tf_transformations import quaternion_from_euler

TEAM_ID = "KR25"
TRAY_POLYGON = Polygon([(57,230), (368,230), (320,384), (0,385)])

class Task1BNode(Node):
    def __init__(self):
        super().__init__('task1b_node')

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None
        self.camera_info_received = False

        # Subscriptions
        self.color_sub = Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_cb, 10)

        self.get_logger().info("Task1B node started. Waiting for camera info...")
        cv2.namedWindow("Bad Fruit Detection", cv2.WINDOW_NORMAL)

    def camera_info_cb(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.camera_frame = msg.header.frame_id
            self.camera_info_received = True
            self.get_logger().info(f"Camera info received: frame={self.camera_frame}")

    def image_callback(self, color_msg: Image, depth_msg: Image):
        if not self.camera_info_received:
            return

        try:
            color_cv = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        depth = depth_cv.astype(np.float32)
        if depth_cv.dtype == np.uint16:
            depth /= 1000.0

        # Greyish/white detection
        hsv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2HSV)
        s_mask = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY_INV)[1]
        v_mask = cv2.threshold(hsv[:, :, 2], 90, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(s_mask, v_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = color_cv.copy()
        fruit_id = 1

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            # Accurate centroid using moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            if not TRAY_POLYGON.contains(Point(u,v)):
                continue
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue

            # Median depth over contour
            mask_contour = np.zeros_like(depth, dtype=np.uint8)
            cv2.drawContours(mask_contour, [cnt], -1, 255, -1)
            zs = depth[mask_contour==255]
            zs = zs[(zs>0.05) & (zs<1.5)]
            if zs.size == 0:
                continue
            z = float(np.median(zs))

            # Pixel -> Camera Optical Frame
            X_opt = (u - self.cx) * z / self.fx
            Y_opt = (v - self.cy) * z / self.fy
            Z_opt = z

            p_cam = PointStamped()
            p_cam.header.frame_id = self.camera_frame
            p_cam.header.stamp = color_msg.header.stamp

            # Optical -> Standard ROS frame
            p_cam.point.x = Z_opt
            p_cam.point.y = -X_opt
            p_cam.point.z = -Y_opt

            try:
                # Transform to world
                transform = self.tf_buffer.lookup_transform(
                    'world',
                    self.camera_frame,
                    color_msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                p_world = do_transform_point(p_cam, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup/transform failed for fruit {fruit_id}: {e}")
                continue

            # --- Publish TF with quaternion for correct RViz axes ---
            # Green=forward, Red=opposite Green, Blue inverted
            roll = np.pi       # invert Blue
            pitch = 0
            yaw = -np.pi/2     # rotate Red/Green
            q = quaternion_from_euler(roll, pitch, yaw)

            fruit_frame = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = fruit_frame
            t.transform.translation.x = p_world.point.x
            t.transform.translation.y = p_world.point.y
            t.transform.translation.z = p_world.point.z
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)

            self.get_logger().info(f"Fruit {fruit_id} @ world: X={p_world.point.x:.3f}, "
                                   f"Y={p_world.point.y:.3f}, Z={p_world.point.z:.3f}")

            # Visualization
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display, f"bad_fruit_{fruit_id}", (x,y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            fruit_id += 1

        cv2.imshow("Bad Fruit Detection", display)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Task1BNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.get_logger().info("Task1B node shutting down.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
