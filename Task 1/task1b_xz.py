#!/usr/bin/env python3
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

        # Subscribers
        self.color_sub = Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.sync.registerCallback(self.image_callback)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_cb, 10)

        cv2.namedWindow("Bad Fruit Detection", cv2.WINDOW_NORMAL)
        self.get_logger().info("Task1B node started. Waiting for camera info...")

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

        hsv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2HSV)
        s_mask = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY_INV)[1]
        v_mask = cv2.threshold(hsv[:, :, 2], 90, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(s_mask, v_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = color_cv.copy()
        fruit_id = 1

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            (x_center, y_center), _ = cv2.minEnclosingCircle(cnt)
            u, v = int(x_center), int(y_center)

            if not TRAY_POLYGON.contains(Point(u, v)):
                continue

            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            z = float(depth[v, u])
            if z == 0 or np.isnan(z) or z > 5.0:
                local = depth[max(0, v-3):v+4, max(0, u-3):u+4]
                nz = local[local > 0.0]
                if nz.size == 0:
                    continue
                z = float(np.median(nz))

            # Pixel → Camera
            X_cam = (u - self.cx) * z / self.fx
            Y_cam = (v - self.cy) * z / self.fy
            Z_cam = z

            p_cam = PointStamped()
            p_cam.header.frame_id = self.camera_frame
            p_cam.header.stamp = color_msg.header.stamp
            p_cam.point.x = X_cam
            p_cam.point.y = Y_cam
            p_cam.point.z = Z_cam

            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', self.camera_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                p_base = do_transform_point(p_cam, transform)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")
                continue

            # Remap axes for RViz
            p_rviz_x = p_base.point.z
            p_rviz_y = -p_base.point.x
            p_rviz_z = -p_base.point.y

            fruit_frame = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = fruit_frame
            t.transform.translation.x = p_rviz_x
            t.transform.translation.y = p_rviz_y
            t.transform.translation.z = p_rviz_z
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"bad_fruit_{fruit_id}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
