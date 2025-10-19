#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time

class UR5ArmWaypoint(Node):
    def __init__(self):
        super().__init__('ur5_waypoint_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ✅ Try publishing to /command instead of /joint_trajectory
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/command',
            qos
        )

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint',
            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.waypoints = [
            [-1.2, -1.5, 1.4, -1.2, 1.6, 0.0],
            [ 0.5, -1.7, 1.6, -1.3, 1.4, 0.2],
            [-2.0, -1.3, 1.7, -1.5, 1.2, 0.0]
        ]

        self.move_through_waypoints()

    def move_through_waypoints(self):
        for i, wp in enumerate(self.waypoints):
            traj = JointTrajectory()
            traj.joint_names = self.joint_names

            point = JointTrajectoryPoint()
            point.positions = wp
            point.velocities = [0.0] * 6
            point.time_from_start = Duration(sec=4)
            traj.points.append(point)

            self.publisher.publish(traj)
            self.get_logger().info(f"Moving to waypoint {i+1}")
            time.sleep(5)

        self.get_logger().info("✅ All waypoints reached!")

def main(args=None):
    rclpy.init(args=args)
    node = UR5ArmWaypoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
