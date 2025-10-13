#!/usr/bin/env python3
"""
Task 1C - Arm Manipulation
Author: Krishi Cobot Team
Description:
Move UR5 robotic arm through 3 waypoints (P1 → P2 → P3)
using joint servoing control.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trac_ik_python.trac_ik import IK
import numpy as np
import time

class UR5ServoController(Node):
    def __init__(self):
        super().__init__('ur5_servo_controller')

        # --- Publishers & Subscribers ---
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # --- UR5 Joint Names ---
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # --- Store current joint states ---
        self.current_joints = [0.0]*6
        self.received_state = False

        # --- IK Solver Setup ---
        self.ik_solver = IK("base_link", "tool0")

        # --- Define Target Waypoints ---
        self.waypoints = [
            {   # P1
                "pos": [-0.214, -0.532, 0.557],
                "ori": [0.707, 0.028, 0.034, 0.707]
            },
            {   # P2
                "pos": [-0.159, 0.501, 0.415],
                "ori": [0.029, 0.997, 0.045, 0.033]
            },
            {   # P3
                "pos": [-0.806, 0.010, 0.182],
                "ori": [-0.684, 0.726, 0.05, 0.008]
            }
        ]

        self.get_logger().info("Waiting for /joint_states...")
        while not self.received_state:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Received joint states. Starting motion...")

        self.execute_waypoints()

    def joint_state_callback(self, msg):
        # update joint feedback
        self.current_joints = list(msg.position[:6])
        self.received_state = True

    def compute_ik(self, pos, ori):
        """Compute IK using TRAC-IK solver."""
        sol = self.ik_solver.get_ik(
            self.current_joints,   # seed state
            pos[0], pos[1], pos[2],
            ori[0], ori[1], ori[2], ori[3]
        )
        if sol is None:
            self.get_logger().warn("IK solution not found for target pose!")
            return None
        return list(sol)

    def send_joint_command(self, target_joints):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start.sec = 1  # short interval servo command
        traj.points.append(point)
        self.joint_pub.publish(traj)

    def close_enough(self, target, current, tol=0.15):
        """Check if joints are within tolerance."""
        return np.all(np.abs(np.array(target) - np.array(current)) < tol)

    def execute_waypoints(self):
        for idx, wp in enumerate(self.waypoints):
            self.get_logger().info(f"Moving to Waypoint {idx+1}")

            target_joints = self.compute_ik(wp["pos"], wp["ori"])
            if target_joints is None:
                self.get_logger().warn(f"Skipping waypoint {idx+1} (no IK found)")
                continue

            # --- Servoing loop ---
            reached = False
            while rclpy.ok() and not reached:
                rclpy.spin_once(self, timeout_sec=0.05)
                self.send_joint_command(target_joints)
                reached = self.close_enough(target_joints, self.current_joints)

            self.get_logger().info(f"Reached waypoint {idx+1} ✅")
            time.sleep(1.0)  # hold for 1 second

        self.get_logger().info("All waypoints reached successfully! 🎯")

def main(args=None):
    rclpy.init(args=args)
    node = UR5ServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
