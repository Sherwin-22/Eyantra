#!/usr/bin/env python3
"""
# Team ID:         KR25
# Theme:           Krishi coBot
# Author List:     Neelay Jain, Sherwin Abraham (Refactored by Gemini)
# Filename:        task1c.py
#
# Logic: This is a hybrid "logic-driven" controller.
# It uses two modes:
# 1. "POTENTIAL_FIELD": (Slow & Smart) Uses a repulsive force to
#    navigate dangerous paths near the base (e.g., P1 -> Center).
# 2. "SIMPLE_P": (Fast & Simple) Uses a high-gain P-controller
#    for safe, open-space moves (e.g., Center -> P2, P2 -> P3).
# This solves the "local minimum" (getting stuck) and "slow speed" issues.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf_transformations import quaternion_multiply, quaternion_conjugate
from tf2_ros import Buffer, TransformListener
import numpy as np
import time

class ArmHybridController(Node):
    def __init__(self):
        super().__init__('task1c_node')

        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Tuning Parameters for "SIMPLE_P" (Fast) Mode ---
        self.Kp_pos_fast = 0.7
        self.Kp_ori_fast = 0.7
        self.clip_val_fast = 0.5 # Fast speed

        # --- Tuning Parameters for "POTENTIAL_FIELD" (Smart) Mode ---
        self.Kp_pos_safe = 0.6
        self.Kp_ori_safe = 0.5
        self.clip_val_safe = 0.4 # Slower, more stable
        self.K_rep = 0.5         # Repulsive strength
        self.R_singular = 0.25   # "Danger Zone" radius (25cm)
        
        self.POS_TOL = 0.15 
        self.ORI_TOL = 0.15
        
        # --- Waypoint Definitions ---
        P1_pos = [-0.214, -0.532, 0.557]
        P1_ori = [0.707, 0.028, 0.034, 0.707]

        P_Center_pos = [-0.2, 0.0, 0.7]
        P_Center_ori = [0.0, 0.707, 0.0, 0.707] # Pointing forward

        P2_pos = [-0.159,  0.501, 0.415]
        P2_ori = [0.029, 0.997, 0.045, 0.033]

        P3_pos = [-0.806,  0.010, 0.182]
        P3_ori = [-0.684, 0.726, 0.05, 0.008]

        # --- Path Plan ---
        # We define the controller to use for each *leg* of the journey.
        # "P1" is the goal, but the *move to* P1 is "SIMPLE_P"
        self.path_plan = [
            (np.array(P1_pos), np.array(P1_ori), "1", "SIMPLE_P"),
            (np.array(P_Center_pos), np.array(P_Center_ori), "Center", "POTENTIAL_FIELD"), # Dangerous move
            (np.array(P2_pos), np.array(P2_ori), "2", "SIMPLE_P"),      # Safe move
            (np.array(P3_pos), np.array(P3_ori), "3", "SIMPLE_P")       # Safe move
        ]

        # --- State Machine ---
        self.target_index = 0
        self.state = "MOVING"
        self.wait_start_time = None
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Hybrid Logic Controller initialized. Waiting for TF...")
        time.sleep(2.0) 
        self.get_logger().info(f"Starting navigation. Moving to Waypoint {self.path_plan[0][2]}...")


    def get_current_pose(self):
        """Looks up the current pose of the end-effector (ee_link) relative to the base (base_link)."""
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', 'ee_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            ori = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            return pos, ori
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None, None


    def control_loop(self):
        """Main state machine, called by the timer."""
        
        if self.target_index >= len(self.path_plan):
            if self.state != "DONE":
                self.get_logger().info("🎯 All waypoints completed! Press Ctrl+C to exit.")
                self.timer.cancel()
                self.pub.publish(Twist())
                self.state = "DONE"
            return

        # Get the current target waypoint and CONTROLLER MODE
        target_pos, target_ori, target_label, controller_mode = self.path_plan[self.target_index]

        pos, ori = self.get_current_pose()
        if pos is None or ori is None:
            self.get_logger().warn("Waiting for TF... stopping motion.")
            self.pub.publish(Twist())
            return
        
        twist = Twist()

        # --- State: MOVING ---
        if self.state == "MOVING":
            # --- 1. Calculate Errors and Attraction Force ---
            err_pos = target_pos - pos
            
            q_conj = quaternion_conjugate(ori)
            q_err = quaternion_multiply(target_ori, q_conj)
            err_ori_vec = np.array(q_err[:3])
            if q_err[3] < 0:
                err_ori_vec *= -1
            
            pos_reached = np.linalg.norm(err_pos) < self.POS_TOL
            ori_reached = np.linalg.norm(err_ori_vec) < self.ORI_TOL

            # --- Check if Goal Reached ---
            if pos_reached and ori_reached:
                self.get_logger().info(f"✅ Reached Waypoint {target_label}. Waiting 1 second...")
                self.state = "WAITING"
                self.wait_start_time = self.get_clock().now()
                self.pub.publish(twist)
                return

            # --- 2. LOGIC-DRIVEN CONTROLLER ---
            v_rep_linear = np.array([0.0, 0.0, 0.0]) # Default: no repulsion
            
            if controller_mode == "POTENTIAL_FIELD":
                # Use "Smart & Safe" gains
                Kp_pos = self.Kp_pos_safe
                Kp_ori = self.Kp_ori_safe
                clip_val = self.clip_val_safe

                # Calculate Repulsive Force
                d_xy = np.linalg.norm(pos[:2]) 
                if d_xy < self.R_singular:
                    self.get_logger().warn(f"Singularity danger! Repelling.", throttle_duration_sec=1.0)
                    rep_dir = pos / (d_xy + 1e-6) # Add epsilon to avoid division by zero
                    rep_dir[2] = 0.0 
                    rep_dir_norm = np.linalg.norm(rep_dir)
                    if rep_dir_norm > 0:
                        rep_dir = rep_dir / rep_dir_norm
                    
                    rep_mag = self.K_rep * (1/d_xy - 1/self.R_singular)
                    v_rep_linear = rep_dir * rep_mag

            else: # controller_mode == "SIMPLE_P"
                # Use "Fast" gains
                Kp_pos = self.Kp_pos_fast
                Kp_ori = self.Kp_ori_fast
                clip_val = self.clip_val_fast
                # v_rep_linear remains [0, 0, 0]

            # --- 3. Calculate Final Velocity ---
            v_attr_linear = Kp_pos * err_pos
            v_attr_angular = Kp_ori * err_ori_vec
            
            v_final_linear = v_attr_linear + v_rep_linear
            v_final_angular = v_attr_angular

            # --- 4. Publish Clipped Command ---
            twist.linear.x = np.clip(v_final_linear[0], -clip_val, clip_val)
            twist.linear.y = np.clip(v_final_linear[1], -clip_val, clip_val)
            twist.linear.z = np.clip(v_final_linear[2], -clip_val, clip_val)
            
            twist.angular.x = np.clip(v_final_angular[0], -clip_val, clip_val)
            twist.angular.y = np.clip(v_final_angular[1], -clip_val, clip_val)
            twist.angular.z = np.clip(v_final_angular[2], -clip_val, clip_val)
            
            self.pub.publish(twist)

        # --- State: WAITING ---
        elif self.state == "WAITING":
            self.pub.publish(twist) # Keep publishing stop
            
            duration = self.get_clock().now() - self.wait_start_time
            if duration.nanoseconds / 1e9 >= 1.0:
                self.target_index += 1 # Move to the next waypoint
                self.state = "MOVING"
                
                if self.target_index < len(self.path_plan):
                    next_label = self.path_plan[self.target_index][2]
                    self.get_logger().info(f"Wait complete. Moving to Waypoint {next_label}...")

def main(args=None):
    rclpy.init(args=args)
    node = ArmHybridController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
