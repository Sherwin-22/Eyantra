#!/usr/bin/env python3
'''
# Team ID:          eYRC#4633
# Theme:            Krishi coBot
# Author List:      Neelay Jain, Sherwin Abraham
# Filename:         ebot_nav.py
# Functions:        odom_callback, scan_callback, control_loop, quaternion_to_yaw,
#                   angle_diff, get_sector_distances, main
# Global variables: WAYPOINTS, POS_TOL, YAW_TOL, KP_LIN, KP_ANG, MAX_LINEAR_SPEED,
#                   MAX_ANGULAR_SPEED, OBSTACLE_DIST_STOP, OBSTACLE_DIST_SLOW
'''

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import csv

# -------- Global Parameters -------- #
# These parameters define the navigation behavior and limits.
# The values are tuned experimentally for stable and responsive movement.
WAYPOINTS = [
    [-1.53, -1.95, 1.57],  # P1
    [0.13, 1.24, 0.0],     # P2
    [0.38, -3.32, -1.57],  # P3
]

POS_TOL = 0.3       # Robot is considered "at" a waypoint when within this radius
YAW_TOL = math.radians(10)  # Acceptable yaw alignment error for final orientation

# Controller gains — chosen to make motion smooth and responsive.
KP_LIN = 0.5        
KP_ANG = 1.1        

# Safety and motion limits to prevent overshooting or jerky motion
MAX_LINEAR_SPEED = 0.6
MAX_ANGULAR_SPEED = 1.2

# Distance thresholds for obstacle handling (in meters)
OBSTACLE_DIST_STOP = 0.4
OBSTACLE_DIST_SLOW = 0.6

# Field of view regions for obstacle detection
FRONT_ANGLE_DEG = 35
SIDE_ANGLE_DEG = 45

# For analysis and debugging of robot behavior
LOG_FILE = "ebot_nav_log.csv"


class EbotNavigator(Node):
    '''
    The EbotNavigator node handles motion control, waypoint tracking, and
    obstacle avoidance. It continuously reads odometry and LIDAR data,
    and publishes velocity commands accordingly.
    '''
    def __init__(self):
        super().__init__('ebot_nav_node')

        # Store robot’s current position and orientation
        self.pose = [0.0, 0.0, 0.0]     
        # LIDAR distance readings
        self.scan_ranges = []           
        # Keep track of which waypoint we’re currently targeting
        self.current_wp = 0             
        # Once all waypoints are done, this prevents further motion commands
        self.reached_all = False        

        # ROS interfaces
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Control loop running every 0.1s (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize a log file to record useful data
        self.init_log()

        self.get_logger().info(f"✅ eBot Navigator Started (KP_LIN={KP_LIN}, KP_ANG={KP_ANG}, FRONT={FRONT_ANGLE_DEG}°, SIDE={SIDE_ANGLE_DEG}°)")

    def init_log(self):
        '''
        A fresh CSV file is created for each run.
        This is helpful for plotting position vs time and analyzing robot dynamics later.
        '''
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'yaw', 'linear_x', 'angular_z', 'min_front_range'])

    def log_state(self, linear_x, angular_z, min_front):
        '''
        Each iteration of the control loop writes the robot’s
        pose and command values into the log file.
        This makes post-run performance visualization possible.
        '''
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_clock().now().nanoseconds / 1e9,
                             self.pose[0], self.pose[1], self.pose[2],
                             linear_x, angular_z, min_front])

    def odom_callback(self, msg):
        '''
        Odometry provides position and orientation feedback.
        This callback updates the robot’s pose in the world frame.
        '''
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)  # Extract yaw angle from quaternion
        self.pose = [x, y, yaw]

    def scan_callback(self, msg):
        '''
        LaserScan callback receives 360° distance readings.
        These are later divided into sectors (front, left, right)
        to make obstacle avoidance decisions simpler.
        '''
        self.scan_ranges = msg.ranges

    def control_loop(self):
        '''
        The brain of the robot — this runs repeatedly and:
        1. Calculates heading and distance to the current waypoint.
        2. Checks for nearby obstacles using LIDAR.
        3. Adjusts linear and angular velocity based on both.
        4. Decides when a waypoint is reached and moves to the next.
        '''
        if self.current_wp >= len(WAYPOINTS):
            # Stop once all waypoints are completed
            self.cmd_pub.publish(Twist())
            if not self.reached_all:
                self.get_logger().info("🎯 All waypoints reached!")
                self.reached_all = True
            return

        # Target waypoint
        goal_x, goal_y, goal_yaw = WAYPOINTS[self.current_wp]
        x, y, yaw = self.pose

        # Compute errors in position and heading
        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)                 # Straight-line distance to goal
        goal_heading = math.atan2(dy, dx)         # Desired direction to move
        heading_error = self.angle_diff(goal_heading, yaw)
        yaw_error = self.angle_diff(goal_yaw, yaw)

        twist = Twist()

        # -------- OBSTACLE AVOIDANCE -------- #
        # Divide the LIDAR data into 3 regions and get minimum distances.
        front, left, right = self.get_sector_distances()

        # If an obstacle is close in front, apply a bias in angular velocity
        # toward the side that’s more open (simple reactive strategy).
        steer_correction = 0.0
        if front < OBSTACLE_DIST_SLOW:
            steer_correction = 0.5 * (left - right)

        # Proportional controller for speed control — slows down as it gets closer
        lin_speed = KP_LIN * dist
        if front < OBSTACLE_DIST_SLOW:
            # Gradually reduce speed if obstacle detected ahead
            lin_speed *= max(front / OBSTACLE_DIST_SLOW, 0.3)

        # Limit velocities to keep robot stable and prevent sharp turns
        twist.linear.x = min(max(lin_speed, 0.15), MAX_LINEAR_SPEED)
        twist.angular.z = max(min(KP_ANG * heading_error + steer_correction,
                                  MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

        # -------- SAFETY HANDLING -------- #
        # If something comes too close, stop and rotate to avoid collision.
        if front < OBSTACLE_DIST_STOP:
            twist.linear.x = 0.1
            twist.angular.z = 0.5 if left > right else -0.5
            self.get_logger().warn(f"🚧 Obstacle too close! Front={front:.2f} m")

        # Slightly increase angular speed if heading error is large.
        if abs(heading_error) > math.radians(25):
            twist.linear.x = max(twist.linear.x, 0.2)

        # -------- WAYPOINT CHECK -------- #
        # If near the waypoint, first slow down, then align orientation.
        if dist < POS_TOL + 0.05:
            if dist > POS_TOL:
                twist.linear.x = max(0.1, KP_LIN * dist)
            else:
                twist.linear.x = 0.0

            # Once close enough, rotate in place to match the target yaw.
            if abs(yaw_error) > YAW_TOL:
                twist.angular.z = max(min(0.8 * yaw_error, MAX_ANGULAR_SPEED),
                                      -MAX_ANGULAR_SPEED)
                self.get_logger().info(f"🌀 Aligning to target yaw at WP{self.current_wp + 1}...")
            else:
                # Fully reached — proceed to next waypoint
                if dist <= POS_TOL:
                    self.get_logger().info(f"✅ Reached waypoint {self.current_wp + 1}")
                    self.current_wp += 1
                    twist = Twist()
                    time.sleep(0.4)  # Small pause for stability before next move

        # Publish computed velocities and log state
        self.cmd_pub.publish(twist)
        self.log_state(twist.linear.x, twist.angular.z, front)

    def get_sector_distances(self):
        '''
        LIDAR gives 360° readings; this method condenses them into
        front, left, and right sectors by taking the minimum distance
        in each region. This helps the robot react intuitively to obstacles.
        '''
        if not self.scan_ranges:
            return float('inf'), float('inf'), float('inf')

        n = len(self.scan_ranges)
        idx_center = n // 2
        half_front = int(math.radians(FRONT_ANGLE_DEG) / (math.pi / n))
        half_side = int(math.radians(SIDE_ANGLE_DEG) / (math.pi / n))

        # Divide the scan data into zones around the robot
        front_indices = range(idx_center - half_front, idx_center + half_front + 1)
        left_indices = range(idx_center + half_front + 1, min(n, idx_center + half_side + 1))
        right_indices = range(max(0, idx_center - half_side), idx_center - half_front)

        # Helper to filter out invalid or extreme readings
        def safe_min(indices):
            vals = [self.scan_ranges[i] for i in indices if 0 < self.scan_ranges[i] < 10.0]
            return min(vals) if vals else 10.0

        return safe_min(front_indices), safe_min(left_indices), safe_min(right_indices)

    def quaternion_to_yaw(self, q):
        '''
        Converts quaternion (4D rotation representation) to yaw angle.
        ROS provides orientation as quaternion, but for navigation,
        we only need the rotation around Z-axis.
        '''
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def angle_diff(self, a, b):
        '''
        Returns the shortest angular difference between two angles.
        Keeps rotation smooth instead of flipping between +π and -π.
        '''
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff


def main(args=None):
    '''
    Initializes ROS2, starts the navigation node, and spins until shutdown.
    Acts as the program’s entry point.
    '''
    rclpy.init(args=args)
    node = EbotNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# Standard Python entry point
if __name__ == "__main__":
    main()
