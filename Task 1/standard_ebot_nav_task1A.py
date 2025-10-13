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
WAYPOINTS = [
    [-1.53, -1.95, 1.57],  # P1
    [0.13, 1.24, 0.0],     # P2
    [0.38, -3.32, -1.57],  # P3
]

POS_TOL = 0.3       # Waypoint position tolerance (meters)
YAW_TOL = math.radians(10)  # Waypoint yaw tolerance (radians)

KP_LIN = 0.5        # Linear velocity proportional gain
KP_ANG = 1.1        # Angular velocity proportional gain

MAX_LINEAR_SPEED = 0.6      # Maximum allowed linear speed
MAX_ANGULAR_SPEED = 1.2     # Maximum allowed angular speed

OBSTACLE_DIST_STOP = 0.4    # Distance threshold to stop near obstacle
OBSTACLE_DIST_SLOW = 0.6    # Distance threshold to slow near obstacle

FRONT_ANGLE_DEG = 35        # Angle for defining the front sector (degrees)
SIDE_ANGLE_DEG = 45         # Angle for defining left/right sectors (degrees)

LOG_FILE = "ebot_nav_log.csv"   # CSV log file name


class EbotNavigator(Node):
    '''
    Purpose:
    ---
    Handles navigation of eBot through a sequence of waypoints while avoiding obstacles
    using ROS 2 publishers and subscribers.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    node = EbotNavigator()
    '''
    def __init__(self):
        super().__init__('ebot_nav_node')
        self.pose = [0.0, 0.0, 0.0]     # [x, y, yaw] of eBot
        self.scan_ranges = []           # List of distance readings from LIDAR
        self.current_wp = 0             # Index of current waypoint
        self.reached_all = False        # Flag to indicate if all waypoints reached

        # ROS publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Timer for control loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize CSV log
        self.init_log()

        self.get_logger().info(f"✅ eBot Navigator Started (KP_LIN={KP_LIN}, KP_ANG={KP_ANG}, FRONT={FRONT_ANGLE_DEG}°, SIDE={SIDE_ANGLE_DEG}°)")

    def init_log(self):
        '''
        Purpose:
        ---
        Initializes the CSV log file to record navigation data such as position, yaw, 
        and command velocities.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.init_log()
        '''
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'yaw', 'linear_x', 'angular_z', 'min_front_range'])

    def log_state(self, linear_x, angular_z, min_front):
        '''
        Purpose:
        ---
        Logs the robot’s current state (pose, velocities, front obstacle distance)
        into the CSV file for post-run analysis.

        Input Arguments:
        ---
        `linear_x` : [float]
            Linear velocity command sent to robot.
        `angular_z` : [float]
            Angular velocity command sent to robot.
        `min_front` : [float]
            Minimum detected distance in front sector.

        Returns:
        ---
        None

        Example call:
        ---
        self.log_state(0.5, 0.1, 0.7)
        '''
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_clock().now().nanoseconds / 1e9,
                             self.pose[0], self.pose[1], self.pose[2],
                             linear_x, angular_z, min_front])

    def odom_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the robot’s pose (x, y, yaw) using odometry data.

        Input Arguments:
        ---
        `msg` : [nav_msgs.msg.Odometry]
            Odometry message from /odom topic.

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically by ROS subscription callback.
        '''
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)
        self.pose = [x, y, yaw]

    def scan_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the LIDAR scan data for obstacle detection.

        Input Arguments:
        ---
        `msg` : [sensor_msgs.msg.LaserScan]
            Laser scan message containing range data.

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically by ROS subscription callback.
        '''
        self.scan_ranges = msg.ranges

    def control_loop(self):
        '''
        Purpose:
        ---
        Main control logic for eBot navigation. Computes linear and angular velocities 
        to move the robot toward waypoints while avoiding obstacles using LIDAR data.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically every 0.1 sec by ROS timer.
        '''
        if self.current_wp >= len(WAYPOINTS):
            # Stop the robot when all waypoints are reached
            self.cmd_pub.publish(Twist())
            if not self.reached_all:
                self.get_logger().info("🎯 All waypoints reached!")
                self.reached_all = True
            return

        # Extract current goal
        goal_x, goal_y, goal_yaw = WAYPOINTS[self.current_wp]
        x, y, yaw = self.pose

        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)
        goal_heading = math.atan2(dy, dx)
        heading_error = self.angle_diff(goal_heading, yaw)
        yaw_error = self.angle_diff(goal_yaw, yaw)

        twist = Twist()

        # -------- OBSTACLE AVOIDANCE -------- #
        front, left, right = self.get_sector_distances()

        # Adjust angular correction based on obstacle distribution
        steer_correction = 0.0
        if front < OBSTACLE_DIST_SLOW:
            steer_correction = 0.5 * (left - right)

        # Base linear and angular speeds
        lin_speed = KP_LIN * dist
        if front < OBSTACLE_DIST_SLOW:
            lin_speed *= max(front / OBSTACLE_DIST_SLOW, 0.3)

        twist.linear.x = min(max(lin_speed, 0.15), MAX_LINEAR_SPEED)
        twist.angular.z = max(min(KP_ANG * heading_error + steer_correction,
                                  MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

        # -------- SAFETY HANDLING -------- #
        if front < OBSTACLE_DIST_STOP:
            twist.linear.x = 0.1
            twist.angular.z = 0.5 if left > right else -0.5
            self.get_logger().warn(f"🚧 Obstacle too close! Front={front:.2f} m")

        if abs(heading_error) > math.radians(25):
            twist.linear.x = max(twist.linear.x, 0.2)

        # -------- WAYPOINT CHECK -------- #
        if dist < POS_TOL + 0.05:
            if dist > POS_TOL:
                twist.linear.x = max(0.1, KP_LIN * dist)
            else:
                twist.linear.x = 0.0

            if abs(yaw_error) > YAW_TOL:
                twist.angular.z = max(min(0.8 * yaw_error, MAX_ANGULAR_SPEED),
                                      -MAX_ANGULAR_SPEED)
                self.get_logger().info(f"🌀 Aligning to target yaw at WP{self.current_wp + 1}...")
            else:
                if dist <= POS_TOL:
                    self.get_logger().info(f"✅ Reached waypoint {self.current_wp + 1}")
                    self.current_wp += 1
                    twist = Twist()
                    time.sleep(0.4)

        # Publish and log robot state
        self.cmd_pub.publish(twist)
        self.log_state(twist.linear.x, twist.angular.z, front)

    def get_sector_distances(self):
        '''
        Purpose:
        ---
        Calculates minimum obstacle distances in the front, left, and right sectors 
        using LIDAR range data.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `front` : [float]
            Minimum distance in front sector.
        `left` : [float]
            Minimum distance in left sector.
        `right` : [float]
            Minimum distance in right sector.

        Example call:
        ---
        front, left, right = self.get_sector_distances()
        '''
        if not self.scan_ranges:
            return float('inf'), float('inf'), float('inf')

        n = len(self.scan_ranges)
        idx_center = n // 2
        half_front = int(math.radians(FRONT_ANGLE_DEG) / (math.pi / n))
        half_side = int(math.radians(SIDE_ANGLE_DEG) / (math.pi / n))

        front_indices = range(idx_center - half_front, idx_center + half_front + 1)
        left_indices = range(idx_center + half_front + 1, min(n, idx_center + half_side + 1))
        right_indices = range(max(0, idx_center - half_side), idx_center - half_front)

        # Function to safely get minimum valid LIDAR values
        def safe_min(indices):
            vals = [self.scan_ranges[i] for i in indices if 0 < self.scan_ranges[i] < 10.0]
            return min(vals) if vals else 10.0

        return safe_min(front_indices), safe_min(left_indices), safe_min(right_indices)

    def quaternion_to_yaw(self, q):
        '''
        Purpose:
        ---
        Converts a quaternion orientation into yaw (rotation about Z-axis).

        Input Arguments:
        ---
        `q` : [geometry_msgs.msg.Quaternion]
            Quaternion representing robot’s orientation.

        Returns:
        ---
        `yaw` : [float]
            Robot’s yaw angle in radians.

        Example call:
        ---
        yaw = self.quaternion_to_yaw(q)
        '''
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def angle_diff(self, a, b):
        '''
        Purpose:
        ---
        Computes the minimal angular difference between two angles, ensuring the 
        result stays within [-π, π].

        Input Arguments:
        ---
        `a` : [float]
            First angle (radians).
        `b` : [float]
            Second angle (radians).

        Returns:
        ---
        `diff` : [float]
            Shortest signed angular difference.

        Example call:
        ---
        diff = self.angle_diff(goal_heading, yaw)
        '''
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff


def main(args=None):
    '''
    Purpose:
    ---
    Initializes ROS2 node, starts EbotNavigator instance, and runs until shutdown.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Called automatically when script is executed.
    '''
    rclpy.init(args=args)
    node = EbotNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# Function Name:    main (built-in)
#        Inputs:    None
#       Outputs:    None
#       Purpose:    Entry point of the program.
if __name__ == "__main__":
    main()
