#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import csv
import os
import time


class JointStateRecorder(Node):
    def __init__(self, logging_rate_hz=10):
        # Initialize the ROS 2 node
        super().__init__('joint_state_recorder')

        # Output file for end-effector pose
        self.output_file = os.path.join(os.path.dirname(__file__), 'end_effector_positions.csv')

        # Open CSV file for writing
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # # Write CSV header
        # self.csv_writer.writerow(['elapsed_time', 'ee_x', 'ee_y', 'ee_z'])

        # Record the start time
        self.start_time = None

        # Store latest received end-effector pose
        self.latest_ee_pose = None

        # Subscribe to the end-effector pose topic
        self.subscription_ee_pose = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.ee_pose_callback, 10
        )

        # Timer for logging data required rate
        self.logging_timer = self.create_timer(1.0 / logging_rate_hz, self.log_data)

        self.get_logger().info("Recording end-effector pose to: %s" % self.output_file)

    def ee_pose_callback(self, msg):
        """ Store latest end-effector position (without writing to CSV immediately). """
        self.latest_ee_pose = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def log_data(self):
        """ Timer-based logging at required rate. """
        if self.latest_ee_pose is None:
            return  # Wait until we have data

        # Get current time and compute elapsed time
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        elapsed_time = current_time - self.start_time

        # Prepare row data
        row = [round(elapsed_time, 6)] + list(self.latest_ee_pose)

        # Write to CSV
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written to file

    def shutdown_hook(self):
        """ Close the CSV file on shutdown. """
        self.csv_file.close()
        self.get_logger().info("Recording stopped. Data saved to: %s" % self.output_file)


def main(args=None):
    rclpy.init(args=args)

    # Initialize the recorder node
    recorder = JointStateRecorder()

    # Register shutdown hook to close the file properly
    rclpy.get_default_context().on_shutdown(recorder.shutdown_hook)

    # Spin the node
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up before exiting
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
