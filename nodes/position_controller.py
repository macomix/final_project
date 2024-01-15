#!/usr/bin/env python3

import math
import numpy as np
from scipy import signal

import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from tf_transformations import euler_from_quaternion


class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        # publisher
        self.thrust_pub = self.create_publisher(ActuatorSetpoint,'thrust_setpoint', 1)
        self.position_setpoint_sub = self.create_subscription(PointStamped, 
            '~/setpoint', 
            self.on_position_setpoint, 
            qos_profile=1)
        
        self.setpoint = Point()
        self.setpoint_timed_out = True

        # subscriber
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        self.pose_counter = 0

    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('setpoint timed out. waiting for new setpoints.')
        self.setpoint_timed_out = True

    def on_position_setpoint(self, msg: PointStamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = msg.point

    def on_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, yaw)

    def apply_control(self, position: Point, yaw: float):
        now = self.get_clock().now()
        x_error = self.setpoint.x - position.x
        y_error = self.setpoint.y - position.y
        z_error = self.setpoint.z - position.z

        x = 1.0 * x_error
        y = 1.0 * y_error
        z = 1.0 * z_error

        msg = ActuatorSetpoint()
        msg.header.stamp = now.to_msg()
        msg.x = math.cos(-yaw) * x - math.sin(-yaw) * y
        msg.x = min(0.5, max(-0.5, msg.x))
        msg.y = math.sin(-yaw) * x + math.cos(-yaw) * y
        msg.y = min(0.5, max(-0.5, msg.y))
        msg.z = z

        self.thrust_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
