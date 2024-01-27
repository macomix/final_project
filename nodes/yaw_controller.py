#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion

def low_pass_filter(x: float, y_old: float, cutoff: float, 
                    dt: rclpy.time.Time) -> float: # type: ignore
    alpha = dt/(dt + 1/(2*np.pi*cutoff))
    y = x * alpha + (1- alpha)*y_old
    return y

def clamp(number: float, smallest: float, largest: float) -> float:
    return max(smallest, min(number, largest))

class YawController(Node):

    def __init__(self):
        super().__init__(node_name='yaw_controller')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        # default value for the yaw setpoint
        self.setpoint = math.pi / 2.0
        self.setpoint_timed_out = True

        # init parameters from config file
        self.gains_yaw = np.zeros(3)
        self.cutoff = 20.0 # cutoff frequency for low pass filter
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        # maybe put in config
        self.output_saturation: float = 0.2

        #
        self.last_filter_estimate= 0 # low pass
        self.last_error = 0
        self.error_integral = 0
        self.last_time = self.get_clock().now().nanoseconds * 10e-9

        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        self.setpoint_sub = self.create_subscription(Float64Stamped,
                                                     topic='~/setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=qos)
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)

        # publisher
        self.torque_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='torque_setpoint',
                                                qos_profile=1)
        
        self.yaw_pub = self.create_publisher(msg_type=Float64Stamped,
                                             topic='current_yaw',
                                             qos_profile=1)

    def init_params(self):
        # load params from config
        self.declare_parameters(namespace='',
                                parameters=[
                                    ("gains_yaw", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("cutoff", rclpy.Parameter.Type.DOUBLE)
                                             ])
        
        param = self.get_parameter('gains_yaw')
        self.gains_yaw = param.get_parameter_value().double_array_value

        param = self.get_parameter('cutoff')
        self.cutoff = param.get_parameter_value().double_value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'gains_yaw':
                self.gains_yaw = param.get_parameter_value().double_array_value
            elif param.name == 'cutoff':
                self.cutoff = param.get_parameter_value().double_value
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')
    
    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('Setpoint timed out. Waiting for new setpoints')
        self.setpoint_timed_out = True

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def on_setpoint(self, msg: Float64Stamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = self.wrap_pi(msg.data)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert the quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw = self.wrap_pi(yaw)

        # publish yaw for debug reasons
        msg_yaw = Float64Stamped()
        msg_yaw.header = msg.header
        msg_yaw.data = yaw
        self.yaw_pub.publish(msg_yaw)

        control_output = self.compute_control_output(yaw)
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp) # type: ignore
        self.publish_control_output(control_output, timestamp)

    def compute_control_output(self, yaw):
        now = self.get_clock().now()
        dt = now.nanoseconds * 10e-9 - self.last_time

        p_gain, i_gain, d_gain = self.gains_yaw

        # sort out all abnormal dt
        if dt > 1:
            dt = 0.0

        # very important: normalize the angle error!
        #self.get_logger().info(f"from: {yaw} to: {self.setpoint}")
        error = self.wrap_pi(self.setpoint - yaw)
        derivative_error = 0
        
        # derivative
        if dt != 0:
            error_change = (error - self.last_error)/dt
            derivative_error = low_pass_filter(self.last_filter_estimate, error_change, cutoff=self.cutoff, dt=dt)
            self.last_filter_estimate = derivative_error

        # integral
        self.error_integral = self.error_integral + dt * error

        # conditional integrator clamping
        # if error and integrator have a different sign
        # turn integrator off
        if (error * self.error_integral) < 0:
            self.error_integral = 0

        # saturation
        thrust = p_gain * error + i_gain * self.error_integral + d_gain * derivative_error
        thrust = clamp(thrust, -self.output_saturation, self.output_saturation)

        # update
        self.last_error = error
        self.last_time = now.nanoseconds * 10e-9
        
        return thrust

    def publish_control_output(self, control_output: float,
                               timestamp: rclpy.time.Time): # type: ignore
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.ignore_x = True
        msg.ignore_y = True
        msg.ignore_z = False  # yaw is the rotation around the vehicle's z axis

        msg.z = control_output
        self.torque_pub.publish(msg)


def main():
    rclpy.init()
    node = YawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
