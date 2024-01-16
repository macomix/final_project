#!/usr/bin/env python3

import math
import numpy as np
from scipy import signal

import rclpy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped, Vector3, Quaternion
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

from final_project.msg import PIDStamped2

def numpy_to_vector3(array: np.ndarray) -> Vector3:
    # function for safe and convenient type conversion
    arr = array.reshape((1,-1))
    if arr.shape != (1, 3):
        raise ValueError("Size of numpy error does not match a Vector3.")

    rosVector = Vector3()
    rosVector.x = np.float64(array[0])
    rosVector.y = np.float64(array[1])
    rosVector.z = np.float64(array[2])
    
    return rosVector

def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    # Hamilton multiplication
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def vector3_rotate(vector3: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    # directly transforms a vector3 using quaternions
    # (much better than euler anglers and faster than rotation matrices)
    q_vec = np.append(0.0, vector3) # 0, x, y, z
    q_inverse = np.concatenate(([quaternion[0]], -quaternion[1:])) # w, -x, -y, -z
    return quaternion_multiply(quaternion, quaternion_multiply(q_vec, q_inverse))[1:]

def low_pass_filter(x: float, y_old: float, cutoff: float, 
                    dt: rclpy.time.Time) -> float: # type: ignore
    alpha = dt/(dt + 1/(2*np.pi*cutoff))
    y = x * alpha + (1- alpha)*y_old
    return y

class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        # gains
        self.gains_x = np.zeros(3)
        self.gains_y = np.zeros(3)
        self.gains_z = np.zeros(3)
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        #
        self.cutoff = 40.0 # cutoff frequency for low pass filter
        self.last_filter_estimate= np.zeros(3) # low pass
        self.last_error = np.zeros(3)
        self.error_integral = np.zeros(3)
        self.last_time = self.get_clock().now().nanoseconds * 10e-9

        # publisher
        self.thrust_pub = self.create_publisher(ActuatorSetpoint,'thrust_setpoint', 1)
        self.position_setpoint_sub = self.create_subscription(PointStamped, 
            '~/setpoint', 
            self.on_position_setpoint, 
            qos_profile=1)
        
        self.setpoint = Point()
        self.setpoint_timed_out = True

        # debug publisher
        self.pid_debug_pub = self.create_publisher(msg_type=PIDStamped2,
                                                topic='pid_gain',
                                                qos_profile=1)

        # subscriber
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        self.pose_counter = 0

    def init_params(self):
        # load params from config
        self.declare_parameters(namespace='',
                                parameters=[
                                    ("gains_x", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("gains_y", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("gains_z", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("cutoff", rclpy.Parameter.Type.DOUBLE)
                                             ])
        
        param = self.get_parameter('gains_x')
        self.gains_x = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_y')
        self.gains_y = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_z')
        self.gains_z = param.get_parameter_value().double_array_value

        param = self.get_parameter('cutoff')
        self.cutoff = param.get_parameter_value().double_value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'gains_x':
                self.gains_x = param.get_parameter_value().double_array_value
            elif param.name == 'gains_y':
                self.gains_y = param.get_parameter_value().double_array_value
            elif param.name == 'gains_z':
                self.gains_z = param.get_parameter_value().double_array_value
            elif param.name == 'cutoff':
                self.cutoff = param.get_parameter_value().double_value
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')
    
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
        #_, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, q)

    def apply_control(self, position: Point, quaternion: Quaternion):
        now = self.get_clock().now()
        gain = np.array([self.gains_x, self.gains_y, self.gains_z])

        dt = self.get_clock().now().nanoseconds * 10e-9 - self.last_time

        # sort out all abnormal dt
        if dt > 1:
            dt = 0.0

        thrust = np.zeros(3)
        error = np.array([self.setpoint.x-position.x, self.setpoint.y-position.y, self.setpoint.z-position.z])
        derivative_error = np.zeros(3)

        # shrink down very big error vectors to limit speed
        # error_size = np.linalg.norm(error).astype(float)
        # max_error_size = 0.3
        # if error_size > max_error_size:
        #     error = max_error_size * error/error_size
        # error_change = np.zeros(3)

        # sort out abnormal dt
        if dt != 0:
            error_change = (error - self.last_error)/dt
        
        for i in range(3):
            if dt != 0:
                derivative_error[i] = low_pass_filter(self.last_filter_estimate[i], error_change[i], cutoff=self.cutoff, dt=dt)

            # integral
            if np.abs(error[i]) < 0.05:
                self.error_integral[i] = self.error_integral[i] + dt * error[i]
            else:
                self.error_integral[i] = 0
        
        # final PID calculation!
        thrust = error * gain[:, 0] + self.error_integral * gain[:, 1] + derivative_error * gain[:, 2]

        # convert to local space of the robot
        q=np.array([quaternion.w,quaternion.x,quaternion.y,quaternion.z])
        thrust = vector3_rotate(vector3=thrust, quaternion=q)
        thrust = thrust.reshape(-1)

        self.last_error = error
        self.last_time = now.nanoseconds * 10e-9
        self.publish_thrust(thrust=thrust, timestamp=now)

        # publish some information for debugging and documentation
        self.publish_pid_info(gain, error, self.error_integral, derivative_error, timestamp=now)

    def publish_thrust(self, thrust: np.ndarray, 
                       timestamp: rclpy.time.Time) -> None: # type: ignore
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.x = np.float64(thrust[0])
        msg.y = np.float64(thrust[1])
        msg.z = np.float64(thrust[2])

        self.thrust_pub.publish(msg)

    def publish_pid_info(self, gains: np.ndarray, error: np.ndarray, 
                         i_error: np.ndarray, d_error:np.ndarray, 
                         timestamp: rclpy.time.Time): # type: ignore
        msg = PIDStamped2()

        msg.gain_p = numpy_to_vector3(np.array([gains[0, 0],gains[1, 0],gains[2, 0]]))
        msg.gain_i = numpy_to_vector3(np.array([gains[0, 1],gains[1, 1],gains[2, 1]]))
        msg.gain_d = numpy_to_vector3(np.array([gains[0, 2],gains[1, 2],gains[2, 2]]))

        msg.error = numpy_to_vector3(error)
        msg.error_integral = numpy_to_vector3(i_error)
        msg.error_derivative = numpy_to_vector3(d_error)

        msg.header.stamp = timestamp.to_msg()
        
        self.pid_debug_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
