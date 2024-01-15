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

class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        # gains
        self.gains_x = np.zeros(3)
        self.gains_y = np.zeros(3)
        self.gains_z = np.zeros(3)
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        self.get_logger().info(f'{self.gains_x}')

        #
        self.quaternion = np.zeros(4) # w, x, y, z

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
                                    ("gains_z", rclpy.Parameter.Type.DOUBLE_ARRAY)
                                             ])
        
        param = self.get_parameter('gains_x')
        self.gains_x = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_y')
        self.gains_y = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_z')
        self.gains_z = param.get_parameter_value().double_array_value

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
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, q)

    def apply_control(self, position: Point, quaternion: Quaternion):
        now = self.get_clock().now()
        x_error = self.setpoint.x - position.x
        y_error = self.setpoint.y - position.y
        z_error = self.setpoint.z - position.z

        x = 1.0 * x_error
        y = 1.0 * y_error
        z = 1.0 * z_error
        thrust = np.array([x,y,z])

        # convert to local space of the robot
        q=np.array([quaternion.w,quaternion.x,quaternion.y,quaternion.z])
        thrust = vector3_rotate(vector3=thrust, quaternion=q)
        thrust = thrust.reshape(-1)

        self.publish_thrust(thrust=thrust, timestamp=now)

    def publish_thrust(self, thrust: np.ndarray, 
                       timestamp: rclpy.time.Time) -> None: # type: ignore
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.x = np.float64(thrust[0])
        msg.y = np.float64(thrust[1])
        msg.z = np.float64(thrust[2])

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
