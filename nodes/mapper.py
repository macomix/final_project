#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Polygon, Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from scenario_msgs.msg import PolygonsStamped

OCCUPIED = 100
DEPTH = -0.5


class MapperNode(Node):

    def __init__(self):
        super().__init__(node_name='mapper')

        # makes the obstacles bigger to account for robot size
        self.obstacle_distance: float = 0.8

        self.tank_size_x: float = 0
        self.tank_size_y: float = 0
        self.cell_resolution: float = 0
        self.init_params()

        self.num_cells_x = int(np.round(self.tank_size_x /
                                        self.cell_resolution))
        self.num_cells_y = int(np.round(self.tank_size_y /
                                        self.cell_resolution))

        self.map_meta_data = self.get_map_meta_data()

        # initialize map cell array
        # we will keep track of two maps:
        # 1. map: includes inflated obstacles to account for robot size
        # 2. map_uninflated: keeps track of original size of obstacles

        # x: columns, y: rows --> switch order here
        empty_map = np.zeros((self.num_cells_y, self.num_cells_x), dtype='int8')
        self.map = self.get_tank_walls(empty_map)
        self.map_uninflated = np.copy(self.map)

        self.get_logger().info(
            f'Using a cell resolution of {self.cell_resolution} m per cell, ' +
            f'resulting in {self.num_cells_x} cells in x-direction and ' +
            f'{self.num_cells_y} cells in y-direction.')

        # publisher
        self.map_pub = self.create_publisher(msg_type=OccupancyGrid,
                                             topic='occupancy_grid',
                                             qos_profile=1)

        self.map_pub_debug = self.create_publisher(
            msg_type=OccupancyGrid,
            topic='/occupancy_grid_uninflated',
            qos_profile=1)

        # subscriber
        self.obstacle_sub = self.create_subscription(msg_type=PolygonsStamped,
                                                     topic='obstacles',
                                                     callback=self.on_obstacles,
                                                     qos_profile=1)

    def init_params(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tank_size_x', rclpy.Parameter.Type.DOUBLE),
                ('tank_size_y', rclpy.Parameter.Type.DOUBLE),
                ('cell_resolution', rclpy.Parameter.Type.DOUBLE),
            ])
        
        param = self.get_parameter('tank_size_x')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tank_size_x = param.get_parameter_value().double_value

        param = self.get_parameter('tank_size_y')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tank_size_y = param.get_parameter_value().double_value

        param = self.get_parameter('cell_resolution')
        self.get_logger().info(f'{param.name}={param.value}')
        self.cell_resolution = param.get_parameter_value().double_value

    def on_obstacles(self, msg: PolygonsStamped):
        num_obstacles = len(msg.polygons)
        if not num_obstacles:
            return

        for obstacle in msg.polygons:
            self.map_uninflated = self.include_obstacle_in_map(
                obstacle, np.copy(self.map_uninflated))

        self.map = self.inflate_obstacles(np.copy(self.map_uninflated))

        self.publish_map(np.copy(self.map), self.map_pub)
        # publish uninflated map for debugging
        self.publish_map(np.copy(self.map_uninflated), self.map_pub_debug)

    def include_obstacle_in_map(self, obstacle: Polygon,
                                cells: np.ndarray) -> np.ndarray:
        num_points = len(obstacle.points)
        obstacle_cells = np.zeros((num_points, 2), dtype='uint32')

        for index, point in enumerate(obstacle.points):
            obstacle_cells[index,
                           0] = int(round(point.x / self.cell_resolution))
            obstacle_cells[index,
                           1] = int(round(point.y / self.cell_resolution))

        img = cells.astype(dtype='uint8')
        img_obstacles = cv2.fillPoly(img,
                                     pts=np.int32([obstacle_cells]),
                                     color=(OCCUPIED))

        return img_obstacles.astype(dtype='int8')

    def inflate_obstacles(self, cells: np.ndarray) -> np.ndarray:
        # robot size is most likely larger than one grid cell
        # therefore, we need to inflate obstacles and walls

        # opencv wants different data type than occupancy grid map msg
        img = cells.astype(dtype='uint8')

        # We're inflating the obstacles by ca. 30cm in this example  # TODO
        dilatation_size = int(np.round(self.obstacle_distance / self.cell_resolution))

        # We will use a circular kernel, feel free to try other kernel shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dilatation_size, dilatation_size))
        img_dilated = cv2.dilate(img, kernel, iterations=1)

        # return convert back data type
        return img_dilated.astype(dtype='int8')

    def get_tank_walls(self, cells: np.ndarray) -> np.ndarray:
        # we will set all borders of the grid as occupied
        cells[:, 0] = OCCUPIED
        cells[:, -1] = OCCUPIED
        cells[0, :] = OCCUPIED
        cells[-1, :] = OCCUPIED
        return cells

    def publish_map(self, data: np.ndarray, pub):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = self.map_meta_data
        msg.data = np.copy(data).flatten()
        pub.publish(msg)

    def get_map_meta_data(self) -> MapMetaData:
        meta_data = MapMetaData()
        meta_data.map_load_time = self.get_clock().now().to_msg()
        meta_data.resolution = self.cell_resolution
        meta_data.width = self.num_cells_x
        meta_data.height = self.num_cells_y
        # origin is at (0,0, our depth), same orientation as map frame
        meta_data.origin = Pose()
        meta_data.origin.position.z = DEPTH
        return meta_data


def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
