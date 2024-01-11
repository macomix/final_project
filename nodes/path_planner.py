#!/usr/bin/env python3

from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scenario_msgs.msg import Viewpoints
from scenario_msgs.srv import MoveToStart, SetPath
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()

def high_speed_normalize(vectors: np.ndarray):
    # normalizing huge amount of vectors in a short time
    norms = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    unit_vectors = vectors.transpose()/norms
    return unit_vectors.transpose()

def quaternion_from_vectors(to_vector: np.ndarray, from_vector=np.array([1,0,0])) -> Quaternion:
    """Outputs the quaternion which describes the rotation 
    from one vector to another vector.

    Args:
        to_vector (np.ndarray): the end 3d vector
        from_vector (_type_, optional): the reference 3d vector. Defaults to np.array([1,0,0]).

    Returns:
        Quaternion: orientation
    """
    # normalize input vectors
    from_vector, to_vector = high_speed_normalize(np.array([from_vector, to_vector]))

    # Calculate the rotation axis
    rotation_axis = np.cross(from_vector, to_vector)
    if np.array_equal(rotation_axis, np.zeros(3)) == False:
        rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot(from_vector, to_vector))

    # Convert the axis-angle representation to quaternion
    quaternion = axis_angle_to_quaternion(rotation_axis, rotation_angle)

    return quaternion

def axis_angle_to_quaternion(axis, angle):
    # Convert the axis-angle representation to quaternion
    sin_half_angle = np.sin(angle / 2.0)
    cos_half_angle = np.cos(angle / 2.0)

    quaternion = Quaternion()
    quaternion.x = axis[0] * sin_half_angle
    quaternion.y = axis[1] * sin_half_angle
    quaternion.z = axis[2] * sin_half_angle
    quaternion.w = cos_half_angle

    return quaternion

def occupancy_grid_to_matrix(grid: OccupancyGrid) -> np.ndarray:
    data = np.array(grid.data, dtype=np.uint8)
    data = data.reshape(grid.info.height, grid.info.width)
    return data


def world_to_matrix(x: float, y: float, grid_size: float) -> tuple[int, int]:
    return round(x / grid_size), round(y / grid_size)


def matrix_index_to_world(x: int, y: int, grid_size: float):
    return [x * grid_size, y * grid_size]


def multiple_matrix_indeces_to_world(points: list[tuple[int, int]], grid_size: float):
    world_points = []
    for point in points:
        world_points.append([point[0] * grid_size, point[1] * grid_size])
    return world_points


def compute_discrete_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x = x0
    y = y0
    points = []
    while True:
        points.append((int(x), int(y)))
        if x == x1 and y == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            if x == x1:
                break
            error += dy
            x += sx
        if doubled_error <= dx:
            if y == y1:
                break
            error += dx
            y += +sy
    return points

class PathPlanner(Node):

    def __init__(self):
        super().__init__(node_name='path_planner')
        self.state = State.UNSET
        self.cell_size = 0.2
        self.recomputation_required = True
        self.target_viewpoint_index = -1
        self.path_marker: Marker
        self.init_path_marker()
        self.path_marker_pub = self.create_publisher(Marker, '~/marker', 1)
        self.viewpoints = []
        self.waypoints = []
        self.orientations = []
        self.occupancy_grid: OccupancyGrid = None # type: ignore
        self.occupancy_matrix: np.ndarray = np.empty(1)
        self.progress = -1.0
        self.remaining_segments = []
        self.init_clients()
        self.init_services()
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.viewpoints_sub = self.create_subscription(Viewpoints, 'viewpoints',
                                                       self.on_viewpoints, 1)

    def init_services(self):
        self.move_to_start_service = self.create_service(
            MoveToStart, '~/move_to_start', self.serve_move_to_start)
        self.start_service = self.create_service(Trigger, '~/start',
                                                 self.serve_start)
        self.stop_service = self.create_service(Trigger, '~/stop',
                                                self.serve_stop)

    def init_clients(self):
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup() # type: ignore
        self.set_path_client = self.create_client(SetPath,
                                                  'path_follower/set_path',
                                                  callback_group=cb_group)
        self.path_finished_client = self.create_client(
            Trigger, 'path_follower/path_finished', callback_group=cb_group)

    def serve_move_to_start(self, request, response):
        self.state = State.MOVE_TO_START
        self.start_pose = request.target_pose
        self.current_pose = request.current_pose
        # we do not care for collisions while going to the start position
        # in the simulation 'collisions' do not matter. In the lab, we
        # can manually make sure that we avoid collisions, by bringing the
        # vehicle in a safe position manually before starting anything.
        response.success = self.move_to_start(request.current_pose,
                                              request.target_pose)
        return response

    def move_to_start(self, p0: Pose, p1: Pose) -> bool:
        path_segment = self.compute_simple_path_segment(p0,
                                                        p1,
                                                        check_collision=False)
        request = SetPath.Request()
        request.path = path_segment['path']
        answer = self.set_path_client.call(request)
        if answer.success:
            self.get_logger().info('Moving to start position')
            return True
        else:
            self.get_logger().info(
                'Asked to move to start position. '
                'But the path follower did not accept the new path.')
            return False

    def has_collisions(self, points_2d: list[tuple[int, int]]) -> list[int]:
        """Checks if any point2d is inside a object and therefore 
        colliding. In that case the index of that point in the list will be returned.

        Args:
            points_2d (list[tuple[int, int]]): list of 2d points

        Returns:
            list(int): indices of points with collision
        """
        if not self.occupancy_grid:
            return []
        collision_indices = [
            i for i, p in enumerate(points_2d)
            if self.occupancy_matrix[p[1], p[0]] >= 50
        ]
        return collision_indices

    def find_grid_neighbors(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []

        # height: y, width: x
        height: int = self.occupancy_grid.info.height
        width: int = self.occupancy_grid.info.width
        
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            x2 = position[0] + dx
            y2 = position[1] + dy
            # check if neighbor is outside the map
            if x2 < 0 or x2 > width-1 or y2 < 0 or y2 > height-1:
                continue
            neighbors.append((x2, y2))
            
        return neighbors
    
    def move_cost(self, posA: tuple[int, int], posB: tuple[int, int]) -> float:
        if self.has_collisions([posA, posB]):
            # if move leads to collision return a high move cost
            return 1000
        else:
            dist = np.asarray(posA) - np.asarray(posB)
            return np.linalg.norm(dist).astype(float)

    def compute_a_star_segment(self, p0: Pose, p1: Pose):
        # calculate the position on the 2d grid map
        startPos = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        endPos = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)

        # TODO: maybe make sure startPos != endPos
        
        matrixPath, cost = self.compute_a_star(startPos, endPos)
        #self.get_logger().info(f'Calculated an A*-segment with a cost of {cost}.')

        worldPath = multiple_matrix_indeces_to_world(matrixPath, self.cell_size)

        # TODO: maybe add last point again separate rotation

        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(worldPath) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(worldPath)
        ]
        # replace the last point with the exact value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position

        # add rotations
        orientations = []
        for i in range(len(worldPath)-1):
            direction = np.array([worldPath[i+1][0]-worldPath[i][0], 
                                  worldPath[i+1][1]-worldPath[i][1],
                                  0])
            q = quaternion_from_vectors(direction)
            orientations.append(q)
            
        orientations.append(p1.orientation)

        ####
        collision_indices = []
        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}

    
    def compute_a_star(self, startPos: tuple[int, int], endPos: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
        """A*-algorithm:
        - pathfinder 
        - graph search
        - horrendous O(b^d) complexity

        Code based on: 
        https://rosettacode.org/wiki/A*_search_algorithm#Python

        Args:
            startPos (tuple[int, int]): start position on matrix map
            endPos (tuple[int, int]): end position on matrix map

        Returns:
            list, float: path and cost
        """
        gScore = {} # cost of each position to the starting position
        fScore = {} # estimated cost of cheapest path from start to end using specific position

        # initialize
        gScore[startPos] = 0
        fScore[startPos] = self.heuristic_cost_2d(startPos, endPos)

        # set of nodes we need to check
        closedNodes = set()
        openNodes = set([startPos])
        cameFrom = {} # the node before the current node on the cheapest path


        while len(openNodes) > 0:
            currentPos = None
            current_fScore = None

            # for every node available check if there is
            # a node with a lower F score
            for pos in openNodes:
                if currentPos is None or fScore[pos] < current_fScore:
                    current_fScore = fScore[pos]
                    currentPos = pos

            if not currentPos: continue # make sure currentPos is set

            # check if the end pose is reached
            if currentPos == endPos:
                # if the end is reached make the path
                path = [currentPos]
                while currentPos in cameFrom:
                    currentPos = cameFrom[currentPos]
                    path.append(currentPos)
                path.reverse()
                return path, fScore[endPos]
                
            # transfer current Node from open set to closed set
            openNodes.remove(currentPos)
            closedNodes.add(currentPos)

            # update scores surrounding the current position
            for neighbor in self.find_grid_neighbors(currentPos):
                if neighbor in closedNodes:
                    # this node was already computed
                    continue

                # this might be the next best node
                candidate_gScore = gScore[currentPos] + self.move_cost(currentPos, neighbor)

                if neighbor not in openNodes:
                    # new node discovered
                    openNodes.add(neighbor)
                elif candidate_gScore >= gScore[neighbor]:
                    # we already found a faster path to the position of neighbor
                    continue

                # add valuable information
                cameFrom[neighbor] = currentPos
                gScore[neighbor] = candidate_gScore
                hCost = self.heuristic_cost_2d(neighbor, endPos)
                fScore[neighbor] = gScore[neighbor] + hCost

        raise RuntimeError("A* failed to find a solution")

    def heuristic_cost_2d(self, start: tuple[int, int], goal: tuple[int, int]) -> float:
        """This is a heuristic function which is supposed to calculate the cost
        to reach a goal based on limited knowledge. 
        There are different ways to calculate our heuristic:
            - Manhatten distance
            - Straight-line (current method of choice)
            - Octile distance
            - Chebyshev distance
            - ...

        Args:
            start (int): start position on 2d grid
            goal (int): end position on 2d grid

        Returns:
            float: estimated cost
        """
        dist = np.asarray(start) - np.asarray(goal)
        return np.linalg.norm(dist).astype(float)

    def compute_simple_path_segment(self, p0: Pose, p1: Pose, check_collision=True):
        p0_2d = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        p1_2d = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)
        # now we should/could apply some sophisticated algorithm to compute
        # the path that brings us from p0_2d to p1_2d. For this dummy example
        # we simply go in a straight line. Not very clever, but a straight
        # line is the shortes path between two points, isn't it?
        line_points_2d = compute_discrete_line(p0_2d[0], p0_2d[1], p1_2d[0],
                                               p1_2d[1])
        if check_collision:
            collision_indices = self.has_collisions(line_points_2d)
        else:
            collision_indices = []

        # Convert back our matrix/grid_map points to world coordinates. Since
        # the grid_map does not contain information about the z-coordinate,
        # the following list of points only contains the x and y component.
        xy_3d = multiple_matrix_indeces_to_world(line_points_2d, self.cell_size)

        # it might be, that only a single grid point brings us from p0 to p1.
        # in this duplicate this point. this way it is easier to handle.
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])
        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        q = quaternion_from_euler(0.0, 0.0, yaw0)
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}

    def reset_internals(self):
        self.target_viewpoint_index = -1
        self.recomputation_required = True
        self.state = State.UNSET

    def serve_start(self, request, response):
        if self.state != State.NORMAL_OPERATION:
            self.get_logger().info('Starting normal operation.')
            self.reset_internals()
        self.state = State.NORMAL_OPERATION
        response.success = True
        return response

    def serve_stop(self, request, response):
        if self.state != State.IDLE:
            self.get_logger().info('Asked to stop. Going to idle mode.')
        response.success = self.do_stop()
        return response

    def do_stop(self):
        self.state = State.IDLE
        if self.path_finished_client.call(Trigger.Request()).success:
            self.reset_internals()
            self.state = State.IDLE
            return True
        return False

    def handle_mission_completed(self):
        self.get_logger().info('Mission completed.')
        if not self.do_stop():
            self.get_logger().error(
                'All waypoints completed, but could not '
                'stop the path_follower. Trying again...',
                throttle_duration_sec=1.0)
            return
        self.state = State.IDLE

    def compute_new_path(self, viewpoints: Viewpoints):
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # start position and is treated differently.
        if i < 1:
            return
        # complete them all.
        # We do nothing smart here. We keep the order in which we received
        # the waypoints and connect them by straight lines.
        # No collision avoidance.
        # We only perform a collision detection and give up in case that
        # our path of straight lines collides with anything.
        # Not very clever, eh?

        # now get the remaining uncompleted viewpoints. In general, we can
        # assume that all the following viewpoints are uncompleted, since
        # we complete them in the same order as we get them in the
        # viewpoints message. But it is not that hard to double-check it.
        viewpoint_poses = [
            v.pose for v in viewpoints.viewpoints[i:] if not v.completed
        ]
        # get the most recently visited viewpoint. Since we do not change
        # the order of the viewpoints, this will be the viewpoint right
        # before the first uncompleted viewpoint in the list, i.e. i-1
        p0 = viewpoints.viewpoints[i - 1].pose
        viewpoint_poses.insert(0, p0)

        # now we can finally call our super smart function to compute
        # the path piecewise between the viewpoints
        path_segments = []
        for i in range(1, len(viewpoint_poses)):
            # segment = self.compute_simple_path_segment(viewpoint_poses[i - 1],
            #                                            viewpoint_poses[i])
            # alternatively call your own implementation
            segment = self.compute_a_star_segment(viewpoint_poses[i - 1],
                                                  viewpoint_poses[i])
            path_segments.append(segment)
        return path_segments

    def handle_no_collision_free_path(self):
        self.get_logger().fatal('We have a collision in our current segment!'
                                'Giving up...')
        if self.do_stop():
            self.state = State.IDLE
        else:
            self.state = State.UNSET

    def do_normal_operation(self, viewpoints: Viewpoints):
        # what we need to do:
        # - check if the viewpoints changed, if so, recalculate the path
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # we completed our mission!
        if i < 0:
            self.handle_mission_completed()
            return

        if (not self.recomputation_required
            ) or self.target_viewpoint_index == i:
            # we are still chasing the same viewpoint. Nothing to do.
            return
        self.get_logger().info('Computing new path segments')
        self.target_viewpoint_index = i
        if i == 0:
            p = viewpoints.viewpoints[0].pose
            if not self.move_to_start(p, p):
                self.get_logger().fatal(
                    'Could not move to first viewpoint. Giving up...')
                if self.do_stop():
                    self.state = State.IDLE
                else:
                    self.state = State.UNSET
            return

        path_segments = self.compute_new_path(viewpoints)
        if not path_segments:
            self.get_logger().error(
                'This is a logic error. The cases that would have lead to '
                'no valid path_segments should have been handled before')
            return
        if path_segments[0]['collision_indices']:
            self.handle_no_collision_free_path()
            return
        self.set_new_path(path_segments[0]['path'])
        return

    def on_viewpoints(self, msg: Viewpoints):
        if self.state == State.IDLE:
            return
        if self.state == State.UNSET:
            if self.do_stop():
                self.state = State.IDLE
            else:
                self.get_logger().error('Failed to stop.')
            return
        if self.state == State.MOVE_TO_START:
            # nothing to be done here. We already did the setup when the
            # corresponding service was called
            return
        if self.state == State.NORMAL_OPERATION:
            self.do_normal_operation(msg)

    def find_first_uncompleted_viewpoint(self, viewpoints: Viewpoints):
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if not viewpoint.completed:
                return i
        # This should not happen!
        return -1

    def on_occupancy_grid(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        if msg.info.resolution != self.cell_size:
            self.get_logger().info('Cell size changed. Recomputation required.')
            self.recomputation_required = True
            self.cell_size = msg.info.resolution

    def init_path_marker(self):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = 'path'
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.header.frame_id = 'map'
        msg.color.a = 1.0
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.02
        self.path_marker = msg

    def set_new_path(self, path):
        request = SetPath.Request()
        if not path:
            return False
        request.path = path
        self.set_new_path_future = self.set_path_client.call_async(request)
        return True

    # NOTE wtf is going on here:
    def publish_path_marker(self, segments):
        msg = self.path_marker
        world_points = self.segments_to_world_points(segments)
        msg.points = [Point(x=p[0], y=p[1], z=-0.5) for p in world_points]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_marker_pub.publish(msg)


def main():
    rclpy.init()
    node = PathPlanner()
    exec = MultiThreadedExecutor()
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
