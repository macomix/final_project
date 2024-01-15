cmake_minimum_required(VERSION 3.5)
project(final_project)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# make sure you dont use the same msg name in a diffrent project already
# if you already made the mistake delete install, build, log folders
# and build the project again (build_ros and source is enough)
# thats is part of the reason you want to put msg in an underlay
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PIDStamped2.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

install(PROGRAMS
  nodes/mapper.py
  nodes/path_follower.py
  nodes/path_planner.py
  nodes/position_controller.py
  nodes/yaw_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
