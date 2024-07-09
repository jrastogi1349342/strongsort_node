cmake_minimum_required(VERSION 3.5)
project (strongsort_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(POLICY CMP0020)
    cmake_policy(SET CMP0020 NEW)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(strongsort_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

SET(Libraries
   cv_bridge
   std_msgs
   sensor_msgs
   nav_msgs
)

ament_python_install_package(${PROJECT_NAME})


# Change what pytorch models to use here
install(
  PROGRAMS
  strongsort_node/track.py
  launch/strongsort_node_launch.py
  osnet_ain_x1_0_msmt17.pt
  osnet_x0_25_market1501.pt
  yolov7.pt
  yolov7x.pt
  requirements.txt
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY yolov7 strong_sort
  DESTINATION lib/${PROJECT_NAME})


install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})

ament_package()