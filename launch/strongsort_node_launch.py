import launch
import launch_ros.actions


# To run: 
# Cmd window 1: colcon build
# Cmd window 2: ros2 launch strongsort_node strongsort_node_launch.py 
#    --> for reference: (ros2 launch {package_name} {name of launch file})
def generate_launch_description(): 
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='strongsort_node', 
            executable='track_ros_setup.py', # connects to entry point in setup.py
            name='strongsort', 
            output='screen',
            emulate_tty=True, 
            parameters=[
                {'video_topic': '/rm_vlc_leftfront/image', 
                 'name_space': 'A'}
            ]
        )
    ])
    