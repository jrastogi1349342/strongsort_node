import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# To run: 
# Cmd window 1: colcon build
# Cmd window 2: ros2 launch strongsort_node strongsort_node_launch.py 
#    --> for reference: (ros2 launch {package_name} {name of launch file})
def launch_setup(context, *args, **kwargs): 
    strongsort_node = Node(
            package='strongsort_node', 
            # connects to entry point in setup.py
            executable='track_ros_setup.py', 
            name='strongsort', 
            output='screen',
            emulate_tty=True, 
            parameters=[
                {
                    'video_topic': '/rm_vlc_leftfront/image', 
                    'name_space': LaunchConfiguration('name_space'), 
                    "robot_id": LaunchConfiguration('robot_id'), 
                    "max_nb_robots": LaunchConfiguration('max_nb_robots'), 
                    "show_video": True
                 }
            ], 
        )
    
    association_node = Node(
        package='strongsort_node', 
        executable='associations_ros_driver.py', 
        name='associations', 
        output='screen', 
        emulate_tty=True,
        parameters=[
            {
                'name_space': LaunchConfiguration('name_space'), 
                "robot_id": LaunchConfiguration('robot_id'), 
                "max_nb_robots": LaunchConfiguration('max_nb_robots'), 
            }
        ]
    )
    
    return [
        strongsort_node, 
        association_node
    ]


def generate_launch_description():     
    return launch.LaunchDescription([
        DeclareLaunchArgument('name_space', default_value='A', description=''), 
        DeclareLaunchArgument('robot_id', default_value='0', description=''), 
        DeclareLaunchArgument('max_nb_robots', default_value='2', description=''), 
        OpaqueFunction(function=launch_setup)
    ])
    