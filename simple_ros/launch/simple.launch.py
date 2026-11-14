import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('simple_ros')
    config = LaunchConfiguration('config')
    pointcloud = LaunchConfiguration('pointcloud')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'livox.yaml')),
        DeclareLaunchArgument('pointcloud', default_value='/livox/lidar'),
        Node(
            package='simple_ros',
            executable='lidar_odometry',
            name='lidar_odometry',
            parameters=[config],
            remappings=[('pointcloud', pointcloud)],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(share, 'rviz', 'simple.rviz')],
        ),
    ])
