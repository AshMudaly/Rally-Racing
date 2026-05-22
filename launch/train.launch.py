import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction, LogInfo,
    RegisterEventHandler, EmitEvent,
)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

BASE_DIR = os.path.expanduser("~/41118_ws/project/racing")
WORLD    = os.path.join(BASE_DIR, "worlds", "oval_track.sdf")
URDF     = os.path.join(BASE_DIR, "urdf",   "car.urdf")


def generate_launch_description():

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo", "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            WORLD,
        ],
        output="screen",
    )

    # If Gazebo dies, kill the whole launch — otherwise training silently
    # runs against a dead simulator and burns hours on garbage data.
    shutdown_on_gazebo_death = RegisterEventHandler(
        OnProcessExit(
            target_action=gazebo,
            on_exit=[
                LogInfo(msg="Gazebo died — shutting down entire launch."),
                EmitEvent(event=Shutdown(reason="Gazebo died")),
            ],
        )
    )

    spawn_car = TimerAction(
        period=5.0,
        actions=[
            Node(
                package    = "gazebo_ros",
                executable = "spawn_entity.py",
                arguments  = [
                    "-file",   URDF,
                    "-entity", "racing_car",
                    "-x",      "0.0",
                    "-y",      "0.0",
                    "-z",      "0.1",
                    "-Y",      "0.0",
                ],
                output="screen",
            )
        ],
    )

    with open(URDF, "r") as f:
        robot_description = f.read()

    robot_state_pub = Node(
        package    = "robot_state_publisher",
        executable = "robot_state_publisher",
        name       = "robot_state_publisher",
        parameters = [{"robot_description": robot_description}],
        output     = "screen",
    )

    train = TimerAction(
        period=12.0,   # +2s vs before to be safe on sensor warmup
        actions=[
            ExecuteProcess(
                cmd=[
                    "python3",
                    os.path.join(BASE_DIR, "src", "train.py"),
                ],
                cwd    = os.path.join(BASE_DIR, "src"),
                output = "screen",
            )
        ],
    )

    return LaunchDescription([
        LogInfo(msg="Launching Gazebo oval track..."),
        gazebo,
        shutdown_on_gazebo_death,
        robot_state_pub,
        LogInfo(msg="Spawning racing car in 5s..."),
        spawn_car,
        LogInfo(msg="Starting PPO training in 12s..."),
        train,
    ])