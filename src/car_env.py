import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time

from reward import RewardCalculator


class RacingCarEnv(gym.Env, Node):
    """
    Gymnasium environment wrapping the Gazebo racing car.

    Observation space (7,):
        [ray_left, ray_front_left, ray_front, ray_front_right, ray_right,
         speed, steering]

    Action space (2,):
        [steering ∈ -1..1, throttle ∈ 0..1]
    """

    metadata = {"render_modes": ["human"]}

    # Physical limits — used both for cmd_vel scaling and observation normalisation
    MAX_SPEED     = 1.5   # m/s
    MAX_TURN_RATE = 1.0   # rad/s
    CONTROL_DT    = 0.05  # seconds per RL step

    def __init__(self):
        rclpy.init(args=None)
        Node.__init__(self, "racing_car_env")

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Internal state ───────────────────────────────────────────────
        self.rays          = np.ones(5, dtype=np.float32)
        self.speed         = 0.0
        self.position      = np.zeros(2, dtype=np.float32)
        self.current_steer = 0.0
        self.step_count    = 0
        self.max_steps     = 1000

        self._ray_received = {
            "front": False, "front_left": False, "front_right": False,
            "left":  False, "right":      False,
        }

        self.reward_calc = RewardCalculator()

        # ── Publishers / Subscribers ─────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, "/racing_car/cmd_vel", 10)

        self.create_subscription(LaserScan, "/racing_car/ray/front",
                                 lambda m: self._ray_cb(m, 2), 10)
        self.create_subscription(LaserScan, "/racing_car/ray/front_left",
                                 lambda m: self._ray_cb(m, 1), 10)
        self.create_subscription(LaserScan, "/racing_car/ray/front_right",
                                 lambda m: self._ray_cb(m, 3), 10)
        self.create_subscription(LaserScan, "/racing_car/ray/left",
                                 lambda m: self._ray_cb(m, 0), 10)
        self.create_subscription(LaserScan, "/racing_car/ray/right",
                                 lambda m: self._ray_cb(m, 4), 10)
        self.create_subscription(Odometry, "/racing_car/odom", self._odom_cb, 10)

        self.reset_client = self.create_client(Empty, "/reset_world")

        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self,), daemon=True
        )
        self._ros_thread.start()

        self.get_logger().info("RacingCarEnv ready — waiting for sensors...")
        # Fail fast at construction time if Gazebo isn't up
        self._wait_for_sensors(timeout=10.0, fatal=True)
        self.get_logger().info("All sensors publishing — env ready.")

    # ── Callbacks ────────────────────────────────────────────────────────

    def _ray_cb(self, msg: LaserScan, index: int):
        raw = msg.ranges[0] if msg.ranges else msg.range_max
        if not np.isfinite(raw):
            raw = msg.range_max
        self.rays[index] = float(
            np.clip(raw / self.reward_calc.max_ray_distance, 0.0, 1.0)
        )
        names = ["left", "front_left", "front", "front_right", "right"]
        self._ray_received[names[index]] = True

    def _odom_cb(self, msg: Odometry):
        self.position[0] = float(msg.pose.pose.position.x)
        self.position[1] = float(msg.pose.pose.position.y)

        # Linear x is forward speed in the body frame from diff_drive plugin
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = float(np.sqrt(vx**2 + vy**2))

        # Yaw from quaternion — needed for signed forward progress
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    # ── Gym interface ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [*self.rays, self.speed / self.MAX_SPEED, self.current_steer],
            dtype=np.float32,
        )

    def _wait_for_sensors(self, timeout: float = 5.0, fatal: bool = False):
        """
        Block until every ray sensor has published at least once.
        If `fatal`, raise on timeout (so training doesn't silently run
        against frozen all-ones observations when Gazebo isn't up).
        """
        start = time.time()
        while not all(self._ray_received.values()):
            if time.time() - start > timeout:
                missing = [k for k, v in self._ray_received.items() if not v]
                msg = (
                    f"Sensor timeout after {timeout:.1f}s — no data on rays: "
                    f"{missing}. Is Gazebo running and the car spawned?"
                )
                if fatal:
                    self.get_logger().error(msg)
                    raise RuntimeError(msg)
                self.get_logger().warn(msg)
                return
            time.sleep(0.05)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._publish_cmd(0.0, 0.0)
        time.sleep(0.1)

        if self.reset_client.wait_for_service(timeout_sec=2.0):
            self.reset_client.call_async(Empty.Request())
            time.sleep(0.5)
        else:
            self.get_logger().warn("Reset service not available.")

        self.step_count    = 0
        self.current_steer = 0.0
        self.speed         = 0.0
        self.rays          = np.ones(5, dtype=np.float32)
        self._ray_received = {k: False for k in self._ray_received}
        self.reward_calc.reset()

        # Non-fatal here — after reset, a brief sensor gap is normal.
        self._wait_for_sensors(timeout=5.0, fatal=False)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1],  0.0, 1.0))

        self.current_steer = steering
        self._publish_cmd(steering, throttle)

        time.sleep(self.CONTROL_DT)

        crashed = self.reward_calc.is_crashed(self.rays)

        reward, info = self.reward_calc.compute(
            rays=self.rays,
            speed=self.speed,
            steering=steering,
            position=self.position,
            yaw=getattr(self, "yaw", 0.0),
            crashed=crashed,
        )

        self.step_count += 1
        terminated = crashed
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, info

    def _publish_cmd(self, steering: float, throttle: float):
        msg = Twist()
        msg.linear.x  = throttle * self.MAX_SPEED
        msg.angular.z = -steering * self.MAX_TURN_RATE
        self.cmd_vel_pub.publish(msg)

    def close(self):
        self._publish_cmd(0.0, 0.0)
        rclpy.shutdown()