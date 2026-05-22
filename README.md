# Oval Track Racing: _PPO Reinforcement Learning Agent_
Bringup for **Oval Track Racing**: a PPO-trained autonomous racing car in Gazebo Classic. A project for *41118 Artificial Intelligence in Robotics*. Launches a differential-drive racing car on a custom oval track world and trains a Proximal Policy Optimization (PPO) agent to lap the track using five ray sensors and odometry. We use **ROS2 Humble**, **Gazebo Classic 11**, and **Stable-Baselines3**.

The car learns to drive from five ray-sensor distance readings (left, front-left, front, front-right, right) plus its current speed and steering. The reward function balances forward progress, speed, smoothness, wall avoidance, and collision penalties.

## Installation
### Installation: Simulation Basics
First install some dependencies:
* If you haven't already, install ROS2 Humble. Follow the instructions here: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
* Install Gazebo Classic and ROS bridge packages
  ```bash
  sudo apt update
  sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
  sudo apt install ros-humble-robot-state-publisher ros-humble-xacro
  ```
* Install development tools
  ```bash
  sudo apt install ros-dev-tools
  ```
* Make sure that your installation is up to date. This is particularly important if you installed ROS a long time ago, such as in another subject. If you get errors here, make sure to resolve these before continuing.
  ```bash
  sudo apt upgrade
  sudo apt update
  ```

### Installation: Racing Package
* Create a new colcon workspace
  ```bash
  mkdir -p ~/41118_ws/project
  ```
* Pull the repository to the `project` directory in this workspace
* Install Python dependencies
  ```bash
  pip install stable-baselines3[extra] gymnasium torch tensorboard
  ```
* Source ROS2 (if you add this to your ~/.bashrc, then you don't need to do this each time)
  ```bash
  source /opt/ros/humble/setup.bash
  ```

## Project Layout
```
~/41118_ws/project/Rally-Racing/
├── launch/
│   └── train.launch.py     # Brings up Gazebo, spawns car, starts training
├── worlds/
│   └── oval_track.sdf      # Oval track world with inner/outer walls
├── urdf/
│   └── car.urdf            # Differential-drive car with 5 ray sensors
├── src/
│   ├── car_env.py          # Gymnasium environment wrapping ROS2 + Gazebo
│   ├── reward.py           # Reward function (progress, speed, swerve, walls)
│   └── train.py            # PPO training script with checkpointing
├── models/                 # Saved PPO weights (resume.zip, best/, checkpoints)
└── logs/                   # TensorBoard logs + monitor.csv
```

## Launching
### Training
* Launch Gazebo, spawn the car, and start training in one command:
  ```bash
  cd ~/41118_ws/project/racing
  ros2 launch launch/train.launch.py
  ```
  The launch file starts Gazebo immediately, spawns the car after 5 seconds, and kicks off `train.py` after 12 seconds. Training resumes from `models/resume.zip` if it exists, otherwise starts fresh.

* To start a completely fresh training run, remove all saved checkpoints first:
  ```bash
  rm -rf ~/41118_ws/project/racing/models/*.zip
  rm -rf ~/41118_ws/project/racing/models/best/*
  rm -rf ~/41118_ws/project/racing/logs/*
  ```

### Monitoring with TensorBoard
* In a separate terminal, launch TensorBoard to watch training curves:
  ```bash
  tensorboard --logdir ~/41118_ws/project/racing/logs
  ```
  Then open http://localhost:6006 in your browser. Key metrics to watch:
    * `rollout/ep_rew_mean` — average episode reward (should trend upward)
    * `rollout/ep_len_mean` — average episode length (1000 = full episode, low values mean frequent crashes)
    * `train/entropy_loss` — policy randomness (should decrease as the agent commits to a strategy)

### Inspecting Episode Logs
* The `Monitor` wrapper writes one row per episode to `logs/monitor.csv`:
  ```bash
  tail -f ~/41118_ws/project/racing/logs/monitor.csv
  ```
  Columns are `r` (episode return), `l` (episode length), `t` (wall-clock time since start).

### Watching the Trained Agent
* The Gazebo GUI opens automatically with the launch file — watch the car drive in the simulation window. To run a saved model deterministically without training:
  ```bash
  # (Requires an eval script — see project source)
  ```

## Errors
If you are getting errors, first check you are following the instructions correctly. Here are a few frequently encountered errors.

### Address Already in Use (Gazebo Master)
If you get an error like:
```bash
[Err] [Master.cc:96] EXCEPTION: Unable to start server[bind: Address already in use].
There is probably another Gazebo process running.
```
This means a previous Gazebo process is still holding port 11345. `killall gazebo` is not always enough. Run:
```bash
killall -9 gzserver gzclient gazebo
pkill -9 -f train.py
pkill -9 -f spawn_entity
fuser -k 11345/tcp
```
Then confirm nothing is bound:
```bash
ss -tlnp | grep 11345
```

### Corrupt Resume Checkpoint
If you get an error like:
```bash
zipfile.BadZipFile: Overlapped entries: 'policy.optimizer.pth' (possible zip bomb)
ValueError: Error: the file ~/41118_ws/project/racing/models/resume.zip wasn't a zip-file
```
The resume checkpoint was corrupted, usually by an interrupted save. The training script automatically moves corrupt files to `resume.zip.broken` and starts fresh, but if you want to manually recover:
```bash
# Verify any saved checkpoint
python3 -c "import zipfile; print(zipfile.ZipFile('~/41118_ws/project/racing/models/SOME_FILE.zip').testzip())"
# Returns None if intact, otherwise prints the corrupt member
```
If a per-step checkpoint exists in `models/`, promote it to `resume.zip`:
```bash
cp ~/41118_ws/project/racing/models/ppo_racing_50000_steps.zip \
   ~/41118_ws/project/racing/models/resume.zip
```

### Sensor Timeout
If you see:
```bash
RuntimeError: Sensor timeout after 10.0s — no data on rays: ['front', ...].
Is Gazebo running and the car spawned?
```
The training script could not detect any ray sensor publishing within 10 seconds of startup. This usually means Gazebo died or the car never spawned. Check the Gazebo terminal output for errors, then clean up and relaunch:
```bash
killall -9 gzserver gzclient
ros2 launch launch/train.launch.py
```

### Car Spawned Inside a Wall
If the car immediately reports a collision on every episode reset, the spawn coordinates in `train.launch.py` may place it inside or beside a track wall. The track surface is the annular region between the inner walls (at |y| = 4, |x| = 9) and the outer walls (at |y| = 8, |x| = 13). A safe spawn is along the bottom straight:
```python
"-x", "0.0",
"-y", "-6.0",
"-z", "0.1",
"-Y", "0.0",
```

### KDL Inertia Warning
You will see this warning on every launch:
```bash
[WARN] [kdl_parser]: The root link base_link has an inertia specified in the URDF,
but KDL does not support a root link with an inertia.
```
This is harmless — KDL just ignores the inertia of the root link. It does not affect the differential-drive plugin or the simulation. You can silence it by adding a massless dummy root link to `car.urdf`, but it is not required.

### Agent Won't Move
If `monitor.csv` shows episode lengths of 1000 with very negative rewards, the agent has learned the "do nothing" local optimum — sitting still avoids the collision penalty. The reward function in `reward.py` includes a per-step time penalty (`w_time`) specifically to break this strategy. If you still see it after 100k+ steps:
* Increase `w_time` from 0.05 to 0.1 in `reward.py`
* Increase `w_progress` to make forward motion more attractive
* Increase `ent_coef` in `train.py` to encourage more exploration

### Agent Crashes Head-On Every Episode
This is normal during early training (first ~50k steps). The agent has learned that forward motion earns reward but has not yet connected ray sensor readings to crash avoidance. If it persists past 200k steps:
* Increase `w_collision` from 50.0 to 100.0 or higher
* Increase `w_wall` and widen `min_ray_distance` so the proximity penalty kicks in earlier
* Raise `ent_coef` temporarily to encourage trying turning actions

## Reward Function
The reward at each timestep is the sum of:
| Component | Sign | Purpose |
|-----------|------|---------|
| `progress` | + | Signed forward velocity along the car's heading — reversing penalised at 2× |
| `speed` | + | Encourages going fast, capped at 3 m/s |
| `swerve_penalty` | − | Penalises rapid changes in steering (squared delta) |
| `wall_proximity` | − | Exponential penalty as any ray drops below 0.3 |
| `time_penalty` | − | Constant per-step cost so standing still bleeds reward |
| `collision` | − | One-shot −50 when any ray drops below 0.05 (terminates episode) |

All weights live at the top of `RewardCalculator.__init__` in `reward.py`.

## Hyperparameters
The PPO configuration lives in `train.py`:
* Network: 3 × 256 MLP for both policy and value heads, ReLU activation
* Learning rate: 3e-4
* Batch size: 64, n_steps 2048, n_epochs 10
* γ = 0.99, GAE λ = 0.95
* Entropy bonus: 0.01
* Total timesteps: 200,000 per run (resumable)

Checkpoints save every 10,000 steps to `models/ppo_racing_*_steps.zip`. The best-evaluated model is saved separately to `models/best/best_model.zip`. The resume checkpoint `models/resume.zip` is written atomically (temp file + `os.replace`) so an interrupted training run cannot corrupt it.
