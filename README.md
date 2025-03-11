# Robot Wall Following with Q-Learning

This ROS package implements Q-learning reinforcement learning for a robot wall following task. The Triton robot learns to follow walls and navigate around corners using a 2D LiDAR sensor.

## Demo Video

Watch the demonstration here: [Robot Wall Following Demo](https://youtu.be/fO-TXPbYNIw)

## Requirements

- ROS Noetic
- Gazebo
- The Triton robot simulation package (stingray_sim)

## Package Overview

This package implements Q-learning with Temporal Difference (TD) updates to teach a robot to follow walls. The state representation is based on the robot's orientation error relative to the wall, and actions control the robot's angular velocity.

## Usage

### Training

To train the Q-learning algorithm:

`roslaunch wall_following_rl train.launch`

This will start the training process with the following default parameters:

- epsilon: 0.9 (exploration rate)
- alpha: 0.2 (learning rate)
- gamma: 0.9 (discount factor)
- max_episodes: 500

The Q-table will be saved to the config directory as `q_table.txt`.

### Testing

To test the trained policy:

`roslaunch wall_following_rl test.launch`

The testing code will load the trained Q-table and use it to control the robot for wall following in different scenarios.

## Implementation Details

### State Representation

The state is represented by the error angle between the robot's current orientation and the desired orientation relative to the wall. This angle is discretized into 6 states using the following thresholds:

- State 0: Error between π and 120°
- State 1: Error between 120° and 80°
- State 2: Error between 80° and 40°
- State 3: Error between 40° and 20°
- State 4: Error between 20° and 5°
- State 5: Error between 5° and 0°

### Actions

The action space consists of 10 different angular velocities: [0, 15, 30, 45, 60, 75, 90, 110, 130, 150] degrees/second.

### Reward Function

The reward function is designed to encourage:

- Small error angles between the robot and the wall
- Consistent distance from the wall
- Penalize collisions with walls

## Results

The training process shows an increasing trend in accumulated rewards over episodes, indicating that the robot is learning the wall-following behavior. The final policy enables the robot to:

1. Follow straight walls
2. Navigate 90-degree corners (left and right both)
3. Navigate 180-degree turns (I-shape and U-shape both)

## File Structure

- `scripts/q_learning_train.py`: Implementation of Q-learning training algorithm
- `scripts/q_learning_test.py`: Code for testing the learned policy
- `launch/train.launch`: Launch file for training mode
- `launch/test.launch`: Launch file for testing mode
- `results/q_table.txt`: Saved Q-table from training
- `results/reward_data.csv`: Saved data for accumulated reward per episode
