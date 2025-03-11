#!/usr/bin/env python3
import rospy
import math
import random
import os
import rospkg
import csv
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

rospack = rospkg.RosPack()
package_path = rospack.get_path('wall_following_rl')
q_table_path = os.path.join(package_path, 'results/q_table.txt')
reward_data_path = os.path.join(package_path, 'results/reward_data.csv')

class QLearningTrainer:
    def __init__(self):
        rospy.init_node('q_learning_trainer', anonymous=True)
        
        # Initialize Q-table
        self.q_table = [
            [0.0]*10,  # State 1
            [0.0]*10,  # State 2
            [0.0]*10,  # State 3
            [0.0]*10,  # State 4
            [0.0]*10,  # State 5
            [0.0]*10   # State 6
        ]
        # Initialize Q-values with small random numbers to encourage exploration
        self.q_table = [
            [random.uniform(-0.1, 0.1) for _ in range(10)] 
            for _ in range(6)
        ]
        
        # ROS Setup
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.get_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Training parameters
        self.state_divisors = [math.pi, 
                              math.radians(120), 
                              math.radians(80),
                              math.radians(40),
                              math.radians(20),
                              math.radians(5),
                              0]
        self.actions = [0, 15, 30, 45, 60, 75, 90, 110, 130, 150]

        self.alpha = rospy.get_param('~alpha', 0.2)  # Default to 0.2 if not specified
        self.gamma = rospy.get_param('~gamma', 0.9)  # Default to 0.9 if not specified
        self.epsilon = rospy.get_param('~epsilon', 0.9)  # Default to 0.9 if not specified
        self.total_episodes = rospy.get_param('~max_episodes', 500)  # Default to 500 if not specified
        # self.alpha = 0.2
        # self.gamma = 0.9
        # self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        # Episode tracking
        self.episode_rewards = []
        self.current_episode = 0
        # self.total_episodes = 500  # Adjust based on needs
        self.steps_per_episode = 400  # Max steps per episode
        self.collision_threshold = 0.20  # Minimum safe distance

        self.start_positions = [
            (0.0, 0.0), 
            (0.0, 2.6),
            (0.0, -2.6),
            (-2.7, 1.2),
            (-2.9, 2.6),
            (-2.9, 2.6),
            (2.8,-3.0)
        ]
        
        self.current_scan = LaserScan()
        self.reward_data = []
        
    def scan_callback(self, data):
        self.current_scan = data
        
    def find_robot_pose(self):
        min_distance = float('inf')
        min_index = 0
        
        # Find closest obstacle
        for i, distance in enumerate(self.current_scan.ranges):
            if distance < min_distance:
                min_distance = distance
                min_index = i
                
        theta_robot_x = 2*math.pi - min_index*self.current_scan.angle_increment
        d_des = 0.4
        d_stan = 0.2
        
        theta_des = math.atan2(d_stan, min_distance - d_des)
        theta_err = abs(theta_des - theta_robot_x)
        
        if theta_err > math.pi:
            theta_err = 2*math.pi - theta_err
            
        return theta_des, theta_err, theta_robot_x
    
    def get_current_state(self, theta_err):
        for i in range(len(self.state_divisors)-1):
            if self.state_divisors[i] >= theta_err >= self.state_divisors[i+1]:
                return i
        return 5
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions)-1)
        else:
            return self.q_table[state].index(max(self.q_table[state]))

    
    def update_q_table(self, state, action, reward, next_state):
        max_next_q = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )

    def reset_robot(self):
        """Reset robot to random starting position"""
        pos = random.choice(self.start_positions)
        state_msg = ModelState()
        state_msg.model_name = 'triton'
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = 0.1
        
        try:
            self.set_state_srv(state_msg)
            rospy.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
    
    def train(self):
        rate = rospy.Rate(10)
    
        # Create results directory if not exists
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Training loop over episodes
        for self.current_episode in range(self.total_episodes):
            self.reset_robot()
            accumulated_reward = 0
            step_count = 0
            episode_done = False
            while not episode_done and step_count < self.steps_per_episode:
                # Get current state
                theta_des, theta_err, theta_robot_x = self.find_robot_pose()
                state = self.get_current_state(theta_err)
                
                # Check for collision
                min_distance = min(self.current_scan.ranges)
                if min_distance < self.collision_threshold:
                    reward = -50  # Large penalty for collision
                    accumulated_reward += reward
                    # Reset position slightly instead of ending episode
                    self.reset_robot() 
                    continue  # Resume learning
                    # episode_done = True
                else:
                    # Choose action
                    action = self.choose_action(state)
                    
                    # Apply action
                    self.apply_action(action, theta_des, theta_robot_x)
                    
                    # Get new state
                    _, new_theta_err, _ = self.find_robot_pose()
                    new_state = self.get_current_state(new_theta_err)
                    
                    # Calculate reward
                    reward = math.pi - abs(theta_err)
                    accumulated_reward += reward
                    
                    # Update Q-table
                    self.update_q_table(state, action, reward, new_state)
                    
                    step_count += 1
                
                rate.sleep()
                
            # Store episode data
            self.episode_rewards.append(accumulated_reward)
            self.save_episode_data(self.current_episode, accumulated_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Print progress
            if self.current_episode % 100 == 0:
                rospy.loginfo(f"Episode {self.current_episode} - Accumulated Reward: {accumulated_reward}")
        
        # Save final data
        self.save_q_table()
        self.save_reward_data()
    
    def save_episode_data(self, episode, reward):
        self.reward_data.append({
            'episode': episode,
            'accumulated_reward': reward
        })
    
    def save_reward_data(self):
        with open(reward_data_path, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'accumulated_reward']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data in self.reward_data:
                writer.writerow(data)
                
        
    def apply_action(self, action_idx, theta_des, current_angle):
        twist = Twist()
        twist.linear.x = 0.3
        
        angular_speed = math.radians(self.actions[action_idx])
        if self.should_rotate_cw(theta_des, current_angle):
            angular_speed *= -1
            
        twist.angular.z = angular_speed
        self.velocity_publisher.publish(twist)
    
    def should_rotate_cw(self, desired, current):
        desired = desired % (2*math.pi)
        current = current % (2*math.pi)
        
        diff = current - desired
        if abs(diff) < math.pi:
            return diff > 0
        else:
            return diff < 0
        
    def save_q_table(self):
        with open(q_table_path, 'w') as f:
            for row in self.q_table:
                f.write('\t'.join(map(str, row)) + '\n')

if __name__ == '__main__':
    try:
        trainer = QLearningTrainer()
        rospy.sleep(5)  # Initial delay
        trainer.train()
    except rospy.ROSInterruptException:
        pass
