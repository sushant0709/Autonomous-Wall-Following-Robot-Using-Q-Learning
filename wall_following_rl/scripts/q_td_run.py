import rospy
import math
import numpy as np
import rospkg
import os
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class QLearningTester:
    def __init__(self, q_table):
        rospy.init_node('q_learning_tester', anonymous=True)
        
        # Load trained Q-table
        self.q_table = q_table
        
        # ROS Setup
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.get_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Testing parameters
        self.state_divisors = [math.pi, 
                              math.radians(120), 
                              math.radians(80),
                              math.radians(40),
                              math.radians(20),
                              math.radians(5),
                              0]
        self.actions = [0, 15, 30, 45, 60, 75, 90, 110, 130, 150]
        self.collision_threshold = 0.20  # Minimum safe distance

        self.current_scan = LaserScan()
        
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
        return self.q_table[state].index(max(self.q_table[state]))
    
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
        
    def test(self):
        rate = rospy.Rate(10)
        
        # Reset robot to a starting position
        pos = (0.0, 0.0, 0.0)  # Center
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
        
        # Testing loop
        for _ in range(2000):  # Run for a certain number of steps
            theta_des, theta_err, theta_robot_x = self.find_robot_pose()
            state = self.get_current_state(theta_err)
            
            # Check for collision
            min_distance = min(self.current_scan.ranges)
            if min_distance < self.collision_threshold:
                rospy.loginfo("Collision detected. Stopping test.")
                break
            
            action = self.choose_action(state)
            self.apply_action(action, theta_des, theta_robot_x)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('wall_following_rl')
        q_table_path = os.path.join(package_path, 'results/q_table.txt')
        # Load trained Q-table from file
        with open(q_table_path, 'r') as f:
            q_table = [[float(x) for x in line.split('\t')] for line in f.readlines()]
        
        tester = QLearningTester(q_table)
        rospy.sleep(5)  # Initial delay
        tester.test()
    except rospy.ROSInterruptException:
        pass
