import threading
import rclpy
import math
import random
import traceback
import sys
import time

from nav_msgs.msg import Odometry 
from rclpy.node import Node
from rclpy.action import ActionClient
import rclpy.executors

import TstML
import TstML.Executor

from nav2_msgs.action import NavigateToPose

import ament_index_python

# Work around for ROS not showing error with a MultiThreadedExecutor
def display_exceptions(func):
  def wrapper(*args):
    try:
      return func(*args)
    except Exception as ex:
      traceback.print_exception(*sys.exc_info()) 
      print(f"Unhandled exception was caught: '{ex}'")
  return wrapper

# Ugly hack to get a new name for each ROS node
ros_name_counter = 0
def gen_name(name):
    global ros_name_counter
    ros_name_counter += 1
    return name + str(ros_name_counter)


class ExploreExecutor(TstML.Executor.AbstractNodeExecutor):
  def __init__(self, node, context):
    super(TstML.Executor.AbstractNodeExecutor, self).__init__(node,
          context)

    self.ros_node = Node(gen_name("explore_node"))

    self.radius = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "radius")
    self.max_count = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "count")
    
    '''
    if self.radius is None:
      self.radius = 3
    
    if self.max_count is None:
      self.max_count = 3
    '''
    
    self.count = 0
    self.start_set = False
    self.subscription_ = self.ros_node.create_subscription(Odometry, '/odom', self.listener_callback, 10)
    self.start_x = 0.0
    self.start_y = 0.0

    self._action_client = ActionClient(self.ros_node, NavigateToPose, 'navigate_to_pose')
    self.executor = rclpy.executors.MultiThreadedExecutor()
    self.executor.add_node(self.ros_node)
    self.thread = threading.Thread(target=self.executor.spin)
    self.thread.start()

    self.keep_exploring = True

  def listener_callback(self, msg):
      
      current_x = msg.pose.pose.position.x
      current_y = msg.pose.pose.position.y
      #self.get_logger().info('Listening: "%s' % msg)
      #self.ros_node.get_logger().info(f'I heard /odom: x={current_x}, y={current_y}')
      
      if self.count == 0 and not self.start_set:
          self.start_x = current_x
          self.start_y = current_y
          self.start_set = True
          self.ros_node.get_logger().info('Start position set!!!')

      

  def finalise(self):
    self.executor.shutdown()

  def start(self):
    

    self.send_goal()

    return TstML.Executor.ExecutionStatus.Started()
  
  def send_goal(self):
    goal_msg = NavigateToPose.Goal()
    x, y, angle = self.generate_random_loc()

    goal_msg.pose.header.frame_id = "map"
    goal_msg.pose.pose.position.x = x
    goal_msg.pose.pose.position.y = y
    goal_msg.pose.pose.orientation.w = math.cos(angle/2)
    goal_msg.pose.pose.orientation.z = math.sin(angle/2)

    self._send_goal_future = self._action_client.send_goal_async(goal_msg, self.feedback_callback)
    self._send_goal_future.add_done_callback(self.goal_response_callback)

  def feedback_callback(self, feedback_msg):
        #self.ros_node.get_logger().info('Navigation time: {0}'.format(feedback_msg.feedback.navigation_time))
        #self.ros_node.get_logger().info('Time remaining: {0}'.format(feedback_msg.feedback.estimated_time_remaining))
        #self.ros_node.get_logger().info('Distance remaining: {0}'.format(feedback_msg.feedback.distance_remaining))
        
        if feedback_msg.feedback.number_of_recoveries > 10:
            self.ros_node.get_logger().info('Recoveries: {0}'.format(feedback_msg.feedback.number_of_recoveries))

            
            self._goal_handle.cancel_goal_async()
            
  
  @display_exceptions
  def goal_response_callback(self, future):
    self._goal_handle = future.result()
    if not self._goal_handle.accepted:
      self.executionFinished(TstML.Executor.ExecutionStatus.Aborted())
      self.ros_node.get_logger().error('Goal rejected :(')
    else:
      self.ros_node.get_logger().error('Goal accepted :)')

      self._get_result_future = self._goal_handle.get_result_async()
      self._get_result_future.add_done_callback(self.handle_result_callback)

  @display_exceptions
  def handle_result_callback(self, future):
    if self.count>=self.max_count:
      self.keep_exploring == False
    else:
      self.count += 1

    if future.result().status == 4:
      
      print("Goal reached!")
     
    else:
      print("Goal NOT reached!")
    
  
    if self.keep_exploring:
      self.send_goal()
    else:
      self._goal_handle.cancel_goal_async()
    #self.executionFinished(TstML.Executor.ExecutionStatus.Finished())

  def generate_random_loc(self):
    angle_dest = random.uniform(-math.pi, math.pi)
    dist = random.uniform(0, self.radius)
    x = self.start_x + (dist* math.cos(angle_dest))
    y = self.start_y + (dist* math.sin(angle_dest))
    

    self.ros_node.get_logger().info('New coordinates generated')
    return x, y, random.uniform(-math.pi, math.pi)

  def pause(self):
    self.ros_node.get_logger().info('Pause is not possible.')
    return TstML.Executor.ExecutionStatus.Running()
  def resume(self):
    return TstML.Executor.ExecutionStatus.Running()
  def stop(self):
    self.keep_exploring = False
    self._goal_handle.cancel_goal_async()
    return TstML.Executor.ExecutionStatus.Finished()
  def abort(self):
    self.keep_exploring = False
    self.ros_node.get_logger().info('Abort requested')
    
    return TstML.Executor.ExecutionStatus.Aborted()