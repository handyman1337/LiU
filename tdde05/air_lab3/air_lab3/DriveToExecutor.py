import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import rclpy.executors
import math
import TstML
import TstML.Executor
import traceback
import sys
import time
import random
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import SpeedLimit
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

class DriveToExecutor(TstML.Executor.AbstractNodeExecutor):
  def __init__(self, node, context):
    super(TstML.Executor.AbstractNodeExecutor, self).__init__(node,
          context)

    self.ros_node = Node(gen_name("drive_to"))
    
   

    self.publisher = self.ros_node.create_publisher(SpeedLimit, '/speed_limit', 10)
    

    self._action_client = ActionClient(self.ros_node, NavigateToPose, 'navigate_to_pose')
    self.executor = rclpy.executors.MultiThreadedExecutor()
    self.executor.add_node(self.ros_node)
    self.thread = threading.Thread(target=self.executor.spin)
    self.thread.start()
    self._abort = False
  

  def finalise(self):
    self.executor.shutdown()

  def start(self):
    goal_msg = NavigateToPose.Goal()
    goal_msg.pose.header.frame_id = "map"

    p = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "p")
    goal_msg.pose.pose.position.x = float(p["x"])
    goal_msg.pose.pose.position.y = float(p["y"])

    msg = SpeedLimit()

    max_speed = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "maximum-speed")
    
    if max_speed is not None:
      msg.speed_limit = float(max_speed)
      self.publisher.publish(msg)
      self.ros_node.get_logger().info('Param speed limit: "%s"' % msg)
    else:
      msg.speed_limit = 0.3
      #msg.speed_limit = random.uniform(0.1, 5)
      self.publisher.publish(msg)
      self.ros_node.get_logger().info('Static speed limit: "%s"' % msg)
    time.sleep(1)
    angle = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "angle")
    if angle is None:
      angle = 0.0
    else:
      angle = float(angle)
    
    goal_msg.pose.pose.orientation.w = math.cos(angle/2)
    goal_msg.pose.pose.orientation.z = math.sin(angle/2)

    self._send_goal_future = self._action_client.send_goal_async(goal_msg, self.feedback_callback)
    self._send_goal_future.add_done_callback(self.goal_response_callback)
    return TstML.Executor.ExecutionStatus.Started()
  
  def feedback_callback(self, feedback_msg):
    self.ros_node.get_logger().info('Navigation time: {0}'.format(feedback_msg.feedback.navigation_time))
    self.ros_node.get_logger().info('Time remaining: {0}'.format(feedback_msg.feedback.estimated_time_remaining))
    self.ros_node.get_logger().info('Distance remaining: {0}'.format(feedback_msg.feedback.distance_remaining))
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
    print("Finished!")
    if self._abort:
      self.ros_node.get_logger().info('Aborted')

      self._goal_handle.cancel_goal()
      self.executionFinished(TstML.Executor.ExecutionStatus.Aborted())

    self.executionFinished(TstML.Executor.ExecutionStatus.Finished())

  def pause(self):
    self.ros_node.get_logger().info('Pause is not possible.')
    return TstML.Executor.ExecutionStatus.Running()
  def resume(self):
    return TstML.Executor.ExecutionStatus.Running()
  def stop(self):
    self._goal_handle.cancel_goal()
    return TstML.Executor.ExecutionStatus.Finished()
  def abort(self):
    self._abort = True
    #self._goal_handle.cancel_goal()
    return TstML.Executor.ExecutionStatus.Aborted()
