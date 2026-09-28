import rclpy
import math
import random
from rclpy.action import ActionClient
from rclpy.node import Node

from nav2_msgs.action import NavigateToPose
import time


class RandomExploration(Node):

    def __init__(self):
        super().__init__('random_exploration')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        


    def send_goal(self, x, y, angle):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = math.cos(angle/2)
        goal_msg.pose.pose.orientation.z = math.sin(angle/2)

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            
            #Try again with new random goal
            x = random.uniform(self._x_min, self._x_max)
            y = random.uniform(self._y_min, self._y_max)
            angle = random.uniform(-math.pi, math.pi)
            self.send_goal(x, y, angle)
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = self._goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        
        status = future.result().status

        self.get_logger().info("Result callback exit: %s" % result)
        self.get_logger().info("Status callback exit: %s" % status)
        
        
        
        self.get_logger().info("Sending new goal")
        [x,y,w] = self.generate_random_loc()
        self.send_goal(x, y, w)
        

    def feedback_callback(self, feedback_msg):
        
        self.get_logger().info('Navigation time: {0}'.format(feedback_msg.feedback.navigation_time))
        self.get_logger().info('Time remaining: {0}'.format(feedback_msg.feedback.estimated_time_remaining))
        self.get_logger().info('Distance remaining: {0}'.format(feedback_msg.feedback.distance_remaining))
        self.get_logger().info('Recoveries: {0}'.format(feedback_msg.feedback.number_of_recoveries))
        if feedback_msg.feedback.number_of_recoveries > 2:
            
            self._goal_handle.cancel_goal_async()
            
        
    def generate_random_loc(self):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        #z = random.uniform(-10, 10)
        w = random.uniform(-math.pi, math.pi)

        self.get_logger().info('New cooridnates generated')


        return [x, y, w]
    
def main(args=None):
    rclpy.init(args=args)

    action_client = RandomExploration()

    action_client.send_goal(0.8, -0.6, 1.0)

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
