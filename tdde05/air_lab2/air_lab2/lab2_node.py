import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry 
import math


class PublisherSubscriber(Node):

    def __init__(self):
        super().__init__('publisher_subscriber')
        # publisher
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_x = None
        self.start_y = None
        
        # subscription
        self.subscription_ = self.create_subscription(Odometry, '/odom', self.listener_callback, 10)
        #self.subscription_ # prevent unused variable warning
        
        # Parameters
        self.declare_parameter('linear', 0.1)
        self.declare_parameter('angular', 0.05)
        self.declare_parameter('distance', 1.0)
        
        self._linear = self.get_parameter('linear').value
        self._angular = self.get_parameter('angular').value
        self._distance = self.get_parameter('distance').value
        
        # timer
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.publisher_callback)

    def publisher_callback(self):
        msg = Twist()
        msg.linear.x = self._linear # linear velocity
        msg.angular.z = self._angular # angular velocity
        self.publisher_.publish(msg)
        self.get_logger().info(
            f'Publishing Twist: linear.x={msg.linear.x}, angular.z={msg.angular.z}'
        )
        
    def listener_callback(self, msg):
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        #self.get_logger().info('Listening: "%s' % msg)
        self.get_logger().info(f'I heard /odom: x={current_x}, y={current_y}')
        
        if self.start_x is None:
            self.start_x = current_x
            self.start_y = current_y
            self.get_logger().info('Start position set!!!')
        
        #Euclidian distance = sqrt((x1-x2)^2 + (y1-y2)^2)
        x_dist = current_x - self.start_x
        y_dist = current_y - self.start_y
        distance = math.sqrt(x_dist**2 + y_dist**2)
        self.get_logger().info(f'Euc_dist is: {distance}')
        
        if distance >= self._distance:
            self.timer.cancel()
            self.get_logger().info('Reached 1m, stopping and exiting!!!')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    publisher_subscriber = PublisherSubscriber()

    rclpy.spin(publisher_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()