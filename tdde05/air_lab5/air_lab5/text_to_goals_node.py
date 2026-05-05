import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from air_lab_interfaces.msg import Goal, Goals

class TextToGoals(Node):

    def __init__(self):
        super().__init__('text_to_goals_node')
        # Publisher for Goals
        self.publisher_ = self.create_publisher(Goals, '/goals_requests', 10)

        # Subscription to text_command
        self.subscription_ = self.create_subscription(
            String,
            '/text_command',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        goals_msg = Goals()
        command = msg.data.strip().lower()

        if "goto" in command:
            destination = command.replace("goto", "").strip().capitalize()
            goal = Goal(type="goto", destination=destination, object="")
            goals_msg.goals = [goal]

        elif "bring" in command:
            object_ = command.replace("bring", "").strip().capitalize()
            goal = Goal(type="bring", object=object_, destination="User")
            goals_msg.goals = [goal]

        elif "explore" in command:
            goal = Goal(type="explore", object="", destination="")
            goals_msg.goals = [goal]

        else:
            self.get_logger().warn(f"Unknown command: {command}")
            return

        self.publisher_.publish(goals_msg)
        self.get_logger().info(f"Published goals: {goals_msg}")

def main(args=None):
    rclpy.init(args=args)
    text_to_goals = TextToGoals()
    rclpy.spin(text_to_goals)
    text_to_goals.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()