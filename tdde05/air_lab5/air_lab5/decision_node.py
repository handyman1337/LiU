import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
from air_lab_interfaces.msg import Goals, Goal
from air_lab_interfaces.srv import ExecuteTst
from ros_kdb_interfaces.srv import QueryDatabase
from ros_kdb_interfaces.msg import Query
import json
import os

class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')
        
        # ReentrantCallBackGroup as this solved blocking in lab4
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribe to goals_requests topic
        self.subscription = self.create_subscription(
            Goals,
            '/goals_requests',
            self.goals_callback,
            10,
            callback_group=self.callback_group
        )
        
        #Subscribe to odom to be able to get current robot pos
        self.current_x = 0.0
        self.current_y = 0.0
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        

        # SPARQL query client
        self.sparql_client = self.create_client(QueryDatabase, '/kdb_server/sparql_query', callback_group=self.callback_group)
        # TST execution client
        self.tst_executor_client = self.create_client(ExecuteTst, '/execute_tst', callback_group=self.callback_group)
        

    def goals_callback(self, msg):
        for goal in msg.goals:
            self.get_logger().info(f"Received goal: {goal.type}, {goal.object}, {goal.destination}")
            
            #Freeze user_position until TST has been generated
            user_position ={
                "x": self.current_x,
                "y": self.current_y
            }
            tst_filename = self.generate_tst(goal, user_position)
            self.get_logger().info(f"TST is generated")
           
            if tst_filename:
                self.execute_tst(tst_filename)
                self.get_logger().info(f"TST is executed")

        self.get_logger().info(f"Goals callback done")
             
                
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        

    def generate_tst(self, goal, user_position):
        # Wait for the SPARQL service to be available
        while not self.sparql_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for DB service...')
        
        # Query the database for destination or object details
        destination_info = None
        object_info = None

        #GET GOAL DESTINATION
        if goal.destination:
            if goal.destination.lower() == "user":
                destination_info = user_position
            else:
                self.get_logger().info('Running DB service... [destination STEP 1]')

                #STEP 1: get obj_id
                query_str = f"""
                SELECT ?obj_id WHERE {{
                    ?obj_id <http://www.ida.liu.se/~TDDE05/propertiestags> ?tag .
                    FILTER(STR(?tag) = "{goal.destination}")
                }}
                LIMIT 1
                """

                q = Query()
                q.graphnames = ['semanticobject']
                q.query = query_str

                req = QueryDatabase.Request()
                req.queries = [q]

                future = self.sparql_client.call_async(req)
                self.get_logger().error("spin_until_future start 1")
                rclpy.spin_until_future_complete(self, future)
                self.get_logger().error("spin_until_future end 1")

                result = future.result()

                if result is None or not result.success:
                    self.get_logger().error("Step 1 query failed")
                    return None

                data = json.loads(result.results[0])
                bindings = data["results"]["bindings"]

                if not bindings:
                    self.get_logger().warn(f"No object found for: {goal.destination}")
                    return None

                obj_id = bindings[0]["obj_id"]["value"]
                self.get_logger().info(f"Found obj_id: {obj_id}")

                #STEP 2: get location
                self.get_logger().info('Running DB service... [destination STEP 2]')

                query_str = f"""
                PREFIX gis: <http://www.ida.liu.se/~TDDE05/gis>

                SELECT ?x ?y WHERE {{
                    <{obj_id}> <http://www.ida.liu.se/~TDDE05/propertieslocation> ?loc .
                    ?loc <http://www.ida.liu.se/~TDDE05/gisx> ?x .
                    ?loc <http://www.ida.liu.se/~TDDE05/gisy> ?y .
                }}
                LIMIT 1
                """

                q = Query()
                q.graphnames = ['semanticobject']
                q.query = query_str

                req = QueryDatabase.Request()
                req.queries = [q]

                future = self.sparql_client.call_async(req)
                self.get_logger().error("spin_until_future start 2")
                rclpy.spin_until_future_complete(self, future)
                self.get_logger().error("spin_until_future end 2")
                result = future.result()

                if result is None or not result.success:
                    self.get_logger().error("Step 2 query failed")
                    return None

                data = json.loads(result.results[0])
                bindings = data["results"]["bindings"]

                if bindings:
                    x = float(bindings[0]["x"]["value"])
                    y = float(bindings[0]["y"]["value"])
                    destination_info = {"name": goal.destination, "x": x, "y": y}
                else:
                    self.get_logger().warn(f"No location found for: {goal.destination}")
                    return None
        

        #GET GOAL OBJECT
        if goal.object:
            self.get_logger().info('Running DB service... [object STEP 1]')

            #STEP 1: get obj_id from tag
            query_str = f"""
            SELECT ?obj_id WHERE {{
                ?obj_id a <vendingmachine> ;
                        <http://www.ida.liu.se/~TDDE05/propertiestags> ?tag .
                FILTER(STR(?tag) = "{goal.object.lower()}")
            }}
            LIMIT 1
            """

            q = Query()
            q.graphnames = ['semanticobject']
            q.query = query_str

            req = QueryDatabase.Request()
            req.queries = [q]

            future = self.sparql_client.call_async(req)
            self.get_logger().error("spin_until_future start 3")
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().error("spin_until_future end 3")

            result = future.result()

            if result is None or not result.success:
                self.get_logger().error("Object STEP 1 query failed")
                return None

            data = json.loads(result.results[0])
            bindings = data["results"]["bindings"]

            if not bindings:
                self.get_logger().warn(f"No object found for: {goal.object}")
                return None

            obj_id = bindings[0]["obj_id"]["value"]
            self.get_logger().info(f"Found object obj_id: {obj_id}")

            #STEP 2: get location
            self.get_logger().info('Running DB service... [object STEP 2]')

            query_str = f"""
            PREFIX gis: <http://www.ida.liu.se/~TDDE05/gis>

            SELECT ?x ?y WHERE {{
                <{obj_id}> <http://www.ida.liu.se/~TDDE05/propertieslocation> ?loc .
                ?loc <http://www.ida.liu.se/~TDDE05/gisx> ?x .
                ?loc <http://www.ida.liu.se/~TDDE05/gisy> ?y .
            }}
            LIMIT 1
            """

            q = Query()
            q.graphnames = ['semanticobject']
            q.query = query_str

            req = QueryDatabase.Request()
            req.queries = [q]

            future = self.sparql_client.call_async(req)
            self.get_logger().error("spin_until_future start 4")
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().error("spin_until_future end 4")
            result = future.result()

            if result is None or not result.success:
                self.get_logger().error("Object STEP 2 query failed")
                return None

            data = json.loads(result.results[0])
            bindings = data["results"]["bindings"]

            if bindings:
                x = float(bindings[0]["x"]["value"])
                y = float(bindings[0]["y"]["value"])
                object_info = {"name": goal.object, "x": x, "y": y}
            else:
                self.get_logger().warn(f"No location found for object: {goal.object}")
                return None
        
        
        # Generate TST based on goal type
        self.get_logger().info('Generating TST...')
        if goal.type == "goto" and destination_info:
            children = [
                {
                    "name": "drive-to",
                    "params": {
                        "p": {
                            "rostype": "Point",
                            "x": destination_info["x"],
                            "y": destination_info["y"],
                            "z": 0
                        }
                    }
                }
            ]
            tst = {"children": children, "name": "seq"}

        elif goal.type == "bring" and object_info and destination_info:
            children = [
                {
                    "name": "drive-to",
                    "params": {
                        "p": {
                            "rostype": "Point",
                            "x": object_info["x"],
                            "y": object_info["y"],
                            "z": 0
                        }
                    }
                },
                {
                    "name": "drive-to",
                    "params": {
                        "p": {
                            "rostype": "Point",
                            "x": destination_info["x"],
                            "y": destination_info["y"],
                            "z": 0
                        }
                    }
                }
            ]
            tst = {"children": children, "name": "seq"}

        elif goal.type == "explore":
            children = [
                {
                    "name": "explore",
                    "params": {"radius": 10.0,
                               "count": 8
                               }
                }
            ]
            tst = {"children": children, "name": "seq"}

        else:
            self.get_logger().warn(f"Unsupported goal type or missing parameters: {goal.type}")
            return None

        # Save the TST to a file
        self.get_logger().info('Saving TST')
        safe_dest = goal.destination.replace(" ", "_") if goal.destination else "unknown" #Spaces not allowed
        tst_filename = f"tst_{goal.type}_{safe_dest}.json"

        script_dir = os.path.dirname(os.path.abspath(__file__))

        tst_path = os.path.join(script_dir, tst_filename)

        with open(tst_path, "w") as f:
            json.dump(tst, f, indent=4)

        self.get_logger().info(f"TST generated: {tst_path}")
        return tst_path

    def execute_tst(self, tst_path):
        # Wait for the TST executor service to be available
        while not self.tst_executor_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for TST executor service...')

        # Call the TST executor service
        req = ExecuteTst.Request()
        req.tst_file = tst_path
        future = self.tst_executor_client.call_async(req)
        self.get_logger().error("spin_until_future start 5")
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().error("spin_until_future end 5")
        result = future.result()
        if result.success:
            self.get_logger().info(f"TST executed successfully: {tst_path}")
        else:
            self.get_logger().error(f"Failed to execute TST: {tst_path}")

def main(args=None):
    '''
    rclpy.init(args=args)
    decision_node = DecisionNode()
    
    #Multithreaded executor, because singlethreaded executor didn't seem to work
    
    executor = MultiThreadedExecutor()
    executor.add_node(decision_node)
    executor.spin()
    
    decision_node.destroy_node()
    rclpy.shutdown()
    '''
    rclpy.init(args=args)
    decision_node = DecisionNode()
    rclpy.spin(decision_node)
    decision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()