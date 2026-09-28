import threading
import rclpy
import traceback
import sys

from rclpy.node import Node
import rclpy.executors
import rclpy.callback_groups

import TstML
import TstML.Executor

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

import rclpy.duration
import json
from ros_kdb_interfaces.msg import Query
from ros_kdb_interfaces.srv import QueryDatabase, InsertTriples
from air_simple_sim_msgs.msg import SemanticObservation


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


class RecordSemanticExecutor(TstML.Executor.AbstractNodeExecutor):
  def __init__(self, node, context):
    super(TstML.Executor.AbstractNodeExecutor, self).__init__(node,
          context)

    self.ros_node = Node(gen_name("record_semantic"))
    
    self.tf_buffer = Buffer()
    self.tf_listener = TransformListener(self.tf_buffer, self.ros_node)
    
    #Parameters given in lab instructions
    self.topic = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "topic")
    if not self.topic.startswith('/'):
      self.topic = '/' + self.topic #Dirty fix because ros2 couldn't find topic otherwise 
      
    self.graphname = self.node().getParameter(TstML.TSTNode.ParameterType.Specific, "graphname")

    self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
    
    self.query_client = self.ros_node.create_client(
        QueryDatabase, '/kdb_server/sparql_query', callback_group=self.callback_group
    )
    
    self.insert_client = self.ros_node.create_client(
        InsertTriples, '/kdb_server/insert_triples', callback_group=self.callback_group
    )
    
    #Wait 1 second for services
    while not self.query_client.wait_for_service(timeout_sec=1.0):
        self.ros_node.get_logger().info('Waiting for query service...')

    while not self.insert_client.wait_for_service(timeout_sec=1.0):
        self.ros_node.get_logger().info('Waiting for insert service...')
    
    self.subscription = self.ros_node.create_subscription(
        SemanticObservation, self.topic, self.semantic_callback, 10
    )
    
    #Executor stuff, same as our other executors in lab3
    self.executor = rclpy.executors.MultiThreadedExecutor()
    self.executor.add_node(self.ros_node)
    self.thread = threading.Thread(target=self.executor.spin)
    self.thread.start()
    
  def finalise(self):
    self.executor.shutdown()

  def start(self):
    self.ros_node.get_logger().info("RecordSemanticExecutor started")
    return TstML.Executor.ExecutionStatus.Started()
  
  def semantic_callback(self, msg):
    #Fetch data from message
    obj_id = msg.uuid
    obj_class = msg.klass
    
    try:
      transformed_point = self.tf_buffer.transform(
        msg.point, "map", timeout=rclpy.duration.Duration(seconds=1.0)
      )
      
      x = transformed_point.point.x
      y = transformed_point.point.y
      
    except TransformException as ex:
      self.ros_node.get_logger().info(f'Could not transform point: {ex}')
      return
    
    #DEBUG LOGGING: do we receive data?
    self.ros_node.get_logger().info(f"Received: {obj_id} ({obj_class}) at ({x}, {y})")
    
    #CHECK IF OBJECT EXISTS IN DATABASE
    query = f"""
    PREFIX gis: <http://www.ida.liu.se/~TDDE05/gis>
    PREFIX properties: <http://www.ida.liu.se/~TDDE05/properties>
    SELECT ?x ?y WHERE {{
        <{obj_id}> a <{obj_class}> ;
        properties:location [ gis:x ?x; gis:y ?y ] . 
        }}
    """
    
    q = Query()
    q.graphnames = [self.graphname]
    q.query = query
    q.bindings = ""
    
    req = QueryDatabase.Request()
    req.queries = [q]
    
    future = self.query_client.call_async(req)
    self.executor.spin_until_future_complete(future)
    
    result = future.result()
    
    try: #bindings = list of result rows from our query
        data = json.loads(result.results[0])
        bindings = data["results"]["bindings"]
    except:
        bindings = []
        
    #Do nothing if already exists in DB
    if len(bindings) > 0:
        #DEBUG LOGGING: object already in DB, skipping insert
        self.ros_node.get_logger().info(f"Already exists: {obj_id}")
        return
    
    #INSERT NEW OBJECT
    ttl = f"""
    @prefix gis: <http://www.ida.liu.se/~TDDE05/gis> .
    @prefix properties: <http://www.ida.liu.se/~TDDE05/properties> .

    <{obj_id}> a <{obj_class}> ;
        properties:location [ gis:x {x}; gis:y {y} ] .
    """
    
    insert_req = InsertTriples.Request()
    insert_req.graphname = self.graphname
    insert_req.format = "ttl"
    insert_req.content = ttl
    
    future = self.insert_client.call_async(insert_req)
    self.executor.spin_until_future_complete(future)
    
    #DEBUG LOGGING: is data actually inserted?
    self.ros_node.get_logger().info(f"Inserted: {obj_id}") 

  def pause(self):
    self.ros_node.get_logger().info('Pause is not possible.')
    return TstML.Executor.ExecutionStatus.Running()
  def resume(self):
    return TstML.Executor.ExecutionStatus.Running()
  def stop(self):
    return TstML.Executor.ExecutionStatus.Finished()
  def abort(self):
    return TstML.Executor.ExecutionStatus.Aborted()