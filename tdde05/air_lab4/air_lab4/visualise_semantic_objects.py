import rclpy
import rclpy.node
import visualization_msgs.msg
import geometry_msgs.msg
import std_msgs.msg

import json
from ros_kdb_interfaces.srv import QueryDatabase
from ros_kdb_interfaces.msg import Query

display_marker_pub = None
query_client = None
node = None
query_future = None


def create_point(x, y, z):
    msg = geometry_msgs.msg.Point()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = z
    return msg


def create_color(r, g, b, a):
    msg = std_msgs.msg.ColorRGBA()
    msg.r = r
    msg.g = g
    msg.b = b
    msg.a = a
    return msg


def timer_callback():
    global display_marker_pub, query_client, node, query_future
    
    #DEBUG logging: does timer_callback() run at all?
    #node.get_logger().info("Timer callback running!")

    #QUERY DATABASE
    #If no request yet, send one
    if query_future is None:
        query_str = """
        PREFIX gis: <http://www.ida.liu.se/~TDDE05/gis>
        PREFIX properties: <http://www.ida.liu.se/~TDDE05/properties>

        SELECT ?obj_id ?class ?x ?y WHERE {
            ?obj_id a ?class ;
                    properties:location [ gis:x ?x; gis:y ?y ] .
        }
        """

        q = Query()
        q.graphnames = ['semanticobject']
        q.query = query_str

        req = QueryDatabase.Request()
        req.queries = [q]

        query_future = query_client.call_async(req)

    #If request not ready, wait
    if not query_future.done():
        return

    #Now result is ready
    result = query_future.result()
    query_future = None #reset for next iteration


    #PARSE RESULTS
    try:
        data = json.loads(result.results[0])
        bindings = data["results"]["bindings"]
    except:
        bindings = []

    #DEBUG LOGGING: gets to parsing?
    #node.get_logger().info(f"Bindings length: {len(bindings)}")

    #CREATE MARKER
    marker = visualization_msgs.msg.Marker()
    marker.id     = 1242 # identifier the marker, should be unique
    marker.header.frame_id = "map"
    marker.type   = visualization_msgs.msg.Marker.CUBE_LIST
    marker.action = 0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0

    for row in bindings:
        x = row["x"]["value"]
        y = row["y"]["value"]
        klass = row["class"]["value"]

        marker.points.append(create_point(x, y, 0.0))

        #choose color based on class
        if "human" in klass:
            marker.colors.append(create_color(1.0, 0.0, 0.0, 1.0))
        elif "table" in klass:
            marker.colors.append(create_color(0.0, 0.0, 1.0, 1.0))
        else:
            marker.colors.append(create_color(0.0, 1.0, 0.0, 1.0))

    marker_array = visualization_msgs.msg.MarkerArray()
    marker_array.markers = [marker]

    display_marker_pub.publish(marker_array)


def main():
    global display_marker_pub, query_client, node, query_future

    rclpy.init()
    node = rclpy.node.Node('visualise_semantic_objects')

    query_client = node.create_client(QueryDatabase, '/kdb_server/sparql_query')

    while not query_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for database service...')

    display_marker_pub = node.create_publisher(
        visualization_msgs.msg.MarkerArray, 'semantic_sensor_visualisation', 10
    )

    timer = node.create_timer(0.5, timer_callback)

    rclpy.spin(node)


if __name__ == '__main__':
    main()