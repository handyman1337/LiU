import json
import rclpy
from rclpy.node import Node
from ros_kdb_interfaces.srv import QueryDatabase
from ros_kdb_interfaces.msg import Query


def main():
    rclpy.init()
    node = Node('generate_tst')

    client = node.create_client(QueryDatabase, '/kdb_server/sparql_query')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for DB service...')

    #QUERY ALL HUMANS
    query_str = """
    PREFIX gis: <http://www.ida.liu.se/~TDDE05/gis>
    PREFIX properties: <http://www.ida.liu.se/~TDDE05/properties>

    SELECT ?x ?y WHERE {
        ?obj_id a ?class ;
                properties:location [ gis:x ?x; gis:y ?y ] .
        FILTER(CONTAINS(STR(?class), "human"))
    }
    """

    q = Query()
    q.graphnames = ['semanticobject']
    q.query = query_str

    req = QueryDatabase.Request()
    req.queries = [q]

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    result = future.result()

    data = json.loads(result.results[0])
    bindings = data["results"]["bindings"]

    #GENERATE TST
    children = []

    for row in bindings:
        x = float(row["x"]["value"])
        y = float(row["y"]["value"])

        children.append({
            "name": "drive-to",
            "params": {
                "p": {
                    "rostype": "Point",
                    "x": x,
                    "y": y,
                    "z": 0
                }
            }
        })

    tst = {
        "children": children,
        "name": "seq"
    }

    #SAVE FILE
    with open("generated_tst.json", "w") as f:
        json.dump(tst, f, indent=4)

    node.get_logger().info("TST generated!")

    rclpy.shutdown()


if __name__ == "__main__":
    main()