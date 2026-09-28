import math
import os
import json
import threading
import time

import rclpy
import rclpy.time
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import rclpy.duration
from copy import deepcopy

from air_simple_sim_msgs.msg import SemanticObservation
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3

HUMAN_CLASS = "human"
HUMAN_FILE = "/tmp/humans.json"
MIN_MOVE_FOR_HEADING = 0.15  #metres, movements smaller than this are considered just noise
HUMAN_TIMEOUT = 1.0          #seconds before a human that left sensor range gets dropped

PERSONAL_SPACE = 1.2   #metres — personal space threshold (cited in lectures)
INTIMATE_SPACE = 0.45  #metres — intimate space threshold (cited in lectures)


class HumanTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._lock = threading.Lock()
        self._humans = {}  #keyed by uuid, stores position + heading per human

        self.create_subscription(
            SemanticObservation,
            'semantic_sensor_hf',  #10hz semantic detections
            self._observation_callback,
            10
        )
        self._marker_pub = self.create_publisher(MarkerArray, 'human_markers', 10)

        self._dist_sum = 0.0
        self._dist_samples = 0
        self._time_personal = 0.0
        self._time_intimate = 0.0
        self._last_check_t = None
        self.create_timer(0.5, self._check_robot_distance)

        self.get_logger().info('HumanTracker ready — listening on semantic_sensor_hf')

    def _observation_callback(self, msg):
        if msg.klass != HUMAN_CLASS:
            return  #ignore chairs, desks, vending machines etc

        #Ugly hack to avoid the "extrapolation into the future" tf2 error
        try:
            point_latest = deepcopy(msg.point)
            point_latest.header.stamp = rclpy.time.Time().to_msg()
            pt = self.tf_buffer.transform(
                point_latest, 'map',
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}', throttle_duration_sec=2.0)
            return

        x, y = pt.point.x, pt.point.y
        uid = msg.uuid

        #lock so the write thread and this callback can't touch _humans at the same time
        with self._lock:
            prev = self._humans.get(uid)
            if prev:
                dx = x - prev['x']
                dy = y - prev['y']
                if math.hypot(dx, dy) > MIN_MOVE_FOR_HEADING:
                    #human moved enough for us to want to calculate reliable heading estimate
                    heading = math.atan2(dy, dx)
                    cos_h = math.cos(heading)
                    sin_h = math.sin(heading)
                else:
                    #barely moved, keep the last known heading
                    cos_h = prev['cos_h']
                    sin_h = prev['sin_h']
            else:
                #first time seeing this human, assume heading is east until we know better
                cos_h, sin_h = 1.0, 0.0

            self._humans[uid] = {
                'x': x, 'y': y,
                'cos_h': cos_h, 'sin_h': sin_h,
                'name': msg.tags[0] if msg.tags else uid,
                'last_seen': time.time()
            }

        self._write_file()
        self._publish_markers()

    #This method publishes markers of humans in Rviz. To see the humans in real-time,
    #add a MarkerArray in Rviz and subcsribe to /human_markers.
    def _publish_markers(self):
        with self._lock:
            humans = list(self._humans.values())

        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for i, h in enumerate(humans):
            
            #orange cylinder for human body
            body = Marker()
            body.header.frame_id = 'map'
            body.header.stamp = now
            body.ns = 'humans'
            body.id = i * 2
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = h['x']
            body.pose.position.y = h['y']
            body.pose.position.z = 0.9
            body.pose.orientation.w = 1.0
            body.scale = Vector3(x=0.2, y=0.2, z=1.0)
            body.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.8)
            body.lifetime.sec = int(HUMAN_TIMEOUT)
            markers.markers.append(body)

            #yellow arrow for human heading
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = now
            arrow.ns = 'humans'
            arrow.id = i * 2 + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = h['x']
            arrow.pose.position.y = h['y']
            arrow.pose.position.z = 1.4
            yaw = math.atan2(h['sin_h'], h['cos_h'])
            
            #conversion from yaw to orientations z and w
            arrow.pose.orientation.z = math.sin(yaw / 2)
            arrow.pose.orientation.w = math.cos(yaw / 2)
            arrow.scale = Vector3(x=0.4, y=0.07, z=0.07)
            arrow.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
            arrow.lifetime.sec = int(HUMAN_TIMEOUT)
            markers.markers.append(arrow)

        self._marker_pub.publish(markers)

    #calculate distance from turtlebot to human for logging purposes.
    #I used this for the data collection for the report.
    def _check_robot_distance(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'Robot TF lookup failed: {e}', throttle_duration_sec=5.0)
            return

        rx = t.transform.translation.x
        ry = t.transform.translation.y

        now = time.time()
        dt = (now - self._last_check_t) if self._last_check_t is not None else 0.0
        self._last_check_t = now

        with self._lock:
            humans = list(self._humans.values())

        if not humans:
            return

        min_d = min(math.hypot(h['x'] - rx, h['y'] - ry) for h in humans)

        self._dist_sum += min_d
        self._dist_samples += 1

        if min_d < PERSONAL_SPACE:
            self._time_personal += dt
        if min_d < INTIMATE_SPACE:
            self._time_intimate += dt

    def _write_file(self):
        now = time.time()
        #same lock as above - only one of these two blocks runs at a time
        with self._lock:
            #drop humans that haven't been seen recently
            active = {uid: h for uid, h in self._humans.items()
                      if now - h['last_seen'] < HUMAN_TIMEOUT}
            self._humans = active
            #remove last_seen attribute before writing, the critic doesn't need it
            data = [{k: v for k, v in h.items() if k != 'last_seen'}
                    for h in active.values()]

        #write to a temp file first then rename - this way the critic
        #never reads a half-written file if we get interrupted mid-write
        tmp = HUMAN_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, HUMAN_FILE)


def main():
    rclpy.init()
    node = HumanTracker()
    try:
        rclpy.spin(node)
    finally:
        avg = node._dist_sum / node._dist_samples if node._dist_samples > 0 else float('nan')
        node.get_logger().info(
            f'\n=== SESSION SUMMARY ===\n'
            f'  Avg distance to nearest human:      {avg:.3f} m  (n={node._dist_samples})\n'
            f'  Time within personal space (1.2m):  {node._time_personal:.1f} s\n'
            f'  Time within intimate space (0.45m): {node._time_intimate:.1f} s'
        )
        rclpy.shutdown()