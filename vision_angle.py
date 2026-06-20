#!/usr/bin/env python3.8
import rospy
import tf
import cv2
import numpy as np
import time
import os
from tf.transformations import euler_from_quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from pcms.openvino_models import HumanPoseEstimation
from RobotChassis import RobotChassis
import dynamic_reconfigure.client
import math


def angle_between(x1, y1, z1_rad, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    target_angle_rad = math.atan2(dx, dy)
    diff_rad = target_angle_rad - z1_rad
    return normalize_angle(diff_rad)
def disable_amcl_alignment():
    print("")


def enable_amcl_alignment():
    print("")
_image = None
_depth = None
CvBr = CvBridge()
net_pose = HumanPoseEstimation(device_name="GPU")
tf_listener = None


def callback_image(msg):
    global _image
    _image = CvBr.imgmsg_to_cv2(msg, "bgr8")


def callback_depth(msg):
    global _depth
    _depth = CvBr.imgmsg_to_cv2(msg, "32FC1")


def say(text):
    print(f"[Robot]: {text}")
    os.system(f"espeak \"{text}\"")


def move(dis, turn_val):
    fb = Twist()
    fb.linear.x = dis
    fb.angular.z = turn_val
    pub_cmd.publish(fb)


def get_yaw_from_odom():
    global _odom
    q = [
        _odom.pose.pose.orientation.x,
        _odom.pose.pose.orientation.y,
        _odom.pose.pose.orientation.z,
        _odom.pose.pose.orientation.w
    ]
    _, _, yaw = euler_from_quaternion(q)
    return yaw


def callback_odom(msg):
    global _odom
    _odom = msg


def turn_to(angle, speed):
    max_speed = 1.82
    start = rospy.get_time()

    while True:
        yaw = get_yaw_from_odom()
        e = normalize_angle(angle - yaw)

        if abs(e) < 0.01 or rospy.get_time() - start > 8:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0, 0)


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def turn(angle):
    # angle 已經是弧度值
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


def is_waving(pose):
    right_up = (pose[5][2] > 0 and pose[9][2] > 0 and pose[9][1] < pose[5][1] - 30)
    left_up = (pose[6][2] > 0 and pose[10][2] > 0 and pose[10][1] < pose[6][1] - 30)
    if right_up: return "right"
    if left_up: return "left"
    return None


def get_map_coords(target_angle,dist_meters, cx=None, img_width=640):
    try:
        (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        robot_yaw = euler[2]
        angle_offset = 0.0
        if cx is not None:
            fov_rad = math.radians(60)
            center_x = img_width / 2.0
            angle_offset = -((cx - center_x) / center_x) * (fov_rad / 2.0)

        total_yaw = target_angle*np.pi/180

        target_x = trans[0] + dist_meters 
        target_y = trans[1] + dist_meters / np.tan(total_yaw)
        return target_x, target_y, robot_yaw
    except Exception as e:
        rospy.logerr(f"TF Error: {e}")
        return None, None, None

def get_real_xyz(dp, x, y, num):
    a1 = 49.5
    b1 = 60.0
    if num == 2:
        a1 = 55.0
        b1 = 86.0
    a = a1 * np.pi / 180
    b = b1 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d

if __name__ == "__main__":
    rospy.init_node("robot_navigator_full")
    tf_listener = tf.TransformListener()
    _odom = None
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    chassis = RobotChassis()

    rospy.sleep(1)
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

    startx, starty, start_yaw = get_map_coords(0,0)
    print("start:", startx, starty, start_yaw)
    state = 0

    say("System ready.")
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()
        if _image is None or _depth is None: continue
        image = _image.copy()
        depth=_depth.copy()
        if state == 0:
            poses = net_pose.forward(image)
            for pose in poses:
                hand = is_waving(pose)
                if hand:
                    p5, p6 = pose[5], pose[6]
                    cx, cy = int((p5[0] + p6[0]) / 2), int((p5[1] + p6[1]) / 2)
                    if cy >= _depth.shape[0] or cx >= _depth.shape[1]: continue
                    dist = _depth[cy][cx] / 1000.0
                    ry, rx, d = get_real_xyz(depth, cx, cy,2)
                    print("real factor",ry, rx, d)
                    
                    target_angle = math.atan2(rx, ry)*180/np.pi
                    if target_angle>90: target_angle=target_angle-180
                    print("vision angle",target_angle)
                    if 0.5 < dist < 10.0:
                        print("Distance to human:", dist)
                        img_w = image.shape[1]
                        hx, hy, _ = get_map_coords(target_angle,dist, cx=cx, img_width=img_w)
                        print("Human map pos:", hx, hy)
                        x1, y1, z1_rad = get_map_coords(target_angle,0)
                        rotation = angle_between(x1, y1, z1_rad, hx, hy)
                        if rotation>3.14/2.0: rotation=rotation-3.14
                        print("Calculated turn rad:", rotation)
                        print("Calculated turn angle:", rotation*180/np.pi)
                        turn(-rotation)
                        time.sleep(5)

        cv2.imshow("Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
