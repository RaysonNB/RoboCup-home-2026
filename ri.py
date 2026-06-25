#!/usr/bin/env python3
from pcms.pytorch_models import *
from std_srvs.srv import Empty
from pcms.pytorch_models import *
import rospy
import tf
import cv2
import os
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from RobotChassis import RobotChassis
import time
from piper_sdk_demo import *
import numpy as np

def get_yaw_from_odom():
    global _odom
    if _odom is None:
        return 0
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
    start = time.time()

    while not rospy.is_shutdown():
        yaw = get_yaw_from_odom()

        e = normalize_angle(angle - yaw)
        print("turn_to_yaw", yaw, "err", e, "time", time.time() - start)
        if abs(e) < 0.015 or time.time() - start > 15:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0, 0)


clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)


def speak(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 130 "{g}"')
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.5)


def callback_image1(msg):
    global _frame1
    _frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def normalize_angle(angle):
    while angle > 2 * np.pi:
        angle -= 2 * np.pi
    while angle < 2 * -np.pi:
        angle += 2 * np.pi
    return angle


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


if __name__ == '__main__':
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    chassis = RobotChassis()

    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    _frame1 = None
    _sub_down_cam_image1 = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image1)
    _depth1 = None
    _sub_down_cam_depth1 = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth1)
    _odom = None
    rospy.Subscriber("/odom", Odometry, callback_odom)
    tf_listener = tf.TransformListener()

    rate = rospy.Rate(20)
    depth_zero = 0
    zero_count = 0
    speak("I am ready")
    while not rospy.is_shutdown():
        if _frame1 is None: continue
        if _depth1 is None: continue
        frame = _frame1.copy()
        depth_frame = _depth1.copy()

        cx, cy = 640 // 2, 320 // 2
        depth = depth_frame[cy, cx]
        frame = cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f"{depth}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if depth_zero >= 400:
            speak("the door is open")
            break
        if depth == 0:
            depth_zero += 1
        else:
            depth_zero = 0
        if depth > 1700:
            speak("the door is open")
            break
        print("depthtt", depth)
        cv2.imshow("door_check", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    _1yaw = get_yaw_from_odom()
    for iii in range(round(200 / 100)):
        move(0.2, 0)

        time.sleep(0.5)
    turn_to(_1yaw, 0.4)
    for iii in range(round(300 / 100)):
        move(0.2, 0)
        time.sleep(0.5)
    clear_costmaps
    chassis.move_to(4.94, -0.391, -3.074)
    while not rospy.is_shutdown():
        # 4. Get the chassis status.
        code = chassis.status_code
        text = chassis.status_text
        if code == 3:
            break
        if code == 4:
            break
    speak("arrived")
    speak("If the robot inspection ends please press my emergency button")
    time.sleep(2)

    while True:
        print("waiting")
        time.sleep(3)
