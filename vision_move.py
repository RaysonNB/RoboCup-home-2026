#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from pcms.openvino_models import HumanPoseEstimation
import cv2
from RobotChassis import RobotChassis
from tf.transformations import euler_from_quaternion
import numpy as np
import math
import os

_image = None
_depth = None
_odom = None
CvBr = CvBridge()

net_pose = HumanPoseEstimation(device_name="GPU")


def callback_image(msg):
    global _image
    _image = CvBr.imgmsg_to_cv2(msg, "bgr8")


def callback_depth(msg):
    global _depth
    _depth = CvBr.imgmsg_to_cv2(msg)


def callback_odom(msg):
    global _odom
    _odom = msg


def say(text):
    print("[Robot says]:", text)
    os.system(f"espeak \"{text}\"")
    rospy.sleep(2)


def move(dis, turn):
    fb = Twist()
    fb.linear.x = dis
    fb.angular.z = turn
    pub_cmd.publish(fb)


def get_yaw_from_odom():
    global _odom
    q = [
        _odom.pose.pose.orientation.x,
        _odom.pose.pose.orientation.y,
        _odom.pose.pose.orientation.z,
        _odom.pose.pose.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    return yaw


def get_pos():
    global _odom
    px = _odom.pose.pose.position.x
    py = _odom.pose.pose.position.y
    return px, py


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def turn_to(angle: float, speed: float):
    global _odom
    max_speed = 1.82
    limit_time = 8
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        if _odom is None:
            move(0.0, 0.0)
            return

        yaw = get_yaw_from_odom()
        e = normalize_angle(angle - yaw)

        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0.0, 0.0)


def turn(angle: float):
    global _odom
    if _odom is None:
        return
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


if __name__ == "__main__":
    rospy.init_node("robot_seek_and_go")
    rospy.loginfo("Node started")
    rate = rospy.Rate(20)

    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    rospy.Subscriber("/odom", Odometry, callback_odom)

    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    chassis = RobotChassis()

    state = 0
    start_pos = None
    human_pos = None
    target_cx = w // 2

    say("Demo start.")

    while not rospy.is_shutdown():
        rate.sleep()

        if _image is None or _depth is None or _odom is None:
            continue

        image = _image.copy()
        depth = _depth.copy()
        h, w = image.shape[0], image.shape[1]

        if state == 0:
            say("mission start")
            start_pos = get_pos()
            state = 1

        elif state == 1:
            poses = net_pose.forward(image)
            found_target = False

            for pose in poses:
                if (pose[5][2] == 0 and pose[9][2] == 0) or (pose[6][2] == 0 and pose[10][2] == 0):
                    continue

                p5 = list(map(int, pose[5][:2]))
                p6 = list(map(int, pose[6][:2]))
                p9 = list(map(int, pose[9][:2]))
                p10 = list(map(int, pose[10][:2]))

                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2

                if 0 <= cx < w and 0 <= cy < h:
                    person_depth = depth[cy][cx]
                    if person_depth and person_depth > 0 and np.isfinite(person_depth):
                        if ((pose[5][2] != 0 and pose[9][2] != 0 and p5[1] > p9[1]) or 
                            (pose[6][2] != 0 and pose[10][2] != 0 and p6[1] > p10[1])):
                            
                            cv2.circle(image, p5, 5, (0, 0, 255), -1)
                            cv2.circle(image, p6, 5, (0, 0, 255), -1)
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.imwrite("image/proof1.jpg", image)
                            
                            target_cx = cx
                            say("Found customer waving hand")
                            state = 2
                            found_target = True
                            break
            
            if found_target:
                continue

        elif state == 2:
            poses = net_pose.forward(image)
            if len(poses) > 0:
                p5 = list(map(int, poses[0][5][:2]))
                p6 = list(map(int, poses[0][6][:2]))
                target_cx = (p5[0] + p6[0]) // 2

            image_mid = w // 2
            diff = 25

            if image_mid - diff <= target_cx <= image_mid + diff:
                say("Facing the customer")
                state = 3
            else:
                if target_cx > image_mid:
                    turn(-10 / 180 * np.pi)
                else:
                    turn(10 / 180 * np.pi)

        elif state == 3:
            navigation_success = False

            for attempt in range(2):
                human_depth = depth[h // 2][w // 2]
                safe_distance = 0.8

                if human_depth > safe_distance and np.isfinite(human_depth):
                    base_theta = normalize_angle(get_yaw_from_odom())
                    robotx, roboty = get_pos()
                    target_depth = human_depth - safe_distance
                    num_steps = 3

                    angle_offsets = [0.0, 15.0 / 180.0 * np.pi, -15.0 / 180.0 * np.pi]

                    for offset in angle_offsets:
                        current_theta = normalize_angle(base_theta + offset)
                        waypoints = []

                        for i in range(1, num_steps + 1):
                            step_depth = (target_depth / num_steps) * i
                            step_x = robotx + step_depth * np.cos(current_theta)
                            step_y = roboty + step_depth * np.sin(current_theta)
                            waypoints.append((step_x, step_y))

                        path_blocked = False
                        for point in waypoints:
                            chassis.move_to(*point, 0)
                            if chassis.status_code == 0 or chassis.status_code == 4:
                                path_blocked = True
                                break

                        if not path_blocked:
                            navigation_success = True
                            break
                        else:
                            print(f"Path with offset {offset} failed. Trying another route.")

                    if navigation_success:
                        break
                    elif attempt == 0:
                        say("Path blocked. Shifting position to retry calculation.")
                        turn(30.0 / 180.0 * np.pi)
                        rospy.sleep(1.0)
                        
                        if _image is not None and _depth is not None:
                            image = _image.copy()
                            depth = _depth.copy()
                else:
                    print("Error: check depth or distance")
                    break

            if navigation_success:
                say("Arrived safely in front of the customer")
                break
            else:
                say("All fallback paths failed twice. Looking for a new customer.")
                state = 1

        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
