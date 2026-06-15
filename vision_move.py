#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ultralytics import YOLO
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

model = YOLO("yolov8n-pose.pt")


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
            result = model(image)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                person_keypoints = result.keypoints.xy.cpu().numpy()

                for keypoint in person_keypoints:
                    if len(keypoint) > 10:
                        nose = keypoint[0]
                        lwrist, rwrist = keypoint[9], keypoint[10]

                        nose_x, nose_y = map(int, nose)
                        lwrist_y = int(lwrist[1])
                        rwrist_y = int(rwrist[1])

                        if (lwrist_y < nose_y and lwrist_y > 0) or (rwrist_y < nose_y and rwrist_y > 0):
                            if 0 <= nose_x < w and 0 <= nose_y < h:
                                x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.imwrite("image/proof1.jpg", image)

                                person_depth = depth[nose_y][nose_x]
                                if person_depth and person_depth > 0 and np.isfinite(person_depth):
                                    say("Found customer waving hand")
                                    state = 2
                                    break

        elif state == 2:
            result = model(image)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                person_keypoints = result.keypoints.xy.cpu().numpy()
                nose = person_keypoints[0][0]
                nose_x = nose[0]

                image_mid = w // 2
                diff = 15

                if image_mid - diff <= nose_x <= image_mid + diff:
                    say("Facing the customer")
                    state = 3
                else:
                    if nose_x > image_mid:
                        turn(-10 / 180 * np.pi)
                    else:
                        turn(10 / 180 * np.pi)

        elif state == 3:
            human_depth = depth[h // 2][w // 2]
            safe_distance = 0.8

            if human_depth > safe_distance and np.isfinite(human_depth):
                base_theta = normalize_angle(get_yaw_from_odom())
                robotx, roboty = get_pos()
                target_depth = human_depth - safe_distance
                num_steps = 3

                angle_offsets = [0.0, 15.0 / 180.0 * np.pi, -15.0 / 180.0 * np.pi]
                navigation_success = False

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
                    say("Arrived safely in front of the customer")
                    break
                else:
                    say("All paths blocked. Looking for a new customer.")
                    state = 1
            else:
                print("Error: check depth or distance")
                state = 1

        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
