#!/usr/bin/env python3.8
from ultralytics import YOLO
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import subprocess
from mr_voice.msg import Voice
from playsound import playsound
import requests
from datetime import datetime
import base64
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
import json

import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from mr_voice.msg import Voice
from ultralytics import YOLO
import cv2
from std_srvs.srv import Empty
from RobotChassis import RobotChassis # Your chassis driver
from tf.transformations import euler_from_quaternion
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pyautogui
from piper_sdk import *
import os
import requests
import json
import re
import base64

from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest


# https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD
import time
from piper_sdk import *
import numpy as np 
import matplotlib.pyplot as plt
import random
import json
clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

tf_listener = None
class KindaPiper(object):        # front/end: x, up/down: y, left/right: z
    L0 = 123.00
    L1 = 285.03
    L2 = 250.75
    L3 =  91.00
    JOINTS_LIMIT = (
        (-2.6179, 2.6179),
        (0.0, 3.14),
        (-2.967, 0.0),
        (-1.745, 1.745),
        (-1.22, 1.22),
        (-2.09439, 2.09439)
    )
    GRIPPER_LIMIT = (0, 1)

    def __init__(self, port="can0"):
        super().__init__()
        self.port = port 

    def connect(self, port=None):
        if port is not None: self.port = port 
        self.piper = C_PiperInterface_V2(self.port)
        time.sleep(0.1)
        # self.piper.DisconnectPort()

        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print(f"Port {self.port} connect success.")

        self.piper.EnableArm(7, 0x02)

    def emergency_stop(self):
        self.piper.EmergencyStop(0x01)

    def resume(self):
        self.piper.EmergencyStop(0x02)

    def reset(self):
        self.piper.ResetPiper()

    def init(self):
        self.piper.PiperInit()

    def check_joints(self, joints):
        if len(joints) != 6:
            print("len(joints) should be 6.")
            return -1, []

        JOINTS_LIMIT = self.JOINTS_LIMIT
        angles = [round(i * 180 / np.pi, 2) for i in joints]
        res = 0
        for i in range(6):
            if not (JOINTS_LIMIT[i][0] <= joints[i] <= JOINTS_LIMIT[i][1]):
                print(f"joint_{i+1} angle problem:", joints[i], angles[i])
                joints[i] = max(JOINTS_LIMIT[i][0], min(joints[i], JOINTS_LIMIT[i][1]))
                res = -2
        return res, joints

    def move_joint(self, joints, speed=30, wait_util_finish=True):
        res, joints = self.check_joints(joints)
        if res == -1: return -1
        # print("move to", [round(i, 2) for i in joints])

        joints = [round(x * 1000 * 180 / np.pi) for x in joints]
        self.piper.ModeCtrl(0x01, 0x01, speed, 0x00)
        self.piper.JointCtrl(*joints)

        while wait_util_finish:
            time.sleep(0.005)
            js = self.piper.GetArmJointMsgs().joint_state
            jx = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
            max_e = np.inf 
            max_e = max([abs(x1 - x2) for x1, x2 in zip(jx, joints)])
            if max_e < 1000:
                wait_util_finish = False
        return res

    def move_to_endpoint2(self, x, y, z, rx, ry, rz, speed=30, wait_util_finish=True, n=1):
        x1 = x - self.L3 * np.cos(rz) * np.cos(ry)
        z1 = z - self.L3 * np.cos(rx) * np.sin(ry)
        y1 = y - self.L3 * np.cos(rx) * np.sin(rz)
        res, j1 = self.calc_joints(x1, y1, z1, rx, ry, rz)

        j0 = np.array(self.get_joints())
        j1 = np.array(j1)
        dj = (j1 - j0) / n
        for i in range(1, n):
            ji = j0 + dj * i 
            self.move_joint(ji, speed, wait_util_finish)
            time.sleep(0.005)
        return self.move_joint(j1, speed, wait_util_finish)

    def calc_joints(self, x, y, z, rx, ry, rz):
        L0 = self.L0
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3

        rz = rz - 5 * np.pi / 180

        if not (L1 + L2) >= (x ** 2 + y ** 2 + z ** 2) ** 0.5 >= 50:
            print("cannot reach that point!", x, y, z)
            return -1, []
        
        c = np.arctan2(z, x)
        x1 = (x * x + z * z) ** 0.5
        if x1 < 0: x1 = -x1
        y1 = y
        z1 = 0

        Lk = (x1 * x1 + y1 * y1) ** 0.5
        b = np.pi - np.arccos((L1 ** 2 + L2 ** 2 - Lk ** 2) / (2 * L1 * L2))
        a = np.pi / 2 - np.arctan2(y1, x1) - np.arccos((L1 ** 2 + Lk ** 2 - L2 ** 2) / (2 * L1 * Lk))
        a = a + np.pi / 2 - 7.78 * np.pi / 180
        b = b - np.pi + 17.5 * np.pi / 180

        # print("angles:", [round(x * 180 / np.pi, 2) for x in [c, a, b]])
        x2 = x1 + L3 * np.cos(-(a + b))
        y2 = y1 + L3 * np.sin(-(a + b))
        z2 = z1

        x3 = x2 * np.cos(c) + z2 * np.sin(c)
        y3 = y2 
        z3 = x2 * np.sin(c) + z2 * np.cos(c)
        
        x4 = x + L3 * np.cos(rz) * np.cos(ry)
        z4 = z + L3 * np.cos(rx) * np.sin(ry)
        y4 = y + L3 * np.cos(rx) * np.sin(rz)
        # print(x, y, z, rx, ry, rz)
        # print(x4, y4, z4)
        # input()

        x5 = x1 * np.cos(c) + z1 * np.sin(c)
        y5 = y1 
        z5 = x1 * np.sin(c) + z1 * np.cos(c)

        # print("x0:", [round(i, 2) for i in [x, y, z]])
        # print("x1:", [round(i, 2) for i in [x1, y1, z1]])
        # print("x2:", [round(i, 2) for i in [x2, y2, z2]])
        # print("x3:", [round(i, 2) for i in [x3, y3, z3]])
        # print("x4:", [round(i, 2) for i in [x4, y4, z4]])
        # print("x5:", [round(i, 2) for i in [x5, y5, z5]])
        # print("L3:", ((x3 - x) ** 2 + (y3 - y) ** 2 + (z3 - z) ** 2) ** 0.5)
        # print("L4:", ((x4 - x) ** 2 + (y4 - y) ** 2 + (z4 - z) ** 2) ** 0.5)

        rx = self.calculate_rotation_angle(
            [x5, y5, z5],
            [x3, y3, z3],
            [x4, y4, z4]
        )
        ry = self.calculate_angle_B(
            [x3, y3, z3],
            [x5, y5, z5],
            [x4, y4, z4]
        )
        if abs(rx - np.pi) < abs(rx): 
            rx = rx - np.pi
            ry = -ry
        if z4 > 0: rx = -rx
        rz = -rx + np.sin(c) * 1.0
        return 0, [c, a, b, rx, ry, rz]

    def calculate_rotation_angle(self, A, B, C):
        O = np.array([0, 0, 0])
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        
        AB = B - A
        AB_norm = np.linalg.norm(AB)
        if AB_norm == 0:
            raise ValueError("A 點與 B 點重合，無法構成旋轉軸。")
        
        n1 = np.cross(A - O, B - O)
        n2 = np.cross(B - A, C - A)
        if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
            raise ValueError("無法構成平面（有點共線的情況）。")
        
        u = AB / AB_norm
        n1_perp = n1 - np.dot(n1, u) * u
        n2_perp = n2 - np.dot(n2, u) * u
        
        n1_perp_norm = np.linalg.norm(n1_perp)
        n2_perp_norm = np.linalg.norm(n2_perp)
        if n1_perp_norm == 0 or n2_perp_norm == 0:
            return 0.0
            
        cos_theta = np.dot(n1_perp, n2_perp) / (n1_perp_norm * n2_perp_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    def calculate_angle_B(self, A, B, C):
        A = np.array(A)
        B = np.array(B)  
        C = np.array(C)
        
        BA = A - B
        BC = C - B

        BA_norm = np.linalg.norm(BA)
        BC_norm = np.linalg.norm(BC)
        if BA_norm == 0 or BC_norm == 0:
            raise ValueError("A 或 C 不能與 B 重合")
        
        dot_product = np.dot(BA, BC)
        cos_B = dot_product / (BA_norm * BC_norm)
        cos_B = np.clip(cos_B, -1.0, 1.0)
        return np.arccos(cos_B)

    def move_point(self, x, y, z, rx, ry, rz, speed=30, wait_util_finish=True):
        res, joints = self.calc_joints(x, y, z, rx, ry, rz)
        return self.move_joint(joints, speed, wait_util_finish)

    def mdh_matrix(self, alpha_prev, a_prev, d_i, theta_i):
        ct = np.cos(theta_i)
        st = np.sin(theta_i)
        ca = np.cos(alpha_prev)
        sa = np.sin(alpha_prev)
        
        return np.array([
            [ct,        -st,         0,   a_prev],
            [st * ca,    ct * ca,   -sa,  -sa * d_i],
            [st * sa,    ct * sa,    ca,   ca * d_i],
            [0,          0,          0,   1]
        ])

    def forward_kinematics(self, joint_angles):
        mdh_params = [
            [0,         0,         self.L0,   0],
            [-np.pi/2,  0,         0,       -172.22 / 180 * np.pi],
            [0,         self.L1,   0,       -102.78 / 180 * np.pi],
            [np.pi/2,   -0.021984, self.L2, 0],
            [-np.pi/2,  0,         0,       0],
            [np.pi/2,   0,         self.L3,   0]
        ]

        joint_radians = [i * np.pi / 180 for i in joint_angles]

        T_base_to_end = np.identity(4)
        for i in range(6):
            alpha_prev, a_prev, d_i, offset = mdh_params[i]
            theta_i = joint_radians[i] + offset
            T_joint = self.mdh_matrix(alpha_prev, a_prev, d_i, theta_i)
            T_base_to_end = np.dot(T_base_to_end, T_joint)
        pos = T_base_to_end[0:3, 3]
        rot = T_base_to_end[0:3, 0:3]
        roll, pitch, yaw = self.extract_euler_angles(rot)
        return pos, (roll, pitch, yaw) 

    def extract_euler_angles(self, R):
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    def move_to_endpoint(self, x, y, z, rx, ry, rz, speed=30, wait_util_finish=True):
        x1 = x - self.L3 * np.cos(rz) * np.cos(ry)
        z1 = z - self.L3 * np.cos(rx) * np.sin(ry)
        y1 = y - self.L3 * np.cos(rx) * np.sin(rz)
        # print(x1, y1, z1)
        self.move_point(x1, y1, z1, rx, ry, rz, speed, wait_util_finish)
        return x, y, z

    def get_end_pose(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        cX = end_pose.X_axis
        cY = end_pose.Y_axis
        cZ = end_pose.Z_axis
        cRX = end_pose.RX_axis
        cRY = end_pose.RY_axis
        cRZ = end_pose.RZ_axis
        return [cX, cY, cZ, cRX, cRY, cRZ]

    def calc_error(self, p1, p2):
        return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5

    def set_gripper(self, distance, effort=500, delay=1.0):
        if not self.valid_gripper(distance):
            print("GRIPPER not valid.", distance)
            distance = self.clip_gripper(distance)

        distance = abs(round(distance * 100 * 1000))
        self.piper.GripperCtrl(distance, effort, 0x01, 0x00)
        time.sleep(delay)

        state = self.get_gripper_state()
        e = self.calc_distance([distance], [state[0]])
        return e > 1000 and abs(state[1]) < effort

    def valid_gripper(self, distance):
        if distance is None: return False 
        return self.GRIPPER_LIMIT[0] <= distance <= self.GRIPPER_LIMIT[1]

    def get_joints(self):
        js = self.piper.GetArmJointMsgs().joint_state
        jx = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        jx = [x * np.pi / 180 / 1000 for x in jx]
        return jx

    def get_end_pose(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        x = end_pose.X_axis / 1000
        y = end_pose.Y_axis / 1000
        z = end_pose.Z_axis / 1000 - self.L0
        rx = end_pose.RX_axis / 1000
        ry = end_pose.RY_axis / 1000 - 90
        rz = end_pose.RZ_axis / 1000
        return [x, z, y, rx, rz, ry]

    def get_gripper_state(self):
        msg = self.piper.GetArmGripperMsgs()
        angle = msg.gripper_state.grippers_angle
        effort = msg.gripper_state.grippers_effort
        return angle, effort 

    def calc_distance(self, p1, p2):
        return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5

_image = None
_depth = None
_voice = None
_odom = None
_arm_img = None
_arm_dth = None
CvBr = CvBridge()
def callback_image(msg):
    global _image
    _image = CvBr.imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global _depth
    _depth = CvBr.imgmsg_to_cv2(msg)

def callback_arm_img(msg):
    global _arm_img
    _arm_img = CvBr.imgmsg_to_cv2(msg, "bgr8")

def callback_arm_dth(msg):
    global _arm_dth
    _arm_dth = CvBr.imgmsg_to_cv2(msg)

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_odom(msg):
    global _odom
    _odom = msg

def say(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 130 "{g}"')
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.5)
def turn(angle: float):
    global _odom
    if _odom is None:
        return

    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)

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

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def move(dis, turn):
    fb = Twist()
    fb.linear.x = dis
    fb.angular.z = turn
    pub_cmd.publish(fb)
def check_wether_turned_while_moving(target_yaw):
    curr_yaw = get_yaw_from_odom()
    error = normalize_angle(target_yaw - curr_yaw)
    if abs(error) > 0.04:
        turn_to(target_yaw, 0.2)
def get_odom_pose():
    """替代原有的 get_map_coords，僅使用里程計 (Odom) 獲取當前位置與角度"""
    try:
        # 改為監聽 /odom 到 /base_link 的轉換，不使用 /map
        (trans, rot) = tf_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        yaw = euler[2]
        return trans[0], trans[1], yaw
    except Exception as e:
        # 如果 TF 抓不到，則從 callback 獲取的 _odom 訊息提取
        if _odom is not None:
            q = _odom.pose.pose.orientation
            euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return _odom.pose.pose.position.x, _odom.pose.pose.position.y, euler[2]
        return 0, 0, 0


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
                    if not (0 <= j < w): continue
                    d = dp[y - k][j]
                    if d > 0: break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h): continue
                    d = dp[i][x + k]
                    if d > 0: break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w): continue
                    d = dp[y + k][j]
                    if d > 0: break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h): continue
                    d = dp[i][x - k]
                    if d > 0: break
            if d > 0: break
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d



if __name__ == "__main__":
    rospy.init_node("robot_started")
    rospy.loginfo("Node started")
    rate = rospy.Rate(20)

    # Globals initialization
    _image = None
    _depth = None
    _voice = None
    _odom = None
    _amcl = None
    _sub_cam_image, _sub_cam_depth, _current_cam = None, None, 1

    # Subscribers
    
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)     # mid cam
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)

    rospy.Subscriber("/cam2/color/image_raw", Image, callback_arm_img)     # arm cam
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_arm_dth)

    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    # rospy.Subscriber("/imu/data", Imu, callback_Imu)

    # Publishers
    pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # Models
    model = YOLO("antony_ws/basket.pt")
    tf_listener = tf.TransformListener()
    # Chassis
    chassis = RobotChassis()

    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

    #roboarm

    piper = KindaPiper("can0")
    piper.connect()

    # Variables
    state = 0
    # end_location = (2.26,3.83,0)  # x, y
    # end_location_2 = (1.429,4.427,1.715) 
    end_location = (2.959,3.836,0.85) 
    end_location2 = (1.339,4.472,-0.001) 
    washing_machine =  (3.44164614366979, 4.850797600265059, 2.1793549701266692)   
    moving = False
    far_d = 0

    say("Demo start.")
    xLaundry = 0
    yLaundry = 0
    piper.move_to_endpoint2(100,100,0,0,0,-30*np.pi/180)
    while not rospy.is_shutdown():
        if _image is None or _depth is None:
            print("image None")
            continue

        rate.sleep()
        image = _image.copy()
        depth = _depth.copy()
        arm_img = _arm_img.copy()
        arm_dth = _arm_dth.copy()
        h, w = image.shape[0], image.shape[1]
        if state == 0:
            say("wait for door open")
            state = 0.25

        elif state == 0.25:
            door_dist = depth[h//2][w//2]
            print(door_dist)
            if door_dist >= 1500 or door_dist == 0: # door opened
                say("door opened, ready to move")
                state = 0.5

        elif state == 0.5:
            if moving: continue
            moving = True
            clear_costmaps
            chassis.move_to(*end_location)
            moving = False
            if chassis.status_code == 3:
                say("arrived")
                state = 1
        elif state == 1:
            # detect laundry basket, need model
            xLaundry, yLaundry = 0, 0
            print(xLaundry,yLaundry)
            step="turn"
            action="find"
            state = 105
        elif state == 105:
            if step == "turn":
                move(0, 0.2)
            if action == "find":
                min_d = 9999
                t_idx = -1
                result = model(arm_img)[0]
                if result.boxes is not None and len(result.boxes) > 0:  # found object
                    result.plot()
                    for box in result.boxes:
                        score = float(box.conf[0])
                        if score > 0.45:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(arm_img, (x1,y1), (x2,y2), (0,0,255),3)
                            fcx = (x1+x2)//2
                            fcy = (y1+y2) //2
                            t_idx=1
                if t_idx != -1:
                    _, _, d = get_real_xyz(depth, fcx, fcy, 2)
                    final_d = d
                    step = "none"
                    e = (image.shape[1] // 2) - fcx
                    v = np.clip(0.001 * e, -0.45, 0.45)
                    move(0, v)
                    if abs(e) <= 40:
                        state=2
                        step="none"
                        action="none"
                        move(0, 0)
                        time.sleep(0.5)
                    else:
                        step = "turn"
        elif state == 2:
            _, _, _1yaw = get_odom_pose()
            distance_vv=depth[h//2][w//2]-1300
            for iii in range(round((distance_vv) / 100)):
                check_wether_turned_while_moving(_1yaw)
                move(0.2, 0)
                time.sleep(0.5)
            
            move(0,0)
            time.sleep(0.5)
            move(0.12,0)
            time.sleep(0.05)
            turn(5 * np.pi / 180)
            time.sleep(0.5)
            state = 3
        elif state == 3:

            piper.set_gripper(1)
            piper.move_to_endpoint2(330,200,0,0,0,-np.pi/2,n=100)
            piper.move_to_endpoint2(330,-320,0,0,0,-np.pi/2,n=100)
            piper.set_gripper(0)
            _, _, _1yaw = get_odom_pose()
            turn(-5 * np.pi / 180)
            time.sleep(0.5)
            for iii in range(round(distance_vv / 100)):
                check_wether_turned_while_moving(_1yaw)
                move(-0.2, 0)
                time.sleep(0.5)
            turn_to(0,0.3)
            time.sleep(0.5)
            turn(48*np.pi/180)
            _, _, _1yaw = get_odom_pose()
            for iii in range(round(1900 / 100)):
                check_wether_turned_while_moving(_1yaw)
                move(0.2, 0)
                time.sleep(0.5)
            # grep the clothing
            # lift up the clothing
            # back to main position
            state = 4

        
        elif state == 4:
            # place the clothing on the table
            piper.move_to_endpoint2(0,500,0,0,0,0,n=100)
            time.sleep(0.5)
            piper.move_to_endpoint2(320,500,0,0,0,0,n=100)
            turn_to(0,0.4)
            time.sleep(0.5)
            move(0.2,0)
            time.sleep(0.5)
            piper.set_gripper(1)
            state = 5
        elif state == 5:
            if moving: continue
            moving = True
            for i in range(5):
                move(-0.2,0)
                time.sleep(0.5)
            piper.move_to_endpoint2(100,200,0,0,0,0)
            time.sleep(0.5)
            state = 6
        elif state == 6:
            clear_costmaps
            chassis.move_to(*washing_machine)
            moving = False
            if chassis.status_code == 3:
                say("arrived")
                time.sleep(0.5)
                state = 7
        elif state == 7:
            move(0.05,0)
            time.sleep(0.035)
            with open('antony_ws/washing_machine.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            x = True
            for i in data:
                piper.move_joint(i["joints"])
                if i["gripper"][0]>50000 and x == True:
                    piper.set_gripper(1,1000)
                    x = False
                elif i["gripper"][0]<=50000 and x == False:
                    piper.set_gripper(0,1000)
                    x = True
                print(i)
            for i in range(7):
                move(-0.2,0)
                time.sleep(0.5)
            piper.move_to_endpoint2(100,200,0,0,0,0)

            state = 8
        elif state == 8:
            clear_costmaps
            chassis.move_to(2.970674933166167, 4.888803384574086, 1.7482325337837898)
            moving = False
            if chassis.status_code == 3:
                say("arrived")
                state = 9
        elif state == 9:

            with open('antony_ws/washing_machine2.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            x = True
            for i in data:
                piper.move_joint(i["joints"])
                if i["gripper"][0]>50000 and x == True:
                    piper.set_gripper(1,1000)
                    x = False
                elif i["gripper"][0]<=50000 and x == False:
                    piper.set_gripper(0,1000)
                    x = True
                print(i)
            for i in range(5):
                move(-0.2,0)
                time.sleep(0.5)
            say("mission end")

        cv2.imshow("image", arm_img)
        key = cv2.waitKey(1)
        if key in [27, ord('q')]:
            break

        print("state wisley", state)
