
#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from mr_voice.msg import Voice
from ultralytics import YOLO
import cv2
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

# Globals
_image = None
_depth = None
_voice = None
_odom = None
_arm_img = None
_arm_dth = None
_low_image = None
_low_depth = None
CvBr = CvBridge()
code = 0 # Navigation result code
_current_cam = 1 # Current camera id
_sub_cam_image = None
_sub_cam_depth = None
camera_pos_to_roboarm_pos_diff = (0, 340, 290)
split_height = [150, 250, 350]


# cam 1, up camera
# cam 2, down camera

color_ranges = {
 'red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
 'orange': [(11, 100, 100), (25, 255, 255)],
 'yellow': [(26, 100, 100), (35, 255, 255)],
 'green': [(36, 100, 100), (85, 255, 255)],
 'cyan': [(86, 100, 100), (95, 255, 255)],
 'blue': [(96, 100, 100), (130, 255, 255)],
 'violet': [(131, 100, 100), (145, 255, 255)],
 'magenta': [(146, 100, 100), (159, 255, 255)],
 'black': [(0, 0, 0), (180, 255, 50)],
 'white': [(0, 0, 200), (180, 30, 255)]
}

# put_pos save the positions of roboarms
# key: name, value: position
put_pos = {
    "dishwasher": (212.48, 444.76, 10.9, -57.38, -54.92, -7.66),
    "kitchen": (212.48, 444.76, 10.9, -57.38, -54.92, -7.66),
    "trashbin": (372.05, 31.77, -20.16, -179.07, -175.71, -89.43),
    "cabinet lv2": (256.02, 6.41, -8.28, -144.1, -145.87, -9.37),
    "cabinet lv3": (177.68, 350.31, -1.41, 44.61, 45.47, -5.11),
    "dining table": (185.83, 283.08, -6.19, 109.34, 107.09, -3.94)
}

position = {
    "dishwasher": (6.986, -0.766, -0.432),
    "kitchen": (7.319, 0.143, -0.545),
    "trashbin": (3.897, 2.483, -1.932),
    "cabinet_see": (7.019, 1.334, 1.253),
    "cabinet_put": (7.4, 2.11, 1.253),
    "dining table": (7.94, 1.26, -0.481),
    "dining table2": (7.58,0.576,-0.481)
}

# category save down category of each item
# key: name, value: items
category = {
    "cutlery": ["fork", "knife", "spoon"],
    "tableware": ["dish", "bowl", "mug", "cup", "plate", "napkin"],
    "trash": ["tissue paper", "empty bottle", "paper cup"],
    "fruits": ["red apple", "green apple", "orange"],
    "snacks": ["cereal box", "nut bag", "pringle"],
    "drinks": ["coke", "green tea", "lemon tea", "red bull", "milk", "coffee", "tea"],
    "other objects": ["bag", "tray", "clothing", "solved rubik's cube", "colgate"]
}

# category2place save which category should place to where
category2place = {
    "cutlery": "dishwasher",
    "tableware": "dishwasher",
    "trash": "trashbin",
    "drinks": "cabinet_put",
    "other": "cabinet_put",
    "snacks": "cabinet_put"
}

item2place = {
    "bowl": "kitchen",
    "spoon": "kitchen",
    "milk": "cabinet",
    "cereal": "cabinet"
}

model = YOLO("antony_ws/pnpobj.pt")                                          # need to change
print(model.names)

def find_item_in_dict(item, dic):
    for key, value in dic.items():
        if item.lower() in value: return key


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

def callback_low_image(msg):
    global _low_image
    _low_image = CvBr.imgmsg_to_cv2(msg, "bgr8")

def callback_low_depth(msg):
    global _low_depth
    _low_depth = CvBr.imgmsg_to_cv2(msg)

def say(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 130 "{g}"')
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.5)

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
    q = [
        _odom.pose.pose.orientation.x,
        _odom.pose.pose.orientation.y,
        _odom.pose.pose.orientation.z,
        _odom.pose.pose.orientation.w
    ]
    return q[0], q[1]

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

def turn_back(speed=0.2):
    global _odom
    if _odom is None:
        return
    yaw = get_yaw_from_odom()
    # Turn 180 degrees (π radians)
    target_angle = yaw + np.pi
    # Normalize angle to [-π, π]
    target_angle = normalize_angle(target_angle)
    turn_to(target_angle, speed)

def turn(angle: float):
    global _odom
    if _odom is None:
        return

    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


def detect_real_xyz(x1,y1,d):
    global image
    h, w, c = image.shape
    y1 -= h/2
    x1 -= w/2
    y = y1/h*2*d*math.tan(45.5*np.pi/360)
    x = x1/w*2*d*math.tan(58.4*np.pi/360)
    return x,y,d

def camera_to_roboarm(x, y, z):
    y += camera_pos_to_roboarm_pos_diff[1]
    z += camera_pos_to_roboarm_pos_diff[2]
    return (z, -y, x, 0, 0, 0)

def roboarm_move_to(x, y, z, rx=0, ry=0, rz=0):
    piper.move_to_endpoint(x, y, z, rx, ry, rz)
    
def roboarm_move(dx, dy, dz, drx=0, dry=0, drz=0):
    x, y, z, rx, ry, rz = piper.get_end_pose()
    x += dx; y += dy; z += dz
    rx += drx; ry += dry; rz += drz
    roboarm_move_to(x, y, z, rx, ry, rz)

def roboarm_grab():
    piper.set_gripper(0, 500)
   
def roboarm_release():
    piper.set_gripper(1, 500)
   
def roboarm_reset_pos():
    piper.move_joint([0, 0, 0, 0, 0, 0])
    
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


def grab_obj(obj_name,image,initd=100,inity=200,initx=0,angle = 0):
   
    result = model(image)[0]
    if result.boxes is not None and len(result.boxes) > 0:   # found object
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = model.names[int(box.cls[0])]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            print(f"I see {name}")
            if name.lower() == obj_name:
                x = (x1+x2)//2
                y = (y1+y2)//2
               
                d = arm_dth[y][x]
                if d == 0:
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if arm_dth[y+j][x+i] != 0:
                                d = arm_dth[y+j][x+i]
                print(x,y,d)
                if x == 0 or y == 0 or d == 0:
                    print(x,y,d)
                    return "not found"
                x,y,d = detect_real_xyz(x,y,d)
                print(x,y,d)
                true_x = x-10
                true_y = y*np.cos((angle)*np.pi/180)-d*np.sin((angle)*np.pi/180)+np.cos(angle)*60-inity
                true_d = y*np.sin((angle)*np.pi/180)+d*np.cos((angle)*np.pi/180)+np.sin(angle)*60+initd
                if angle == 0:
                    true_y-=170
                elif angle == -15:
                    true_d+=30
                    
                elif angle == -30:
                    true_y-=50
                    true_d-=70
                elif angle == -45:
                    true_y+=30
                print("x:",true_x,"y:",true_y,"arm_dth",true_d)
                piper.move_to_endpoint2(true_d-100,-int(true_y),-int(true_x),0,0,0,n=100)
                time.sleep(0.5)
                piper.move_to_endpoint2(true_d,-int(true_y),-int(true_x),0,0,0,n=100)
                time.sleep(0.5)
                print(piper.get_end_pose())
                roboarm_grab()
                piper.move_to_endpoint2(true_d-30,-int(true_y)+100,-int(true_x),0,0,0,n=100)
                time.sleep(0.5)
                piper.move_joint([0,0,0,0,0,0])
                return "success"
    return "not found"


def grab_down(obj_name,image,initd=100,inity=200,initx=0,angle = 0):


    result = model(image)[0]
    if result.boxes is not None and len(result.boxes) > 0:   # found object
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = model.names[int(box.cls[0])]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            print(f"I see {name}")
            if name.lower() == obj_name:
                x = (x1+x2)//2
                y = (y1+y2)//2
               
                d = arm_dth[y][x]
                if d == 0:
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if arm_dth[y+j][x+i] != 0:
                                d = arm_dth[y+j][x+i]
                print(x,y,d)
                if x == 0 or y == 0 or d == 0:
                    print(x,y,d)
                    return "not found"
                x,y,d = detect_real_xyz(x,y,d)
                print(x,y,d)
                true_x = x-10
                true_y = y*np.cos((angle)*np.pi/180)-d*np.sin((angle)*np.pi/180)+np.sin(angle)*50+np.cos(angle)*60-inity
                true_d = y*np.sin((angle)*np.pi/180)+d*np.cos((angle)*np.pi/180)-np.cos(angle)*50+np.sin(angle)*60+initd
                if angle == 0:
                    true_y-=170
                elif angle == -15:
                    true_d+=30
                    true_y-=50
                elif angle == -30:
                    true_y-=50
                    true_d-=70
                elif angle == -45:
                    true_y+=30
                elif angle == -60:
                    true_y+=130
                    true_d-=140
                true_y-=250
                true_d+=250
                print("x:",true_x,"y:",true_y,"arm_dth",true_d)
                piper.move_to_endpoint2(true_d,-int(true_y)+50,-int(true_x),0,0,-1.57,n=100)
                time.sleep(0.5)
                piper.move_to_endpoint2(true_d,-int(true_y),-int(true_x),0,0,-1.57,n=100)
                time.sleep(0.5)
                print(piper.get_end_pose())
                roboarm_grab()
                piper.move_to_endpoint2(true_d,-int(true_y)+50,-int(true_x),0,0,-1.57,n=100)
                time.sleep(0.5)
                piper.move_joint([0,0,0,0,0,0])
                return "success"
    return "not found"


def grab(obj,image,initd=100,inity=200,initx=0,angle=0):
    piper.move_to_endpoint2(initd,inity,0,0,0,np.pi*angle/180)
    roboarm_release()
    
    if obj in ["pringle", "red bull", "coffee", "coke", "cereal box"]:
        grab_obj(obj,image,initd,inity,initx,angle)
        print("normal grab")
    else:
        grab_down(obj,image,initd,inity,initx,angle)


if __name__ == "__main__":
    rospy.init_node("robot started")
    rospy.loginfo("Node started")
    rate = rospy.Rate(20)


    # Globals initialization
    _image = None
    _depth = None
    _arm_img = None
    _arm_dth = None
    _voice = None
    _odom = None
    _sub_cam_image, _sub_cam_depth, _current_cam = None, None, 1
    piper = KindaPiper()
    piper.connect("can0")


    # Subscribers
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    # rospy.Subscriber("/imu/data", Imu, callback_Imu)
   
    # Publishers
    pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)


    # Initialize Camera
    rospy.Subscriber("/cam2/color/image_raw", Image, callback_image)     # mid cam
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

    rospy.Subscriber("/cam1/color/image_raw", Image, callback_arm_img)     # arm cam
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_arm_dth)

    # Chassis
    chassis = RobotChassis()


    # Variables
    will_grab = [9999, "", (0, 0, 0)]
    will_grab_category = "drinks"
    place_to_put_will_grab = ""
    count_try = 0
   
    item_in_process = [9999, "", (0, 0, 0)]
    need_kitchen = ["bowl", "spoon"]
    need_cabinet = ["cereal", "milk"]
    state = 0
    count_try = 0
    table_state = 0

    cabinet_item_category = {"lv1": "None", "lv2": "None", "lv3": "None", "lv4": "None"}
    category_appeared = []
    detected_first_spot=False

    say("Demo start. Ready to receive guest.")
    piper.set_gripper(1)
    piper.move_to_endpoint2(100,200,0,0,0,0)

    while not rospy.is_shutdown():
        rate.sleep()
       
        if _image is None or _depth is None:
            print("image None")
            continue
       
        image = _image.copy()
        depth = _depth.copy()
        arm_img = _arm_img.copy()
        arm_dth = _arm_dth.copy()
        h, w, _ = image.shape

        print("state", state)
       
        if state == 0:
            say("code start")
            roboarm_reset_pos()
            state = 1
       
        elif state == 1: # wait for door open
            door_dist = depth[h//2][w//2]
            print(door_dist)
            if door_dist >= 1500 or door_dist == 0: # door opened
                say("door opened, ready to move")
               
                state = 10125
       
        elif state == 10125:   # navigate to cabinet to check every level
            chassis.move_to(*position["cabinet_see"])
            if chassis.status_code == 3:
                state = 1025
                   
        elif state == 1025:    # check cabinet
            result = model(image)[0]
               
            if result.boxes is not None and len(result.boxes) > 0: # found object
                image = result.plot()
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    name = model.names[int(box.cls[0])]
                   
                    current_item_category = find_item_in_dict(name, category)
                    print("current", current_item_category)


                    y = (y1 + y2) // 2
                    cabinet_level = "lv"
                    if y <= split_height[0]:
                        cabinet_level += "4"
                    elif y <= split_height[1]:
                        cabinet_level += "3"
                    elif y <= split_height[2]:
                        cabinet_level += "2"
                    else:
                        cabinet_level += "1"
                   
                   
                    if cabinet_item_category[cabinet_level] == "None":    # not set yet
                        cabinet_item_category[cabinet_level] = current_item_category
                    elif cabinet_item_category[cabinet_level] == current_item_category:    # same
                        continue
                    else:     # find different item in same lvl
                        cabinet_item_category[cabinet_level] = "Unknown"  

            else:
                continue


            category_appeared.extend([v for k, v in cabinet_item_category.items() if v != "None" or v != "Unknown"])    # add lv1,2,3,4 category
            print(category_appeared)
            state = 105

            print(cabinet_item_category)
            
           
       
        elif state == 105:
            if table_state == 0:
                print("tbs 1")
                table_state = 1
                chassis.move_to(*position["dining table"])
                if chassis.status_code == 3:
                    state = 2
            elif table_state == 1:
                print("tbs 2")
                table_state = 2
                chassis.move_to(*position["dining table2"])
                if chassis.status_code == 3:
                    state = 2
        
           
        elif state == 2: # detect items

            piper.move_to_endpoint2(100,200,0,0,0,-np.pi*15/180)
        
            result = model(image)[0]
            if result.boxes is not None and len(result.boxes) > 0: # found object
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    name = model.names[int(box.cls[0])]
                
                    say(f"I see {name}")
            state = 3
        
        elif state == 3: # find item to grip
            print("ctry", count_try)

            result = model(arm_img)[0]
            will_grab = [9999, "", (0, 0, 0)] # depth, name, pos
           
            if result.boxes is not None and len(result.boxes) > 0: # found object
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    midx, midy = (x1+x2)//2, (y1+y2)//2
                    name = model.names[int(box.cls[0])].lower()
                   
                    item_dist = depth[midy][midx]
                    if find_item_in_dict(name, category) in category_appeared:
                        will_grab[0] = item_dist
                        will_grab[1] = name
                        will_grab[2] = camera_to_roboarm(*detect_real_xyz(midx, midy, item_dist))
                        print(will_grab[2])
                        print("will_grab 1", name)
                    else:
                        count_try += 1
            else:
                count_try += 1

            if count_try >= 10: # really nothing but not checking error
                count_try = 0
                if table_state == 2:
                    state = 7
                    say("no item found, now serving breakfast")
                else:
                    state = 105
                continue
           
            if not will_grab[1]: continue
            say(f"i'm going to grab {will_grab[1]} now")
            will_grab_category = find_item_in_dict(will_grab[1], category)
            place_to_put_will_grab = category2place[will_grab_category]
            say(f"{will_grab[1]} is {will_grab_category}, need to put in {place_to_put_will_grab}")
           
            state = 4
       
        elif state == 4: # grab item
            grab(will_grab[1], arm_img, angle=-15)
            #piper.set_gripper(0)
            state = 5
        
        elif state == 5: # navigate
            chassis.move_to(*position[place_to_put_will_grab])
            if chassis.status_code == 3:
                state = 6
           
        elif state == 6: # put
            if will_grab_category == "tableware" or will_grab_category == "cutlery":
                say("please help me to open the dishwasher door thank you, you have 10 seconds")
                rospy.sleep(10)
           
           
            print(place_to_put_will_grab)
            if place_to_put_will_grab == "cabinet_put":
                place_to_put_will_grab = [k for k, v in cabinet_item_category.items() if v == will_grab_category][0]
            
            state = 6025
        elif state == 6025:
            if depth[h//2][w//2]>1300:
                move(0.1,0)
            else:
                state = 605
        elif state == 605:
            if place_to_put_will_grab == "lv1":
                piper.move_to_endpoint2(100,200,0,0,0,0)

                piper.move_to_endpoint2(170,0,0,0,0,-np.pi/2)
                piper.move_to_endpoint2(170,-360,0,0,0,-np.pi/2,n=100)
                piper.move_to_endpoint2(330,-360,0,0,0,0)

                time.sleep(0.5)
                for iii in range(round(100 / 100)):
                        move(0.2, 0)
                        time.sleep(0.5)
                time.sleep(0.5)
                piper.set_gripper(1,1000)
                time.sleep(0.5)
                for iii in range(round(100 / 100)):
                        move(-0.2, 0)
                        time.sleep(0.5)
                time.sleep(0.5)
                piper.move_to_endpoint2(170,-360,0,0,0,-np.pi/2)
                piper.move_to_endpoint2(170,0,0,0,0,-np.pi/2, n=100)

                piper.move_to_endpoint2(160,160,0,0,0,-np.pi/2)
                piper.move_to_endpoint2(100,200,0,0,0,0)
                
            elif place_to_put_will_grab == "lv2":
                piper.set_gripper(0,1000)
                piper.move_to_endpoint2(100,200,0,0,0,0)

                piper.move_to_endpoint2(230,70,0,0,0,0,n=100)
                piper.move_to_endpoint2(470,0,0,0,0,0,n=100)

                time.sleep(0.5)
                piper.set_gripper(1,1000)
                time.sleep(0.5)

                piper.move_to_endpoint2(230,70,0,0,0,0,n=100)
                piper.move_to_endpoint2(100,200,0,0,0,0,n=100)
                
            elif place_to_put_will_grab == "lv3":
                piper.set_gripper(0,1000)
                piper.move_to_endpoint2(100,200,0,0,0,0)
                piper.move_to_endpoint2(100,350,0,0,0,0,n=100)
                piper.move_to_endpoint2(450,350,0,0,0,0,n=100)
                time.sleep(0.5)
                piper.set_gripper(1,1000)
                time.sleep(0.5)
                piper.move_to_endpoint2(100,350,0,0,0,0,n=100)

                piper.move_to_endpoint2(100,200,0,0,0,0)
                
            else:
                say("unable to reach")
            state = 105
            table_state = 0
        elif state == 7: # sets breakfast - navigate to kitchen
            chassis.move_to(*position["kitchen"])
            if chassis.status_code == 3:
                state = 9
           
        elif state == 8: # sets breakfast - navigate to cabinet
            chassis.move_to(*position["cabinet"])
            if chassis.status_code == 3:
                state = 9
           
       
        elif state == 9:
            if not need_kitchen and not need_cabinet: state = 12
            elif not need_kitchen: state = 8
            result = model(arm_img)[0]
           
            if result.boxes is not None and len(result.boxes) > 0: # found object
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    name = model.names[int(box.cls[0])]
                   
                    if name in need_kitchen+need_cabinet:
                        midx, midy = (x1+x2)//2, (y1+y2)//2
                        item_dist = depth[midy][midx]
                       
                        item_in_process[0] = item_dist
                        item_in_process[1] = name
                        item_in_process[2] = camera_to_roboarm(detect_real_xyz(midx, midy, item_dist))
               
                state = 10
           
        elif state == 10: # sets breakfast - get item
            grab(item_in_process[1], arm_img)
           
            state = 1005
       
        elif state == 1005:
            chassis.move_to(*position["dining table"])
            if chassis.status_code == 3:
                state = 11
           
        elif state == 11: # sets breakfast - place item
            roboarm_move_to(*put_pos["dining table"])
            roboarm_release()
           
            if item_in_process[1] in need_kitchen: need_kitchen.remove(item_in_process[1])
            if item_in_process[1] in need_cabinet: need_cabinet.remove(item_in_process[1])
           
            state = 8 if need_kitchen == [] else 7
       
        elif state == 12:
            say("code done")
            break


        cv2.imshow("image", arm_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




