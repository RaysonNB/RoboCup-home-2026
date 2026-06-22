#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, os
from ultralytics import YOLO
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import subprocess
from geometry_msgs.msg import Twist
import math
import time
import re
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
import datetime
from std_srvs.srv import Empty
from gtts import gTTS
from playsound import playsound
import requests
import speech_recognition as sr
import json
import os
from datetime import datetime
import base64
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
import time
from piper_sdk import *
import numpy as np
import matplotlib.pyplot as plt
import random
import json


def callback_image1(msg):
    global _frame1
    _frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


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


'''
def callback_voice(msg):
    global s
    s = msg.text
'''


def speak(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 130 "{g}"')
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.5)


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def callback_voice(msg):
    global s
    s = msg.text


API_KEY = "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"
MODEL = "xiaomi/mimo-v2.5"
AUDIO_PATH = "/home/pcms/catkin_ws/src/beginner_tutorials/src/recording_2s.wav"

SAMPLE_RATE = 44100
RECORD_SECONDS = 7
ALSA_DEVICE = "pulse"

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


def call_ai_api(a):
    print(f"\n[AI] 正在分析訂單錄音檔...")
    try:
        base64_audio = encode_audio(AUDIO_PATH)
        if a == 1:
            optimized_prompt = '''
                You are an expert order-taking assistant. The provided audio is a recording of a customer placing an order,
                but it contains a lot of noisy background sounds. Please perform the following tasks:
                1. Ignore all background noise, wind, music, or chimes.
                2. Listen closely to the human voice and transcribe exactly what the customer said (English).
                answer it in ths format for example: ["apple","coffee"]
                just answer me with a python list
                if the order is two apple, please answer me in ["apple", "apple"]
                if nothing in the audio answer: ["nothing']
                '''
        else:
            optimized_prompt = '''
            Listen to the customer's response in the audio. Reply with only one of these three exact phrases:
            1. robot yes
            2. robot no
            3. no answer from the customer
            Do not add any extra words or punctuation.
            '''

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": optimized_prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ]
        }

        start_time = time.time()
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=20
        )
        end_time = time.time()

        result = response.json()
        print("\n" + "=" * 40)
        if 'choices' in result:
            content = result['choices'][0]['message']['content']
            print(f"[AI 訂單分析結果]:\n{content}")
            print("-" * 40)
            return content
            print(f"[API 執行時間]: {end_time - start_time:.2f} 秒")
        else:
            print("[AI 錯誤]: API 回傳異常")
            print(json.dumps(result, indent=2))
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"[AI 發生錯誤]: {e}")
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
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


def is_waving(pose):
    right_up = (pose[5][2] > 0 and pose[9][2] > 0 and pose[9][1] < pose[5][1] - 30)
    left_up = (pose[6][2] > 0 and pose[10][2] > 0 and pose[10][1] < pose[6][1] - 30)
    if right_up: return "right"
    if left_up: return "left"
    return None


def get_map_coords(dist_meters):
    try:
        (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        yaw = euler[2]
        print("angle", yaw)
        target_x = trans[0] + dist_meters  # test np.cos(yaw)
        target_y = trans[1] + dist_meters * np.tan(yaw)  # test np.sin(yaw)
        return target_x, target_y, yaw
    except Exception as e:
        rospy.logerr(f"TF Error: {e}")
        return None, None, None


class KindaPiper(object):
    L0 = 123.00
    L1 = 285.00
    L2 = 255.50
    L3 = 90.00
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
                print(f"joint_{i + 1} angle problem:", joints[i], angles[i])
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
            time.sleep(0.01)
            js = self.piper.GetArmJointMsgs().joint_state
            jx = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
            max_e = np.inf
            max_e = max([abs(x1 - x2) for x1, x2 in zip(jx, joints)])
            if max_e < 1000:
                wait_util_finish = False
        return res

    def calc_joints(self, x, y, z, rx, ry, rz):
        L0 = self.L0
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3

        if not (L1 + L2) >= (x ** 2 + y ** 2 + z ** 2) ** 0.5 >= 150:
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
        a = a + np.pi / 2 - 8 * np.pi / 180
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
def move_to_bar(a):
    if a==1:
        turn_to(0, 0.2)
        nigga_meter=1000
        for iii in range(round(nigga_meter / 100)):
            move(-0.2, 0)
            time.sleep(0.5)
    else:
        nigga_meter = 1000
        for iii in range(round(nigga_meter / 100)):
            move(0.2, 0)
            time.sleep(0.5)


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    # open things
    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("speak")
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("gemini rgb")
    _frame1 = None
    _sub_down_cam_image1 = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image1)
    print("gemini depth")
    _depth1 = None
    _sub_down_cam_depth1 = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth1)
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    _odom = None
    rospy.Subscriber("/odom", Odometry, callback_odom)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    chassis = RobotChassis()
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    follow_cnt = 0
    action = 0
    step_action = 0
    print("yolov8")
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo1 = Yolov8("yolov8n", device_name="GPU")
    # s = ""
    print("sever1")
    # gg = post_message_request("-1", "", "")
    print("sever2")
    step = "none"
    confirm_command = 0
    # left positive
    turn(35)
    '''
    for i in range(35):
        move(0, 0.2)
        time.sleep(0.125)'''
    nigga_meter = 2000
    for iii in range(round(nigga_meter / 100)):
        move(0.2, 0)
        time.sleep(0.5)
    speak("I am ready")
    meter = 1000
    print("0")
    finded_niggasss = 0
    speak_j = ""
    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
    model = YOLO("yolov8n-pose.pt")
    speak("To all audience please put your hand under your shoulder if you are not calling for an order")
    speak("repeat")
    speak("To all audience please put your hand under your shoulder if you are not calling for an order")
    time.sleep(1)
    speak("thank you")
    startx, starty, start_yaw = get_map_coords(0)
    arm = PiPER_Arm()
    arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
    arm.open_gripper()
    target_max_depth=6000
    evidence_cnt=1
    for ifffff in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6]:
        step_action = 0
        depth_ddd = 0
        skip_voice_cnt = 0
        repeat_cnt = 0
        repeatnigga = 0
        if finded_niggasss >= 2:
            break
        while not rospy.is_shutdown():
            # voice check
            # break
            now1 = datetime.now()
            current_time = now1.strftime("%H:%M:%S")
            rospy.Rate(10).sleep()
            confirm_command = 0
            if _frame1 is None:
                print("no frame")
            if _depth1 is None:
                print("no depth")
            code_image = _frame1.copy()
            code_depth = _depth1.copy()
            catch_image = _frame1.copy()
            h, w, c = code_image.shape
            if step_action == 0:
                speak("hello customers, please raise your hand over your head if you need a order")
                speak("don't put down your hands, until the robot come through, thank you")
                step = "turn"
                action = "find"
                speak("finding customer")
                step_action = 1
            if step_action == 1:
                # walk in front of the guy
                if step == "turn":
                    move(0, -0.2)
                if action == "find":
                    image = _frame1.copy()
                    poses = net_pose.forward(code_image)
                    result = model(image)[0]
                    if result.boxes is not None and len(result.boxes) > 0:  # found person
                        person_keypoints = result.keypoints.xy.cpu().numpy()
                        print(person_keypoints)

                        for keypoint in person_keypoints:
                            kx, ky = keypoint
                            cv2.circle(image, (kx, ky), 10, (255, 0, 0), -1)

                        try:
                            nose = person_keypoints[0]
                            lwrist, rwrist = person_keypoints[9], person_keypoints[10]

                            nose_x, nose_y = map(int, nose)
                            lwrist_y = int(lwrist[1])
                            rwrist_y = int(rwrist[1])

                            if lwrist_y < nose_y or rwrist_y < nose_y:  # wrist above nose
                                if 0 <= nose_x <= w and 0 <= nose_y <= h:  # check if position is valid
                                    # draw boxes for partial points
                                    x1, y1, x2, y2 = map(int, result.boxes.xyxy)
                                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                                    name_photo= "proof"+str(evidence_cnt)+".jpg"
                                    file_path = file_path+name_photo
                                    cv2.imwrite(file_path)

                                    person_depth = code_depth[nose_y][nose_x]
                                    if person_depth and person_depth > 1500 and np.isfinite(person_depth) and person_depth<target_max_depth:
                                        step_action = -6
                        except Exception as e:
                            print(f"error: {e}")
            if step_action == -6:  # find & go to the customer - facing the customer
                result = model(image[0])
                person_keypoints = result.keypoints.xy.cpu().numpy()
                nose = person_keypoints[0]
                h, w, c = image.shape
                image_mid = w // 2
                diff = 10
                e = w // 2 - nose[0]
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(0, v)
                print(e)
                if abs(e) <= 25:
                    speak("found you the customer who need order")
                    meter = person_depth - 1400 -400
                    time.sleep(2)
                    print("walking", meter)
                    for walk in range(round(meter / 100)):
                        move(0.2, 0)
                        time.sleep(0.5)
                    step_action=2
            if step_action == 2:
                speak("dear customer, please tell me what you want")
                speak("please tell me your order in complete sentence, for example: I want an apple and orange juice")
                yn = 0
                start_time = time.time()
                playsound(
                    '/home/pcms/catkin_ws/src/beginner_tutorials/src/mixkit-modern-classic-door-bell-sound-113.mp3')
                try:
                    cmd = [
                        "arecord",
                        "-D", ALSA_DEVICE,
                        "-d", str(RECORD_SECONDS),
                        "-f", "S16_LE",
                        "-r", str(SAMPLE_RATE),
                        "-c", "1",
                        AUDIO_PATH
                    ]

                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                    print("⏹️ 錄音結束！")
                    data = call_ai_api(1).lower()

                    if "Nothing" not in data or "nothing" not in data:
                        step_action = 99
                except Exception as e:
                    print(f"\n執行失敗: {e}")
                if abs((start_time - time.time())) >= 25:
                    speak("I can't get it, I gonna go back now")
                    time.sleep(3)
                    for walk in range(round(meter / 100)):
                        move(-0.2, 0)
                        time.sleep(0.5)
                    time.sleep(2)
                    for turn in range(30):
                        move(0, -0.6)
                        time.sleep(0.026)
                    time.sleep(1)
                    speak("hello customers, please raise your hand over your head if you need a order")
                # data = command_list[i]
                # continue
            if step_action == 99:
                speak("dear customer your order is")
                time.sleep(0.3)
                content_clean = data.replace("```json", "").replace("```", "").strip()
                data = json.loads(content_clean)
                final_speak_to_guest = data[0]+" and " +data[1]
                s = ""
                print("Your order is **********************")
                print(data)

                speak(final_speak_to_guest)
                print("********************")
                speak("please answer robot yes or robot no thank you")
                speak("sound")
                time.sleep(2)
                playsound('/home/pcms/catkin_ws/src/beginner_tutorials/src/mixkit-modern-classic-door-bell-sound-113.mp3')
                yn=-1
                for i in range(3):
                    try:
                        cmd = [
                            "arecord",
                            "-D", ALSA_DEVICE,
                            "-d", str(RECORD_SECONDS),
                            "-f", "S16_LE",
                            "-r", str(SAMPLE_RATE),
                            "-c", "1",
                            AUDIO_PATH
                        ]

                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                        print("⏹️ 錄音結束！")
                        data = call_ai_api(2)
                        if "Nothing" not in data or "nothing" not in data:
                            yn=1
                        elif "no" in data:
                            yn=2

                    except Exception as e:
                        print(f"\n執行失敗: {e}")
                        step_action = 100
                    speak("please repeat")
                if yn==-1:
                    speak("I can't get it, I gonna go back now")
                    time.sleep(3)
                    for walk in range(round(meter / 100)):
                        move(-0.2, 0)
                        time.sleep(0.5)
                    step_action = 100
                    for turn in range(30):
                        move(0, -0.6)
                        time.sleep(0.026)
                    time.sleep(1)
                    speak("hello customers, please raise your hand over your head if you need a order")
                elif yn==2:
                    step_action = 2
                else:
                    step_action = 9
            if step_action == 9:
                time.sleep(3)
                for walk in range(round(meter / 100)):
                    move(-0.2, 0)
                    time.sleep(0.5)
                move(0, 0)
                speak("the customer want " + final_speak_to_guest)
                time.sleep(2)
                step_action = 10
            if step_action == 10:
                startx, starty, _yaw = get_map_coords(0, 0)
                move_to_bar(1)
                speak("dear bar man please put "+final_speak_to_guest[0]+" between my gripper until it close")
                time.sleep(5)
                speak("5")
                speak("4")
                speak("3")
                speak("2")
                speak("1")
                arm.set_gripper(1,1000)
                move_to_bar(2)
                turn_to(_yaw,0.2)
                speak("I gonna move")
                move(0, 0)
                for walk in range(round((meter) / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                speak("dear customer, this is "+final_speak_to_guest[0])
                arm.move_joint([600, 50, 0, 0, 0, 0], 10, 30)
                arm.open_gripper()
                arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
                for walk in range(round((meter) / 100)):
                    move(-0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                move_to_bar(1)
                speak("dear bar man please put " + final_speak_to_guest[1] + " between my gripper until it close")
                time.sleep(5)
                speak("5")
                speak("4")
                speak("3")
                speak("2")
                speak("1")
                arm.set_gripper(1, 1000)
                speak("I gonna move")
                turn_to(_yaw, 0.2)
                move(0, 0)
                for walk in range(round((meter) / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                speak("dear customer, this is " + final_speak_to_guest[1])
                arm.move_joint([600, 50, 0, 0, 0, 0], 10, 30)
                arm.open_gripper()
                arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
                for walk in range(round((meter) / 100)):
                    move(-0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                finded_niggasss += 1
                evidence_cnt+=1
                break
            cv2.imshow("frame", code_image)
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                break
    speak("mission end")
