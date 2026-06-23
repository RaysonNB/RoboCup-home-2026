#!/usr/bin/env python3
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


def callback_image1(msg):
    global _frame1
    _frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


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
                if nothing is in the audio answer: ["nothing'], don't give me extral text
                Reply with only one of these two exact phrases:
                1. ["nothing']
                2. ["item1", "item2"]
                '''
        else:
            optimized_prompt = '''
            Listen to the customer's response in the audio. Reply with only one of these three exact phrases:
            1. robot yes
            2. robot no
            3. nothing answer from the customer
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
tf_listener = None


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


def rpy_to_rotation_matrix(rpy):
    """將 (roll, pitch, yaw) 轉為旋轉矩陣 (Z-Y-X 順序)"""
    r, p, y = rpy

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])

    # 航太標準順序：R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def rotation_matrix_to_rpy(R):
    """將 3x3 旋轉矩陣反解為 (roll, pitch, yaw)"""
    # Pitch
    p = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

    # 檢查奇異點（Gimbal Lock 萬向鎖現象，當 Pitch 接近 +-90 度時）
    if np.abs(np.cos(p)) < 1e-6:
        # 當 Pitch = 90° 或 -90° 時，Roll 與 Yaw 會共面
        r = 0.0
        y = np.arctan2(-R[0, 1], R[1, 1])
    else:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])

    return np.array([r, p, y])


def solve_6dof_wrist_zyz(R_wrist):
    """
    針對 6DOF 手腕 (Z-Y-Z 構型，如 J4-J5-J6) 反解關節歐拉角
    :param R_wrist: 3x3 旋轉矩陣 (numpy array)
    :return: dict 包含兩種可能的關節角度組合 (theta4, theta5, theta6) 單位：度
    """
    # 提取矩陣元素
    r11, r12, r13 = R_wrist[0, 0], R_wrist[0, 1], R_wrist[0, 2]
    r21, r22, r23 = R_wrist[1, 0], R_wrist[1, 1], R_wrist[1, 2]
    r31, r32, r33 = R_wrist[2, 0], R_wrist[2, 1], R_wrist[2, 2]

    # 檢查是否遭遇奇異點 (Singularity / Gimbal Lock)
    # 當 J5 關節接近 0 度或 180 度時
    sin_theta5_sq = r31 ** 2 + r32 ** 2

    if sin_theta5_sq < 1e-6:
        print("⚠️ 觸發手腕奇異點 (Wrist Singularity)！J4 與 J6 軸平行。")
        # 遭遇奇異點時，令 theta4 = 0
        t4 = 0.0
        if r33 > 0:  # theta5 = 0
            t5 = 0.0
            t6 = np.arctan2(r21, r11)
        else:  # theta5 = 180 度
            t5 = np.pi
            t6 = np.arctan2(-r21, r11)

        return {
            "solution_1": np.degrees([t4, t5, t6]),
            "solution_2": None
        }

    # 一般情況：存在正反兩組解 (Wrist Up / Wrist Down)
    # 解法一 (正弦值為正)
    t5_1 = np.arctan2(np.sqrt(sin_theta5_sq), r33)
    t4_1 = np.arctan2(r23, r13)
    t6_1 = np.arctan2(r32, -r31)

    # 解法二 (正弦值為負)
    t5_2 = np.arctan2(-np.sqrt(sin_theta5_sq), r33)
    t4_2 = np.arctan2(-r23, -r13)
    t6_2 = np.arctan2(-r32, r31)

    return {
        "solution_1": np.degrees([t4_1, t5_1, t6_1]),
        "solution_2": np.degrees([t4_2, t5_2, t6_2])
    }


def get_zyz_matrix(t4, t5, t6):
    R4 = np.array([[np.cos(t4), -np.sin(t4), 0], [np.sin(t4), np.cos(t4), 0], [0, 0, 1]])
    R5 = np.array([[np.cos(t5), 0, np.sin(t5)], [0, 1, 0], [-np.sin(t5), 0, np.cos(t5)]])
    R6 = np.array([[np.cos(t6), -np.sin(t6), 0], [np.sin(t6), np.cos(t6), 0], [0, 0, 1]])
    return R4 @ R5 @ R6


def calc_distance(self, p1, p2):
    return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5


def set_joints(self, joints, speed=30):
    if not self.valid_joints(joints):
        print("JOINTS not valid.", joints)
        joints = self.clip_joints(joints)

    joints = [round(x * 1000 * 180 / np.pi) for x in joints]
    self.piper.ModeCtrl(0x01, 0x01, speed, 0x00)
    time.sleep(0.005)
    self.piper.JointCtrl(*joints)

    e = self.calc_distance(joints, self.get_joints())
    return e > 1000


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


def set_end_pose(self, x, y, z, rx, ry, rz, speed=50):
    position = [x, y, z, rx, ry, rz]
    position = [i * 1000 for i in position]

    self.piper.ModeCtrl(0x01, 0x00, speed, 0x00)
    time.sleep(0.005)
    self.piper.EndPoseCtrl(*position)

    status = self.piper.GetArmStatus().arm_status.arm_status
    if status == 0x04:
        return False
    return True


def calc_joints(self, x0, y0, z0, rx, ry, rz):
    L01 = self.L01
    L12 = self.L12
    L23 = self.L23
    L34 = self.L34

    x1 = x0 - L34 * np.cos(ry) * np.cos(rz)
    y1 = y0 - L34 * np.cos(ry) * np.sin(rz)
    z1 = z0 - L34 * np.sin(ry)

    c = np.arctan2(y1, x1)
    x2 = (x1 ** 2 + y1 ** 2) ** 0.5
    y2 = 0
    z2 = z1

    print(x0, y0, z0)
    print(x1, y1, z1)
    print(x2, y2, z2)
    if not (L12 + L23) >= (x2 ** 2 + z2 ** 2) ** 0.5 >= 150:
        print("cannot reach that point!", x0, y0, z0)
        return []

    Lk = (x2 ** 2 + z2 ** 2) ** 0.5
    b = np.pi - np.arccos((L12 ** 2 + L23 ** 2 - Lk ** 2) / (2 * L12 * L23))
    a = np.pi / 2 - np.arctan2(z2, x2) - np.arccos((L12 ** 2 + Lk ** 2 - L23 ** 2) / (2 * L12 * Lk))
    a = a + np.pi / 2 - 8 * np.pi / 180
    b = b - np.pi + 17.5 * np.pi / 180

    R1 = rpy_to_rotation_matrix([0, 0, c])
    R2 = rpy_to_rotation_matrix([0, a, 0])
    R3 = rpy_to_rotation_matrix([0, b, 0])
    R_total = R1 @ R2 @ R3
    rpy = rotation_matrix_to_rpy(R_total)

    rpy = rpy_to_rotation_matrix([rx, ry, rz]) @ np.linalg.inv(R_total)
    rpy = rotation_matrix_to_rpy(rpy)
    # R_curr = rpy_to_rotation_matrix(rpy)
    # R_targ = rpy_to_rotation_matrix([rx, ry, rz])
    # R_rel = R_curr.T @ R_targ
    # # R_rel = R_targ @ R_curr.T
    # rpy = rotation_matrix_to_rpy(R_rel)
    print(rpy[1], -a - b - ry)
    print(np.linalg.inv(R_total) @ R_total)

    R_test = get_zyz_matrix(rx, ry, rz)
    solutions = solve_6dof_wrist_zyz(R_test)
    print(solutions)
    rpy = [x * np.pi / 180 for x in solutions['solution_1']]
    rpy[1] = -rpy[1]
    return [c, a, b, *rpy]


def move_to_bar(a):
    if a == 1:
        turn_to(0.12831112955412388, 0.2)
        nigga_meter = 1000
        for iii in range(round(nigga_meter / 100)):
            move(-0.2, 0)
            time.sleep(0.5)
    else:
        nigga_meter = 1000
        for iii in range(round(nigga_meter / 100)):
            move(0.2, 0)
            time.sleep(0.5)
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
    
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("speak")
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    print("gemini rgb")
    _frame1 = None
    _sub_down_cam_image1 = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image1)
    print("gemini depth")
    _depth1 = None
    _sub_down_cam_depth1 = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth1)
    net_pose = HumanPoseEstimation(device_name="GPU")
    _odom = None
    rospy.Subscriber("/odom", Odometry, callback_odom)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    follow_cnt = 0
    action = 0
    step_action = 0
    step = "none"
    confirm_command = 0
    arm = KindaPiper()
    arm.connect("can0")
    tf_listener = tf.TransformListener()
    arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
    arm.set_gripper(1)
    # left positive
    startx, starty, _1yaw = get_map_coords(0)
    print(_1yaw)
    
    turn(90*np.pi/180)
    nigga_meter = 2000
    for iii in range(round(nigga_meter / 100)):
        move(0.2, 0)
        time.sleep(0.5)
    speak("I am ready")
    meter = 1000
    print("0")
    finded_niggasss = 0

    speak_j = ""
    '''
    speak("To all audience please put your hand under your shoulder if you are not calling for an order")
    speak("repeat")
    speak("To all audience please put your hand under your shoulder if you are not calling for an order")
    time.sleep(1)
    speak("thank you")'''

    target_max_depth = 6000
    evidence_cnt = 1
    mini_depth = 9999999
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
                #speak("hello customers, please raise your hand over your head if you need a order")
                #speak("don't put down your hands, until the robot come through, thank you")
                step = "turn"
                action = "find"
                #speak("finding customer")
                step_action = 1
            if step_action == 1:
                # walk in front of the guy
                if step == "turn":
                    move(0, -0.2)
                if action == "find":
                    code_image = _frame1.copy()
                    poses = net_pose.forward(code_image)
                    min_d = 9999
                    t_idx = -1
                    for i, pose in enumerate(poses):
                        if (pose[5][2] == 0 and pose[9][2] == 0) or (pose[6][2] == 0 and pose[10][2] == 0):
                            continue
                        p5 = list(map(int, pose[5][:2]))
                        p6 = list(map(int, pose[6][:2]))
                        p9 = list(map(int, pose[9][:2]))
                        p10 = list(map(int, pose[10][:2]))
                        print(p5, p9, "p5,p9")
                        print(p6, p10, "p6,p10")
                        cx = (p5[0] + p6[0]) // 2
                        cy = (p5[1] + p6[1]) // 2
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        if d >= 5000 or d == 0: continue
                        cv2.circle(code_image, p5, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, p6, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, p9, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, p10, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)

                        if (((pose[5][2] != 0 and pose[9][2] != 0) and (p5[1] > p9[1])) or (
                                (p6[1] > p10[1]) and (pose[6][2] != 0 and pose[10][2] != 0))) and d <= min_d:
                            t_idx = i
                            final_d = d
                            min_d = d

                    x, z = 0, 0
                    if t_idx != -1:
                        print("nigga")
                        p5 = list(map(int, poses[t_idx][5][:2]))
                        p6 = list(map(int, poses[t_idx][6][:2]))
                        cx = (p5[0] + p6[0]) // 2
                        cy = (p5[1] + p6[1]) // 2
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        cv2.circle(code_image, (cx, cy), 5, (0, 255, 255), -1)

                        print("people_d", d)
                        final_d = d
                        step = "none"
                        h, w, c = code_image.shape
                        e = w // 2 - cx
                        v = 0.001 * e
                        if v > 0:
                            v = min(v, 0.45)
                        if v < 0:
                            v = max(v, -0.45)
                        move(0, v)
                        print(e)
                        if abs(e) <= 25:
                            # speak("walk")
                            action = "front"
                            time.sleep(1)
                            speak("found you the customer who need order")
                            time.sleep(3)
                            if ifffff == 1:
                                speak("there is no path for me to move")
                                time.sleep(2)
                                for turn in range(30):
                                    move(0, -0.6)
                                    time.sleep(0.026)
                                time.sleep(1)
                                break

                            print("turned")
                            move(0, 0)
                    else:
                        step = "turn"
                if action == "front":
                    meter = final_d - 1400
                    time.sleep(2)
                    print("walking", meter)
                    for walk in range(round(meter / 100)):
                        move(0.2, 0)
                        time.sleep(0.5)
                    action = "speak"
                if action == "speak":
                    step = "none"
                    action = "none"
                    step_action = 2

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
                    data1 = call_ai_api(1).lower()

                    if "Nothing" not in data1 or "nothing" not in data1:
                        step_action = 99
                except Exception as e:
                    print(f"\n執行失敗: {e}")
                if abs((start_time - time.time())) >= 35 or "nothing" in data1:
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
                    step_action = 1
                    mini_depth = 99999
                # data = command_list[i]
                # continue
            if step_action == 99:
                speak("dear customer your order is")
                time.sleep(0.3)
                content_clean = data1.replace("```json", "").replace("```", "").strip()
                data1 = json.loads(content_clean)
                if len(data1)!=2:
                    step_action=2
                else:
                    final_speak_to_guest = data1[0] + " and " + data1[1]
                    s = ""
                    print("Your order is **********************")
                    print(data1)

                    speak(final_speak_to_guest)
                    print("********************")
                    speak("please answer robot yes or robot no thank you")
                    playsound(
                        '/home/pcms/catkin_ws/src/beginner_tutorials/src/mixkit-modern-classic-door-bell-sound-113.mp3')
                    yn = -1
                    for i in range(3):
                        try:
                            cmd = [
                                "arecord",
                                "-D", ALSA_DEVICE,
                                "-d", str(3),
                                "-f", "S16_LE",
                                "-r", str(SAMPLE_RATE),
                                "-c", "1",
                                AUDIO_PATH
                            ]

                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                            print("⏹️ 錄音結束！")
                            data = call_ai_api(2)
                            if "no" in data and "robot" in data:
                                yn = 2
                                break
                            elif "yes" in data:
                                step_action=9
                                yn=1
                                break

                        except Exception as e:
                            print(f"\n執行失敗: {e}")
                        speak("please repeat")
                if yn == -1:
                    speak("I can't get it, I gonna go back now")
                    time.sleep(3)
                    for walk in range(round(meter / 100)):
                        move(-0.2, 0)
                        time.sleep(0.5)
                    for turn in range(30):
                        move(0, -0.6)
                        time.sleep(0.026)
                    time.sleep(1)
                    speak("hello customers, please raise your hand over your head if you need a order")
                    step_action = 1
                    mini_depth = 99999
                elif yn == 2:
                    step_action = 2
                else:
                    step_action = 9
            if step_action == 9:
                for walk in range(round(meter / 100)):
                    move(-0.2, 0)
                    time.sleep(0.5)
                move(0, 0)
                speak("the customer want " + final_speak_to_guest)
                time.sleep(2)
                step_action = 10
            if step_action == 10:
                startx, starty, _yaw = get_map_coords(0)
                move_to_bar(1)
                speak("dear bar man please put " + data1[0] + " between my gripper until it close")
                time.sleep(5)
                speak("5")
                speak("4")
                speak("3")
                speak("2")
                speak("1")
                arm.set_gripper(0, 1000)
                move_to_bar(2)
                turn_to(_yaw, 0.2)
                speak("I gonna move")
                move(0, 0)
                for walk in range(round((meter) / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                speak("dear customer, this is " + final_speak_to_guest[0])
                arm.move_to_endpoint2(200, 400,0, 0, 0, 0)
                arm.set_gripper(1)
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
                arm.set_gripper(0, 1000)
                speak("I gonna move")
                move_to_bar(2)
                turn_to(_yaw, 0.2)
                move(0, 0)
                for walk in range(round((meter) / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                speak("dear customer, this is " + final_speak_to_guest[1])
                arm.move_to_endpoint2(200, 400,0, 0, 0, 0)
                arm.set_gripper(1)
                arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
                for walk in range(round((meter) / 100)):
                    move(-0.2, 0)
                    time.sleep(0.5)
                time.sleep(3)
                finded_niggasss += 1
                evidence_cnt += 1
                break
            cv2.imshow("frame", code_image)
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                break
    speak("mission end")
