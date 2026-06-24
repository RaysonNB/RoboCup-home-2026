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

angle_bar = 0.4620277986947793
first_angle = 90  # left positive
first_meter = 2000
bar_dins_distance = 2000
table_distance= 800

def check_wether_turned_while_moving(target_yaw):
    curr_yaw = get_yaw_from_odom()
    error = normalize_angle(target_yaw - curr_yaw)
    if abs(error) > 0.05:
        turn_to(target_yaw, 0.2)
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
        print("turn_to_yaw", yaw,"err",e,"time",time.time() - start)
        if abs(e) < 0.01 or time.time() - start > 15:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0, 0)


def normalize_angle(angle):
    while angle > 2*np.pi:
        angle -= 2 * np.pi
    while angle < 2*-np.pi:
        angle += 2 * np.pi
    return angle


def turn(angle):
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


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


def rpy_to_rotation_matrix(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_rpy(R):
    p = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if np.abs(np.cos(p)) < 1e-6:
        r = 0.0
        y = np.arctan2(-R[0, 1], R[1, 1])
    else:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    return np.array([r, p, y])


def solve_6dof_wrist_zyz(R_wrist):
    r11, r12, r13 = R_wrist[0, 0], R_wrist[0, 1], R_wrist[0, 2]
    r21, r22, r23 = R_wrist[1, 0], R_wrist[1, 1], R_wrist[1, 2]
    r31, r32, r33 = R_wrist[2, 0], R_wrist[2, 1], R_wrist[2, 2]
    sin_theta5_sq = r31 ** 2 + r32 ** 2
    if sin_theta5_sq < 1e-6:
        t4 = 0.0
        if r33 > 0:
            t5 = 0.0
            t6 = np.arctan2(r21, r11)
        else:
            t5 = np.pi
            t6 = np.arctan2(-r21, r11)
        return {"solution_1": np.degrees([t4, t5, t6]), "solution_2": None}
    t5_1 = np.arctan2(np.sqrt(sin_theta5_sq), r33)
    t4_1 = np.arctan2(r23, r13)
    t6_1 = np.arctan2(r32, -r31)
    t5_2 = np.arctan2(-np.sqrt(sin_theta5_sq), r33)
    t4_2 = np.arctan2(-r23, -r13)
    t6_2 = np.arctan2(-r32, r31)
    return {"solution_1": np.degrees([t4_1, t5_1, t6_1]), "solution_2": np.degrees([t4_2, t5_2, t6_2])}


def get_zyz_matrix(t4, t5, t6):
    R4 = np.array([[np.cos(t4), -np.sin(t4), 0], [np.sin(t4), np.cos(t4), 0], [0, 0, 1]])
    R5 = np.array([[np.cos(t5), 0, np.sin(t5)], [0, 1, 0], [-np.sin(t5), 0, np.cos(t5)]])
    R6 = np.array([[np.cos(t6), -np.sin(t6), 0], [np.sin(t6), np.cos(t6), 0], [0, 0, 1]])
    return R4 @ R5 @ R6

def move_to_bar(a):
    if a == 1:
        turn_to(angle_bar, 0.4)
        _, _, _1yaw = get_odom_pose()
        for iii in range(round(bar_dins_distance / 100)):
            move(-0.2, 0)
            time.sleep(0.5)
            check_wether_turned_while_moving(_1yaw)
    else:
        _, _, _1yaw = get_odom_pose()
        for iii in range(round(bar_dins_distance / 100)):
            move(0.2, 0)
            time.sleep(0.5)
            check_wether_turned_while_moving(_1yaw)


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
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    _frame1 = None
    _sub_down_cam_image1 = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image1)
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
    time.sleep(1)  # 等待 TF 緩衝
    startx, starty, _11yaw = get_odom_pose()
    print("Initial Yaw:", _11yaw)
    speak("I am ready")
    turn(first_angle * np.pi / 180)
    nigga_meter = first_meter
    for iii in range(round(nigga_meter / 100)):
        move(0.2, 0)
        time.sleep(0.5)

    finded_niggasss = 0

    target_max_depth = 6000
    evidence_cnt = 1
    mini_depth = 9999999
    speak("To all audience please put your hand under your shoulder if you are not calling for a order")
    speak("repeat")
    speak("To all audience please put your hand under your shoulder if you are not calling for a order")
    speak("thank you")
    for ifffff in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6,11,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6,11]:
        step_action = 0
        depth_ddd = 0
        if finded_niggasss >= 2:
            break
        while not rospy.is_shutdown():
            rospy.Rate(10).sleep()
            if _frame1 is None or _depth1 is None:
                continue
            code_image = _frame1.copy()
            code_depth = _depth1.copy()
            if step_action == 0:
                speak("hello customers, please raise your hand over your head if you need a order")
                speak("please don't put down your hands, until the robot come through, thank you")
                step = "turn"
                action = "find"
                speak("finding customer")
                step_action = 1

            if step_action == 1:
                if step == "turn":
                    move(0, -0.2)
                if action == "find":
                    poses = net_pose.forward(code_image)
                    min_d = 9999
                    t_idx = -1
                    for i, pose in enumerate(poses):
                        if (pose[5][2] == 0 and pose[9][2] == 0) or (pose[6][2] == 0 and pose[10][2] == 0):
                            continue
                        p5, p6, p9, p10 = list(map(int, pose[5][:2])), list(map(int, pose[6][:2])), list(
                            map(int, pose[9][:2])), list(map(int, pose[10][:2]))
                        cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        if d >= 5000 or d == 0: continue
                        if (((pose[5][2] != 0 and pose[9][2] != 0) and (p5[1] > p9[1])) or (
                                (p6[1] > p10[1]) and (pose[6][2] != 0 and pose[10][2] != 0))) and d <= min_d:
                            t_idx, final_d, min_d = i, d, d

                    if t_idx != -1:
                        p5, p6 = list(map(int, poses[t_idx][5][:2])), list(map(int, poses[t_idx][6][:2]))
                        cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        final_d = d
                        step = "none"
                        e = (code_image.shape[1] // 2) - cx
                        v = np.clip(0.001 * e, -0.45, 0.45)
                        move(0, v)
                        if abs(e) <= 25:
                            action = "front"
                            time.sleep(1)
                            speak("found you the customer who need order")
                            time.sleep(3)
                            move(0, 0)
                    else:
                        step = "turn"

                if action == "front":
                    meter = final_d - 350 - 50 - table_distance
                    time.sleep(2)
                    _, _, _1yaw = get_odom_pose()
                    for walk in range(round(meter / 100)):
                        check_wether_turned_while_moving(_1yaw)
                        move(0.2, 0)
                        time.sleep(0.5)
                    action = "speak"
                if action == "speak":
                    step = "none"
                    action = "none"
                    step_action = 2

            if step_action == 2:
                speak("dear customer, please tell me what you want")

                try:
                    #playsound('/home/pcms/catkin_ws/src/beginner_tutorials/src/mixkit-modern-classic-door-bell-sound-113.mp3')
                    cmd = ["arecord", "-D", ALSA_DEVICE, "-d", str(RECORD_SECONDS), "-f", "S16_LE", "-r",
                           str(SAMPLE_RATE), "-c", "1", AUDIO_PATH]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                    data1 = call_ai_api(1).lower()
                    if "nothing" not in data1:
                        step_action = 99
                except Exception as e:
                    print(f"Error: {e}")

            if step_action == 99:
                
                content_clean = data1.replace("```json", "").replace("```", "").strip()
                try:
                    data1 = json.loads(content_clean)
                except:
                    speak("please repeat")
                    step_action = 2
                    continue

                if len(data1) != 2:
                    speak("please repeat")
                    step_action = 2
                else:
                    speak("dear customer your order is")
                    final_speak_to_guest = data1[0] + " and " + data1[1]
                    speak(final_speak_to_guest)
                    speak("please answer robot yes robot yes or robot no robot no, thank you")
                    yn = -1
                    for i in range(3):
                        try:
                            #playsound('/home/pcms/catkin_ws/src/beginner_tutorials/src/mixkit-modern-classic-door-bell-sound-113.mp3')
                            cmd = ["arecord", "-D", ALSA_DEVICE, "-d", "3", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                                   "-c", "1", AUDIO_PATH]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

                            data = call_ai_api(2)
                            if "no" in data and "robot" in data:
                                yn = 2;
                                break
                            elif "yes" in data:
                                yn = 1;
                                break
                        except:
                            pass
                        speak("please repeat wether i am correct")

                    if yn == 1:
                        step_action = 9
                    elif yn == 2:
                        step_action = 2
                    else:
                        speak("I can't get it, I gonna go back now")
                        _, _, _1yaw = get_odom_pose()
                        for walk in range(round(meter / 100)):
                            check_wether_turned_while_moving(_1yaw)
                            move(-0.2, 0)
                            time.sleep(0.5)
                        step_action = 1

            if step_action == 9:
                _, _, _1yaw = get_odom_pose()
                for walk in range(round(meter / 100)):
                    check_wether_turned_while_moving(_1yaw)
                    move(-0.2, 0)
                    time.sleep(0.5)
                move(0, 0)
                speak("the customer want " + final_speak_to_guest)
                time.sleep(2)
                step_action = 10

            if step_action == 10:
                # 獲取當前里程計坐標與朝向 (當前位置與朝向)
                curr_x, curr_y, _guest_yaw = get_odom_pose()

                # 去吧檯
                move_to_bar(1)
                speak("dear bar man please put " + data1[0] + " between my gripper until it close")
                time.sleep(5)
                speak("5")
                speak("4")
                speak("3")
                speak("2")
                speak("1")
                arm.set_gripper(0, 1000)

                # 從吧檯回來
                move_to_bar(2)
                # 轉回剛才面對客席的方向 (使用里程計記錄的角度)
                turn_to(_guest_yaw, 0.4)

                speak("I gonna move")
                _, _, _1yaw = get_odom_pose()
                for walk in range(round(meter / 100)):
                    check_wether_turned_while_moving(_1yaw)
                    move(0.2, 0)
                    time.sleep(0.5)

                speak("dear customer, this is " + data1[0])
                arm.move_to_endpoint2(300, 400, 0, 0, 0, 0)
                arm.set_gripper(1)
                arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
                _, _, _1yaw = get_odom_pose()
                for walk in range(round(meter / 100)):
                    check_wether_turned_while_moving(_1yaw)
                    move(-0.2, 0)
                    time.sleep(0.5)

                # 第二次去吧檯
                move_to_bar(1)
                speak("dear bar man please put " + data1[1] + " between my gripper until it close")
                time.sleep(5)
                speak("5")
                speak("4")
                speak("3")
                speak("2")
                speak("1")
                arm.set_gripper(0, 1000)

                move_to_bar(2)
                turn_to(_guest_yaw, 0.4)
                _, _, _1yaw = get_odom_pose()
                for walk in range(round(meter / 100)):
                    check_wether_turned_while_moving(_1yaw)
                    move(0.2, 0)
                    time.sleep(0.5)

                speak("dear customer, this is " + data1[1])
                arm.move_to_endpoint2(300, 400, 0, 0, 0, 0)
                arm.set_gripper(1)
                arm.move_joint([0, 0, 0, 0, 0, 0], 10, 30)
                _, _, _1yaw = get_odom_pose()
                for walk in range(round(meter / 100)):
                    check_wether_turned_while_moving(_1yaw)
                    move(-0.2, 0)
                    time.sleep(0.5)

                finded_niggasss += 1
                break

            cv2.imshow("frame", code_image)
            if cv2.waitKey(1) in [ord('q'), 27]: break

    speak("mission end")
