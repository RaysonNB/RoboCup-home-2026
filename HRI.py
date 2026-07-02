#!/usr/bin/env python3.8
from std_msgs.msg import String
from std_srvs.srv import Empty
from piper_sdk_demo import KindaPiper
from pcms.openvino_models import *
from pcms.pytorch_models import *
from typing import Tuple
import tempfile
from ultralytics import YOLO
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import subprocess
from mr_voice.msg import Voice
import requests
import rospy
import cv2
import os
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from RobotChassis import RobotChassis
import time
from piper_sdk_demo import *
import numpy as np
import json
from cv_bridge import CvBridge
bridge = CvBridge()
clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
_image = None
_depth = None
_image_seq = 0
_depth_seq = 0
_voice = None
_odom = None
code = 0
_current_cam = 1
_sub_cam_image = None
_sub_cam_depth = None
_audio_path = None
_new_audio_flag = False
model = YOLO("yolov8n-pose.pt")
SAMPLE_RATE = 44100
ALSA_DEVICE = "pulse"
prp = 0
BASE_URL = "http://192.168.88.89:8888"

def callback_voice(msg):
    global _voice
    _voice = msg
def callback_odom(msg):
    global _odom
    _odom = msg
def callback_path(msg):
    global _audio_path, _new_audio_flag
    _audio_path = msg.data
    _new_audio_flag = True
    print("new audio path =", _audio_path)
def callback_image(msg):
    global _image, _image_seq
    try:
        _image = bridge.imgmsg_to_cv2(msg, "bgr8")
        _image_seq += 1
    except Exception as e:
        print("callback_image error:", e)
def callback_depth(msg):
    global _depth, _depth_seq
    try:
        _depth = bridge.imgmsg_to_cv2(msg, "passthrough")
        _depth_seq += 1
    except Exception as e:
        print("callback_depth error:", e)
def callback_Imu(msg):
    global _Imu
    _Imu = msg
def move(dis, turn):
    fb = Twist()
    fb.linear.x = dis
    fb.angular.z = turn
    pub_cmd.publish(fb)
def say(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 130 "{g}"')
    print(g)
    time.sleep(0.5)
def speak(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 110 "{g}"')
    print(g)
    time.sleep(0.5)
def get_pose_box(user, image_shape, padding=20):
    if user is None or len(user) == 0:
        return None

    h, w = image_shape[:2]
    xs = []
    ys = []

    for p in user:
        x, y = map(int, p)
        if 0 <= x < w and 0 <= y < h:
            xs.append(x)
            ys.append(y)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w - 1, max(xs) + padding)
    y2 = min(h - 1, max(ys) + padding)

    return x1, y1, x2, y2
def record_audio_arecord(filename="/tmp/guest.wav", seconds=5):
    if os.path.exists(filename):
        print("file exists")
        os.remove(filename)
        print("deleted file")

    cmd = [
        "arecord",
        "-D", ALSA_DEVICE,
        "-d", str(seconds),
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        filename
    ]

    print("[ARECORD]", " ".join(cmd))

    try:
        print('start recording')
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print('finish recording')
    except Exception as e:
        print("[ARECORD] failed:", e)
        return None

    if not os.path.exists(filename):
        print("[ARECORD] file not exists")
        return None

    size = os.path.getsize(filename)
    print("[ARECORD] file size =", size)

    if size < 1000:
        print("[ARECORD] file too small")
        return None

    return filename
def save_user_crop(image, user, save_path=None):
    global prp
    box = get_pose_box(user, image.shape)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if save_path is None:
        fd, save_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

    cv2.imwrite(save_path, crop)
    if prp == 0:
        cv2.imwrite("/tmp/person0.jpg", crop)
        prp += 1
    return save_path
def find_user(img, depth):
    global model
    h, w = depth.shape
    keypoints = model(img, verbose=False)[0].keypoints
    user = []
    user_d = -1
    for pose in keypoints.xy:
        mid_d = -1
        for p in pose:
            x, y = map(int, p)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            if x >= w or y >= h:
                continue
            if depth[y][x] > 0 and depth[y][x] <= 3000:
                if mid_d == -1 or depth[y][x] < mid_d:
                    mid_d = depth[y][x]
        if (user_d == -1 or mid_d < user_d) and mid_d != -1:
            user_d = mid_d
            user = pose
    return user
def change_to_cam(idx):
    global _sub_cam_image, _sub_cam_depth, _current_cam
    _current_cam = idx

    if _sub_cam_image is not None:
        _sub_cam_image.unregister()
    if _sub_cam_depth is not None:
        _sub_cam_depth.unregister()

    if idx == 1:
        _sub_cam_image = rospy.Subscriber("/cam2/color/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    elif idx == 2:
        _sub_cam_image = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
def send_media_task(endpoint, file_path, prompt, file_key):
    try:
        if not os.path.exists(file_path):
            print("[AI] file not exists:", file_path)
            return None

        with open(file_path, "rb") as f:
            files = {file_key: f}
            data = {"prompt": prompt}

            print(f"[AI] sending {file_path} to {endpoint}")

            response = requests.post(
                f"{BASE_URL}{endpoint}",
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code == 200:
            return response.json().get("ans")
        else:
            print("[AI] server error:", response.text)
            return None

    except Exception as e:
        print("[AI] connection failed:", e)
        return None
def audio_drink_name(audio_path):
    prompt = '''
    Listen to the audio carefully.

    Task:
    Extract the person's name and their favorite drink.

    Rules:
    - Ignore background noise.
    - If a specific field is unclear or missing, use the word "nothing".
    - Return only the result in a list format: ["name", "drink"]
    - Use lowercase for all text.
    - Do not add any extra text, explanations, or punctuation outside the brackets.

    Format Examples:
    ["tom", "milk"]
    ["nothing", "coffee"]
    ["sarah", "nothing"]
    ["nothing", "nothing"]
    """
    '''

    model_output = send_media_task("/Fambot", audio_path, prompt, "audio")
    try:
        data = json.loads(model_output)
        customer_name = data[0]
        favorite_drink = data[1]
        print(f"Customer Name: {customer_name}")
        print(f"Favorite Drink: {favorite_drink}")

    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error processing model output: {e}")
        customer_name = "nothing"
        favorite_drink = "nothing"

    return customer_name, favorite_drink
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
def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
def turn_to(angle, speed):
    max_speed = 1.82
    start = rospy.get_time()

    while True:
        yaw = get_yaw_from_odom()
        e = normalize_angle(angle - yaw)

        if abs(e) < 0.01 or rospy.get_time() - start > 10:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0, 0)
def turn(angle):
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)
class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])

        if len(p) == 0:
            return -1, -1, -1
        return int(p[0][0]), int(p[0][1]), 1

    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0
        a1 = 55.0
        b1 = 86.0
        a = a1 * np.pi / 180
        b = b1 * np.pi / 180

        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.15)
        if x < 0:
            x = max(x, -0.15)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
        return z

    def calc_cmd_vel(self, image, depth, cx, cy) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        frame = image
        if cx == 2000:
            cur_x, cur_z = 0, 0
            return cur_x, cur_z, frame, "no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 800)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.15)
        if dx < 0:
            dx = max(dx, -0.15)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.4)
        if dz < 0:
            dz = max(dz, -0.4)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z, frame, "yes"
def turn_back(speed=0.4):
    for iii in range(round(300 / 100)):
        move(-0.2, 0)
        time.sleep(0.5)
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + np.pi)
    turn_to(target, speed)
def get_person_feature(image, user, api_key=None):
    img_path = save_user_crop(image, user)
    if img_path is None:
        print("save_user_crop failed")
        return None
    prompt = """
Describe the main person in the image.

Return JSON only:
{
  "hat": false,
  "glasses": false,
  "upper_color": "blue",
  "skin_tone": "light",
  "hair_length": "short",
  "age_group": "adult"
}

Rules:
- Output JSON only.
- Use one-word colors for upper_color.
- skin_tone must be one of: light, medium, dark, unclear.
- age_group must be one of: child, adult, elderly, unclear.
- If uncertain, use "unclear".
"""

    ans = send_media_task("/Vision", img_path, prompt, "image")
    print("[AI feature raw] =", ans)

    if ans is None:
        return None

    try:
        ans = str(ans).strip()
        ans = ans.replace("```json", "").replace("```", "").strip()
        return json.loads(ans)

    except Exception as e:
        print("feature parse error:", e)
        print("raw answer =", ans)
        return None
def feature_to_sentence(feature):
    if feature is None:
        return "I cannot describe the appearance"

    desc = []

    upper = feature.get("upper_color", "unclear")
    skin = feature.get("skin_tone", "unclear")
    hair = feature.get("hair_length", "unclear")
    age = feature.get("age_group", "unclear")

    if upper != "unclear":
        desc.append(f"wearing a {upper} top")

    if skin != "unclear":
        desc.append(f"with {skin} skin tone")

    if hair != "unclear":
        desc.append(f"with {hair} hair")

    if age != "unclear":
        desc.append(f"an {age}")

    if len(desc) == 0:
        return "I cannot describe the appearance"

    return ", ".join(desc)
def arm_open():
    global piper
    piper.set_gripper(1)
def arm_close():
    global piper
    piper.set_gripper(0.2)
def arm_extend():
    global piper
    piper.move_joint([0.592, 0.202, -1.418, 0.276, 0.584, 0.009])
def arm_retract():
    global piper
    piper.move_joint([0, 0, 0, 0, 0, 0], 50)
def receive_bag():
    arm_extend()
    arm_open()

    say("Please put your bag between my gripper.")

    say("I detect the object in my hand.")
    say("I will close my gripper in 3 seconds. Please be careful.")
    rospy.sleep(3)

    arm_close()
    rospy.sleep(1)

    arm_retract()
def get_door_depth_and_angle(image, depth):
    h, w = depth.shape[:2]

    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    y1 = int(h * 0.30)
    y2 = int(h * 0.80)

    crop_depth = depth[y1:y2, x1:x2]

    vals = crop_depth[crop_depth > 0]

    if len(vals) == 0:
        return None, 0

    door_depth = float(np.median(vals))
    min_d = np.percentile(vals, 20)

    mask = (crop_depth > 0) & (crop_depth < min_d + 100)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return door_depth, 0

    cx_crop = int(np.mean(xs))
    cx = x1 + cx_crop

    error_x = cx - w // 2

    # camera horizontal FOV 約 58 度
    angle = -error_x / (w / 2) * (58.0 / 2.0) * np.pi / 180.0

    print("[DOOR] depth =", door_depth)
    print("[DOOR] cx =", cx, "error_x =", error_x)
    print("[DOOR] angle rad =", angle)

    return door_depth, angle
def get_center_depth_and_turn(depth):
    h, w = depth.shape[:2]

    # 看畫面中間偏上/中間，避免地面
    x1 = int(w * 0.40)
    x2 = int(w * 0.60)
    y1 = int(h * 0.35)
    y2 = int(h * 0.65)

    roi = depth[y1:y2, x1:x2]
    vals = roi[roi > 0]

    if len(vals) == 0:
        return None, 0

    center_depth = float(np.median(vals))

    # 左右比較，判斷要不要微調角度
    left_roi = depth[y1:y2, int(w * 0.25):int(w * 0.45)]
    right_roi = depth[y1:y2, int(w * 0.55):int(w * 0.75)]

    left_vals = left_roi[left_roi > 0]
    right_vals = right_roi[right_roi > 0]

    turn_angle = 0

    if len(left_vals) > 0 and len(right_vals) > 0:
        left_d = float(np.median(left_vals))
        right_d = float(np.median(right_vals))

        print("[DOOR] left_d =", left_d, "right_d =", right_d)

        diff = left_d - right_d

        if abs(diff) > 120:
            if diff > 0:
                # 左邊較遠，向左少少
                turn_angle = +np.pi / 72
            else:
                # 右邊較遠，向右少少
                turn_angle = -np.pi / 72

    print("[DOOR] center_depth =", center_depth)
    print("[DOOR] turn_angle =", turn_angle)

    return center_depth, turn_angle
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
def align_and_approach_door():
    global _image, _depth

    image = _image.copy()
    depth = _depth.copy()

    door_depth, angle = get_door_depth_and_angle(image, depth)

    if door_depth is None:
        print("[DOOR] no valid depth")
        return False

    if abs(angle) > 0.05:
        turn(angle)

    rospy.sleep(0.5)

    target_depth = 850

    move_dist = door_depth - target_depth

    print("[DOOR] move_dist =", move_dist)

    if move_dist > 150:
        move_time = min(move_dist / 400.0, 2.0)
        move(0.15, 0)
        rospy.sleep(move_time)
        move(0, 0)

    return True
def detect_door_handle(image):
    results = door_handle_model(image, verbose=False)
    if len(results) == 0:
        return None
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    best_box = max(boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    conf = float(best_box.conf)
    return {
        "center": (cx, cy),
        "box": (x1, y1, x2, y2),
        "conf": conf
    }
def align_to_door_handle():
    global _image
    if _image is None:
        return False
    image = _image.copy()
    result = detect_door_handle(image)
    if result is None:
        print("No door handle.")
        return False
    h, w = image.shape[:2]
    cx = result["center"][0]
    error = cx - w // 2
    print("handle x =", cx)
    print("error =", error)
    x1, y1, x2, y2 = result["box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(image, (cx, result["center"][1]), 5, (0, 0, 255), -1)
    cv2.imshow("door_handle", image)
    cv2.waitKey(1)
    if abs(error) < 25:
        print("Door handle aligned.")
        return True
    angle = error * 0.002
    angle = max(min(angle, np.deg2rad(4)), -np.deg2rad(4))
    turn(-angle)
    rospy.sleep(0.4)
    return False
def get_front_door_depth():
    global _depth

    depth = _depth.copy()
    h, w = depth.shape[:2]

    x1 = int(w * 0.40)
    x2 = int(w * 0.60)
    y1 = int(h * 0.35)
    y2 = int(h * 0.65)

    roi = depth[y1:y2, x1:x2]
    vals = roi[roi > 0]

    if len(vals) == 0:
        print("[DOOR] no valid depth")
        return None

    d = float(np.median(vals))
    print("[DOOR] front depth =", d)

    return d
def adjust_door_distance():
    d = get_front_door_depth()

    if d is None:
        return False

    # 你話 arm 到門大約 800-900mm
    if d > 870:
        print("[DOOR] too far, move forward")
        move(0.12, 0)
        rospy.sleep(0.4)
        move(0, 0)
        return False

    elif d < 800:
        print("[DOOR] too close, move back")
        move(-0.08, 0)
        rospy.sleep(0.3)
        move(0, 0)
        return False

    else:
        print("[DOOR] good distance")
        return True
def open_door1():
    with open('/home/pcms/antony_ws/koreadoor1.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    x = True
    for i in data:
        piper.move_joint(i["joints"])
        if i["gripper"][0] > 30000 and x == True:
            piper.set_gripper(1, 1000)
            x = False
        elif i["gripper"][0] <= 30000 and x == False:
            piper.set_gripper(0, 1000)
            x = True
def open_door2():
    with open('/home/pcms/antony_ws/koreadoor2.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    x = True
    for i in data:
        piper.move_joint(i["joints"])
        if i["gripper"][0] > 30000 and x == True:
            piper.set_gripper(1, 1000)
            x = False
        elif i["gripper"][0] <= 30000 and x == False:
            piper.set_gripper(0, 1000)
            x = True
def open_door3():
    with open('/home/pcms/antony_ws/koreadoor3.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    x = True
    for i in data:
        piper.move_joint(i["joints"])
        if i["gripper"][0] > 30000 and x == True:
            piper.set_gripper(1, 1000)
            x = False
        elif i["gripper"][0] <= 30000 and x == False:
            piper.set_gripper(0, 1000)
            x = True


def empty_seatt(img, depth_img, yolo_model):
    person_in_roi = False
    need_position = None
    code_image = img.copy()
    h, w, _ = code_image.shape

    roi_w, roi_h = 200, 60
    roi_cx, roi_cy = w // 2, int(h * 0.8)

    roi_x1 = roi_cx - (roi_w // 2)
    roi_y1 = roi_cy - (roi_h // 2)
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    cv2.rectangle(code_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    results = yolo_model.forward(code_image)[0]["det"]

    for detection in results:
        dx1, dy1, dx2, dy2, score, class_id = detection
        ix1, iy1, ix2, iy2 = map(int, [dx1, dy1, dx2, dy2])

        person_cy = (iy1 + iy2) / 2.0

        if score > 0.5 and class_id == 0 and person_cy > h * 0.5:
            if ix1 < roi_x2 and ix2 > roi_x1 and iy1 < roi_y2 and iy2 > roi_y1:
                person_in_roi = True
                cv2.rectangle(code_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), -1)
                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2
                need_position = [ix1, iy1, ix2, iy2, cx, cy]

            cv2.rectangle(code_image, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)

    if person_in_roi:
        move(0, 0.3)
    else:
        move(0, 0)

    cv2.imshow("Bottom ROI", code_image)
    cv2.waitKey(1)
    return person_in_roi, need_position
if __name__ == "__main__":
    rospy.init_node("mission1")
    rospy.loginfo("mission1 start")
    chassis = RobotChassis()
    clear_costmaps
    _image = None
    _depth = None
    _voice = None
    _odom = None
    _Imu = None

    _sub_cam_image, _sub_cam_depth = None, None
    _current_cam = 1

    rospy.Subscriber("/voice/text", Voice, callback_voice)
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/respeaker/audio_path", String, callback_path)

    pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    change_to_cam(2)

    door_handle_model = YOLO("/home/pcms/antony_ws/door.pt")
    dnn_pose = HumanPoseEstimation(device_name="GPU")

    piper = KindaPiper()
    piper.connect("can0")
    followmecnt = 0
    guest1_name = ""
    guest1_drink = ""
    guest2_name = ""
    guest2_drink = ""
    speak_nack = 0
    host_seat = None

    empty_seat = None
    seat_names = ["left", "middle", "right"]
    point_dir = 0
    user = None
    _fw = FollowMe()
    lsp = 0
    front_angle = None
    seat_index = 0
    seat_angles = []

    guest1_feature = None
    guest2_feature = None
    host_feature = None

    guest1_now_seat = None
    host_now_seat = None

    handover_ready_count = 0
    guest2_seat = None
    guest1_seat = None
    target_bagx = 0
    target_bagy = 0
    target_bag_depth = 0
    best_guest1_score = -1
    best_host_score = -1
    host_user = None
    host_x = 0
    host_y = 0
    host_depth = -1
    host_search_count = 0
    recording_done = False
    arm_retract()
    state = 1
    dnn_yolo1 = Yolov8("yolov8n", device_name="GPU")
    scan_step = 45 * np.pi / 180
    far_right_step = 3 * np.pi / 4
    say("I am ready")
    number_seat = 5
    # start point: 4.049847164040595, 1.7364938075665215, -2.844931046420003)

    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if _image is None:
            print("image none")
            continue
        if _depth is None:
            print("depth none")
            continue
        if _odom is None:
            print("odom none")
            continue

        if len(_image) == 0 or len(_image[0]) == 0:
            continue
        if len(_depth) == 0 or len(_depth[0]) == 0:
            continue

        image = _image.copy()
        depth = _depth.copy()
        h, w, c = image.shape

        # state check
        if state == 1:
            print("state 0: ready")
            state = 2
            say("dear host please close the door")
            time.sleep(3)
            say("moving to door")
        elif state == 2:
            clear_costmaps
            chassis.move_to(1.1639211287046005, 0.4159578853659312, -3.1190504833965833)
            if chassis.status_code == 3:
                print("arrived")
                state = 3
            print(chassis.status_code)
        elif state == 3:
            change_to_cam(1)
            print("Align to door handle")
            ok = align_to_door_handle()
            if not ok:
                continue
            print("Door handle aligned.")
            ok = adjust_door_distance()
            if not ok:
                continue
            print("Ready to open.")
            turn(3 * np.pi / 180)
            open_door1()
            for iii in range(round(200 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            open_door2()
            for iii in range(round(300 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            open_door3()
            arm_retract()
            say("please wait for a moment")
            change_to_cam(2)
            state = 4
            say("moving to room")
        elif state == 4:
            print("[STATE 9999] MOVE_TO_ROOM")
            clear_costmaps
            chassis.move_to(1.9089344791625311, 0.06949383359396662, 3.0934575863400777)
            if chassis.status_code == 3:
                print("arrived")
                state = 5
                for iii in range(round(300 / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
            print(chassis.status_code)
        elif state == 5:
            print("state 1: find guest 1")
            poses = dnn_pose.forward(image)
            min_d = 9999
            t_idx = -1
            for i, pose in enumerate(poses):
                if (pose[5][2] == 0 or pose[6][2] == 0):
                    continue
                p5, p6 = list(map(int, pose[5][:2])), list(map(int, pose[6][:2]))

                cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                if d >= 2000 or d == 0: continue
                if d <= min_d:
                    t_idx, final_d, min_d = i, d, d

            if t_idx != -1:
                p5, p6 = list(map(int, poses[t_idx][5][:2])), list(map(int, poses[t_idx][6][:2]))
                cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                final_d = d
                step = "none"
                e = (image.shape[1] // 2) - cx
                v = np.clip(0.001 * e, -0.45, 0.45)
                move(0, v)
                if abs(e) <= 40:
                    state = 6
                    move(0, 0)
        elif state == 6:
            user = find_user(image, depth)
            if user is not None and len(user) > 0:
                say("hello I am detecting your feature now, please wait")
                feat = get_person_feature(image, user)
                print("guest1_feature =", feat)
                if feat is not None:
                    guest1_feature = feat
                    state = 7
                    for iii in range(round(300 / 100)):
                        move(0.2, 0)
                        time.sleep(0.5)
                else:
                    print("feature failed, retry")
        elif state == 7:
            print("state 3: ask guest1 name and drink")
            speak(
                "dear guest, What is your name and what is your favourite drink? Please say it in complete sentence with close and louder voice, for example my name is fambot and I like milk")
            os.system("play -q -n synth 0.2 sin 880")
            guest1_name, guest1_drink = "nothing", "nothing"
            for i in range(3):
                print(f"[Guest1 Name] trial {i + 1}/3")
                wav_path = record_audio_arecord(
                    "/tmp/guest1_name.wav",
                    12
                )
                if wav_path is None:
                    print("xxxxx")
                    say("Please repeat")
                    continue
                guest1_name1, guest1_drink1 = audio_drink_name(wav_path)
                if guest1_name1 != "nothing": guest1_name = guest1_name1
                if guest1_drink1 != "nothing": guest1_drink = guest1_drink1
                if "nothing" == guest1_name or "nothing" == guest1_drink:
                    say("Please repeat")
                else:
                    break
            speak_tet = "your name is " + guest1_name
            if guest1_name == "nothing":
                speak_tet = "I cannot get your name"
            say(speak_tet)
            speak_tet = "your favourite is " + guest1_drink
            if guest1_drink == "nothing":
                speak_tet = "I cannot get your favourite drink"
            say(speak_tet)
            turn_back()
            say("guest please follow me")
            state = 8
        elif state == 8:
            print("state 8: move to main point")
            clear_costmaps
            

            chassis.move_to(2.739,-0.327,-2.302)
            if chassis.status_code == 3:
                say("arrived")
                front_angle = get_yaw_from_odom()
                print("front_angle =", front_angle)
                seat_names = ["left", "middle", "right"]
                seat_angles = [
                    normalize_angle(front_angle + scan_step),
                    normalize_angle(front_angle),
                    normalize_angle(front_angle - scan_step)
                ]
                seat_index = 0
                empty_seat = None
                guest1_seat = None
                host_seat = None
                host_feature = None
                state = 9
                change_to_cam(1)

        elif state == 9:
              # 1 下2 上
            in_roi, position = empty_seatt(_image, _depth, dnn_yolo1)
            if in_roi:
                print("no seats")
            else:
                say("dear guest here is your seat")
                # time out add later
                time.sleep(2)
                state = 10

        # state10 : turn back to front
        elif state == 10:
            print("state 10: print out guest1 info and host info")
            print("guest1_name =", guest1_name)
            print("guest1_drink =", guest1_drink)
            print("guest1_feature =", guest1_feature)
            print("guest1_seat =", guest1_seat)
            print("host_seat =", host_seat)
            print("host_feature =", host_feature)
            state = 11

        elif state == 11:
            print("[STATE 10.5] MOVE_TO_GUEST2")
            say("moving to guest 2 interaction point")
            clear_costmaps
            chassis.move_to(1.9089344791625311, 0.06949383359396662, 3.0934575863400777)
            if chassis.status_code == 3:
                print("arrived")
                start = time.time()
                say("dear host please close the door")
                time.sleep(3)
                state = 12
            print(chassis.status_code)

        elif state == 12:
            print("[STATE 99999] move to door")
            say("moving to door")
            clear_costmaps
            chassis.move_to(1.1639211287046005, 0.4159578853659312, -3.1190504833965833)
            if chassis.status_code == 3:
                print("arrived")
                state = 13
            print(chassis.status_code)
        elif state == 13:
            change_to_cam(1)
            print("Align to door handle")
            ok = align_to_door_handle()
            if not ok:
                continue
            print("Door handle aligned.")
            ok = adjust_door_distance()
            if not ok:
                continue
            print("Ready to open.")
            turn(3 * np.pi / 180)
            open_door1()
            for iii in range(round(200 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            open_door2()
            for iii in range(round(300 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            open_door3()
            arm_retract()
            change_to_cam(2)
            state = 14

        # state 114: walk into the room
        elif state == 14:
            print("[STATE 114] MOVE_TO_ROOM")
            say("moving to room")
            clear_costmaps
            chassis.move_to(1.9089344791625311, 0.06949383359396662, 3.0934575863400777)
            if chassis.status_code == 3:
                print("arrived")
                state = 15
                for iii in range(round(300 / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)
            print(chassis.status_code)

        # state 11 : find guest 2
        elif state == 15:
            print("state 11: find guest 2")
            poses = dnn_pose.forward(image)
            min_d = 9999
            t_idx = -1
            for i, pose in enumerate(poses):
                if (pose[5][2] == 0 or pose[6][2] == 0):
                    continue
                p5, p6 = list(map(int, pose[5][:2])), list(map(int, pose[6][:2]))

                cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                if d >= 2000 or d == 0: continue
                if d <= min_d:
                    t_idx, final_d, min_d = i, d, d

            if t_idx != -1:
                p5, p6 = list(map(int, poses[t_idx][5][:2])), list(map(int, poses[t_idx][6][:2]))
                cx, cy = (p5[0] + p6[0]) // 2, (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                final_d = d
                step = "none"
                e = (image.shape[1] // 2) - cx
                v = np.clip(0.001 * e, -0.45, 0.45)
                move(0, v)
                if abs(e) <= 40:
                    state = 16
                    move(0, 0)

        elif state == 16:
            user = find_user(image, depth)
            if user is not None and len(user) > 0:
                say("hello")
                state = 17

        # state 1275: move to guest2 interaction point
        elif state == 17:
            print("[STATE 1275] MOVE_TO_GUEST2 interaction point")
            say("moving interaction point")
            clear_costmaps
            chassis.move_to(1.1639211287046005, 0.4159578853659312, -3.1190504833965833)
            if chassis.status_code == 3:
                print("arrived")
                state = 18
                for iii in range(round(300 / 100)):
                    move(0.2, 0)
                    time.sleep(0.5)

            print(chassis.status_code)
        elif state == 18:
            speak(
                "dear guest, What is your name and what is your favourite drink? Please say it in complete sentence with close and louder voice, for example my name is fambot and I like milk")
            os.system("play -q -n synth 0.2 sin 880")
            guest2_name, guest2_drink = "nothing", "nothing"
            for i in range(3):
                print(f"[Guest1 Name] trial {i + 1}/3")
                wav_path = record_audio_arecord(
                    "/tmp/guest2_name.wav",
                    12
                )
                if wav_path is None:
                    print("xxxxx")
                    say("Please repeat")
                    continue
                guest2_name1, guest2_drink1 = audio_drink_name(wav_path)
                if guest2_name1 != "nothing": guest2_name = guest2_name1
                if guest2_drink1 != "nothing": guest2_drink = guest2_drink1
                if "nothing" == guest2_name or "nothing" == guest2_drink:
                    say("Please repeat")
                else:
                    break
            speak_tet = "your name is " + guest2_name
            if guest2_name == "nothing":
                speak_tet = "I cannot get your name"
            say(speak_tet)
            speak_tet = "your favourite is " + guest2_drink
            if guest2_drink == "nothing":
                speak_tet = "I cannot get your favourite drink"
            say(speak_tet)
            state = 19

        elif state == 19:
            say(f"now I will tell you the feature of the first guest")
            guest1_desc = feature_to_sentence(guest1_feature)
            say(f"the first guest is {guest1_desc}")
            change_to_cam(1)
            print("state 155: receive bag")
            receive_bag()
            change_to_cam(2)
            state = 20
        elif state == 20:
            print("state 15: seat guest 2")
            turn_back()
            say("please follow me")
            print("state 18: move to main point")
            clear_costmaps
            chassis.move_to(2.739,-0.327,-2.302)
            if chassis.status_code == 3:
                say("arrived")

                empty_seat = None
                guest2_seat = None
                guest1_now_seat = None

                state = 21
                change_to_cam(1)  

        # scan seats (loop state)
        # state 19 : scan chairs, and check where guest1 is every turn
        elif state == 21:
            # 1 下2 上
            in_roi, position = empty_seatt(_image, _depth, dnn_yolo1)
            if in_roi:
                print("no position")
            else:
                say("dear guest here is your seat")
            time.sleep(2)
            state = 22
        # facing
        elif state == 9999999:
            print("state 20 : introduce everyone")
            guest1_desc = feature_to_sentence(guest1_feature)
            guest2_desc = feature_to_sentence(guest2_feature)
            if host_now_seat == "left":
                turn(+scan_step)
            elif host_now_seat == "middle":
                pass
            elif host_now_seat == "right":
                turn(-scan_step)
            rospy.sleep(1)
            say(
                f"Host, this is {guest2_name}. "
                f"{guest2_name} likes {guest2_drink}. "
                f"{guest2_name} is {guest2_desc}."
            )
            rospy.sleep(1)
            if guest1_now_seat == "left":
                if host_now_seat == "middle":
                    turn(+scan_step)
                elif host_now_seat == "right":
                    turn(+2 * scan_step)

            elif guest1_now_seat == "middle":
                if host_now_seat == "left":
                    turn(-scan_step)
                elif host_now_seat == "right":
                    turn(+scan_step)

            elif guest1_now_seat == "right":
                if host_now_seat == "middle":
                    turn(-scan_step)
                elif host_now_seat == "left":
                    turn(-2 * scan_step)
            rospy.sleep(1)
            say(
                f"{guest1_name}, this is {guest2_name}. "
                f"{guest2_name} likes {guest2_drink}. "
                f"{guest2_name} is {guest2_desc}."
            )
            rospy.sleep(1)

            # ---------- Introduce Guest1 to Guest2 ----------
            if guest2_seat == "left":
                if guest1_now_seat == "middle":
                    turn(+scan_step)
                elif guest1_now_seat == "right":
                    turn(+2 * scan_step)

            elif guest2_seat == "middle":
                if guest1_now_seat == "left":
                    turn(-scan_step)
                elif guest1_now_seat == "right":
                    turn(+scan_step)

            elif guest2_seat == "right":
                if guest1_now_seat == "middle":
                    turn(-scan_step)
                elif guest1_now_seat == "left":
                    turn(-2 * scan_step)
            rospy.sleep(1)
            say(
                f"{guest2_name}, this is {guest1_name}. "
                f"{guest1_name} likes {guest1_drink}. "
                f"{guest1_name} is {guest1_desc}."
            )
            rospy.sleep(1)

            # ---------- Turn to host ----------
            if host_now_seat == "left":
                turn(+scan_step)
            elif host_now_seat == "middle":
                pass
            elif host_now_seat == "right":
                turn(-scan_step)
            rospy.sleep(1)
            say("Host, please guide me.")
            for iii in range(round(300 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            turn(-90)
            state = 22
        elif state == 22:
            turn(-180)
            say("dear host, please stand in front of me, and please please don't walk too fast")
            say("please say robot stop robot stop,when you arrive")
            say("3")
            say("2")
            say("1")
            state = 22
        # state 21 : follow host
        elif state == 23:
            msg = Twist()
            code_image = _image.copy()
            poses = dnn_pose.forward(code_image)
            depth = _depth.copy()
            min_d = 9999
            t_idx = -1
            for i, pose in enumerate(poses):
                if pose[5][2] == 0 or pose[6][2] == 0:
                    continue
                p5 = list(map(int, pose[5][:2]))
                p6 = list(map(int, pose[6][:2]))

                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                cv2.circle(code_image, p5, 5, (0, 0, 255), -1)
                cv2.circle(code_image, p6, 5, (0, 0, 255), -1)
                cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                if d >= 1800 or d == 0: continue
                if (d != 0 and d < min_d):
                    t_idx = i
                    min_d = d

            x, z = 0, 0
            if t_idx != -1:
                p5 = list(map(int, poses[t_idx][5][:2]))
                p6 = list(map(int, poses[t_idx][6][:2]))
                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(depth, cx, cy, 2)
                cv2.circle(code_image, (cx, cy), 5, (0, 255, 255), -1)

                print("people_d", d)
                if d >= 1800 or d == 0: continue

                x, z, code_image, yn = _fw.calc_cmd_vel(code_image, depth, cx, cy)
                print("turn_x_z:", x, z)
            if x == 0 and z == 0:
                if speak_nack >= 20:
                    say("don't walk too fast, please come back")
                    speak_nack = 0
                speak_nack += 1
                followmecnt += 1
            else:
                followmecnt = 0
            if followmecnt >= 100:
                step = "back"
                say("I can't find you I gonna go back now")
                followmecnt = 0
            if "stop" in _voice:
                state = -1111
                piper.move_to_endpoint2(300, 0, 0, 0, -1.57, 0)
                piper.set_gripper(1)
                say("mission end")
            print("follow", followmecnt)
            move(x, z)
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rospy.sleep(0.05)
