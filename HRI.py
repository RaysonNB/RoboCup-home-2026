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
from RobotChassis import RobotChassis  # Your chassis driver
from tf.transformations import euler_from_quaternion
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import subprocess
import sounddevice as sd
from std_srvs.srv import Empty
from scipy.io.wavfile import write
from piper_sdk_demo import KindaPiper
from pcms.openvino_models import *
from pcms.pytorch_models import *
import requests
import json
import base64
import time
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pyautogui
import os
import tempfile
import subprocess
import librosa
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

clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
OPENROUTER_API_KEY = "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"
VISION_MODEL = "qwen/qwen3.5-flash-02-23"
MODEL = "xiaomi/mimo-v2.5"
CHAIR_MODEL = "qwen/qwen3.5-flash-02-23"
AUDIO_DEVICE = ""
AUDIO_FS = 16000
AUDIO_CHANNELS = 1
API_KEY = "sk-lm-Z7kmhaBy:URWIhcO11V4clG2ex936"

_image = None
_depth = None
_image_seq = 0
_depth_seq = 0
_voice = None
_odom = None
_low_image = None
_low_depth = None
CvBr = CvBridge()
code = 0
_current_cam = 1
_sub_cam_image = None
_sub_cam_depth = None
bridge = CvBridge()
_audio_path = None
_new_audio_flag = False

NAME_LIST = [
    "sophie", "kevin", "julia", "gabrielle", "emma", "robin",
    "noah", "sara", "john", "harrie", "laura", "liam",
    "peter", "lucas", "susan", "william"
]

DRINK_LIST = [
    "lemonade", "coffee can", "fanta", "yakult",
    "green tea", "milk", "red bull can"
]

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
model = YOLO("yolov8n-pose.pt")


# def tezhengyi():
def annotate_empty_seats(frame, output_dir=".", filename="temp_annotated.jpg"):
    check_empty_img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (255, 255, 255)
    font_thickness = 5

    height, width = check_empty_img.shape[:2]

    positions = {
        1: (int(width * 0.3), int(height * 0.5)),
        2: (int(width * 0.5), int(height * 0.5)),
        3: (int(width * 0.7), int(height * 0.5)),
    }

    for number, pos in positions.items():
        text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
        rectangle_start = (pos[0] - 10, pos[1] + 10)
        rectangle_end = (pos[0] + text_size[0] + 10, pos[1] - text_size[1] - 10)
        cv2.rectangle(check_empty_img, rectangle_start, rectangle_end, (0, 0, 0), cv2.FILLED)
        cv2.putText(check_empty_img, str(number), pos, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, check_empty_img)

    return output_path


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


def callback_low_image(msg):
    global _low_image
    _low_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_low_depth(msg):
    global _low_depth
    _low_depth = CvBridge().imgmsg_to_cv2(msg)


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
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.5)
def speak(g):
    print("[robot said]: ", end=" ")
    os.system(f'espeak -s 120 "{g}"')
    # rospy.loginfo(g)
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


def wait_for_new_audio(timeout=5):
    global _audio_path, _new_audio_flag, _voice

    start = time.time()
    while time.time() - start < timeout:
        path = None
        print(_voice, end="\r", flush=False)
        if _voice is not None and _audio_path is not None:
            path = _audio_path
            _voice = None
            _audio_path = None
            print(path)
            if os.path.exists(path):
                _new_audio_flag = False
                print("use audio file:", path)
                break
        # rospy.sleep(0.05)
    print(path)
    return path


def encode_audio(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


SAMPLE_RATE = 44100
RECORD_SECONDS = 5
ALSA_DEVICE = "pulse"


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


BELL_THRESHOLD = 160000
BELL_TARGET_PATH = "/home/pcms/antony_ws/bellring"
_last_checked_audio_path = None
BELL_SAMPLE_RATE = 44100

def match_from_list(text, valid_list):
    if text is None:
        return None

    text = text.lower().strip()

    for item in valid_list:
        if item in text:
            return item

    return None


def wait_new_frame(old_seq, timeout=3.0):
    start = rospy.get_time()
    while rospy.get_time() - start < timeout:
        if _image is not None and _image_seq > old_seq:
            return True
        rospy.sleep(0.05)
    return False


prp = 0


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


def bag_any_xy(image):
    h, w, c = image.shape
    result = luggage_model(image, verbose=False)[0]
    boxes = result.boxes

    best_box = None
    best_score = -1

    for cls, xyxy in zip(boxes.cls, boxes.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        if cy > h * 0.9:
            continue

        score = area - abs(cx - w // 2) * 150

        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return 0, 0, None

    x1, y1, x2, y2 = best_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return cx, cy, (x1, y1, x2, y2)


def is_bag_handover_from_guest2(image, depth, user):
    h, w = depth.shape

    person_box = get_pose_box(user, image.shape)
    if person_box is None:
        return False, 0, 0, 0

    px1, py1, px2, py2 = person_box
    pcx = (px1 + px2) // 2
    pcy = (py1 + py2) // 2
    pw = px2 - px1
    ph = py2 - py1

    cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 255), 2)

    bagx, bagy, bag_box = bag_any_xy(image)
    if bag_box is None:
        return False, 0, 0, 0

    bx1, by1, bx2, by2 = bag_box
    bcx = (bx1 + bx2) // 2
    bcy = (by1 + by2) // 2

    if bagx < 0 or bagx >= w or bagy < 0 or bagy >= h:
        return False, 0, 0, 0

    bag_depth = depth[bagy][bagx]

    if bag_depth == 0:
        return False, bagx, bagy, 0

    if abs(bcx - pcx) > pw * 1.0:
        return False, bagx, bagy, bag_depth

    if bcy > py2 + ph * 0.15:
        return False, bagx, bagy, bag_depth

    if bcy < py1 + ph * 0.10:
        return False, bagx, bagy, bag_depth

    if bag_depth > 1200:
        return False, bagx, bagy, bag_depth

    if bag_depth < 250:
        return False, bagx, bagy, bag_depth

    return True, bagx, bagy, bag_depth
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


def user_point_to_LR(pose):
    if pose is None: return 0
    if len(pose) == 0: return 0
    x0, y0 = map(int, pose[0])
    x8, y8 = map(int, pose[10])
    if x0 == x8: return 0
    k = (y0 - y8) / (x0 - x8)
    if abs(k) > 1.2: return 0
    if k > 0: return 1
    return -1


def bag_xy(dir, image):
    h, w, cc = image.shape
    result = luggage_model(image, verbose=False)[0]
    boxes = result.boxes

    for c, xyxy in zip(boxes.cls, boxes.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        b = 0
        if int((x1 + x2) / 2) > w // 2:
            b = 1
        if dir == b:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            return int((x1 + x2) / 2), int((y1 + y2) / 2)
    return 0, 0


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
def draw_chair_numbers(image):
    check_empty_img = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_color = (255, 255, 255)
    font_thickness = 5

    h, w = check_empty_img.shape[:2]

    print("image size =", w, h)

    # 🔥 用比例，而不是寫死 640, 320
    positions = {
        1: (int(w * 0.15), int(h * 0.65)),  # 左椅
        2: (int(w * 0.50), int(h * 0.70)),  # 中間椅
        3: (int(w * 0.80), int(h * 0.65)),  # 右椅
    }

    for number, pos in positions.items():
        text = str(number)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        x, y = pos

        # 黑底
        rectangle_start = (x - 15, y + 15)
        rectangle_end = (x + text_size[0] + 15, y - text_size[1] - 15)
        cv2.rectangle(check_empty_img, rectangle_start, rectangle_end, (0, 0, 0), cv2.FILLED)

        # 數字
        cv2.putText(
            check_empty_img,
            text,
            pos,
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA
        )

        # debug點
        cv2.circle(check_empty_img, pos, 5, (0, 255, 0), -1)

    return check_empty_img


def chair_id_to_seat(chair_id):
    if chair_id == 1:
        return "left"
    elif chair_id == 2:
        return "middle"
    elif chair_id == 3:
        return "right"
    return None


BASE_URL = "http://192.168.88.89:8888"


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

    return customer_name,favorite_drink

def audio_extract_name(audio_path, api_key=None, model=None):
    prompt = """
Listen to the audio carefully.

Task:
Extract the person's name.

Rules:
- Ignore background noise.
- If unclear, answer: nothing
- Return only the name, lowercase.
- Do not add extra text.
"""

    ans = send_media_task("/Fambot", audio_path, prompt, "audio")
    print("[AI name raw] =", ans)

    if ans is None:
        return {"success": False, "name": None}

    ans = str(ans).lower().strip()
    name = match_from_list(ans, NAME_LIST)

    if name is None:
        return {"success": False, "name": None}

    return {"success": True, "name": name}


# def audio_extract_name(audio_path, api_key, model=MODEL):
#     try:
#         base64_audio = encode_audio(audio_path)

#         prompt = """
# Transcribe exactly what the speaker says.

# Rules:
# - Do NOT guess.
# - Do NOT invent names.
# - If unclear, output exactly: unclear
# - Output only the spoken sentence.

# NAME_LIST = [
#      "sophie", "kevin", "julia", "gabrielle", "emma", "robin",
#     "noah", "sara", "john", "harrie", "laura", "liam",
#     "peter", "lucas", "susan", "william"
# ]


# DRINK_LIST= [
#  "lemonade", "coffee can", "fanta", "yakult",
#     "green tea", "milk", "red bull can"
# ]

# These are the name list and drink list. When you are replying a name, make sure it is in this list. If not, find a closest name in this list comparing with the audio.
# This also occurs to the drink.
# """

#         payload = {
#             "model": model,
#             "messages": [{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "input_audio",
#                         "input_audio": {
#                             "data": base64_audio,
#                             "format": "wav"
#                         }
#                     }
#                 ]
#             }],
#             "temperature": 0
#         }

#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             },
#             data=json.dumps(payload),
#             timeout=15
#         )

#         result = response.json()

#         if "choices" not in result:
#             return {"success": False, "name": None}

#         text = result["choices"][0]["message"].get("content")
#         if text is None:
#             return {"success": False, "name": None}

#         text = text.strip().lower()
#         print("name transcript =", text)

#         name = match_from_list(text, NAME_LIST)

#         if name is None:
#             return {"success": False, "name": None}

#         return {"success": True, "name": name}

#     except Exception as e:
#         print("audio_extract_name exception:", e)
#         return {"success": False, "name": None}

def extract_name(text):
    text = text.lower().strip()

    patterns = ["my name is", "i am", "i'm"]

    for p in patterns:
        if p in text:
            name = text.split(p)[-1].strip()
            if name:
                return name.split()[0]

    words = text.split()
    if len(words) == 1:
        return words[0]

    return None


def get_name_with_retry(default_name="friend", max_retry=2):
    for attempt in range(max_retry + 1):
        print(f"[Name] attempt {attempt + 1}/{max_retry + 1}")

        wav_path = record_audio_arecord(f"/tmp/guest_name_{attempt}.wav", 5)
        if wav_path is None:
            continue

        res = audio_extract_name(wav_path, OPENROUTER_API_KEY)
        print("name result =", res)

        if res["success"] and res["name"] is not None:
            return str(res["name"]).strip()

    return default_name


def get_drink_with_retry(default_drink="water", max_retry=2):
    for attempt in range(max_retry + 1):
        print(f"[Drink] attempt {attempt + 1}/{max_retry + 1}")

        wav_path = record_audio_arecord(f"/tmp/guest_drink_{attempt}.wav", 5)
        if wav_path is None:
            continue

        res = audio_extract_drink(wav_path, OPENROUTER_API_KEY)
        print("drink result =", res)

        if res["success"] and res["drink"] is not None:
            return str(res["drink"]).strip()

    return default_drink


# def audio_extract_drink(audio_path, api_key, model=MODEL):
#     try:
#         base64_audio = encode_audio(audio_path)

#         prompt = """
# Transcribe exactly what the speaker says.

# Rules:
# - Do NOT guess.
# - Do NOT invent drinks.
# - If unclear, output exactly: unclear
# - Output only the spoken sentence.


# DRINK_LIST= [
#  "lemonade", "coffee can", "fanta", "yakult",
#     "green tea", "milk", "red bull can"
# ]

# These are the name list and drink list. When you are replying a name, make sure it is in this list. If not, find a closest drink in this list comparing with the audio.

# """

#         payload = {
#             "model": model,
#             "messages": [{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "input_audio",
#                         "input_audio": {
#                             "data": base64_audio,
#                             "format": "wav"
#                         }
#                     }
#                 ]
#             }],
#             "temperature": 0
#         }

#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             },
#             data=json.dumps(payload),
#             timeout=15
#         )

#         result = response.json()

#         if "choices" not in result:
#             return {"success": False, "drink": None}

#         text = result["choices"][0]["message"].get("content")
#         if text is None:
#             return {"success": False, "drink": None}

#         text = text.strip().lower()
#         print("drink transcript =", text)

#         drink = match_from_list(text, DRINK_LIST)

#         if drink is None:
#             return {"success": False, "drink": None}

#         return {"success": True, "drink": drink}

#     except Exception as e:
#         print("audio_extract_drink exception:", e)
#         return {"success": False, "drink": None}

def audio_extract_drink(audio_path, api_key=None, model=None):
    prompt = """
Listen to the audio carefully.

Task:
Extract the drink.

Rules:
- Ignore background noise.
- If unclear, answer: nothing
- Return only the drink, lowercase.
- Do not add extra text.
"""

    ans = send_media_task("/Fambot", audio_path, prompt, "audio")
    print("[AI drink raw] =", ans)

    if ans is None:
        return {"success": False, "drink": None}

    ans = str(ans).lower().strip()
    drink = match_from_list(ans, DRINK_LIST)

    if drink is None:
        return {"success": False, "drink": None}

    return {"success": True, "drink": drink}


def get_name_with_retry(default_name="friend", max_retry=2):
    for attempt in range(max_retry + 1):
        print(f"[Name] attempt {attempt + 1}/{max_retry + 1}")

        wav_path = record_audio(filename=f"/tmp/guest_name_{attempt}.wav", seconds=3)
        if wav_path is None:
            print("record failed")
            continue

        res = audio_extract_name(wav_path)
        print("name result =", res)

        if res["success"] and res["name"] is not None:
            name = str(res["name"]).strip()
            if name != "":
                return name

    print("fallback name =", default_name)
    return default_name


# ===== odom turn =====
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

        if abs(e) < 0.01 or rospy.get_time() - start > 8:
            break

        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()

    move(0, 0)


def turn(angle):
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


def detect_real_xyz(x1, y1, d):
    global image
    h, w, c = image.shape
    y1 -= h / 2
    x1 -= w / 2
    y = y1 / h * 2 * d * math.tan(45.5 * 3.14159 / 360)
    x = x1 / w * 2 * d * math.tan(58.4 * 3.14159 / 360)
    return x, y, d
door_baseline = None


def set_joints(j1, j2, j3, j4, t):
    rospy.wait_for_service("/goal_joint_space_path")
    try:
        s = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)
        req = SetJointPositionRequest()
        req.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
        req.joint_position.position = [j1, j2, j3, j4]
        req.path_time = t
        return s(req)
    except:
        return False

def draw_chair_numbers(image):
    img = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_color = (255, 255, 255)
    font_thickness = 5

    h, w = img.shape[:2]

    positions = {
        1: (int(w * 0.15), int(h * 0.65)),  # 左
        2: (int(w * 0.50), int(h * 0.70)),  # 中
        3: (int(w * 0.80), int(h * 0.65)),  # 右
    }

    for number, (x, y) in positions.items():
        text = str(number)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        cv2.rectangle(img, (x - 15, y + 15), (x + text_size[0] + 15, y - text_size[1] - 15), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return img


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def follow1():
    global _voice
    say("I will follow you now.")
    print("Starting to follow the person.")

    # PID variables for angular control (keep person centered)
    ang_kp = 0.7  # stronger response
    ang_ki = 0.001
    ang_kd = 0.02  # Increased slightly for damping
    ang_prev_error = 0.0
    ang_integral = 0.0
    ang_last_time = time.time()

    # PID variables for linear control
    desired_dist_mm = 600
    lin_kp = 0.0005
    lin_ki = 0.00005
    lin_kd = 0.0003
    lin_prev_error = 0.0
    lin_integral = 0.0
    lin_last_time = time.time()

    stop_threshold_mm = 50  # Stop if within 50mm of target
    braking_zone_mm = 150  # Start braking 200mm from target
    max_forward_speed = 0.5  # Max speed (reduced for better control)
    max_backward_speed = -0.25

    person_lost_counter = 0
    max_lost_frames = 30

    while not rospy.is_shutdown():
        msg = Twist()
        image = _image.copy()
        depth = _depth.copy()

        if image is None or depth is None:
            rospy.sleep(0.05)
            continue

        h, w = image.shape[:2]
        center_x = w // 2

        # Detect persons
        poses = dnn_pose.forward(image)
        targetx, targety, target_depth_mm = None, None, None
        best_depth = float('inf')

        for pose in poses:
            x1, y1, x2, y2 = 10000, 10000, -1, -1
            for cx, cy, score in pose:
                if cx != 0 and cy != 0:
                    x1 = int(min(x1, cx))
                    y1 = int(min(y1, cy))
                    x2 = int(max(x2, cx))
                    y2 = int(max(y2, cy))
            if x1 == 10000 or x2 == -1:
                continue

            midx = (x1 + x2) // 2
            midy = (y1 + y2) // 2
            d = depth[midy, midx]
            if np.isnan(d) or d == 0:
                region = depth[y1:y2, x1:x2]
                if region.size > 0:
                    d = np.median(region[region > 0])
                else:
                    d = float('inf')

            if d < best_depth:
                best_depth = d
                targetx, targety = midx, midy
                target_depth_mm = d

        # Voice stop check
        if _voice is not None:
            sentence = _voice.text.strip()
            if "stop" in sentence.lower():
                say("Stopping following.")
                _voice = None
                return
            _voice = None

        # Person lost handling
        if targetx is None:
            person_lost_counter += 1
            if person_lost_counter > max_lost_frames:
                say("I lost you. Stopping.")
                return
            msg.angular.z = 0.3
            msg.linear.x = 0.0
            pub_cmd.publish(msg)
            rospy.sleep(0.05)
            continue
        else:
            person_lost_counter = 0

        # Angular PID
        ang_error = (targetx - center_x) / (w / 2)
        ang_error = max(-1.0, min(1.0, ang_error))
        now = time.time()
        dt_ang = now - ang_last_time
        if dt_ang > 0.1:
            dt_ang = 0.05
        ang_last_time = now
        ang_integral += ang_error * dt_ang
        ang_derivative = (ang_error - ang_prev_error) / dt_ang if dt_ang > 0 else 0
        ang_vel = ang_kp * ang_error + ang_ki * ang_integral + ang_kd * ang_derivative
        ang_vel = max(-0.8, min(0.8, ang_vel))
        ang_prev_error = ang_error

        # Linear PID with IMPROVED STOPPING LOGIC
        lin_error = target_depth_mm - desired_dist_mm

        # CRITICAL FIX: Stop if within threshold
        if abs(lin_error) < stop_threshold_mm:
            # Within 50mm of desired distance - STOP
            lin_vel = 0.0
            lin_integral = 0.0  # Reset integral to prevent windup
            # print(f"Stopped at {target_depth_mm}mm (within {stop_threshold_mm}mm of target)")
        else:
            # Calculate PID output
            dt_lin = now - lin_last_time
            if dt_lin > 0.1:
                dt_lin = 0.05
            lin_last_time = now

            # Anti-windup: limit integral accumulation
            lin_integral += lin_error * dt_lin
            lin_integral = max(-100, min(100, lin_integral))  # Clamp integral

            lin_derivative = (lin_error - lin_prev_error) / dt_lin if dt_lin > 0 else 0
            lin_vel = lin_kp * lin_error + lin_ki * lin_integral + lin_kd * lin_derivative

            # APPLY BRAKING ZONE (gradual slowdown)
            distance_to_target = abs(lin_error)
            if distance_to_target < braking_zone_mm:
                # Progressive braking: slower as you get closer
                brake_factor = distance_to_target / braking_zone_mm
                lin_vel = lin_vel * brake_factor
                # print(f"Braking: {distance_to_target}mm from target, speed={lin_vel:.3f}")

            # EMERGENCY STOP if too close
            if target_depth_mm < desired_dist_mm - 100:  # 100mm too close
                lin_vel = max(lin_vel, -0.1)  # Allow only backward or zero
                if target_depth_mm < 500:  # Dangerously close
                    lin_vel = -0.05  # Small reverse
                    # print("Emergency reverse!")

            # Limit speeds
            lin_vel = max(max_backward_speed, min(max_forward_speed, lin_vel))

            # Special case: moving forward but too close
            if target_depth_mm < desired_dist_mm and lin_vel > 0:
                lin_vel = 0  # Immediate stop
                # print("Too close - stopping forward movement")

        lin_prev_error = lin_error

        # Send commands
        msg.angular.z = -ang_vel
        msg.linear.x = lin_vel
        pub_cmd.publish(msg)

        # Debug info
        # print(f"Dist: {target_depth_mm:.0f}mm | Target: {desired_dist_mm}mm | Error: {lin_error:.0f}mm | Vel: {lin_vel:.3f}")

        # Display
        cv2.circle(image, (targetx, targety), 8, (0, 0, 255), -1)
        cv2.putText(image, f"dist={target_depth_mm:.0f}/{desired_dist_mm}mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"vel={lin_vel:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Color code based on distance
        if target_depth_mm < desired_dist_mm - 50:
            cv2.putText(image, "TOO CLOSE!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif target_depth_mm > desired_dist_mm + 50:
            cv2.putText(image, "TOO FAR", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rospy.sleep(0.05)


def ask_empty_chair_api(image_path):
    base64_image = encode_image(image_path)

    prompt = """
Look at the image.
There are three chairs labeled 1, 2, and 3.
Tell me which chair is empty.

Return JSON only:
{"empty_chair": 1}
"""

    payload = {
        "model": CHAIR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload),
            timeout=20
        )

        result = response.json()

        if "choices" not in result:
            print("API error:", result)
            return None

        answer = result["choices"][0]["message"]["content"].strip()

        # clean ```json```
        answer = re.sub(r"^```json\s*", "", answer)
        answer = re.sub(r"^```\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer)

        data = json.loads(answer)
        return data.get("empty_chair")

    except Exception as e:
        print("API fail:", e)
        return None


def detect_empty_seat(image):
    # 存暫存圖
    path = "/tmp/chair.jpg"
    labeled = draw_chair_numbers(image)
    cv2.imwrite(path, labeled)

    results = []
    for i in range(1):
        cid = ask_empty_chair_api(path)
        print(f"trial {i + 1} =", cid)
        if cid in [1, 2, 3]:
            results.append(cid)

    if len(results) == 0:
        return None

    cid = max(set(results), key=results.count)

    if cid == 1:
        return "left"
    elif cid == 2:
        return "middle"
    elif cid == 3:
        return "right"


def turn_back(speed=0.4):
    for iii in range(round(300 / 100)):
        move(-0.2, 0)
        time.sleep(0.5)
    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + np.pi)
    turn_to(target, speed)


def user_point_to_LR(pose):
    if pose is None: return 2
    if len(pose) == 0: return 2
    x0, y0 = map(int, pose[8])
    x8, y8 = map(int, pose[10])
    if x0 == x8: return 2
    k = (y0 - y8) / (x0 - x8)
    if abs(k) > 1.2: return 2
    if k > 0: return 1
    return 0


def wait_until_door_open():
    global _image, _depth

    depth_zero = 0

    say("I am ready.")

    start_time = rospy.get_time()

    while not rospy.is_shutdown():

        if rospy.get_time() - start_time > 20:
            print("[DOOR] timeout")
            cv2.destroyWindow("door_check")
            return False

        if _image is None or _depth is None:
            rospy.sleep(0.05)
            continue

        frame = _image.copy()
        depth = _depth.copy()

        h, w = depth.shape[:2]

        cx = w // 2
        cy = h // 2

        d = depth[cy, cx]

        print("[DOOR] depth =", d)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame,
                    str(d),
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        cv2.imshow("door_check", frame)
        cv2.waitKey(1)

        if d == 0:
            depth_zero += 1
        else:
            depth_zero = 0

        if depth_zero >= 20:
            print("[DOOR] door opened")
            cv2.destroyWindow("door_check")
            return True

        if d > 1700:
            print("[DOOR] door opened")
            cv2.destroyWindow("door_check")
            return True

        rospy.sleep(0.05)

    cv2.destroyWindow("door_check")
    return False


def bag_xy(dir, image):
    h, w, cc = image.shape
    result = luggage_model(image, verbose=False)[0]

    boxes = result.boxes

    res = zip(boxes.cls, boxes.xyxy)
    for c, xyxy in res:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        b = 0
        if int((x1 + x2) / 2) > w // 2: b = 1
        print("dir,b                           ", dir, b)
        if dir == b:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            return int((x1 + x2) / 2), int((y1 + y2) / 2)
    return 0, 0


def bag_any_xy(image):
    h, w, c = image.shape
    result = luggage_model(image, verbose=False)[0]

    boxes = result.boxes

    best_box = None
    best_score = -1

    for cls, xyxy in zip(boxes.cls, boxes.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        box_area = (x2 - x1) * (y2 - y1)
        # underground
        if cy > h * 0.9:
            continue

        # score = area + if in the middle
        center_bias = abs(cx - w // 2)
        score = box_area - center_bias * 200

        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy

    return 0, 0


def is_bag_handover_ready(image, depth):
    h, w, c = image.shape

    bagx, bagy = bag_any_xy(image)

    if bagx == 0 and bagy == 0:
        return False, 0, 0, 0

    if bagx < 0 or bagx >= w or bagy < 0 or bagy >= h:
        return False, 0, 0, 0

    bag_depth = depth[bagy][bagx]
    if bag_depth == 0:
        return False, bagx, bagy, 0
    if bag_depth > 1200:
        return False, bagx, bagy, bag_depth
    if bag_depth < 250:
        return False, bagx, bagy, bag_depth
    if bagy > h * 0.85:
        return False, bagx, bagy, bag_depth
    return True, bagx, bagy, bag_depth
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


def feature_score(f1, f2):
    score = 0

    if f1 is None or f2 is None:
        return -1

    if f1.get("hat") == f2.get("hat"):
        score += 2

    if f1.get("glasses") == f2.get("glasses"):
        score += 2

    if f1.get("upper_color") == f2.get("upper_color"):
        score += 5

    if f1.get("skin_tone") == f2.get("skin_tone"):
        score += 3

    if f1.get("hair_length") == f2.get("hair_length"):
        score += 2

    if f1.get("age_group") == f2.get("age_group"):
        score += 1

    return score


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


def detect_person_in_view(image, depth):
    user = find_user(image, depth)
    if user is not None and len(user) > 0:
        return user
    return None


def find_user_seated(image, depth, conf_thres=0.25, min_box_area=4000):
    """
    return format:
        None 或 [best_person]

    best_person = {
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "depth": d,
        "conf": conf
    }
    """
    try:
        result = model(image)[0]
    except Exception as e:
        print("[find_user_seated] model error:", e)
        return None

    if result.boxes is None or len(result.boxes) == 0:
        return None

    print("find user seated:        found person")

    h, w = image.shape[:2]
    img_center_x = w / 2

    best_person = None
    best_score = -1e9

    for box in result.boxes:
        try:
            cls_id = int(box.cls[0]) if len(box.cls.shape) > 0 else int(box.cls)
            conf = float(box.conf[0]) if len(box.conf.shape) > 0 else float(box.conf)

            if cls_id != 0:
                continue
            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < min_box_area:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            print("depth:", depth[cy][cx])
            if depth[cy][cx] > 2500:
                continue

            x_min = max(0, cx - 10)
            x_max = min(w, cx + 10)
            y_min = max(0, cy - 10)
            y_max = min(h, cy + 10)

            patch = depth[y_min:y_max, x_min:x_max]
            vals = patch[patch > 0]
            d = float(vals.mean()) if len(vals) > 0 else -1

            depth_score = 0 if d < 0 else (-d * 0.01)
            center_score = -abs(cx - img_center_x) * 0.5
            area_score = area * 0.0005
            conf_score = conf * 100

            total_score = depth_score + center_score + area_score + conf_score

            if total_score > best_score:
                best_score = total_score
                best_person = {
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "depth": d,
                    "conf": conf
                }

        except Exception as e:
            print("[find_user_seated] skip error:", e)
            continue

    if best_person is None:
        return None

    print("[find_user_seated] result =", best_person)
    return [best_person]
people_cnt = 0


def get_person_feature_seated(image, user, api_key=None):
    global people_cnt
    if user is None or len(user) == 0:
        return None

    x1, y1, x2, y2 = user[0]["box"]
    crop = image[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        print("invalid crop")
        return None

    img_path = "/tmp/person1.jpg"
    people_cnt += 1
    cv2.imwrite(img_path, crop)

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
    print("[AI seated feature raw] =", ans)

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


def crop_center_third(image, depth):
    h, w = image.shape[:2]

    x1 = int(w * 0.33)
    x2 = int(w * 0.67)

    y1 = int(h * 0.10)
    y2 = int(h * 0.95)

    crop_img = image[y1:y2, x1:x2]
    crop_depth = depth[y1:y2, x1:x2]

    return crop_img, crop_depth, x1, y1


def scan_people_roles(image, depth, scan_step, guest1_feature, host_feature):
    roles = []  # [(seat, role)]

    # -------- LEFT --------
    turn(+scan_step)
    rospy.sleep(1.5)

    image = _image.copy()
    depth = _depth.copy()

    user = find_user_seated(image, depth)
    if user is not None and len(user) > 0:
        role, _ = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("left", role))
    else:
        roles.append(("left", "empty"))

    # -------- MIDDLE --------
    turn(-scan_step)
    rospy.sleep(1.5)
    image = _image.copy()
    depth = _depth.copy()
    mid_img, mid_depth, ox, oy = crop_center_third(image, depth)
    user = find_user_seated(mid_img, mid_depth)
    if user is not None and len(user) > 0:
        user = seated_user_crop_to_original(user, ox, oy)
        role, feat = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("middle", role, feat))
    else:
        roles.append(("middle", "empty", None))

    # -------- RIGHT --------
    turn(-scan_step)
    rospy.sleep(1.5)

    image = _image.copy()
    depth = _depth.copy()

    user = find_user_seated(image, depth)
    if user is not None and len(user) > 0:
        role, _ = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("right", role))
    else:
        roles.append(("right", "empty"))

    # -------- 回中間 --------
    turn(+scan_step)
    rospy.sleep(1)


def classify_person(user, image, guest1_feature, host_feature, threshold=5):
    feat = get_person_feature_seated(image, user)

    print("detected feature =", feat)

    if feat is None:
        return "unknown", None

    g1_score = feature_score(feat, guest1_feature) if guest1_feature else -1
    h_score = feature_score(feat, host_feature) if host_feature else -1

    print("g1_score =", g1_score, "host_score =", h_score)

    if g1_score >= threshold and g1_score >= h_score:
        return "guest1", feat

    if h_score >= threshold and h_score > g1_score:
        return "host", feat

    return "unknown", feat


def scan_people_roles(scan_step, guest1_feature, host_feature):
    roles = []

    global _image, _depth, _image_seq

    # -------- LEFT --------
    old_seq = _image_seq
    turn(+scan_step)
    if not wait_new_frame(old_seq, timeout=3.0):
        print("warning: no new image for left")

    image = _image.copy()
    depth = _depth.copy()

    user = find_user_seated(image, depth)
    if user is not None and len(user) > 0:
        role, feat = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("left", role, feat))
    else:
        roles.append(("left", "empty", None))

    # -------- MIDDLE --------
    old_seq = _image_seq
    turn(-scan_step)
    if not wait_new_frame(old_seq, timeout=3.0):
        print("warning: no new image for middle")

    image = _image.copy()
    depth = _depth.copy()

    user = find_user_seated(image, depth)
    if user is not None and len(user) > 0:
        role, feat = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("middle", role, feat))
    else:
        roles.append(("middle", "empty", None))

    # -------- RIGHT --------
    old_seq = _image_seq
    turn(-scan_step)
    if not wait_new_frame(old_seq, timeout=3.0):
        print("warning: no new image for right")

    image = _image.copy()
    depth = _depth.copy()

    user = find_user_seated(image, depth)
    if user is not None and len(user) > 0:
        role, feat = classify_person(user, image, guest1_feature, host_feature)
        roles.append(("right", role, feat))
    else:
        roles.append(("right", "empty", None))

    # -------- 回中間 --------
    old_seq = _image_seq
    turn(+scan_step)
    wait_new_frame(old_seq, timeout=2.0)

    return roles



def crop_middle_seat(image, depth):
    h, w = image.shape[:2]

    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    y1 = int(h * 0.15)
    y2 = int(h * 0.95)

    crop_img = image[y1:y2, x1:x2]
    crop_depth = depth[y1:y2, x1:x2]

    return crop_img, crop_depth, x1, y1


def seated_user_crop_to_original(user, offset_x, offset_y):
    if user is None or len(user) == 0:
        return user

    x1, y1, x2, y2 = user[0]["box"]
    cx, cy = user[0]["center"]

    user[0]["box"] = (
        x1 + offset_x,
        y1 + offset_y,
        x2 + offset_x,
        y2 + offset_y
    )

    user[0]["center"] = (
        cx + offset_x,
        cy + offset_y
    )

    return user


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


def find_double_door_center(depth):
    h, w = depth.shape[:2]

    # 只看中間高度，不看地板和天花
    y1 = int(h * 0.35)
    y2 = int(h * 0.65)

    # 只看中間大範圍
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    roi = depth[y1:y2, x1:x2]

    # 每一條 x column 取 median depth
    col_depths = []
    xs = []

    for x in range(roi.shape[1]):
        col = roi[:, x]
        vals = col[col > 0]

        if len(vals) < 10:
            continue

        d = float(np.median(vals))
        col_depths.append(d)
        xs.append(x1 + x)

    if len(col_depths) < 20:
        print("[DOOR] not enough depth columns")
        return None, None

    col_depths = np.array(col_depths)
    xs = np.array(xs)

    # 平滑
    smooth = np.convolve(col_depths, np.ones(15) / 15, mode="same")

    # 找深度變化最大的位置，通常是兩門中間縫
    diff = np.abs(np.diff(smooth))
    idx = int(np.argmax(diff))
    door_center_x = int(xs[idx])

    center_depth = float(np.median(col_depths))

    print("[DOOR] door_center_x =", door_center_x)
    print("[DOOR] image_center_x =", w // 2)
    print("[DOOR] center_depth =", center_depth)

    return door_center_x, center_depth


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
def empty_seat(img, depth_img, yolo_model):
    person_in_roi = False
    need_position = None
    code_image = img.copy()
    h, w, _ = code_image.shape
    roi_w, roi_h = 60, 60
    roi_x1 = (w // 2) - (roi_w // 2)
    roi_y1 = int(h * 0.8)
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h
    if roi_y2 >= h: roi_y2 = h - 1
    cv2.rectangle(code_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    results = yolo_model.forward(code_image)[0]["det"]
    for detection in results:
        dx1, dy1, dx2, dy2, score, class_id = detection
        dx1, dy1, dx2, dy2, class_id = map(int, [dx1, dy1, dx2, dy2, class_id])
        if score > 0.5 and class_id == 0 and (dy1+dy2)/2.0<h*0.4:
            if dx1 < roi_x2 and dx2 > roi_x1 and dy1 < roi_y2 and dy2 > roi_y1:
                person_in_roi = True
                cv2.rectangle(code_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), -1)
                cx = (dx1 + dx2) // 2
                cy = (dy1 + dy2) // 2
                need_position = [dx1, dy1, dx2, dy2, cx, cy]
            cv2.rectangle(code_image, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
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
    rate = rospy.Rate(20)

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

    model = YOLO("yolov8n-pose.pt")
    luggage_model = YOLO("bag_model_2026.pt")
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
        rate.sleep()

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
            for iii in range(round(300 / 100)):
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
                _, _, d = get_real_xyz(image, cx, cy, 2)
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
        elif state==6:
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
            speak("dear guest, What is your name and what is your favourite drink? Please say it in complete sentence with close and louder voice, for example my name is fambot and I like milk")
            guest1_name, guest1_drink = "nothing","nothing"
            for i in range(3):
                print(f"[Guest1 Name] trial {i + 1}/3")
                wav_path = record_audio_arecord(
                    "/tmp/guest1_name.wav",
                    9
                )
                if wav_path is None:
                    print("xxxxx")
                    say("Please repeat")
                    continue
                guest1_name1, guest1_drink1 = audio_drink_name(wav_path)
                if guest1_name1 != "nothing": guest1_name=guest1_name1
                if guest1_drink1 != "nothing": guest1_drink=guest1_drink1
                if "nothing" == guest1_name or "nothing" == guest1_drink:
                    say("Please repeat")
                else:
                    break
            speak_tet="your name is "+ guest1_name 
            if guest1_name=="nothing":
                speak_tet = "I cannot get your name"
            say(speak_tet)
            speak_tet="your favourite is "+ guest1_name 
            if guest1_drink=="nothing":
                speak_tet = "I cannot get your favourite drink"
            say(speak_tet)
            turn_back()
            say("guest please follow me")
            state = 8
        elif state == 8:
            print("state 8: move to main point")
            clear_costmaps
            chassis.move_to(2.533, -0.586,-3.086)
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

        elif state == 9:
            change_to_cam(1)  # 1 下2 上
            in_roi, position = empty_seat(_image, _depth, dnn_yolo1)
            if in_roi:
                print("no seats")
            else:
                say("dear guest here is your seat")
            #time out add later
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
            for iii in range(round(300 / 100)):
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
                _, _, d = get_real_xyz(image, cx, cy, 2)
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
            speak("dear guest, What is your name and what is your favourite drink? Please say it in complete sentence with close and louder voice, for example my name is fambot and I like milk")
            guest2_name, guest2_drink = "nothing","nothing"
            for i in range(3):
                print(f"[Guest1 Name] trial {i + 1}/3")
                wav_path = record_audio_arecord(
                    "/tmp/guest2_name.wav",
                    9
                )
                if wav_path is None:
                    print("xxxxx")
                    say("Please repeat")
                    continue
                guest2_name1, guest2_drink1 = audio_drink_name(wav_path)
                if guest2_name1 != "nothing": guest2_name=guest2_name1
                if guest2_drink1 != "nothing": guest2_drink=guest2_drink1
                if "nothing" == guest2_name or "nothing" == guest2_drink:
                    say("Please repeat")
                else:
                    break
            speak_tet="your name is "+ guest2_name 
            if guest2_name=="nothing":
                speak_tet = "I cannot get your name"
            say(speak_tet)
            speak_tet="your favourite is "+ guest2_name 
            if guest2_drink=="nothing":
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
            chassis.move_to(2.533, -0.586,-3.086)

            if chassis.status_code == 3:
                say("arrived")

                empty_seat = None
                guest2_seat = None
                guest1_now_seat = None

                state = 21

        # scan seats (loop state)
        # state 19 : scan chairs, and check where guest1 is every turn
        elif state == 21:
            change_to_cam(1)  # 1 下2 上
            in_roi, position = empty_seat(_image, _depth, dnn_yolo1)
            if in_roi:
                print("no position")
            else:
                say("dear guest here is your seat")
            time.sleep(2)
            state = 22
        #facing
        elif state == 9999999:
            print("state 20 : introduce everyone")
            guest1_desc = feature_to_sentence(guest1_feature)
            guest2_desc = feature_to_sentence(guest2_feature)

            # ---------- Introduce Guest2 to Host ----------
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

            # ---------- Introduce Guest2 to Guest1 ----------
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
                piper.move_to_endpoint2(300,0,0,0,-1.57,0)
                piper.set_gripper(1)
                # ******************************************************************
                say("mission end")
            print("follow", followmecnt)
            move(x, z)
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rospy.sleep(0.05)
