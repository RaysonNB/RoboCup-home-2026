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
from scipy.io.wavfile import write
from piper_sdk_demo import KindaPiper
from pcms.openvino_models import *
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


def process_and_find_seat(guest_img_path, raw_room_img_path, prompt1, prompt2, server_ip="172.22.11.147"):
    """
    負責「畫圖」+「傳送資料給Server」的整合 Function
    """
    print("[*] 正在 Client 端為座位圖片加上號碼標籤...")

    # 1. 讀取尚未畫上數字的客廳照片
    raw_frame = cv2.imread(raw_room_img_path)
    if raw_frame is None:
        return f"Error: 找不到或無法讀取座位圖片 {raw_room_img_path}"

    # 2. 呼叫 function 畫上數字，並取得存好的暫存路徑
    annotated_room_path = annotate_empty_seats(raw_frame)

    print("[*] 標籤添加完成，準備傳送給 Server 分析...")

    # 3. 準備發送給 Server
    url = f"http://{server_ip}:8888/FindSeat"
    try:
        files = {
            'guest_image': open(guest_img_path, 'rb'),
            'seats_image': open(annotated_room_path, 'rb')  # 這裡傳送的是已經畫好數字的圖片
        }
        data = {
            'prompt1': prompt1,
            'prompt2': prompt2
        }

        response = requests.post(url, files=files, data=data)

        # 關閉檔案
        for f in files.values():
            f.close()

        if response.status_code == 200:
            return response.json()['ans']
        else:
            return f"Server Error: {response.status_code}, {response.text}"

    except requests.exceptions.ConnectionError:
        return f"Error: 無法連線到伺服器 {url}，請確認 Server 已啟動！"
    except FileNotFoundError as e:
        return f"Error: 找不到客人圖片 - {e}"

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


def say(text):
    os.system(f"espeak -s 120 \"{text}\"")


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


def detect_bell_from_file(audio_path):
    try:
        target_path = BELL_TARGET_PATH
        SAMPLE_RATE = BELL_SAMPLE_RATE

        if not os.path.exists(target_path):
            print("[BELL] target file not exists:", target_path)
            return False

        if not os.path.exists(audio_path):
            print("[BELL] audio file not exists:", audio_path)
            return False

        target_y, _ = librosa.load(target_path, sr=SAMPLE_RATE)
        audio_y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

        target_mfcc = librosa.feature.mfcc(
            y=target_y,
            sr=SAMPLE_RATE,
            n_mfcc=40
        ).T

        audio_mfcc = librosa.feature.mfcc(
            y=audio_y,
            sr=SAMPLE_RATE,
            n_mfcc=40
        ).T

        n = min(len(target_mfcc), len(audio_mfcc))
        if n <= 0:
            print("[BELL] empty mfcc")
            return False

        target_mfcc = target_mfcc[-n:]
        audio_mfcc = audio_mfcc[-n:]

        distance = np.sum(
            np.abs(audio_mfcc[:, 1:] - target_mfcc[:, 1:])
        )

        rms = float(np.sqrt(np.mean(audio_y ** 2)))
        peak = float(np.max(np.abs(audio_y)))
        duration = len(audio_y) / SAMPLE_RATE

        print("[BELL] audio_path =", audio_path)
        print("[BELL] target_path =", target_path)
        print("[BELL] distance =", int(distance))
        print("[BELL] rms =", rms)
        print("[BELL] peak =", peak)
        print("[BELL] duration =", duration)

        if distance < BELL_THRESHOLD and peak > 0.01:
            print("[BELL] detected")
            return True
        else:
            print("[BELL] not bell")
            return False

    except Exception as e:
        print("[BELL] detect error:", e)
        return False


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

prp=0
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
    if prp==0:
        cv2.imwrite("/tmp/person0.jpg",crop)
        prp+=1
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


def crop_middle_robot(image, depth):
    h, w = image.shape[:2]

    # whole middle area
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    y1 = int(h * 0.10)
    y2 = int(h * 0.95)

    middle_img = image[y1:y2, x1:x2]
    middle_depth = depth[y1:y2, x1:x2]

    return middle_img, middle_depth


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
            if depth[y][x] > 0:
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


def ai_vision_request(mode, prompt, api_key=OPENROUTER_API_KEY, image_path=None, model=VISION_MODEL):
    url = "https://openrouter.ai/api/v1/chat/completions"

    if mode == 1 and image_path:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    else:
        content = prompt

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0
    }

    start_time = time.time()

    try:
        response = requests.post(
            url=url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )
        print(response)

        execution_time = time.time() - start_time
        result = response.json()

        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]

            # remove json
            answer = re.sub(r"^```json\s*", "", answer.strip())
            answer = re.sub(r"^```\s*", "", answer.strip())
            answer = re.sub(r"\s*```$", "", answer.strip())

            return {
                "success": True,
                "answer": answer.strip(),
                "time": f"{execution_time:.2f}s"
            }
        else:
            return {"success": False, "error": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


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


BASE_URL = "http://172.22.11.147:8888"


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


def audio_extract_name(audio_path, api_key=None, model=None):
    prompt = """
Listen to the audio carefully.

Task:
Extract the person's name.

Rules:
- Ignore background noise.
- The name must be one of this list:
sophie, kevin, julia, gabrielle, emma, robin,
noah, sara, john, harrie, laura, liam,
peter, lucas, susan, william
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
- The drink must be one of this list:
lemonade, coffee can, fanta, yakult,
green tea, milk, red bull can
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


def find_human_xy(user, image):
    x1, y1, x2, y2 = 0, 0, 0, 0
    num = 0
    if user is not None:
        for pp in user:
            x, y = map(int, pp)
            if num == 11:
                x1, y1 = x, y
            elif num == 12:
                x2, y2 = x, y
            num += 1

    if (x1 or x2) and (y1 or y2):
        x3 = int((x1 + x2) / 2) if x1 and x2 else (x1 or x2)
        y3 = int((y1 + y2) / 2) if y1 and y2 else (y1 or y2)
        return x3, y3
    return 0, 0


door_baseline = None


def detect_door_open_simple():
    global _depth, door_baseline
    if _depth is None:
        return False
    depth = _depth.copy()
    h, w = depth.shape[:2]
    cx = w // 2
    cy = h // 2
    d = depth[cy, cx]
    if d == 0:
        return False
    print("[DOOR] current depth =", d)
    if door_baseline is None:
        door_baseline = d
        print("[DOOR] baseline set =", door_baseline)
        return False
    diff = abs(d - door_baseline)
    print("[DOOR] diff =", diff)
    if diff > 150:
        print("[DOOR] OPEN detected")
        return True
    return False


def find_speed(depth, x3, y3, lsp):
    dis = depth[y3][x3]
    v = dis - 650
    speed = float(v * 0.0005)
    if speed > 0.3: speed = 0.3
    if speed - lsp > 0.01: speed = lsp + 0.01
    if lsp - speed > 0.01: speed = lsp - 0.01
    return speed


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


def extract_after_phrase(data, phrase, options):
    data = data.lower()
    if phrase in data:
        words = data.split()
        phrase_words = phrase.split()
        for i in range(len(words) - len(phrase_words)):
            if words[i:i + len(phrase_words)] == phrase_words:
                for j in range(1, 4):
                    if i + len(phrase_words) + j <= len(words):
                        guess = ' '.join(words[i + len(phrase_words): i + len(phrase_words) + j])
                        if not options or guess in options:
                            return guess
    return None


def extract_name_after_phrase(data, phrase):
    return extract_after_phrase(data, phrase, nameslist)


def extract_drink(data, drinklist):
    d = extract_after_phrase(data, "i want", drinklist)
    if not d:
        d = extract_after_phrase(data, "i would like", drinklist)
    if not d:
        d = extract_after_phrase(data, "give me", drinklist)
    return d


def detect_people_positions(image, model):
    results = model(image, verbose=False)[0]
    people = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > 0.6:
            x1, y1, x2, y2 = map(int, box)
            people.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return people


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


def turn_back(speed=0.2):
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


# def get_person_feature(image, user, api_key=OPENROUTER_API_KEY):
#     img_path = save_user_crop(image, user)
#     if img_path is None:
#         print("save_user_crop failed")
#         return None

#     prompt = """
# You are helping a RoboCup@Home robot identify a seated person.
# Look only at the main person in the image crop.
# Return JSON only, with exactly these fields:

# {
#   "hat": false,
#   "glasses": false,
#   "upper_color": "blue",
#   "lower_color": "black",
#   "hair_length": "short",
#   "gender_like": "unclear"
# }

# Rules:
# - Output JSON only.
# - Use one-word colors.
# - If uncertain, use "unclear".
# - Focus only on the main person.
# """

#     res = ai_vision_request(
#         mode=1,
#         prompt=prompt,
#         api_key=api_key,
#         image_path=img_path
#     )

#     print("vision result =", res)

#     if not res["success"]:
#         print("vision API failed:", res["error"])
#         return None

#     try:
#         feat = json.loads(res["answer"])
#         return feat
#     except Exception as e:
#         print("json parse failed:", e)
#         print("raw answer:", res["answer"])
#         return None

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


# def get_person_feature_seated(image, user, api_key=OPENROUTER_API_KEY):
#     if user is None or len(user) == 0:
#         return None

#     x1, y1, x2, y2 = user[0]["box"]
#     crop = image[y1:y2, x1:x2]

#     if crop is None or crop.size == 0:
#         print("invalid crop")
#         return None

#     img_path = "/tmp/person.jpg"
#     cv2.imwrite(img_path, crop)

#     prompt = """
# Describe the person.

# Return JSON only:
# {
#   "hat": false,
#   "glasses": false,
#   "upper_color": "blue",
#   "lower_color": "black",
#   "hair_length": "short",
#   "gender_like": "male"
# }
# """

#     result = ai_vision_request(1, prompt, api_key, img_path)
#     print("vision result =", result)

#     if not result or not result.get("success", False):
#         print("vision API failed:", result.get("error", "unknown"))
#         return None

#     try:
#         content = result["answer"]
#         if content is None:
#             return None

#         content = content.strip()
#         content = content.replace("```json", "").replace("```", "").strip()

#         return json.loads(content)

#     except Exception as e:
#         print("feature parse error:", e)
#         print("raw answer =", result["answer"])
#         return None
people_cnt=0
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
    people_cnt+=1
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

    say("Please put your bag on my hand.")
    rospy.sleep(3)

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


def find_all_users(image, depth):
    global model

    h, w = depth.shape
    keypoints = model(image)[0].keypoints
    users = []

    for pose in keypoints.xy:
        mid_d = -1
        valid_points = 0

        for p in pose:
            x, y = map(int, p)
            if x >= w or y >= h or x < 0 or y < 0:
                continue
            if depth[y][x] > 0:
                valid_points += 1
                if mid_d == -1 or depth[y][x] < mid_d:
                    mid_d = depth[y][x]

        if valid_points > 0 and mid_d != -1:
            users.append((pose, mid_d))

    return users


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

def approach_door_simple():
    global _depth

    depth = _depth.copy()

    center_depth, turn_angle = get_center_depth_and_turn(depth)

    if center_depth is None:
        print("[DOOR] no valid center depth")
        return False

    # 先微調方向
    if abs(turn_angle) > 0:
        turn(turn_angle)
        rospy.sleep(0.5)

    # arm 到門希望 800~900mm
    # camera depth 通常比 arm 位置不同，所以先用 850 作目標
    target_depth = 850

    if center_depth > 900:
        print("[DOOR] too far, move forward")

        move(0.12, 0)
        rospy.sleep(0.5)
        move(0, 0)

        return False

    elif center_depth < 750:
        print("[DOOR] too close, move back")

        move(-0.08, 0)
        rospy.sleep(0.4)
        move(0, 0)

        return False

    else:
        print("[DOOR] good distance")
        return True


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


def align_to_double_door_center():
    global _depth

    depth = _depth.copy()
    h, w = depth.shape[:2]

    door_center_x, center_depth = find_double_door_center(depth)

    if door_center_x is None:
        print("[DOOR] cannot find door center")
        return False, None

    error_x = door_center_x - w // 2

    print("[DOOR] error_x =", error_x)

    # 如果已經差不多中間，不轉
    if abs(error_x) < 25:
        print("[DOOR] already centered")
        return True, center_depth

    # pixel error 轉角度，58度 FOV
    angle = -error_x / (w / 2) * (58.0 / 2.0) * np.pi / 180.0

    # 限制每次最多轉 8 度，避免過頭
    max_turn = 8 * np.pi / 180
    angle = max(-max_turn, min(max_turn, angle))

    print("[DOOR] turn angle =", angle)

    turn(angle)
    rospy.sleep(0.5)

    return False, center_depth


def align_double_door_by_lr_depth():
    global _depth

    depth = _depth.copy()
    h, w = depth.shape[:2]

    y1 = int(h * 0.35)
    y2 = int(h * 0.65)

    left_roi = depth[y1:y2, int(w * 0.20):int(w * 0.45)]
    right_roi = depth[y1:y2, int(w * 0.55):int(w * 0.80)]

    left_vals = left_roi[left_roi > 0]
    right_vals = right_roi[right_roi > 0]

    if len(left_vals) < 50 or len(right_vals) < 50:
        print("[DOOR] not enough depth")
        return False, None

    left_d = float(np.median(left_vals))
    right_d = float(np.median(right_vals))

    center_roi = depth[y1:y2, int(w * 0.40):int(w * 0.60)]
    center_vals = center_roi[center_roi > 0]
    center_d = float(np.median(center_vals)) if len(center_vals) > 0 else None

    diff = left_d - right_d

    print("[DOOR] left_d =", left_d)
    print("[DOOR] right_d =", right_d)
    print("[DOOR] diff =", diff)
    print("[DOOR] center_d =", center_d)

    if abs(diff) < 80:
        print("[DOOR] aligned")
        return True, center_d

    if diff > 0:
        print("[DOOR] turn left")
        turn(+np.pi / 90)  # 2 degrees
    else:
        print("[DOOR] turn right")
        turn(-np.pi / 90)  # 2 degrees

    rospy.sleep(0.3)
    return False, center_d


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
    if abs(error) < 20:
        print("Door handle aligned.")
        return True
    angle = error * 0.0012
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
    if d > 900:
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
    with open('/home/pcms/antony_ws/door1.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    x = True
    for i in data:
        piper.move_joint(i["joints"])
        if i["gripper"][0] > 50000 and x == True:
            piper.set_gripper(1, 1000)
            x = False
        elif i["gripper"][0] <= 50000 and x == False:
            piper.set_gripper(0, 1000)
            x = True


def open_door2():
    with open('/home/pcms/antony_ws/door2.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    x = True
    for i in data:
        piper.move_joint(i["joints"])
        if i["gripper"][0] > 50000 and x == True:
            piper.set_gripper(1, 1000)
            x = False
        elif i["gripper"][0] <= 50000 and x == False:
            piper.set_gripper(0, 1000)
            x = True


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
    door_handle_model = YOLO("/home/pcms/antony_ws/best(4).pt")
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
    state = 99995
    scan_step = np.pi / 4
    far_right_step = 3 * np.pi / 4
    say("I am ready")

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
        if state == 999999:
            print("state 0: ready")
            state = 999

        elif state == 999:
            if wait_until_door_open():
                say("Door is open")
                state = 995
            else:
                say("Door is still closed")
                state = 999

        # state 9999: walk into the room
        elif state == 995:
            print("[STATE 9999] MOVE_TO_ROOM")
            say("moving to room")
            chassis.move_to(3.69, 0.31, 2.69)
            if chassis.status_code == 3:
                print("reach point")
                state = 0
            print(chassis.status_code)

        # state0 : hear the bell
        elif state == 0:
            print("state 0: wait for bell")
            if _new_audio_flag and _audio_path is not None and os.path.exists(_audio_path):

                if _audio_path == _last_checked_audio_path:
                    _new_audio_flag = False
                    continue

                _last_checked_audio_path = _audio_path
                _new_audio_flag = False

                if detect_bell_from_file(_audio_path):
                    say("I heard the doorbell")
                    state = 99999
                else:
                    print("not bell")

        elif state == 99999:
            print("[STATE 99999] move to door")
            say("moving to door")
            chassis.move_to(2.53, 4.423, 1.251)
            if chassis.status_code == 3:
                print("reach point")
                state = 99995
            print(chassis.status_code)

        # state 99995: open door function
        elif state == 99995:
            # change_to_cam(1)
            # print("Align to door handle")
            # ok = align_to_door_handle()
            # if not ok:
            #     continue
            # print("Door handle aligned.")
            # ok = adjust_door_distance()
            # if not ok:
            #     continue
            # print("Ready to open.")
            open_door1()
            for iii in range(round(800 / 100)):
                move(-0.2, 0)
                time.sleep(0.5)
            open_door2()
            arm_retract()

        # state 9999: walk into the room
        elif state == 9999:
            print("[STATE 9999] MOVE_TO_ROOM")
            say("moving to room")
            chassis.move_to(3.69, 0.31, 2.69)
            if chassis.status_code == 3:
                print("reach point")
                state = 1
            print(chassis.status_code)

        # state1 : find user
        elif state == 1:
            print("state 1: find guest 1")
            user = find_user(image, depth)
            if user is not None and len(user) > 0:
                say("hello I am detecting your feature now, please wait")
                feat = get_person_feature(image, user)
                print("guest1_feature =", feat)
                if feat is not None:
                    guest1_feature = feat
                    state = 1225
                else:
                    print("feature failed, retry")

        # state 1255: move to guest1 first point
        elif state == 1225:
            print("[STATE 1225] MOVE_TO_GUEST1 first point")
            say("moving to guest1 first point")
            chassis.move_to(2.5, 0.663, 2.73)
            if chassis.status_code == 3:
                print("reach point")
                state = 2
            print(chassis.status_code)

        # state2 : move to guest1
        elif state == 2:
            print("[STATE 2] MOVE_TO_GUEST1")
            say("moving to guest1")
            chassis.move_to(2.64, 0.81, 2.73)
            if chassis.status_code == 3:
                print("reach guest1 interaction point")
                state = 3
            print(chassis.status_code)

        # elif state == 25:
        # turn_back()
        # state = 3

        # state3 : ask guest1(name)
        elif state == 3:
            print("state 3: ask guest1 name")
            guest1_name = None
            for i in range(3):
                print(f"[Guest1 Name] trial {i + 1}/3")
                say("Please say in complete sentence")
                rospy.sleep(1)
                say("What is your name?")
                wav_path = record_audio_arecord(
                    "/tmp/guest1_name.wav",
                    5
                )
                if wav_path is None:
                    say("I cannot hear you. Please repeat your name.")
                    continue
                res = audio_extract_name(wav_path, OPENROUTER_API_KEY)
                print("AI name result =", res)
                if res["success"] and res["name"] is not None:
                    name = res["name"].strip().lower()
                    if name in NAME_LIST:
                        guest1_name = name
                        break
                say("Please repeat your name.")
            if guest1_name is None:
                guest1_name = "kevin"
            print("guest1_name =", guest1_name)
            say(f"Hello {guest1_name}")
            state = 4

        # state4 : ask guest1(drink)
        elif state == 4:
            print("state 4: ask guest1 drink")
            guest1_drink = None
            for i in range(3):
                print(f"[Guest1 Drink] trial {i + 1}/3")
                say("Please say in complete sentence")
                rospy.sleep(1)
                say("What is your favorite drink?")
                wav_path = record_audio_arecord(
                    "/tmp/guest1_drink.wav",
                    5
                )
                if wav_path is None:
                    say("I cannot hear you. Please repeat your drink.")
                    continue
                res = audio_extract_drink(wav_path, OPENROUTER_API_KEY)
                print("AI drink result =", res)
                if res["success"] and res["drink"] is not None:
                    drink = res["drink"].strip().lower()
                    valid_drinks = [d.lower() for d in DRINK_LIST]
                    if drink in valid_drinks:
                        guest1_drink = DRINK_LIST[valid_drinks.index(drink)]
                        break
                say("Please repeat your drink.")
            if guest1_drink is None:
                guest1_drink = "coffee can"
            print("guest1_drink =", guest1_drink)
            say(f"{guest1_name} likes {guest1_drink}")
            state = 5

            # state5 : seat guest
        elif state == 5:
            print("state 5: seat guest 1")
            say("please follow me")
            state = 6

        # state 6 : turn back
        elif state == 6:
            turn_back()
            print("success turning 180")
            state = 7

        # state 7 : move to point 1
        elif state == 7:
            print("starting to move to point 1")
            chassis.move_to(2.5, 0.663, 0)
            if chassis.status_code == 3:
                print("reach point 1")
                state = 8

        # state 8 : move to main point
        elif state == 8:
            print("state 8: move to main point")
            chassis.move_to(3.41, 0.086, -1.85)
            if chassis.status_code == 3:
                say("reach main point")
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

        # elif state == 85:
        # turn_back()
        # turn(10/180*np.pi)
        # print("state85: turn back and turn 10")
        # state=9

        # state 9 : find empty seat first, then find host
        elif state == 9:
            print("state 9: find empty chair for guest1 and find host feature")

            empty_seat = None
            guest1_seat = None
            host_feature = None

            # -------- LEFT --------
            old_seq = _image_seq
            turn(+scan_step)
            wait_new_frame(old_seq)

            image = _image.copy()
            depth = _depth.copy()
            user = find_user_seated(image, depth)

            if user is None or len(user) == 0:
                print("left empty")
                if guest1_seat is None:
                    empty_seat = "left"
                    guest1_seat = "left"
                    say("this is your seat")
                    rospy.sleep(2)
            else:
                print("left occupied, maybe host")
                if host_feature is None:
                    host_feature = get_person_feature_seated(image, user)
                    print("host_feature =", host_feature)
            # -------- MIDDLE --------
            old_seq = _image_seq
            turn(-scan_step)
            wait_new_frame(old_seq)

            image = _image.copy()
            depth = _depth.copy()

            middle_img, middle_depth, ox, oy = crop_middle_seat(image, depth)
            user = find_user_seated(middle_img, middle_depth)
            user = seated_user_crop_to_original(user, ox, oy)

            if user is None or len(user) == 0:
                print("middle empty")
                if guest1_seat is None:
                    empty_seat = "middle"
                    guest1_seat = "middle"
                    say("this is your seat")
                    rospy.sleep(2)
            else:
                print("middle occupied, maybe host")
                if host_feature is None:
                    host_feature = get_person_feature_seated(image, user)
                    print("host_feature =", host_feature)

                # -------- RIGHT --------
                old_seq = _image_seq
                turn(-scan_step)
                wait_new_frame(old_seq)

                image = _image.copy()
                depth = _depth.copy()
                user = find_user_seated(image, depth)

                if user is None or len(user) == 0:
                    print("right empty")
                    if guest1_seat is None:
                        empty_seat = "right"
                        guest1_seat = "right"
                        say("this is your seat")
                        rospy.sleep(2)
                else:
                    print("right occupied, maybe host")
                    if host_feature is None:
                        host_feature = get_person_feature_seated(image, user)
                        print("host_feature =", host_feature)

            # -------- 回 MIDDLE --------
            old_seq = _image_seq
            turn(+scan_step)
            wait_new_frame(old_seq)

            print("empty_seat =", empty_seat)
            print("guest1_seat =", guest1_seat)
            print("host_feature =", host_feature)

            if guest1_seat is None:
                say("sorry, no empty chair found")

            if host_feature is None:
                print("WARNING: host_feature is None")

            state = 10

        # state10 : turn back to front
        elif state == 10:
            print("state 10: turn back to front")
            turn_back()
            print("guest1_name =", guest1_name)
            print("guest1_drink =", guest1_drink)
            print("guest1_feature =", guest1_feature)
            print("guest1_seat =", guest1_seat)
            print("host_seat =", host_seat)
            print("host_feature =", host_feature)
            state = 105

        elif state == 105:
            print("[STATE 10.5] MOVE_TO_GUEST2")
            say("wait the bell ring")
            chassis.move_to(3.69, 0.31, 2.69)
            if chassis.status_code == 3:
                print("reach guest2 interaction point")
                state = 111
            print(chassis.status_code)

        # state two : find guest 2
        # state0 : hear the bell
        elif state == 111:
            print("state 0: wait for bell")
            if _new_audio_flag and _audio_path is not None and os.path.exists(_audio_path):

                if _audio_path == _last_checked_audio_path:
                    _new_audio_flag = False
                    continue

                _last_checked_audio_path = _audio_path
                _new_audio_flag = False

                if detect_bell_from_file(_audio_path):
                    say("I heard the doorbell")
                    state = 112
                else:
                    print("not bell")

        elif state == 112:
            print("[STATE 99999] move to door")
            say("moving to door")
            chassis.move_to(2.53, 4.423, 1.251)
            if chassis.status_code == 3:
                print("reach point")
                state = 113
            print(chassis.status_code)

        # state 99995: open door function
        elif state == 113:
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
            open_door1()
            move(-0.18, 0)
            open_door2()
            arm_retract()
            change_to_cam(2)
            state = 114

        # state 9999: walk into the room
        elif state == 114:
            print("[STATE 9999] MOVE_TO_ROOM")
            say("moving to room")
            chassis.move_to(3.69, 0.31, 2.69)
            if chassis.status_code == 3:
                print("reach point")
                state = 115
            print(chassis.status_code)

        # state 11 : find guest 2
        elif state == 115:
            print("state 11: find guest 2")
            user = find_user(image, depth)
            if user is not None and len(user) > 0:
                say("hello")
                feat = get_person_feature(image, user)
                print("guest2_feature =", feat)
                if feat is not None:
                    guest2_feature = feat
                    state = 1275
                else:
                    print("feature failed, retry")

        # state 1275:move to guest2 first point
        elif state == 1275:
            print("[STATE 1275] MOVE_TO_GUEST2 first point")
            say("moving to guest2 first point")
            chassis.move_to(2.5, 0.663, 2.73)
            if chassis.status_code == 3:
                print("reach point")
                state = 12
            print(chassis.status_code)

        # state = 12: navigate to point
        elif state == 12:
            print("[STATE 12] MOVE_TO_GUEST2")
            say("moving to guest2")
            chassis.move_to(2.64, 0.81, 2.73)
            if chassis.status_code == 3:
                print("reach guest2 interaction point")
                state = 13
            print(chassis.status_code)

        # elif state == 125:
        # turn_back()
        # state = 13

        # state 13 : ask guest 2 name
        elif state == 13:
            print("state 13: ask guest2 name")
            guest2_name = None
            for i in range(3):
                print(f"[Guest2 Name] trial {i + 1}/3")
                say("Please say in complete sentence")
                rospy.sleep(1)
                say("What is your name?")
                rospy.sleep(0.5)
                wav_path = record_audio_arecord(
                    "/tmp/guest2_name.wav",
                    5
                )
                if wav_path is None:
                    say("I cannot hear you. Please repeat your name.")
                    continue
                res = audio_extract_name(wav_path, OPENROUTER_API_KEY)
                print("AI name result =", res)
                if res["success"] and res["name"] is not None:
                    name = res["name"].strip().lower()
                    if name in NAME_LIST:
                        guest2_name = name
                        break
                say("Please repeat your name.")
            if guest2_name is None:
                guest2_name = "friend"
            print("guest2_name =", guest2_name)
            say(f"Hello {guest2_name}")
            state = 14

        # state14 : ask guest2(drink)
        elif state == 14:
            print("state 14: ask guest2 drink")
            guest2_drink = None
            for i in range(3):
                print(f"[Guest2 Drink] trial {i + 1}/3")
                say("Please say in complete sentence")
                rospy.sleep(1)
                say("What is your favorite drink?")
                rospy.sleep(0.5)
                wav_path = record_audio_arecord(
                    "/tmp/guest2_drink.wav",
                    5
                )
                if wav_path is None:
                    say("I cannot hear you. Please repeat your drink.")
                    continue
                res = audio_extract_drink(wav_path, OPENROUTER_API_KEY)
                print("AI drink result =", res)
                if res["success"] and res["drink"] is not None:
                    drink = res["drink"].strip().lower()
                    valid_drinks = [d.lower() for d in DRINK_LIST]
                    if drink in valid_drinks:
                        guest2_drink = DRINK_LIST[valid_drinks.index(drink)]
                        break
                say("Please repeat your drink.")
            if guest2_drink is None:
                guest2_drink = "water"
            print("guest2_drink =", guest2_drink)
            say(f"{guest2_name} likes {guest2_drink}")
            state = 155

        elif state == 155:
            change_to_cam(1)
            print("state 155: receive bag")
            receive_bag()
            change_to_cam(2)
            state = 15

        # state 15: seat guest 2
        elif state == 15:
            print("state 15: seat guest 2")
            say("please follow me")
            state = 17

        # state 16: turn 180
        # elif state == 16:
        # turn_back()
        # print("success turning 180")
        # state = 17

        # state 17: move to point 1
        elif state == 17:
            print("starting to move to point 1")
            chassis.move_to(2.5, 0.663, 0)
            if chassis.status_code == 3:
                print("reach point 1")
                state = 18

        # state 18: move to main point
        elif state == 18:
            print("state 18: move to main point")
            chassis.move_to(3.41, 0.086, -1.85)

            if chassis.status_code == 3:
                say("reach main point")

                empty_seat = None
                guest2_seat = None
                guest1_now_seat = None

                state = 19

        # scan seats (loop state)
        # state 19 : scan chairs, and check where guest1 is every turn
        elif state == 19:
            print("state 19 : find host, guest1, and empty chair for guest2")
            host_now_seat = None
            guest1_now_seat = None
            host_photo_path = "/tmp/person1.jpg"
            guest1_photo_path = "/tmp/person0.jpg"
            guest2_seat = None
            p1 = "this is the host"
            p2 = "here have 3 seats(3 green chairs) please tell me where is he/she sitting, just give me number in [1,2,3], there should be 1 numbers only, no require extra information, answer format: 1"
            guest_photo = host_photo_path
            img_path = "/tmp/chair.jpg"
            cv2.imwrite(img_path, image)
            room_photo = "/tmp/chair.jpg"
            host_now_seat = process_and_find_seat(host_photo_path, room_photo, p1, p2, server_ip="172.22.11.147")
            print("host position",host_now_seat)
            p1 = "this is the first guest"
            guest1_now_seat = process_and_find_seat(guest1_photo_path, room_photo, p1, p2, server_ip="172.22.11.147")
            print("guest position", guest1_now_seat)
            niggggga = '123'
            niggggga = niggggga.replace(guest1_now_seat,"")
            niggggga = niggggga.replace(host_now_seat,"")
            if len(niggggga) >0:
                guest2_seat= niggggga[0]
            if "1" in guest2_seat:
                guest2_seat = "left"
            if "2" in guest2_seat:
                guest2_seat = "middle"
            if "3" in guest2_seat:
                guest2_seat = "right"
            if "1" in host_now_seat:
                host_now_seat = "left"
            if "2" in host_now_seat:
                host_now_seat = "middle"
            if "3" in host_now_seat:
                host_now_seat = "right"
            if "1" in guest1_now_seat:
                guest1_now_seat = "left"
            if "2" in guest1_now_seat:
                guest1_now_seat = "middle"
            if "3" in guest1_now_seat:
                guest1_now_seat = "right"
            print("host_now_seat =", host_now_seat)
            print("guest1_now_seat =", guest1_now_seat)
            print("guest2_now_seat =", guest2_seat)
            # ================= FIND EMPTY SEAT =================

            if guest2_seat is None:
                print("[STATE 19] no empty seat found, fallback to middle")
                guest2_seat = "middle"

            print("guest2_seat =", guest2_seat)

            # ================= TELL GUEST2 WHERE TO SIT =================
            if guest2_seat == "left":
                say("Please sit on the left chair.")
            elif guest2_seat == "middle":
                say("Please sit on the middle chair.")
            elif guest2_seat == "right":
                say("Please sit on the right chair.")

            rospy.sleep(2)

            state = 20

        # state 20 : introduce everyone
        elif state == 20:
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
            say("dear host, please don't stand in front of me, and please please don't walk too fast")
            say("please say robot stop robot stop")
            state = 21

        # state 21 : follow host
        elif state == 21:
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
                    say("don't walk too fast, pleae come back")
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
                state=-1111
                #******************************************************************
                say("mission end")
            print("follow", followmecnt)
            move(x, z)
            cv2.imshow("image", code_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rospy.sleep(0.05)