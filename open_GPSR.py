# robot_gpsr.py
import json
import base64
import urllib.request
import urllib.error
import time
import math
import cv2
from datetime import datetime
import numpy as np
import requests
import subprocess
import re
import os
# ------------------------------
# Robot Library
# ------------------------------
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from mr_voice.msg import Voice
from RobotChassis import RobotChassis # Your chassis driver
from tf.transformations import euler_from_quaternion
# from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from pcms.openvino_models import *

# ------------------------------
# Global variables
# ------------------------------
_image = None
_depth = None
_voice = None
_odom = None
CvBr = CvBridge()
code = 0 # Navigation result code
_current_cam = 1 # Current camera id
_sub_cam_image = None
_sub_cam_depth = None
# cam 1, up camera
# cam 2, down camera

# ------------------------------
# Model initialization
# ------------------------------
dnn_yolo = Yolov8("rcj2026",device_name="GPU")
dnn_yolo.classes = ['Apple', 'Bowl', 'Cereal', 'Coffee Can', 'Cup Noodle', 'Fanta', 'Fork', 'Green Tea', 'Jagarico', 'Knife', 'Laundry Basket', 'Lemon', 'Lemonade', 'Milk', 'Mug', 'Orange', 'Peach', 'Plate', 'Red Bull can', 'Rubiks cube', 'Spam', 'Sponge', 'Spoon', 'Strawberry', 'Tomato soup can', 'Toothpaste box']
obj_list = ['Apple', 'Bowl', 'Cereal', 'Coffee Can', 'Cup Noodle', 'Fanta', 'Fork', 'Green Tea', 'Jagarico', 'Knife', 'Laundry Basket', 'Lemon', 'Lemonade', 'Milk', 'Mug', 'Orange', 'Peach', 'Plate', 'Red Bull can', 'Rubiks cube', 'Spam', 'Sponge', 'Spoon', 'Strawberry', 'Tomato soup can', 'Toothpaste box']

dnn_pose = HumanPoseEstimation(device_name="GPU")
# ------------------------------
# Object categories and Locations
# ------------------------------
objects = {
    "Drink": [
        "Lemonade", "Coffee Can", "Fanta", "Yakult", "Green Tea", "Milk", "Red Bull can"
    ],
    "Food": [
        "Tomato soup can", "Spam", "Cereal", "Cup Noodle", "Cup Rice"
    ],
    "Fruit": [
        "Apple", "Lemon", "Peach", "Orange", "Strawberry"
    ],
    "Kitchen Item": [
        "Fork", "Knife", "Spoon", "Plate", "Bowl", "Mug", "Sponge", "Dishwasher Tab"
    ],
    "Snack": [
        "Chocolate Snack", "Jagarico", "Gummy"
    ],
    "Task Item": [
        "Rubik's cube", "Toothpaste", "Bag", "Clothes", "Laundry Basket", "Tray"
    ]
}
place = {
    "Host": [0.726,-0.22,0],
    "Bedroom": [0.726,-0.22,0],
    "Bed": [0.18,-0.0432, 0],
    "Bookcase": [-0.284,-0.395,0],
    "Coat rack": [1.33,-0.419,0],
    "Kitchen": [3.68,0.0613,0],
    "Side Table": [2.6,0.219,0],
    "Trash Bin": [2.37,-0.417,0],
    "Dining Table": [3.65,-0.273,0],
    "Dishwasher": [4.95,-0.151,0],
    "Microwave": [5.22,-0.086,0],
    "Cabinet": [5.03,0.552,0],
    "Laundry Room": [4.31,2.8,0],
    "Table": [3.87,2.16,0],
    "Washing Machine": [5.05,2.74,0],
    "Living Room": [0.71,2.76,0],
    "Coffee Table": [0.142,3.32,0],
    "Couch": [1.26,1.86,0],
    "Chair": [1.21,2.67,0]
}
jointsPos = {
    "idle": [0,-1.049,0.374,0.706], # use this when not going to grasp shits or already grasped shits
    "ready": [-0.029,0.430,-0.821,0.149] # use this when going to grasp shits
}
# ------------------------------
# Callback functions
# ------------------------------
def callback_image(msg):
    global _image
    _image = CvBr.imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global _depth
    _depth = CvBr.imgmsg_to_cv2(msg)

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_odom(msg):
    global _odom
    _odom = msg

def callback_path(msg):
    global _audio_path
    _audio_path = msg


# ------------------------------
# LLM Server
# ------------------------------
# SERVER_URL = "http://172.22.11.127:5000"  # replace with actual IP

# def call_llm(messages, model="gemini-2.5-flash-lite"):
#     """Send messages to the LLM proxy and return the response content."""
#     payload = {'messages': messages, 'model': model}
#     data = json.dumps(payload).encode('utf-8')
#     req = urllib.request.Request(f"{SERVER_URL}/chat", data=data, headers={'Content-Type': 'application/json'})
#     try:
#         with urllib.request.urlopen(req) as response:
#             resp = json.loads(response.read().decode('utf-8'))
#             return resp.get('content')
#     except urllib.error.URLError as e:
#         print(f"LLM proxy error: {e}")
#         return None

def call_llm(messages, model="google/gemini-2.5-flash-lite"):
    """
    Drop-in replacement for call_llm() that uses OpenRouter directly.
    Returns the assistant's content string, or None on error.
    """
    # Map the simple model name to OpenRouter's full identifier
    openrouter_model = model
    if model == "gemini-2.5-flash-lite":
        openrouter_model = "google/gemini-2.5-flash-lite"
    # Add other mappings here if needed, e.g.:
    # elif model == "gpt-4o-mini":
    #     openrouter_model = "openai/gpt-4o-mini"

    payload = {
        "model": openrouter_model,
        "messages": messages
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )
        result = response.json()
    except Exception as e:
        print(f"LLM proxy error: {e}")
        return None

    if "choices" not in result:
        print("call_llm error:", result)
        return None

    # Return the content without stripping (matches call_llm's behaviour)
    return result["choices"][0]["message"]["content"]
# ------------------------------
# Sentence Recognition
# ------------------------------

def encode_audio(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

OPENROUTER_API_KEY = "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"

def audio_extract_sentence(audio_path, api_key=OPENROUTER_API_KEY, model="xiaomi/mimo-v2-omni"):
    try:
        base64_audio = encode_audio(audio_path)

        prompt = """
You will hear a sentence of a command, you have to transcribe the sentence.
Return JSON only in this format:
{"text": "string or null"}

Rules:
- If unclear or unfinished sentence, return null.
- Do not include filler sounds (eg. um uh ah er mm) in the sentence
- If the speaker correct himself in the audio, output the sentence without errors
- Output JSON only.
"""
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )

        result = response.json()

        if "choices" not in result:
            print("audio_extract_text API error:", result)
            return {"success": False, "text": None}

        content = result["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        return {"success": True, "text": data.get("text")}

    except Exception as e:
        print("audio_extract_text exception:", e)
        return {"success": False, "text": None}
    

def audio_extract_name(audio_path, api_key=OPENROUTER_API_KEY, model="xiaomi/mimo-v2-omni"):
    try:
        base64_audio = encode_audio(audio_path)

        prompt = """
You will hear a short answer to the question: what is your name?
Return JSON only in this format:
{"name": "string or null"}

Rules:
- If unclear, return null.
- Output JSON only.
- Pronuciation might not be really correct, if they sound alike, you can directly match the name

Name List: ["Yoshimura", "Angel", "Basil", "Chikako", "Andrew", "Sophia", "Jack", "Mike", "Leo", "Tom"]
"""

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30
        )

        result = response.json()

        if "choices" not in result:
            print("audio_extract_name API error:", result)
            return {"success": False, "name": None}

        content = result["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        return {"success": True, "name": data.get("name")}

    except Exception as e:
        print("audio_extract_name exception:", e)
        return {"success": False, "name": None}
# ------------------------------
# Camera switching
# ------------------------------
def change_to_cam(idx):
    global _sub_cam_image, _sub_cam_depth, _current_cam
    _current_cam = idx

    if _sub_cam_image is not None:
        _sub_cam_image.unregister()
    if _sub_cam_depth is not None:
        _sub_cam_depth.unregister()

    if idx == 1:
        _sub_cam_image = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    elif idx == 2:
        _sub_cam_image = rospy.Subscriber("/cam2/color/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

# ------------------------------
# OpenManipulator control
# ------------------------------
# def set_joints(joint1, joint2, joint3, joint4, t):
#     service_name = "/goal_joint_space_path"
#     rospy.wait_for_service(service_name)
    
#     try:
#         service = rospy.ServiceProxy(service_name, SetJointPosition)
        
#         request = SetJointPositionRequest()
#         request.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
#         request.joint_position.position = [joint1, joint2, joint3, joint4]
#         request.path_time = t
        
#         response = service(request)
#         return response
#     except Exception as e:
#         rospy.loginfo("%s" % e)
#         return False
    
# def set_gripper(angle, t):
#     service_name = "/goal_tool_control"
#     rospy.wait_for_service(service_name)
    
#     try:
#         service = rospy.ServiceProxy(service_name, SetJointPosition)
        
#         request = SetJointPositionRequest()
#         request.joint_position.joint_name = ["gripper"]
#         request.joint_position.position = [angle]
#         request.path_time = t
        
#         response = service(request)
#         return response
#     except Exception as e:
#         rospy.loginfo("%s" % e)
#         return False

# def open_gripper(t):
#     return set_gripper(0.01, t)

# def close_gripper(t):
#     return set_gripper(-0.01, t)

# ------------------------------
# Helper functions
# ------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def nav_to(obj_name):
    # Navigate to a known object or location.
    def dfs_search(data, parent_key=None, parent_obj=None):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == obj_name:
                    coords = None
                    if parent_obj and obj_name in parent_obj:
                        coords = parent_obj[obj_name]
                    elif isinstance(value, list):
                        coords = value
                    print(f"Navigating to {obj_name} with coordinates {coords}")
                    tell(f"Navigating to {obj_name}")
                    chassis.move_to(coords[0], coords[1], coords[2])
                    return True
                if isinstance(value, dict):
                    found = dfs_search(value, parent_key=key, parent_obj=value)
                    if found:
                        return True
        return False
    return dfs_search(place)

def count_object(category):
    # change_to_cam(2)
    rospy.sleep(1)
    image = _image.copy()
    # Get the list of object names for the given category
    objs_in_category = objects.get(category, [])
    tell(f"Counting how many {category} are there.")
    if not objs_in_category:
        return 0
    
    # Run YOLO detection
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = dnn_yolo.forward(bgr_image) 
    
    # Extract detected class names
    detected_names = []
    for boxes in results:
        for i, (x1, y1, x2, y2, score, cls) in enumerate(results[0]["det"]):
            x1, y1, x2, y2, cls = map(int, (x1, y1, x2, y2, cls))
            obj_name = obj_list[cls]  # e.g., "cup", "cookie"
            cv2.rectangle(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
            cv2.putText(image, obj_name, (x1,y1),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            detected_names.append(obj_name)
    cv2.imwrite("{}.jpg".format(int(time.time())),image)
    # Count matches
    count = 0
    for det_name in detected_names:
        # Compare lower‑case, and optionally map common variants
        det_lower = det_name.lower()
        for target in objs_in_category:
            if det_lower == target.lower():
                count += 1
                break
    tell("Finished counting")
    print(f"Count of {category}: {count}")
    change_to_cam(1)
    return count
def approach_person():
    state = 1
    print("Approaching the person.")
    tell("I am approaching you.")
    while not rospy.is_shutdown():
        image = _image.copy()
        depth = _depth.copy()
        height, width, channels = image.shape
        msg = Twist()  # Reset command
        poses = dnn_pose.forward(image)
        shortest_distance = float('inf')
        targetx, targety = 0, 0
        for pose in poses:
            x1 = 10000
            y1 = 10000
            x2 = -1
            y2 = -1
            for cx, cy, score in pose:
                if cx!=0 and cy!=0:
                    x1 = int(min(x1, cx))
                    y1 = int(min(y1, cy))
                    x2 = int(max(x2, cx))
                    y2 = int(max(y2, cy))
            midx = (x1 + x2) // 2
            midy = (y1 + y2) // 2
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.circle(image, (midx, midy), 5, (0, 255, 0), -1)
            shortest_distance = min(shortest_distance, depth[midy, midx])
            if shortest_distance == depth[midy, midx]:
                 targetx, targety = midx, midy
        cv2.putText(image, str(depth[targety][targetx]), (targetx,targety), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            
        if state == 1:
            if targetx < width//2 - 30:
                msg.angular.z = 0.3
            elif targetx > width//2 + 30:
                msg.angular.z = -0.3
            else:
                msg.angular.z = 0
                state = 2
            # rotate to align with person
        elif state == 2:
            msg.linear.x = 0.2
            if depth[targety, targetx] < 800:
                msg.linear.x = 0
                state = 3
            else:
                msg.linear.x = 0.2
            # move forward
        elif state == 3:
            # stop and return
            return True
        pub_cmd.publish(msg)
        cv2.imshow("image",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def approach(obj_name):
    state = 1
    timeout = 20
    tell("i am approaching the %s." % obj_name)
    print("Approaching the object:", obj_name)
    # change_to_cam(2)
    time.sleep(0.5)
    startTime = time.time()
    while time.time() - startTime < timeout:
        image = _image.copy()
        depth = _depth.copy()
        msg = Twist()  # Reset command
        h, w, c = image.shape
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = dnn_yolo.forward(bgr_image)
        for boxes in results:
            for i, (x1, y1, x2, y2, score, cls) in enumerate(results[0]["det"]):
                x1, y1, x2, y2, cls = map(int, (x1, y1, x2, y2, cls))      
                obj_name_detected = obj_list[cls]  # e.g., "cup", "cookie"
                if obj_name_detected.lower() == obj_name.lower():
                    midx = (x1 + x2) // 2
                    midy = (y1 + y2) // 2
                    cv2.circle(image, (midx, midy), 5, (0, 255, 0), -1)
                    if state == 1:
                        if midx < w//2 - 30:
                            msg.angular.z = 0.3
                        elif midx > w//2 + 30:
                            msg.angular.z = -0.3
                        else:
                            msg.angular.z = 0
                            state = 2
                        #rotate to align with object
                    elif state == 2:
                        msg.linear.x = 0.2
                        if depth[midy, midx] < 600:
                            msg.linear.x = 0
                            state = 3
                        else:
                            msg.linear.x = 0.2
                        #move forward
                    elif state == 3:
                        #stop and return
                        change_to_cam(1)
                        time.sleep(0.5)
                        return True
        pub_cmd.publish(msg)
    print("Timeout")
    tell("Timeout")

def tell(sentence):
    os.system(f'espeak "{sentence}"')
    print("Telling:", sentence)

def ask(question):
    image = _image.copy()
    
    # Convert numpy array to JPEG bytes in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    """Ask a question about an image, returns the answer."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
    return call_llm(messages)

def grasp():
    print("Grasping the object.")
    tell("Sorry, the arm is not functioning well, please put it on the robot")
    time.wait(3)

def release():
    print("Releasing the object.")
    tell("Please take you object")
    time.wait(3)
    

def follow():
    global _voice
    tell("I will follow you now.")
    print("Starting to follow the person.")

    # PID variables for angular control (keep person centered)
    ang_kp = 0.7      # stronger response
    ang_ki = 0.001
    ang_kd = 0.02      # Increased slightly for damping
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

    stop_threshold_mm = 50    # Stop if within 50mm of target
    braking_zone_mm = 150     # Start braking 200mm from target
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
                tell("Stopping following.")
                _voice = None
                return
            _voice = None

        # Person lost handling
        if targetx is None:
            person_lost_counter += 1
            if person_lost_counter > max_lost_frames:
                tell("I lost you. Stopping.")
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(image, f"vel={lin_vel:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Color code based on distance
        if target_depth_mm < desired_dist_mm - 50:
            cv2.putText(image, "TOO CLOSE!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        elif target_depth_mm > desired_dist_mm + 50:
            cv2.putText(image, "TOO FAR", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rospy.sleep(0.05)

def guide(destination):
    print(f"Guiding person to {destination}.")
    tell(f"Please follow me to the {destination}.")
    nav_to(destination)
    tell(f"We have arrived at the {destination}.")
    print(f"Arrived at {destination}. Guiding completed.")

def detect(type):
    if type == "person":
        print("Detecting person in the image.")
    elif type == "object":
        print("Detecting objects in the image.")

def api_check_posture(posture):
    change_to_cam(2)
    rotate()
    tell("Checking posture.")

    image = _image.copy()
    
    # Convert numpy array to JPEG bytes in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    question = f"Is the person in the image {posture}? Only respond with yes or no."
    print(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
    answer = call_llm(messages).strip().lower()
    if answer == "yes":
        tell("Target Found")
        change_to_cam(1)
        return True
    elif answer == "no":
        api_check_posture(posture)
    else:
        print(f"Unexpected response: {answer}")
        return None

def api_count_posture(posture):
    image = _image.copy()
    tell(f"Counting {posture}")

    # Convert numpy array to JPEG bytes in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    question = f"How many people in the image are {posture}? Answer in one short sentence"
    print(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
    response = call_llm(messages)
    print(response)
    try:
        return response.strip()
    except ValueError:
        return 0

def api_count(obj):
    image = _image.copy()
    tell(f"Counting {obj}")
    # Convert numpy array to JPEG bytes in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    question = f"How many {obj} are there? Answer in one short sentence"
    print(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
    response = call_llm(messages).strip()
    print(response)
    return response

def api_describe_image(obj):
    image = _image.copy()
    tell(f"Getting description of the {obj}.")
    # Convert numpy array to JPEG bytes in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    question = f"Describe the {obj} in the image in one sentence."
    print(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
    response = call_llm(messages).strip()
    print(response)
    return response

def capture_image():
    image = _image.copy()
    image_path = "current_view.jpg"
    print(f"Capturing image to {image_path}")
    cv2.imwrite(image_path, image)
    return image_path

def move_to_instruction_point():
    tell("Returning to instruction point.")
    print("Moving to Instruction Point...")
    nav_to("Host")

def confirm_command(command_text):
    tell(f"Command received: {command_text}")
    print(f"Robot confirms: {command_text}")

def rotate():
    first_person_ignored = False   # state to track if we've already ignored the initial centered person
    tell("Rotating to look for the person.")
    while not rospy.is_shutdown():
        image = _image.copy()
        depth = _depth.copy()
        h, w, c = image.shape
        half = w // 2
        msg = Twist()
        poses = dnn_pose.forward(image)

        # Default: keep rotating (if no person centered or first person not yet ignored)
        msg.angular.z = -0.3

        for pose in poses:
            x1 = 10000
            y1 = 10000
            x2 = -1
            y2 = -1
            for cx, cy, score in pose:
                if cx != 0 and cy != 0:
                    x1 = int(min(x1, cx))
                    y1 = int(min(y1, cy))
                    x2 = int(max(x2, cx))
                    y2 = int(max(y2, cy))

            midx = (x1 + x2) // 2
            midy = (y1 + y2) // 2
            
            # IGNORE PERSONS ON THE LEFT HALF OF THE SCREEN
            if midx < half:
                # print(f"Ignoring person on left side at x={midx}")
                continue  # Skip this person, don't process them
                
            if depth[midy][midx] > 2500:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(image, (midx, midy), 5, (0, 255, 0), -1)

            # Check if person is in the center region (only check right half now)
            # Center region relative to the right half's perspective
            if half + 20 >= midx >= half - 20:  # This checks if person is near the vertical center line
                if not first_person_ignored:
                    # This is the initial centered person – ignore and continue rotating
                    first_person_ignored = True
                    # print("Ignoring first centered person, continuing rotation...")
                    # Do NOT break – keep searching for the next person
                else:
                    # Second (or later) person centered – stop rotating
                    msg.angular.z = 0
                    # print("Found another person in the center. Stopping rotation.")
                    return "found"
                    break   # exit the for loop (stop processing more poses)

        pub_cmd.publish(msg)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print("Rotating robot to look for another person.")

def ask_person_name():
    tell("What is your name? Please say it in a complete sentence. For example, My name is Adam.")
    # Wait for voice input with timeout
    global _voice, _audio_path
    state = 0
    while not rospy.is_shutdown():
        if state == 0:
            if _voice is not None:
                tell("Analyzing name")
                response = audio_extract_name(audio_path=_audio_path)
                success, name = response["success"], response["name"]
                if not success or name is None:
                    state = 1
                else:
                    return name
                _voice = None
        elif state == 1:
            # Wait a bit and then ask again
            tell("Sorry, Please say it again.")
            state = 0
                    
        rospy.sleep(0.1)

def find_person_by_name(name, max_attempts=5):
    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1} to find {name}")
        # capture_image()
        approach_person()
        person_name = ask_person_name()
        if person_name.lower() == name.lower():
            print(f"Found {name}!")
            return True
        else:
            print(f"Not {name} (said '{person_name}'), rotating.")
            rotate()
    print(f"Could not find {name} after {max_attempts} attempts.")
    return False

# ------------------------------
# Planner
# ------------------------------
def get_plan_from_ai(command_text):
    # Read system prompt from external file
    with open("gpsr_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": command_text}
    ]
    tell("Generating plan.")
    content = call_llm(messages)
    if not content:
        return None
    # Remove possible markdown code block markers
    content = content.strip()
    if content.startswith('```json'):
        content = content[7:]
    if content.endswith('```'):
        content = content[:-3]
    try:
        tell("Plan generated.")
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {content}")
        return None

# ------------------------------
# Plan executor
# ------------------------------
def execute_plan(plan):
    result = None
    for step in plan["steps"]:
        action = step["action"]
        params = step.get("parameters", {})
        # Replace $result placeholder in parameters
        for key, value in params.items():
            if isinstance(value, str) and "$result" in value:
                params[key] = value.replace("$result", str(result))
        # Call functions
        if action == "nav_to":
            nav_to(**params)
        elif action == "grasp":
            grasp()
        elif action == "release":
            release()
        elif action == "count_object":
            result = count_object(**params)
        elif action == "approach":
            result = approach(**params)
        elif action == "approach_person":
            approach_person()
        elif action == "tell":
            tell(**params)
        elif action == "ask":
            result = ask(**params)
        elif action == "follow":
            follow()
        elif action == "guide":
            guide(**params)
        elif action == "detect":
            detect(**params)
        elif action == "api_check_posture":
            result = api_check_posture(**params)
        elif action == "api_count_posture":
            result = api_count_posture(**params)
        elif action == "api_describe_image":
            result = api_describe_image(**params)
        elif action == "capture_image":
            result = capture_image()
        elif action == "move_to_instruction_point":
            move_to_instruction_point()
        elif action == "find_person_by_name":
            result = find_person_by_name(**params)
        elif action == "api_count":
            result = api_count(**params)
        else:
            print(f"Unknown action: {action}")
    return result

# ------------------------------
# Main GPSR routine
# ------------------------------
if __name__ == "__main__":
    rospy.init_node("Robot")
    rospy.loginfo("Node started")
    rate = rospy.Rate(20)

    # Globals initialization
    _image = None
    _depth = None
    _voice = None
    _odom = None
    _sub_cam_image, _sub_cam_depth, _current_cam = None, None, 1

    # Subscribers
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    rospy.Subscriber("/respeaker/audio_path", String, callback_path)
    
    # Publishers
    # pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # Initialize Camera
    change_to_cam(1)

    # Chassis
    chassis = RobotChassis()

    # set_joints(jointsPos["idle"][0], jointsPos["idle"][1], jointsPos["idle"][2], jointsPos["idle"][3], 2.5)
    # time.sleep(2.5)

    print("=== GPSR Task Started ===")
    tell("Robot Ready")

    # tell("moving to the entrance")
    # nav_to("entrance")
    tell("waiting for the door to be opened")
    initial_door_distance, doorMarked = 0, False

    # # Door check
    # change_to_cam(2)
    # while not rospy.is_shutdown():
    #     if _image is None or _depth is None:
    #         continue
    #     image = _image.copy()
    #     depth = _depth.copy()
    #     h, w, c = image.shape
    #     midx, midy = w//2, h//2
    #     midy += 100
    #     cv2.circle(image, (midx, midy), 5, (0,255, 0), -1)
    #     if doorMarked == False:
    #         initial_door_distance = depth[midy][midx]
    #         print(initial_door_distance)
    #         doorMarked = True
    #     elif doorMarked:
    #         if depth[midy][midx] > initial_door_distance*1.7 or depth[midy][midx] == 0:
    #             tell("The door is opened. Now im moving to the instruction point")
    #             move_to_instruction_point()
    #             break
    #     cv2.imshow("image",image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     rate.sleep()
        
    change_to_cam(1)
    # tell("Ready for the command.")
    commands = [
        "Describe the person on the sofa"
    ]
    for i, command in enumerate(commands, start=1):
        tell("Command Recieved: {}".format(command))
        print(f"\n--- Command {i}: {command} ---")
        plan = get_plan_from_ai(command)
        if not plan:
            print("Skipping command due to plan failure.")
            continue
        print(f"Plan: {json.dumps(plan, indent=2)}")
        confirm_command(command)
        try:
            execute_plan(plan)
        except Exception as e:
            print(f"Error during execution: {e}. Might need restart or skip.")
        move_to_instruction_point()
    print("\nAll commands executed. Exiting arena.")
