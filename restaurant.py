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
# from qreader import QReader
import requests
import base64
import json
import os

# Globals
_image = None
_depth = None
_voice = None
_odom = None
_low_image = None
_low_depth = None
CvBr = CvBridge()
code = 0 # Navigation result code
_current_cam = 1 # Current camera id
_sub_cam_image = None
_sub_cam_depth = None
start_pos = None

# qreader = QReader()

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

model = YOLO("yolov8n-pose.pt")

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

def callback_low_image(msg):
    global _low_image
    _low_image = CvBr.imgmsg_to_cv2(msg, "bgr8")

def callback_low_depth(msg):
    global _low_depth
    _low_depth = CvBr.imgmsg_to_cv2(msg)

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

def turn(angle: float):   # 左正右負
    global _odom
    if _odom is None:
        return

    yaw = get_yaw_from_odom()
    target = normalize_angle(yaw + angle)
    turn_to(target, 0.5)


def change_to_cam(idx):
    global _sub_cam_image, _sub_cam_depth, _current_cam
    _current_cam = idx

    if _sub_cam_image is not None:
        _sub_cam_image.unregister()
    if _sub_cam_depth is not None:
        _sub_cam_depth.unregister()

    if idx == 2:
        _sub_cam_image = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    elif idx == 1:
        _sub_cam_image = rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)
        _sub_cam_depth = rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

def detect_real_xyz(x1,y1,d):
    global image
    h, w, c = image.shape
    y1 -= h/2
    x1 -= w/2
    y = y1/h*2*d*math.tan(45.5*np.pi/360)
    x = x1/w*2*d*math.tan(58.4*np.pi/360)
    return x,y,d

class AI:
    API_KEY = "sk-lm-Z7kmhaBy:URWIhcO11V4clG2ex936"
    # BASE_URL = "http://fablabai.oake.dpdns.org/v1/chat/completions"
    BASE_URL = "http://fablabai.pcms.dpdns.org/v1/chat/completions"
    MODEL_NAME = "qwen/qwen3.5-9b"

    def __init__(self, system_prompt=None):
        self.messages = []
        self.new_chat(system_prompt)

    def new_chat(self, system_prompt=None):
        self.messages = []
        if system_prompt is not None:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

    def send_request(self, prompt, image_path=None):
        content = [{"type": "text", "text": prompt}]

        if image_path is not None:
            try:
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:
                return print(f"❌ Image Error: {e}")

        self.messages.append({"role": "user", "content": content})

        response = requests.post(
            url = self.BASE_URL,
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            },
            data = json.dumps({
                "model": self.MODEL_NAME,
                "messages": self.messages
            })
        )

        for k, v in response.json().items():
            print(k, v)
        response_msg = response.json()["choices"][0]["message"]
        print("ai:", response_msg["content"])
        self.messages.append(response_msg)
        return response_msg["content"]

if __name__ == "__main__":
    rospy.init_node("robot started")
    rospy.loginfo("Node started")
    rate = rospy.Rate(20)

    # Globals initialization
    _image = None
    _depth = None
    _voice = None
    _odom = None
    _sub_cam_image, _sub_cam_depth, _current_cam = None, None, 1

    # Subscribers
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    # rospy.Subscriber("/imu/data", Imu, callback_Imu)
    
    # Publishers
    pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # Initialize Camera
    change_to_cam(1)

    # Chassis
    chassis = RobotChassis()
    
    # AI
    prompt = """
You are a waiter in a restaurant. Listen to the customer's words, and understand what he/she want. Remember to be polite to the customer.
Note: The customer's audio may be misrecognized. 
New orders will not overwrite previous orders unless requested by the customer.
You have to give me the response with the form following the rules: 
There are three lines of response in total. The first line is a python format list, with all the order which the customer mentioned cumulatively (words corrected if needed).
For example: ["apple juice", "sandwich"] is a valid list.
The second line is your response to the customer.
For example, the customer said "I want coffee and rice" and you should response "I understand, you want a cup of coffee and a bowl of rice". 
The third line include only one word, "yes" or "no", lowercase letter. You have to judge if you can serve the food now, ask the customer if the food can serve now.
Important: All output should be in plain text, no extra markdown.
If there is some vague prompt from user, ask him/her.
"""
    ai = AI()
    ai.new_chat(prompt)

    # Variables
    state = 0

    say("Demo start.")

    while not rospy.is_shutdown():
        rate.sleep()
        image = _image.copy()
        depth = _depth.copy()
        w, h = image.shape[0], image.shape[1]


        if image is None or depth is None:
            print("image None")
            continue
        
        if state == 0:
            say("mission start")
            
            start_pos = get_pos()
            
            state = 1
            
        elif state == 1:   # check customer waving hand
            result = model(image)[0]
            if result.boxes is not None and len(result.boxes) > 0:   # found person
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
                    
                    if lwrist_y < nose_y or rwrist_y < nose_y:   # wrist above nose
                        if 0 <= nose_x <= w and 0 <= nose_y <= h:   # check if position is valid
                            # draw boxes for partial points
                            x1, y1, x2, y2 = map(int, result.boxes.xyxy)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.imwrite("image/proof1.jpg")
                            
                            person_depth = depth[nose_y][nose_x]
                            if person_depth and person_depth > 0 and np.isfinite(person_depth):   # check if depth is valid
                                state = 2
                except Exception as e:
                    print(f"error: {e}")

        elif state == 2:  # find & go to the customer - facing the customer
            result = model(image[0])
            person_keypoints = result.keypoints.xy.cpu().numpy()
            nose = person_keypoints[0]
            image_mid = w // 2
            diff = 10   # 置中誤差
            
            # 正對 customer
            if image_mid-diff <= nose[0] <= image_mid+diff:
                state == 3
            else:
                if nose[0] > image_mid:   # customer at right
                    turn(10/180*np.pi)
                else:
                    turn(-10/180*np.pi)
                
        elif state == 3:  # find & go to the customer - navigate to customer's table
            human_depth = depth[h//2][w//2]
            theta = normalize_angle(get_yaw_from_odom())     # 0度夾角
            robotx, roboty = get_pos()    # robot position
            
            human_pos = (robotx + human_depth*np.cos(theta), roboty + human_depth*np.sin(theta))
            
            chassis.move_to(*human_pos, 0)
            if chassis.status_code == 3:
                state = 4
                
        elif state == 4:    # use AI to understand customer
            cur_prompt = None
            # decoded = qreader.detect_and_decode(image=image)
            # if decoded and decoded[0] is not None:
            #     print("decoded", decoded[0])
            #     cur_prompt = decoded[0]
            #     break
            
            cur_prompt = _voice.text        # use if JapanOpen does not allow qrcode
            ai_response:str = ai.send_request(cur_prompt)
            text = ai_response.split('\n')
            
            order_list = eval(text[-3])
            ai_say = text[-2]
            can_deliver_order = True if text[-1] == "yes" else False
            
            say(ai_say)
            
            if can_deliver_order:
                state = 5
                
        elif state == 5:      # deliver food - navigate to the barman
            chassis.move_to(*start_pos, 0)
            if chassis.status_code == 3:
                state = 6

        elif state == 6:      # deliver food - get item
            say("the customer want the following items:")
            say(f"{', '.join(order_list)}")
            say("please help to put the items inside my head, thank you")
            
            state = 7
            
        elif state == 7:      # deliver food - return to the customer
            chassis.move_to(*human_pos, 0)
            if chassis.status_code == 3:
                state = 8
                
        elif state == 8:      # deliver food - give item
            say("please get the items inside my head, thank you")
            
            state == 9
            
        elif state == 9:        # back to the start position
            chassis.move_to(*start_pos, 0)


        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break