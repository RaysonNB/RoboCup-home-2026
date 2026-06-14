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
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
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
dnn_yolo = Yolov8("tidy_up_model_v2",device_name="GPU")
dnn_yolo.classes = ["bowl", "cereal", "fork", "knife", "milk", "plate", "spoon", "trash"]
obj_list = ["bowl", "cereal", "fork", "knife", "milk", "plate", "spoon", "trash"]

dnn_pose = HumanPoseEstimation(device_name="GPU")
# ------------------------------
# Object categories and Locations
# ------------------------------
objects = {
    "Food Item": ["cereal", "milk"],
    "Kitchen Item": ["fork", "knife", "spoon"],
    "Task Item": ["bowl", "plate", "trash"]
}
place = {
    "Host":[-5.185, -1.966, -1.469],
    "living room": [2.029, 0.298, -0.837],
    "dining table": [-0.181, 0.544, 1.661],
    "bedroom": [-0.182, 0.544, 1.661],
    "cafe table": [-3.410, -4.011, -3.052],
    "desk": [2.735, -2.829, 0.7789],
    "entrance": [-6.014, 4.026, -1.405]
}
jointsPos = {
    "idle": [3.068, 0.601, 0.186, 0.554],
    "ready": [3.105, -0.291, 0.535, -0.739]
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


# ------------------------------
# LLM Server
# ------------------------------
SERVER_URL = "http://172.22.11.127:5000"  # replace with actual IP

def call_llm(messages, model="gemini-2.5-flash-lite"):
    """Send messages to the LLM proxy and return the response content."""
    payload = {'messages': messages, 'model': model}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{SERVER_URL}/chat", data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            resp = json.loads(response.read().decode('utf-8'))
            return resp.get('content')
    except urllib.error.URLError as e:
        print(f"LLM proxy error: {e}")
        return None
    
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
def set_joints(joint1, joint2, joint3, joint4, t):
    service_name = "/goal_joint_space_path"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)
        
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
        request.joint_position.position = [joint1, joint2, joint3, joint4]
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False
    
def set_gripper(angle, t):
    service_name = "/goal_tool_control"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)
        
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [angle]
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False

def open_gripper(t):
    return set_gripper(0.01, t)

def close_gripper(t):
    return set_gripper(-0.01, t)

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
    change_to_cam(2)
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
            detected_names.append(obj_name)
    
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

def approach(obj_name):
    state = 1
    tell("i am approaching the %s." % obj_name)
    print("Approaching the object:", obj_name)
    change_to_cam(2)
    time.sleep(0.5)
    while not rospy.is_shutdown():
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

def tell(sentence):
    os.system(f'espeak "{sentence}"')
    print("Telling:", sentence)

def ask(question, image_path):
    """Ask a question about an image, returns the answer."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
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
    t = 2.5
    set_joints(jointsPos["ready"][0], jointsPos["ready"][1], jointsPos["ready"][2], jointsPos["ready"][3], t)
    time.sleep(t)
    open_gripper(1.0)
    tell("Please place the object in my gripper. I'm closing in 3 seconds.")
    print("Please place the object in my gripper. I'm closing in 3 seconds.")
    time.sleep(3)
    tell("Closing gripper now.")
    print("Closing gripper.")
    close_gripper(1.0)
    time.sleep(1)
    set_joints(jointsPos["idle"][0], jointsPos["idle"][1], jointsPos["idle"][2], jointsPos["idle"][3], t)
    time.sleep(t)

def release():
    print("Releasing the object.")
    t = 2.5
    set_joints(jointsPos["ready"][0], jointsPos["ready"][1], jointsPos["ready"][2], jointsPos["ready"][3], t)
    time.sleep(t)
    tell("Please take the object in my gripper. I'm opening in 3 seconds.")
    print("Please take the object in my gripper. I'm opening in 3 seconds.")
    time.sleep(3)
    tell("Opening gripper now.")
    print("Opening gripper.")
    open_gripper(1.0)
    time.sleep(1)
    set_joints(jointsPos["idle"][0], jointsPos["idle"][1], jointsPos["idle"][2], jointsPos["idle"][3], t)
    time.sleep(t)

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
    global _voice
    state = 0
    while not rospy.is_shutdown():
        if state == 0:
            if _voice is not None:
                sentence = _voice.text.strip()
                name = sentence.split()[-1]  # Assume the name is the last word spoken
                if name[0].isupper():
                    print(f"Person said their name is: {name}")
                    _voice = None
                    return name
                else:
                    state = 1
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
    
    # Publishers
    # pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # Initialize Camera
    change_to_cam(1)

    # Chassis
    chassis = RobotChassis()

    use_of_qr = False
    qr_detector = None
    if use_of_qr:
        qr_detector = cv2.QRCodeDetector()
    state_for_voice_command = 0

    set_joints(jointsPos["idle"][0], jointsPos["idle"][1], jointsPos["idle"][2], jointsPos["idle"][3], 2.5)
    time.sleep(2.5)

    print("=== GPSR Task Started ===")
    tell("Robot Ready")

    # tell("moving to the entrance")
    # nav_to("entrance")
    # tell("waiting for the door to be opened")
    # initial_door_distance, doorMarked = 0, False

    # Door check
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
        
    tell("Ready for the command.")
    _voice = None

    while not rospy.is_shutdown():
        # Wait until we have a valid image
        if _image is None:
            rospy.sleep(0.1)
            continue

        # Copy the latest image for QR detection (avoid threading issues)
        image = _image.copy()
        if image is None:
            continue
        
        if use_of_qr:
            # Detect and decode QR code
            data, bbox, _ = qr_detector.detectAndDecode(image)

            if data:
                # Valid QR code detected
                command = data.strip()
                rospy.loginfo(f"QR Code detected: {command}")
                
                if bbox is not None:
                    points = bbox[0] if len(bbox.shape) == 3 else bbox 
                    points = points.astype(int)

                    cv2.polylines(image, [points], isClosed = True, color=(0, 255, 0), thickness = 2)
                # Confirm command to user
                tell(f"Command received: {command}")

                # Get plan from AI
                plan = get_plan_from_ai(command)
                if not plan:
                    tell("Sorry, I could not understand the command.")
                    # Return to instruction point and wait for next QR
                    move_to_instruction_point()
                    tell("Ready for the next command.")
                    continue

                rospy.loginfo(f"Plan: {json.dumps(plan, indent=2)}")

                # Execute the plan
                try:
                    execute_plan(plan)
                except Exception as e:
                    rospy.logerr(f"Error during execution: {e}")
                    tell("An error occurred while executing the command. Please try input again.")

                # After finishing (or failing), go back to instruction point
                move_to_instruction_point()
                tell("Ready for the next command.")

                # Small delay to avoid re-reading the same QR code immediately
                rospy.sleep(2)
        else:
            if state_for_voice_command == 0:
                if _voice is not None:
                    command = _voice.text.strip()
                    _voice = None
                    rospy.loginfo(f"Command Recieved: {command}")
                    tell(f"Command received: {command}. Is this your command?")
                    state_for_voice_command = 1
            elif state_for_voice_command == 1:
                if _voice is not None:
                    if "yes" in _voice.text:
                        state_for_voice_command = 2
                    elif "no" in _voice.text:
                        state_for_voice_command = 0
                        tell("Please input again")
                    _voice = None
            elif state_for_voice_command == 2:
                plan = get_plan_from_ai(command)
                if not plan:
                    tell("Sorry, I could not understand the command.")
                    move_to_instruction_point()
                    tell("Ready for the next command.")
                    state_for_voice_command = 0

                rospy.loginfo(f"Plan: {json.dumps(plan, indent=2)}")

                # Execute the plan
                try:
                    execute_plan(plan)
                except Exception as e:
                    rospy.logerr(f"Error during execution: {e}")
                    tell("An error occurred while executing the command. Please try input again.")

                # After finishing (or failing), go back to instruction point
                move_to_instruction_point()
                tell("Ready for the next command.")
                state_for_voice_command = 0
                # Small delay to avoid re-reading the same QR code immediately
                rospy.sleep(2)
        # Maintain ROS rate (optional, but keeps loop responsive)
        cv2.imshow("image",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rate.sleep()