#!/usr/bin/env python3.8
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

def get_clockwise_rotation(x1, y1, z1_deg, x2, y2):
    # 1. 計算目標角度 (使用 atan2)
    # 注意：標準 atan2 是逆時針為正，我們取負號改為順時針為正
    dx = x2 - x1
    dy = y2 - y1
    target_angle_deg = -math.degrees(math.atan2(dx,dy))
    
    # 2. 計算轉動差值
    diff = target_angle_deg - z1_deg
    
    # 3. 標準化到 -180 到 180 度之間
    # 這樣可以保證轉動的是「最短路徑」
    diff = (diff + 180) % 360 - 180
    
    return diff
def disable_amcl_alignment():
    print("")
def enable_amcl_alignment():
    print("")
# --- 全域變數 ---
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

        if abs(e) < 0.01 or rospy.get_time()-start > 8:
            break

        move(0.0, max_speed*speed*e)
        rospy.Rate(20).sleep()

    move(0,0)
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
    # 判斷舉手：手腕 Y < 肩膀 Y (畫面越上方 Y 值越小)
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
        print("angle",yaw)
        target_x = trans[0] + dist_meters #test np.cos(yaw)
        target_y = trans[1] + dist_meters * np.tan(yaw) #test np.sin(yaw)
        return target_x, target_y, yaw
    except Exception as e:
        rospy.logerr(f"TF Error: {e}")
        return None, None, None

if __name__ == "__main__":
    rospy.init_node("robot_navigator_full")
    tf_listener = tf.TransformListener()
    _odom = None
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image)
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    chassis = RobotChassis()
    
    # 紀錄初始位置 (假設開機時就是起始點)
    rospy.sleep(1) # 等待 TF 初始化
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    
    startx, starty, start_yaw = get_map_coords(0)
    print("start", startx, starty, start_yaw)
    state = 0 # 0:找人/偵測, 1:導航去客人那, 2:回到起點
    
    say("System ready.")
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()
        if _image is None or _depth is None: continue
        image = _image.copy()
        #turn
        # --- State 0: 找人 ---
        if state == 0:
            poses = net_pose.forward(image)
            for pose in poses:
                hand = is_waving(pose)
                if hand:
                    say(f"Detected {hand} hand. Moving to customer.")
                    p5, p6 = pose[5], pose[6]
                    cx, cy = int((p5[0]+p6[0])/2), int((p5[1]+p6[1])/2)
                    dist = _depth[cy][cx] / 1000.0
                    
                    if 0.5 < dist < 10.0:
                        print("distance",dist)
                        hx, hy, hyaw = get_map_coords(dist)
                        print("human pos",hx,hy,hyaw)
                        tx, ty, tyaw = get_map_coords(dist - 0.5-0.5)
                        print(tx, ty, tyaw )
                        
                        clear_costmaps
                        disable_amcl_alignment()
                        chassis.move_to(tx, ty, 0)
                        while not rospy.is_shutdown():
                            code = chassis.status_code
                            if code == 3:
                                state = 0 # 重啟任務
                                break
                            if code in [4, 0]:
                                say("arrive failed.")
                                state = 0
                                break
                            rospy.sleep(0.1)
                        state = 2
                        x1, y1, z1 = get_map_coords(0)
                        rotation = get_clockwise_rotation(x1,y1,z1, hx, hy)
                        print("angle",rotation)
                        say("facing the customer")
                        turn(rotation*np.pi/180)
                        say("I arrive in front of the customer")
                        time.sleep(2)
                        say("Returned to start.")


        # --- State 2: 回到起點 ---
        elif state == 2:
            clear_costmaps
            chassis.move_to(startx, starty, start_yaw)
            while not rospy.is_shutdown():
                code = chassis.status_code
                if code == 3:
                    say("Returned to start.")
                    state = 0 # 重啟任務
                    break
                if code in [4, 0]:
                    say("Return failed.")
                    state = 0
                    break
                rospy.sleep(0.1)
            enable_amcl_alignment()
        cv2.imshow("Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
