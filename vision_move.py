#!/usr/bin/env python3.8
import rospy
import tf
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from pcms.openvino_models import HumanPoseEstimation
from RobotChassis import RobotChassis

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
        target_x = trans[0] + dist_meters #test np.cos(yaw
        target_y = trans[1] + dist_meters * np.tan(yaw) #test np.sin(yaw)
        return target_x, target_y, yaw
    except Exception as e:
        rospy.logerr(f"TF Error: {e}")
        return None, None, None

if __name__ == "__main__":
    rospy.init_node("robot_navigator_full")
    tf_listener = tf.TransformListener()
    
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    chassis = RobotChassis()
    
    # 紀錄初始位置 (假設開機時就是起始點)
    rospy.sleep(1) # 等待 TF 初始化
    startx, starty, start_yaw = get_map_coords(0)
    
    state = 0 # 0:找人/偵測, 1:導航去客人那, 2:回到起點
    
    say("System ready.")
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()
        if _image is None or _depth is None: continue
        image = _image.copy()

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
                    
                    if 0.5 < dist < 4.0:
                        tx, ty, tyaw = get_map_coords(dist - 0.5)
                        chassis.move_to(tx, ty, tyaw)
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
                        state = 2
                        time.sleep(2)


        # --- State 2: 回到起點 ---
        elif state == 2:
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

        cv2.imshow("Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
