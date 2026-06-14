#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *

def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        

dnn_yolo = Yolov8("tidy_up_model_v2",device_name="GPU")
dnn_yolo.classes = classes =[0, 1, 2, 3, 4, 5, 6, 7]
obj = ["bowl", "cereal", "fork", "knife", "milk", "plate", "spoon", "trash"]

if __name__=='__main__':
    rospy.init_node("node_test")
    _image = None
    _topic_image = "/cam1/color/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        image = _image.copy()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = dnn_yolo.forward(bgr_image)
        for boxes in results:
            for i, (x1, y1, x2, y2, score, cls) in enumerate(results[0]["det"]):
                x1, y1, x2, y2, cls = map(int, (x1, y1, x2, y2, cls))
                text = "Object: {} Score: {} ".format(obj[cls], score)
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0),1)
                cv2.putText(image, text, (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 1)  
        cv2.imshow('detection',image)
        if cv2.waitKey(1) == 27: break
        
