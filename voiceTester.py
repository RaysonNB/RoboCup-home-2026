import rospy
from mr_voice.msg import Voice
from datetime import datetime
from std_msgs.msg import String
_voice = None

def callback_voice(msg):
    global _voice 
    _voice = msg
                 
def callback_path(msg):
    global _audio_path
    _audio_path = msg

if __name__ == "__main__":
    rospy.init_node("Robot")
    rospy.loginfo("Node Start")
    rate = rospy.Rate(20)

    rospy.Subscriber("/voice/text",Voice,callback_voice)
    rospy.Subscriber("/respeaker/audio_path",String, callback_path)
    while not rospy.is_shutdown():
        rate.sleep()
        if _voice is not None:
            print(_voice.text)
            print(_audio_path.data)
            _voice = None