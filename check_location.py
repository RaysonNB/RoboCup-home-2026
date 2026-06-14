import rospy
from RobotChassis import RobotChassis

if __name__ == "__main__":
    rospy.init_node("location_printer")
    rospy.loginfo("start")
    chassis = RobotChassis()
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rate.sleep()
        print(chassis.get_current_pose())