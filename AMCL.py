import rospy
import dynamic_reconfigure.client

def disable_amcl_alignment():
    try:
        client = dynamic_reconfigure.client.Client("amcl", timeout=5)
        
        client.update_configuration({"update_min_d": 999.0, "update_min_a": 999.0})
        rospy.loginfo("【State 1】成功鎖定 AMCL 自動對位（保持雷達避障）")
    except rospy.ROSException:
        rospy.logerr("無法連接到 AMCL 節點，請檢查 AMCL 是否有啟動！")


def enable_amcl_alignment():
    try:
        client = dynamic_reconfigure.client.Client("amcl", timeout=5)
        client.update_configuration({"update_min_d": 0.05, "update_min_a": 0.2})
        rospy.loginfo("【State 2】成功回復 AMCL 自動對位機制！")
    except rospy.ROSException:
        rospy.logerr("無法連接到 AMCL 節點，請檢查 AMCL 是否有啟動！")


# ==========================================================
# 測試用 Main 流程
# ==========================================================
if __name__ == "__main__":
    rospy.init_node("amcl_tuning_node") 
    disable_amcl_alignment()
    
    rospy.sleep(3)
    
    enable_amcl_alignment()
