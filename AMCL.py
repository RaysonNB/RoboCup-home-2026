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

import math

def calculate_rotation(x1, y1, z1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    target_angle = math.atan2(dy, dx)
    diff = target_angle - z1
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    
    return diff

x1, y1, z1 = 0, 0, 0  # 當前在原點，面向右 (0弧度)
x2, y2 = 1, 1        # 目標在右上角

rotation = calculate_rotation(x1, y1, z1, x2, y2)
print(f"需要轉動弧度: {rotation:.4f}")
print(f"需要轉動角度: {math.degrees(rotation):.2f} 度")
# ==========================================================
# 測試用 Main 流程
# ==========================================================
if __name__ == "__main__":
    rospy.init_node("amcl_tuning_node") 
    disable_amcl_alignment()
    
    rospy.sleep(3)
    
    enable_amcl_alignment()
