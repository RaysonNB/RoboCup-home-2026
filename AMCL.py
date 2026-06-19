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

def get_clockwise_rotation(x1, y1, z1_deg, x2, y2):
    # 1. 計算目標角度 (使用 atan2)
    # 注意：標準 atan2 是逆時針為正，我們取負號改為順時針為正
    dx = x2 - x1
    dy = y2 - y1
    target_angle_deg = -math.degrees(math.atan2(dy, dx))
    
    # 2. 計算轉動差值
    diff = target_angle_deg - z1_deg
    
    # 3. 標準化到 -180 到 180 度之間
    # 這樣可以保證轉動的是「最短路徑」
    diff = (diff + 180) % 360 - 180
    
    return diff

# 測試
# x1, y1 = 0, 0
# z1 = 0 (假設面向正右方，根據你的定義可能需要調整初始方向)
# x2, y2 = 0, 1 (在正上方)
rotation = get_clockwise_rotation(0, 0, 0, 0, 1)
print(f"需要順時針轉動: {rotation:.2f} 度")
# ==========================================================
# 測試用 Main 流程
# ==========================================================
if __name__ == "__main__":
    rospy.init_node("amcl_tuning_node") 
    disable_amcl_alignment()
    
    rospy.sleep(3)
    
    enable_amcl_alignment()
