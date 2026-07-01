#!/usr/bin/env python3
import json
import os
import time
import rospy
from RobotChassis import RobotChassis
from std_srvs.srv import Empty
def speak(g):
    os.system(f'espeak "{g}"')
    print(g)
clear_costmaps = None
try:
    rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
except rospy.ROSException:
    rospy.logwarn("Costmap clearing service not available.")


def walk_to(name, target_locations):
    name = name.lower()
    if "none" in name or "unknown" in name:
        rospy.logwarn(f"Invalid location: {name}")
        return

    if name in target_locations:
        speak(f"going to {name}")
        coords = target_locations[name]
        chassis.move_to(coords[0], coords[1], coords[2])
        while not rospy.is_shutdown():
            code = chassis.status_code
            if code == 3:
                speak("arrived")
                break
            elif code == 4:
                speak("failed to reach destination")
                break
            time.sleep(0.2)

        time.sleep(1)
        if clear_costmaps:
            try:
                clear_costmaps
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
    else:
        rospy.logerr(f"Location '{name}' not found in database.")
other_mission = {
    "tv table 1": [1.282, 1.508, 1.569],
    "tv table 2": [1.881, 1.260, 1.703],
    "dish washer": [7.417, -1.169, -1.512],
    "cabinet 1": [6.164, 1.755, -3.003],
    "cabinet 2": [1.065, 3.790, -1.481],
    "taking clothes": [3.006, 3.916, 1.652],
    "ri": [6.11, -0.46, 0]
}
locations = {}
json_path = os.path.join(os.path.dirname(__file__), "locations.json")
try:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        for room, items in raw_data.items():
            for item_name, coordinates in items.items():
                locations[item_name.lower()] = coordinates
    else:
        rospy.logwarn("locations.json not found.")
except Exception as e:
    rospy.logerr(f"Failed to load locations.json: {e}")

if __name__ == "__main__":
    rospy.init_node("demo_node")
    chassis = RobotChassis()
    if locations:
        speak("Starting GPSR locations")
        for x in locations.keys():
            walk_to(x, locations)
    speak("Starting other locations")
    for x in other_mission.keys():
        walk_to(x, other_mission)

    speak("all positions completed")
