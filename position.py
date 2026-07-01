#!/usr/bin/env python3
import json
import os
import time
import rospy
from RobotChassis import RobotChassis
from std_srvs.srv import Empty


def speak(g):
    os.system(f'espeak "{g}"')
    print(f"[VOICE] {g}")

clear_costmaps = None


def init_services():
    global clear_costmaps
    try:
        rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
        clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    except rospy.ROSException:
        rospy.logwarn("Costmap clearing service not available.")


def walk_to(name, target_locations, chassis):
    name_key = name.lower()
    if "none" in name_key or "unknown" in name_key:
        return

    if name_key in target_locations:
        coords = target_locations[name_key]
        speak(f"Going to {name}")

        chassis.move_to(coords[0], coords[1], coords[2])

        while not rospy.is_shutdown():
            code = chassis.status_code
            if code == 3:
                speak(f"Arrived at {name}")
                break
            elif code == 4:
                speak(f"Failed to reach {name}")
                break
            time.sleep(0.5)

        if clear_costmaps:
            try:
                clear_costmaps()
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        time.sleep(1)
    else:
        rospy.logerr(f"Location '{name}' not found.")


log_positions = {
    "tv table 1": [1.415, 1.083, 1.591],
    "tv table 2": [1.949, 1.041, 1.510],
    "sofa table": [4.492, -0.794, -1.331],
    "sink table 1": [5.827, -0.947, -1.572],
    "sink table 2": [6.583, -1.118, -1.649],
    "dish washer": [7.501, -1.065, -1.613],
    "cab1": [1.438, 3.457, -1.561],
    "washing machine": [2.945, 4.846, 1.687],
    "table 1": [1.695, 4.408, 1.624],
    "table 2": [1.035, 4.330, 1.677],
    "finding": [2.400, 3.761, -1.482],
    "cab2": [6.370, 1.411, -3.092]
}

other_mission = {
    "laundry": {
        "laundry": [1.990, 3.655, 2.367],
        "laundry table": [1.459, 4.204, 1.574],
        "washing machine": [2.790, 4.580, 1.495],
        "shelf": [1.179, 3.042, -1.554],
        "laundry trash bin": [2.298, 2.696, -1.799]
    },
    "bedroom": {
        "bedroom": [6.476, 4.525, -1.919],
        "bed": [5.330, 4.019, -1.376],
        "host": [6.688, 4.250, -3.062]
    },
    "living room": {
        "living room": [2.650, -0.111, 2.105],
        "tv stand": [1.724, 0.755, 1.568],
        "sofa": [2.842, -0.764, -1.586],
        "coffee table": [4.416, -1.109, -1.557]
    },
    "kitchen": {
        "kitchen": [7.156, 1.411, 0.117],
        "cabinet": [6.411, 1.527, -3.121],
        "refrigerator": [7.324, 1.937, 0.242],
        "counter": [5.644, -1.478, -1.653],
        "sink": [6.203, -1.444, -1.626],
        "cooking table": [7.023, -1.453, -2.123],
        "dishwasher": [8.306, -1.300, -2.099],
        "kitchen trash bin": [8.266, 1.616, 0.117],
        "dinner table": [7.891, -0.548, 2.305]
    }
}

all_locations = {}
for category in other_mission.values():
    for name, coords in category.items():
        all_locations[name.lower()] = coords

for name, coords in log_positions.items():
    all_locations[name.lower()] = coords

if __name__ == "__main__":
    try:
        rospy.init_node("gpsr_navigation_node")
        chassis = RobotChassis()
        init_services()

        speak("Starting log positions.")
        for loc in log_positions.keys():
            if rospy.is_shutdown(): break
            walk_to(loc, all_locations, chassis)

        speak("GPSR start")
        time.sleep(1)

        speak("Continuing with GPSR locations.")
        for loc in all_locations.keys():
            if rospy.is_shutdown(): break
            if loc not in [k.lower() for k in log_positions.keys()]:
                walk_to(loc, all_locations, chassis)

        speak("All navigation tasks finished.")

    except rospy.ROSInterruptException:
        pass
