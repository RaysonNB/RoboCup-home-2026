roslaunch orbbec_camera multi_gemini2.launch


roslaunch open_manipulator_controller open_manipulator_controller.launch usb_port:=/dev/arm
roslaunch open_manipulator_control_gui open_manipulator_control_gui.launch


roslaunch lingao_bringup robot.launch
roslaunch lingao_navigation navigate.launch map_file:=/home/pcms/maps/bestopl.yaml
rviz

roslaunch xpkg_demo chassis_navigation.launch map_file:=/home/pcms/maps/restaurant.yaml


roslaunch mr_voice voice.launch

rosservice call /move_base/clear_costmaps 


roslaunch lingao_slam slam.launch

rosrun map_server map_saver -f /home/pcms/maps/japan2027


roslaunch xpkg_demo chassis_rplidar.launch

roslaunch xpkg_demo chassis_bringup.launch

bash pyAgxArm/pyAgxArm/scripts/ubuntu/can_activate.sh can0 1000000

rosparam get /amcl/update_min_a #0.2
rosparam get /amcl/update_min_d #0.05
rosparam set /amcl/update_min_a 9999.0
rosparam set /amcl/update_min_d 9999.0
