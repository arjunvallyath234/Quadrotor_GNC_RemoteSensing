
----------------------------
Source all terminals
----------------------------
source /opt/ros/noetic/setup.bash
source ~/px4_ws/devel/setup.bash


   TEST 1:
   
---------------------------
Terminal 1:
--------------------------
~/px4_ws/run_sim.sh

---------------------------
Terminal 2:
--------------------------

rosrun uav_sim_env flight_strategy_master.py


------------------------------------------
 Additional commands

rosrun uav_sim_env high_alt_flight.py

rosrun uav_sim_env detect_cubes.py

-------------------------------------------



 TEST 2:



---------------------------
Terminal 1:
--------------------------
# Navigate to the PX4 directory
cd ~/PX4-Autopilot

# Source the Gazebo setup script
source Tools/simulation/gazebo-classic/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default

# Add PX4 to your ROS Package Path
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot:~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic

---------------------------
Terminal 2:
--------------------------
#To Launch gazebo world and quadrotor iris 

roslaunch px4 mavros_posix_sitl.launch world:=~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/warehouse.world vehicle:=iris

# To Launch gazebo world and quadrotor iris with downward facing depth camera

   roslaunch px4 mavros_posix_sitl.launch vehicle:=iris_downward_depth_camera


---------------------------
Terminal 3:
--------------------------
# to see the camera output

rosrun rqt_image_view rqt_image_view

