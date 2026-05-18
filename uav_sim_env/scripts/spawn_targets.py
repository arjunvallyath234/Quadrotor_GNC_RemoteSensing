#!/usr/bin/env python3
import rospy
import random
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

def generate_cube_sdf(size, green_shade):
    # Generates an SDF string for a static cube with a specific size and shade of green
    return f"""<?xml version="1.0" ?>
    <sdf version="1.6">
      <model name="target_cube">
        <static>true</static>
        <link name="link">
          <pose>0 0 {size/2.0} 0 0 0</pose>
          <collision name="collision">
            <geometry><box><size>{size} {size} {size}</size></box></geometry>
          </collision>
          <visual name="visual">
            <geometry><box><size>{size} {size} {size}</size></box></geometry>
            <material>
              <ambient>0.0 {green_shade} 0.0 1</ambient>
              <diffuse>0.0 {green_shade} 0.0 1</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>"""

def main():
    rospy.init_node('cube_spawner')
    rospy.loginfo("Waiting for Gazebo spawn service...")
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    rospy.loginfo("Spawning 50 random green cubes...")

    for i in range(200):
        # Randomize parameters
        size = random.uniform(0.01, 1.0)
        x = random.uniform(-50.0, 50.0) # 10x10 field centered at 0
        y = random.uniform(-50.0, 50.0)
        green_shade = random.uniform(0.3, 1.0) # 0.3 (dark green) to 1.0 (bright green)

        cube_name = f"green_cube_{i}"
        sdf_xml = generate_cube_sdf(size, green_shade)

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = 0.0 # SDF handles shifting Z so it sits on the floor

        try:
            spawn_model_prox(cube_name, sdf_xml, "", initial_pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn service failed: {e}")

    rospy.loginfo("All cubes spawned successfully.")

if __name__ == '__main__':
    main()
