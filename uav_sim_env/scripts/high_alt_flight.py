#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg

def main():
    rospy.init_node("fly_10m_node")
    
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    
    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
    
    rate = rospy.Rate(20) # Must be faster than 2Hz
    
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    # Target position: 30 meters straight up
    pose = PoseStamped()
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 30.0 

    # Send dummy setpoints before starting to satisfy the OFFBOARD requirement
    for _ in range(100):
        if rospy.is_shutdown(): break
        local_pos_pub.publish(pose)
        rate.sleep()

    last_req = rospy.Time.now()

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            if set_mode_client.call(0, 'OFFBOARD').mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        else:
            if not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if arming_client.call(True).success:
                    rospy.loginfo("Vehicle armed")
                last_req = rospy.Time.now()

        local_pos_pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    main()

