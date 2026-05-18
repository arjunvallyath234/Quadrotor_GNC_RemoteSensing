#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import csv
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CubeDetector:
    def __init__(self):
        rospy.init_node('cube_detector_node', anonymous=True)
        self.bridge = CvBridge()
        
        self.latest_cv_image = None
        self.latest_target_coords = []
        
        self.img_width = None
        self.img_height = None
        self.meters_per_pixel = 0.025 

        self.image_sub = rospy.Subscriber("/iris_downward_depth_camera/camera/rgb/image_raw", Image, self.image_callback)
        rospy.loginfo("Subscribed to camera feed. Waiting for images...")
        rospy.loginfo("Press 's' in the OpenCV window to save the map and target data!")

    def pixel_to_gazebo_coords(self, cX, cY):
        if self.img_width is None or self.img_height is None:
            return 0.0, 0.0
            
        center_x = self.img_width / 2.0
        center_y = self.img_height / 2.0
        
        gazebo_x = (cY - center_y) * -self.meters_per_pixel 
        gazebo_y = (cX - center_x) * -self.meters_per_pixel
        
        return round(gazebo_x, 3), round(gazebo_y, 3)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        if self.img_width is None:
            self.img_height, self.img_width = cv_image.shape[:2]

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 140]) 
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display_image = cv_image.copy()
        current_targets = []
        valid_contours = []
        max_area = 0.0

        # PASS 1: Filter noise and find the largest cube area for normalization
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50: # Filter out tiny noise
                valid_contours.append((contour, area))
                if area > max_area:
                    max_area = area

        # PASS 2: Calculate scores, draw UI, and save coordinates
        if max_area > 0:
            for contour, area in valid_contours:
                # Normalize the score between 0.0 and 1.0 (rounded to 2 decimal places)
                # We add a small baseline (e.g., max(0.1, ...)) so even tiny cubes are worth something
                raw_score = area / max_area
                score = round(max(0.1, raw_score), 2)

                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw the bounding box
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Draw the score text right above the bounding box
                cv2.putText(display_image, f"{score}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(display_image, (cX, cY), 5, (255, 0, 0), -1)
                    
                    gX, gY = self.pixel_to_gazebo_coords(cX, cY)
                    # Append the score to the data list!
                    current_targets.append([gX, gY, score])

        self.latest_cv_image = display_image
        self.latest_target_coords = current_targets

        cv2.imshow("Bright Green Cubes Detected", display_image)
        cv2.imshow("HSV Filter Mask", mask)
        
        key = cv2.waitKey(3) & 0xFF
        if key == ord('s'):
            self.save_mapping_data()

    def save_mapping_data(self):
        if self.latest_cv_image is None or not self.latest_target_coords:
            rospy.logwarn("No image or targets to save yet!")
            return

        save_dir = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
        
        img_path = os.path.join(save_dir, "high_alt_map.jpg")
        cv2.imwrite(img_path, self.latest_cv_image)
        
        csv_path = os.path.join(save_dir, "target_coordinates.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Added the Score column to the header
            writer.writerow(["Target_ID", "Gazebo_X", "Gazebo_Y", "Score"])
            for idx, coord in enumerate(self.latest_target_coords):
                writer.writerow([idx, coord[0], coord[1], coord[2]])
                
        rospy.loginfo(f"SUCCESS: Saved map to {img_path}")
        rospy.loginfo(f"SUCCESS: Saved {len(self.latest_target_coords)} targets with scores to {csv_path}")

if __name__ == '__main__':
    try:
        detector = CubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
