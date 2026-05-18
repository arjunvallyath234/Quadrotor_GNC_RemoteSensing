#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt_subplots
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tf_trans

# ADDED FOR RL
import torch
import torch.nn as nn

# ==========================================
# 1. EEPC-ACO ROUTER (Optimized + Home Node)
# ==========================================
class ACO_Router:
    def __init__(self):
        self.MAXN = 100
        self.n_a = 1000        
        self.n_t = 8        
        self.alpha = 1.0    
        self.beta = 5.0     
        self.gamma = 6.0    
        self.rho0 = 0.10
        self.rhoMax = 0.8
        self.phi = 0.10
        self.q0 = 0.85
        
        self.tao = np.ones((self.MAXN, self.MAXN))
        
        self.ETA_HOVER = 0.75
        self.RHO_AIR = 1.225
        self.A_DISK_HOVER = 0.2027
        self.THRUST_N = 1.597 * 9.81
        self.HOVER_TIME_SEC = 0.2
        self.c_D = 0.9
        self.A_mambo = 0.00536
        self.mod_vd = 3.0
        self.T_lift = 0.063 * 9.81

    def compute_route(self, uav_pos, targets, scores, wind_dir, wind_speed, e_remain, home_pos):
        if len(targets) == 0:
            return [uav_pos, home_pos]

        Ntot = len(targets) + 2
        
        if Ntot > self.tao.shape[0]:
            new_size = Ntot + 50
            new_tao = np.ones((new_size, new_size))
            new_tao[:self.tao.shape[0], :self.tao.shape[1]] = self.tao
            self.tao = new_tao

        nodes = np.vstack([uav_pos, targets, home_pos])
        
        node_scores = np.insert(scores, 0, 0.0)
        node_scores = np.append(node_scores, 0.0)

        wind_speed_ms = wind_speed / 3.6
        wind_vec = np.array([math.cos(math.radians(270 - wind_dir)), math.sin(math.radians(270 - wind_dir))]) * wind_speed_ms
        
        P_hover = (abs(self.THRUST_N)**1.5) / (self.ETA_HOVER * math.sqrt(2.0 * self.RHO_AIR * self.A_DISK_HOVER) + 1e-9)

        dist = np.zeros((Ntot, Ntot))
        L = np.zeros((Ntot, Ntot))
        
        for i in range(Ntot):
            for j in range(Ntot):
                if i != j:
                    d = np.linalg.norm(nodes[i] - nodes[j])
                    dist[i, j] = d
                    if d > 1e-9:
                        move_vec = (nodes[j] - nodes[i]) / d
                        v_d = self.mod_vd * move_vec
                        v_l = v_d - wind_vec
                        T_drag = 0.5 * self.c_D * self.A_mambo * self.RHO_AIR * (np.linalg.norm(v_l)**2)
                        
                        E_move = (self.T_lift + T_drag) * d / 0.55
                        t_flight = d / max(self.mod_vd, 1e-6)
                        E_hover = P_hover * (t_flight + self.HOVER_TIME_SEC)
                        
                        raw_joules = max(0.0, E_move + E_hover)
                        cost_in_battery_units = raw_joules / P_hover
                        
                        L[i, j] = cost_in_battery_units

        best_reward = -1e30
        best_path = []
        end_node = Ntot - 1

        for t in range(self.n_t):
            rho = self.rho0 + (t / self.n_t) * (self.rhoMax - self.rho0)
            self.tao = (1.0 - rho) * self.tao
            self.tao = np.clip(self.tao, 1e-12, None)

            for a in range(self.n_a):
                visited = [False] * Ntot
                current = 0
                visited[0] = True
                visited[end_node] = False
                tour = [0]
                total_E, total_R = 0.0, 0.0

                while current != end_node:
                    P = np.zeros(Ntot)
                    
                    for v in range(1, end_node):
                        if not visited[v]:
                            cost_to_target = L[current, v]
                            cost_target_to_home = L[v, end_node]
                            
                            if total_E + cost_to_target + cost_target_to_home <= e_remain:
                                eta = 1.0 / (cost_to_target + 1e-6)
                                P[v] = (self.tao[current, v]**self.alpha) * (eta**self.beta) * (max(node_scores[v], 1e-6)**self.gamma)
                    
                    if np.sum(P) == 0: 
                        next_node = end_node
                    else:
                        P = P / np.sum(P)
                        if np.random.rand() <= self.q0:
                            next_node = np.argmax(P)
                        else:
                            next_node = np.random.choice(range(Ntot), p=P)

                    tour.append(next_node)
                    visited[next_node] = True
                    total_E += L[current, next_node]
                    total_R += node_scores[next_node]
                    
                    self.tao[current, next_node] = (1.0 - self.phi) * self.tao[current, next_node] + self.phi * 0.1
                    current = next_node

                if total_R > best_reward:
                    best_reward = total_R
                    best_path = tour

            if len(best_path) > 1:
                for k in range(len(best_path) - 1):
                    i, j = best_path[k], best_path[k+1]
                    self.tao[i, j] += rho * best_reward

        return [nodes[i] for i in best_path]

# ==========================================
# 2. MLP ARCHITECTURE & RL EXPLORER
# ==========================================
class ExplorerMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, activation_name='ReLU'):
        super(ExplorerMLP, self).__init__()
        activations = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'LeakyReLU': nn.LeakyReLU()}
        activation = activations.get(activation_name, nn.ReLU())
        
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class RL_Explorer:
    def __init__(self, map_size=100.0, grid_res=20.0, model_path="explorer_mlp.pth"):
        self.res = grid_res
        self.bounds = map_size / 2.0
        x_coords = np.arange(-self.bounds + self.res/2, self.bounds, self.res)
        y_coords = np.arange(-self.bounds + self.res/2, self.bounds, self.res)
        
        self.sectors = [{'center': np.array([x, y]), 'n_j': 0, 'cube_score': 0.0} for x in x_coords for y in y_coords]
        self.max_visits = 2 
        
        # Load the PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Loading RL Explorer Model from {model_path} onto {self.device}...")
        
        try:
            # We construct the path properly using ROS filesystem standards
            save_dir = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
            full_model_path = os.path.join(save_dir, model_path)
            
            checkpoint = torch.load(full_model_path, map_location=self.device)
            hp = checkpoint['hyperparams']
            self.model = ExplorerMLP(
                hidden_dim=hp['hidden_dim'], 
                num_layers=hp['num_layers'], 
                activation_name=hp['activation']
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval() # Set to evaluation mode
        except FileNotFoundError:
            rospy.logerr(f"{model_path} not found in {save_dir}! Please run train_rl_explorer.py first.")
            raise

    def get_best_sector(self):
        valid_sectors = [s for s in self.sectors if s['n_j'] < self.max_visits and not (s['center'][0] == 0.0 and s['center'][1] == 0.0)]
        if not valid_sectors: return None 
            
        best_score = -float('inf')
        best_center = None
        
        for sec in valid_sectors:
            score = self._calculate_score(sec)
            if score > best_score:
                best_score = score
                best_center = sec['center']
        return best_center

    def _calculate_score(self, sec):
        if sec['center'][0] == 0.0 and sec['center'][1] == 0.0:
            return 0.0
        if sec['n_j'] >= self.max_visits: 
            return -1.0
            
        state = [
            sec['center'][0] / self.bounds, 
            sec['center'][1] / self.bounds, 
            sec['n_j'] / float(self.max_visits), 
            sec['cube_score']
        ]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_value = self.model(state_tensor).item()
            
        return q_value

    def register_found_cube(self, cube_xy, reward):
        for sec in self.sectors:
            if abs(sec['center'][0] - cube_xy[0]) <= self.res/2.0 and abs(sec['center'][1] - cube_xy[1]) <= self.res/2.0:
                sec['cube_score'] += reward
                break

    def mark_visited(self, center_xy):
        for sec in self.sectors:
            if abs(sec['center'][0] - center_xy[0]) <= self.res/2.0 and abs(sec['center'][1] - center_xy[1]) <= self.res/2.0:
                sec['n_j'] += 1
                break

    def get_heatmap_image(self, img_size=500):
        grid_dim = int(100.0 / self.res)
        scores = [self._calculate_score(sec) for sec in self.sectors]
            
        active_scores = [s for s in scores if s >= 0.0]
        max_s = max(active_scores) if active_scores else 1.0
        
        heat_color = np.zeros((grid_dim, grid_dim, 3), dtype=np.uint8)
        
        idx = 0
        for i in range(grid_dim):
            for j in range(grid_dim):
                val = scores[idx]
                
                if val == -1.0: 
                    heat_color[j, i] = [0, 0, 0] 
                elif val == 0.0 and self.sectors[idx]['center'][0] == 0.0 and self.sectors[idx]['center'][1] == 0.0:
                    heat_color[j, i] = [128, 128, 128] # Display Home as gray
                else:
                    norm_val = min(255, int((val / max_s) * 200.0 + 50))
                    pixel = np.array([[[norm_val]]], dtype=np.uint8)
                    color_pixel = cv2.applyColorMap(pixel, cv2.COLORMAP_JET)
                    heat_color[j, i] = color_pixel[0, 0]
                idx += 1
                
        resized_heat = cv2.resize(heat_color, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        resized_heat = cv2.flip(resized_heat, 0)
        
        for i in range(0, img_size, img_size//grid_dim):
            cv2.line(resized_heat, (i, 0), (i, img_size), (0, 0, 0), 1)
            cv2.line(resized_heat, (0, i), (img_size, i), (0, 0, 0), 1)
            
        cv2.putText(resized_heat, "MLP Belief Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        bar_width = 110
        color_bar_bg = np.ones((img_size, bar_width, 3), dtype=np.uint8) * 240 
        
        pad_top = 40
        pad_bottom = 80
        bar_h = img_size - pad_top - pad_bottom
        grad_w = 20
        grad_x = 15
        
        grad_values = np.linspace(250, 50, bar_h, dtype=np.uint8).reshape(-1, 1)
        grad_img = np.repeat(grad_values, grad_w, axis=1)
        grad_colored = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)
        
        color_bar_bg[pad_top:pad_top+bar_h, grad_x:grad_x+grad_w] = grad_colored
        cv2.rectangle(color_bar_bg, (grad_x, pad_top), (grad_x+grad_w, pad_top+bar_h), (0, 0, 0), 1)
        
        cv2.putText(color_bar_bg, "Q-Score", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        
        num_ticks = 6
        for i in range(num_ticks):
            y_offset = pad_top + int(i * (bar_h / (num_ticks - 1)))
            if i == num_ticks - 1: y_offset -= 1 
            
            tick_val = max_s * (1.0 - (i / (num_ticks - 1)))
            cv2.line(color_bar_bg, (grad_x + grad_w, y_offset), (grad_x + grad_w + 5, y_offset), (0, 0, 0), 1)
            cv2.putText(color_bar_bg, f"{tick_val:.2f}", (grad_x + grad_w + 10, y_offset + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.rectangle(color_bar_bg, (10, img_size - 60), (30, img_size - 45), (255, 255, 255), -1)
        cv2.rectangle(color_bar_bg, (10, img_size - 60), (30, img_size - 45), (0, 0, 0), 1)
        cv2.putText(color_bar_bg, "Empty", (38, img_size - 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.rectangle(color_bar_bg, (10, img_size - 35), (30, img_size - 20), (128, 128, 128), -1)
        cv2.rectangle(color_bar_bg, (10, img_size - 35), (30, img_size - 20), (0, 0, 0), 1)
        cv2.putText(color_bar_bg, "Base", (38, img_size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        resized_heat = np.hstack((resized_heat, color_bar_bg))
        return resized_heat


# ==========================================
# 3. MASTER MISSION NODE & DASHBOARD
# ==========================================
class AutonomousMission:
    def __init__(self):
        rospy.init_node('autonomous_mission_master')
        self.bridge = CvBridge()
        
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        self.image_sub = rospy.Subscriber("/iris_downward_depth_camera/camera/rgb/image_raw", Image, self.process_image)
        
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        
        self.current_state = State()
        self.current_pose = PoseStamped()
        
        self.aco = ACO_Router()
        # RL INTEGRATION
        self.rl_explorer = RL_Explorer(grid_res=20.0, model_path="explorer_mlp.pth") 
        
        self.battery = 150.0 
        self.wind_dir = 45.0
        self.wind_speed = 20.0
        self.home_xy = None
        
        self.initial_search_battery = 0.0
        self.allocated_budget = 0.0
        
        self.known_macro_cubes = []   
        self.known_macro_scores = []
        self.known_micro_cubes = []   
        self.known_micro_scores = []
        
        self.unvisited_macro_cubes = []
        self.unvisited_macro_scores = []
        self.unvisited_micro_cubes = []
        self.unvisited_micro_scores = []
        
        self.visited_cubes = [] 
        self.trajectory_x = []
        self.trajectory_y = []
        self.current_aco_route = [] 
        self.micro_discovery_history = [] 
        self.current_fov_polygon = None
        
        self.rl_history = [] 
        self.current_rl_target = None 
        
        self.latest_cv_image = None
        self.needs_reroute = False 
        self.last_visited_sector = None 
        
        self.f_x = 554.25  
        self.f_y = 554.25
        self.img_width = None
        self.img_height = None
        
        self.mission_state = "TAKEOFF"

    def state_cb(self, msg): self.current_state = msg
    def pose_cb(self, msg): 
        self.current_pose = msg
        self.trajectory_x.append(msg.pose.position.x)
        self.trajectory_y.append(msg.pose.position.y)

    def gazebo_to_pixel(self, gz_x, gz_y, map_size_px=500):
        px = int((gz_x + 50.0) * (map_size_px / 100.0))
        py = int((-gz_y + 50.0) * (map_size_px / 100.0)) 
        return px, py

    def check_fov_coverage(self):
        z = max(self.current_pose.pose.position.z, 0.1)

        img_w = self.img_width if self.img_width else 640.0
        img_h = self.img_height if self.img_height else 480.0

        fov_w = z * (img_w / self.f_x)
        fov_h = z * (img_h / self.f_y)

        q = [self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, 
             self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w]
        _, _, yaw = tf_trans.euler_from_quaternion(q)

        yaw += (math.pi / 2.0)

        cx, cy = self.current_pose.pose.position.x, self.current_pose.pose.position.y

        corners = np.array([
            [-fov_w/2, -fov_h/2],
            [ fov_w/2, -fov_h/2],
            [ fov_w/2,  fov_h/2],
            [-fov_w/2,  fov_h/2]
        ])

        R_yaw = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        
        self.current_fov_polygon = np.dot(corners, R_yaw.T) + np.array([cx, cy])

        if self.mission_state == "MICRO_SEARCH":
            contour = np.array(self.current_fov_polygon, dtype=np.float32)
            drone_pos = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
            
            # 2) RL Target: Considered covered if inside FOV, decrease score by 80%
            if self.current_rl_target is not None:
                if cv2.pointPolygonTest(contour, (float(self.current_rl_target[0]), float(self.current_rl_target[1])), False) >= 0:
                    for sec in self.rl_explorer.sectors:
                        if sec['center'][0] == self.current_rl_target[0] and sec['center'][1] == self.current_rl_target[1]:
                            sec['cube_score'] *= 0.2
                            self.needs_reroute = True
                            rospy.loginfo("RL target covered in FOV! Score reduced by 80%.")
                            self.current_rl_target = None
                            break

            def sweep_targets(unvisited_cubes, unvisited_scores, type_str):
                new_cubes, new_scores = [], []
                swept_any = False
                for c, s in zip(unvisited_cubes, unvisited_scores):
                    dist = math.hypot(c[0] - drone_pos[0], c[1] - drone_pos[1])
                    
                    in_fov = cv2.pointPolygonTest(contour, (float(c[0]), float(c[1])), False) >= 0
                    
                    if dist < 1.0 or in_fov:
                        self.visited_cubes.append([c[0], c[1], s, type_str])
                        swept_any = True
                    else:
                        new_cubes.append(c)
                        new_scores.append(s)
                        
                if swept_any:
                    self.needs_reroute = True
                    rospy.loginfo(f"[{type_str.upper()}] Target(s) successfully covered! Triggering ACO Reroute.")
                    
                return new_cubes, new_scores

            self.unvisited_macro_cubes, self.unvisited_macro_scores = sweep_targets(self.unvisited_macro_cubes, self.unvisited_macro_scores, "macro")
            self.unvisited_micro_cubes, self.unvisited_micro_scores = sweep_targets(self.unvisited_micro_cubes, self.unvisited_micro_scores, "micro")

    def update_live_dashboard(self, cv_image):
        map_size = 500
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 240 
        
        for i in range(0, map_size, 100):
            cv2.line(map_img, (i, 0), (i, map_size), (200, 200, 200), 1)
            cv2.line(map_img, (0, i), (map_size, i), (200, 200, 200), 1)

        if self.home_xy is not None:
            hx, hy = self.gazebo_to_pixel(self.home_xy[0], self.home_xy[1], map_size)
            cv2.drawMarker(map_img, (hx, hy), (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(map_img, "Base", (hx+10, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

        w_x = int(math.cos(math.radians(270 - self.wind_dir)) * 30)
        w_y = int(-math.sin(math.radians(270 - self.wind_dir)) * 30)
        cv2.arrowedLine(map_img, (450, 50), (450 + w_x, 50 + w_y), (128, 0, 128), 3, tipLength=0.3)
        cv2.putText(map_img, "WIND", (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1, cv2.LINE_AA)

        for rl_pt in self.rl_history:
            px, py = self.gazebo_to_pixel(rl_pt[0], rl_pt[1], map_size)
            cv2.circle(map_img, (px, py), 4, (255, 0, 0), 1) 

        if self.current_rl_target is not None:
            px, py = self.gazebo_to_pixel(self.current_rl_target[0], self.current_rl_target[1], map_size)
            cv2.circle(map_img, (px, py), 6, (255, 0, 0), -1) 
            cv2.circle(map_img, (px, py), 8, (255, 255, 255), 1) 

        for c, s in zip(self.known_macro_cubes, self.known_macro_scores):
            px, py = self.gazebo_to_pixel(c[0], c[1], map_size)
            radius = max(3, int(s * 8))
            cv2.circle(map_img, (px, py), radius, (0, 100, 0), -1)
            
        for c, s in zip(self.known_micro_cubes, self.known_micro_scores):
            px, py = self.gazebo_to_pixel(c[0], c[1], map_size)
            radius = max(3, int(s * 8))
            cv2.circle(map_img, (px, py), radius, (0, 255, 255), -1)
            cv2.circle(map_img, (px, py), radius, (0, 150, 150), 1) 

        if self.current_fov_polygon is not None:
            pts = [self.gazebo_to_pixel(p[0], p[1], map_size) for p in self.current_fov_polygon]
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(map_img, [pts], True, (0, 165, 255), 2, cv2.LINE_AA)

        if len(self.current_aco_route) > 1:
            aco_pts = [self.gazebo_to_pixel(p[0], p[1], map_size) for p in self.current_aco_route]
            for i in range(1, len(aco_pts)):
                cv2.line(map_img, aco_pts[i-1], aco_pts[i], (255, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(map_img, aco_pts[i], 3, (255, 0, 255), -1)

        for i in range(1, len(self.trajectory_x)):
            p1 = self.gazebo_to_pixel(self.trajectory_x[i-1], self.trajectory_y[i-1], map_size)
            p2 = self.gazebo_to_pixel(self.trajectory_x[i], self.trajectory_y[i], map_size)
            cv2.line(map_img, p1, p2, (255, 0, 0), 2)

        curr_px, curr_py = self.gazebo_to_pixel(self.current_pose.pose.position.x, self.current_pose.pose.position.y, map_size)
        cv2.circle(map_img, (curr_px, curr_py), 6, (0, 0, 255), -1)
        
        cv2.putText(map_img, f"State: {self.mission_state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(map_img, "Dark Green: Known Targets", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)
        cv2.putText(map_img, "Yellow: Discovered Targets", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(map_img, "Magenta: Active ACO Route", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(map_img, "Blue: RL suggestion", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(map_img, "Orange: Camera FOV", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        if cv_image is not None:
            aspect_ratio = cv_image.shape[1] / cv_image.shape[0]
            cam_resized = cv2.resize(cv_image, (int(map_size * aspect_ratio), map_size))
            rl_heat = self.rl_explorer.get_heatmap_image(map_size)
            
            dashboard = np.hstack((cam_resized, map_img, rl_heat))
            cv2.imshow("Live Mission Telemetry", dashboard)
            cv2.waitKey(1)

    def process_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            return

        if self.img_width is None:
            self.img_height, self.img_width = cv_image.shape[:2]

        self.latest_cv_image = cv_image.copy()

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, np.array([50, 100, 140]), np.array([70, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0.0
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                valid_contours.append((contour, area))
                if area > max_area: max_area = area

        if max_area > 0:
            for contour, area in valid_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    score = round(max(0.1, area / max_area), 2)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(cv_image, f"{score}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    Z_uav = self.current_pose.pose.position.z
                    if Z_uav < 0.5: continue
                    
                    X_c = ((cX - (self.img_width / 2.0)) * Z_uav) / self.f_x
                    Y_c = ((cY - (self.img_height / 2.0)) * Z_uav) / self.f_y
                    
                    q = [self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, 
                         self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w]
                    
                    R = tf_trans.quaternion_matrix(q)
                    world_vec = np.dot(R, np.array([-Y_c, X_c, 0.0, 1.0]))
                    
                    gazebo_x = self.current_pose.pose.position.x + world_vec[0]
                    gazebo_y = self.current_pose.pose.position.y + world_vec[1]
                    
                    cube_coord = np.array([gazebo_x, gazebo_y])
                    is_new = True
                    
                    all_known_cubes = self.known_macro_cubes + self.known_micro_cubes
                    for kc in all_known_cubes:
                        if np.linalg.norm(kc - cube_coord) < 4.0: is_new = False; break
                            
                    if is_new:
                        if self.mission_state in ["TAKEOFF", "CLIMBING", "MACRO_VISION"]:
                            self.known_macro_cubes.append(cube_coord)
                            self.known_macro_scores.append(score)
                            self.unvisited_macro_cubes.append(cube_coord)
                            self.unvisited_macro_scores.append(score)
                        elif self.mission_state == "MICRO_SEARCH":
                            self.known_micro_cubes.append(cube_coord)
                            self.known_micro_scores.append(score)
                            
                            self.unvisited_micro_cubes.append(cube_coord)
                            self.unvisited_micro_scores.append(score)
                            
                            self.micro_discovery_history.append([gazebo_x, gazebo_y, score, len(self.trajectory_x)])
                            rospy.loginfo(f"MICRO SEARCH: NEW CUBE FOUND IN FOV AT [{gazebo_x:.2f}, {gazebo_y:.2f}]. Added to targets.")
                            
                            self.rl_explorer.register_found_cube(cube_coord, score)
                            self.needs_reroute = True 

        self.check_fov_coverage()
        self.update_live_dashboard(cv_image)

    def take_high_altitude_snapshot(self):
        rospy.loginfo("Saving data gathered during the 50m climb...")
        save_dir = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
        img_path = os.path.join(save_dir, "high_alt_map.jpg")
        csv_path = os.path.join(save_dir, "target_coordinates.csv")
        
        if self.latest_cv_image is not None:
            cv2.imwrite(img_path, self.latest_cv_image)
            
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Target_ID", "Gazebo_X", "Gazebo_Y", "Score"])
            for idx, (c, s) in enumerate(zip(self.known_macro_cubes, self.known_macro_scores)):
                writer.writerow([idx+1, c[0], c[1], s])
                
        rospy.loginfo(f"MACRO SUCCESS: Saved {len(self.known_macro_cubes)} targets to {csv_path}")

    def fly_to_altitude(self, target_z):
        pose = PoseStamped()
        pose.pose.position.x = self.current_pose.pose.position.x
        pose.pose.position.y = self.current_pose.pose.position.y
        pose.pose.position.z = target_z
        
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.local_pos_pub.publish(pose)
            self.battery -= 0.05
            if abs(self.current_pose.pose.position.z - target_z) < 0.5: return True
            rate.sleep()

    def fly_route(self, target_xy):
        pose = PoseStamped()
        pose.pose.position.x = target_xy[0]
        pose.pose.position.y = target_xy[1]
        pose.pose.position.z = 10.0
        
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            self.local_pos_pub.publish(pose)
            self.battery -= 0.05 
            if self.battery < 5.0: return "BATTERY_LOW" 
            
            if getattr(self, 'needs_reroute', False):
                return "INTERRUPTED"
                
            if self.mission_state == "MICRO_SEARCH":
                curr_x = self.current_pose.pose.position.x
                curr_y = self.current_pose.pose.position.y
                curr_xy = np.array([curr_x, curr_y])
                
                for sec in self.rl_explorer.sectors:
                    if abs(sec['center'][0] - curr_x) <= self.rl_explorer.res/2.0 and abs(sec['center'][1] - curr_y) <= self.rl_explorer.res/2.0:
                        sec_tuple = tuple(sec['center'])
                        if getattr(self, 'last_visited_sector', None) != sec_tuple:
                            self.rl_explorer.mark_visited(curr_xy)
                            self.last_visited_sector = sec_tuple
                        break
                
            dist = math.hypot(target_xy[0] - self.current_pose.pose.position.x, target_xy[1] - self.current_pose.pose.position.y)
            
            in_fov = False
            if self.current_fov_polygon is not None:
                contour = np.array(self.current_fov_polygon, dtype=np.float32)
                
                # Any target in FOV grants early REACHED status
                if cv2.pointPolygonTest(contour, (float(target_xy[0]), float(target_xy[1])), False) >= 0:
                    in_fov = True

            if dist < 1.0 or in_fov: 
                return "REACHED"
            rate.sleep()
        return "FAILED"

    def generate_reports(self):
        rospy.loginfo("Generating Final Mission Reports...")
        save_dir = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
        
        fig_static, ax_static = plt.subplots(figsize=(10, 10))
        ax_static.set_xlim(-50, 50); ax_static.set_ylim(-50, 50)
        ax_static.set_title("Mission Summary: EEPC-ACO Route & Discoveries", fontsize=14, weight='bold')
        
        num_cells = int(100.0 / self.rl_explorer.res) 
        grid_ticks = np.linspace(-50, 50, num_cells + 1)
        ax_static.set_xticks(grid_ticks)
        ax_static.set_yticks(grid_ticks)
        ax_static.grid(True, linestyle='--', alpha=0.6)
        
        ax_static.plot(self.trajectory_x, self.trajectory_y, 'b-', lw=2, alpha=0.8, label='UAV Trajectory')
        
        if self.rl_history:
            rl_arr = np.array(self.rl_history)
            ax_static.scatter(rl_arr[:,0], rl_arr[:,1], c='blue', s=100, edgecolors='white', zorder=4)

        if self.known_macro_cubes:
            mac_arr = np.array(self.known_macro_cubes)
            ax_static.scatter(mac_arr[:,0], mac_arr[:,1], c='green', s=np.array(self.known_macro_scores)*200, edgecolors='black', zorder=5)

        if self.micro_discovery_history:
            mic_arr = np.array(self.micro_discovery_history)
            ax_static.scatter(mic_arr[:,0], mic_arr[:,1], c='yellow', s=mic_arr[:,2]*200, edgecolors='black', zorder=5)

        if self.home_xy is not None:
            ax_static.plot(self.home_xy[0], self.home_xy[1], 'rX', markersize=15, label='Home')

        for vc in self.visited_cubes:
            rect = plt.Rectangle((vc[0]-2.0, vc[1]-2.0), 4.0, 4.0, fill=False, edgecolor='red', linewidth=1.5, zorder=6)
            ax_static.add_patch(rect)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', lw=2, label='Flight Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Known Targets'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Discovered Targets'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='RL suggestions'),
            Line2D([0], [0], color='red', marker='X', linestyle='None', markersize=10, label='Base'),
            Line2D([0], [0], color='red', lw=1.5, marker='s', fillstyle='none', linestyle='None', markersize=10, label='Visited (Bounding Box)')
        ]
        ax_static.legend(handles=custom_lines, loc='upper right')
        
        static_path = os.path.join(save_dir, "mission_summary_plot.png")
        fig_static.savefig(static_path, dpi=300, bbox_inches='tight')
        rospy.loginfo(f"STATIC PLOT SAVED TO: {static_path}")
        plt.close(fig_static)

        rl_heat = self.rl_explorer.get_heatmap_image(1000) 
        rl_path = os.path.join(save_dir, "rl_belief_map.png")
        cv2.imwrite(rl_path, rl_heat)
        rospy.loginfo(f"RL BELIEF MAP SAVED TO: {rl_path}")
        
        cv2.destroyAllWindows()

    def run(self):
        while self.current_pose.pose.position.z == 0.0: rospy.sleep(0.1)
        self.home_xy = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
        rospy.loginfo(f"Home Coordinates Locked: [{self.home_xy[0]:.2f}, {self.home_xy[1]:.2f}]")

        rospy.loginfo("Initiating Mission: Climbing to 50m Macro Altitude...")
        self.mission_state = "CLIMBING"
        pose = PoseStamped(); pose.pose.position.z = 50.0
        rate = rospy.Rate(20)
        
        for _ in range(100): 
            self.local_pos_pub.publish(pose)
            self.battery -= 0.05
            rate.sleep()
            
        self.set_mode_client(0, 'OFFBOARD')
        self.arming_client(True)
        self.fly_to_altitude(50.0)
        
        self.battery -= 2.0 
        rospy.sleep(2.0) 
        
        self.mission_state = "MACRO_VISION"
        self.take_high_altitude_snapshot()
        
        rospy.loginfo("Descending to 10m before starting ACO routing...")
        self.mission_state = "DESCENDING"
        self.fly_to_altitude(10.0)
        
        rospy.loginfo("Initiating EEPC-ACO Routing... Diagonal descent will begin shortly.")
        self.mission_state = "MICRO_SEARCH"
        
        self.initial_search_battery = self.battery
        self.allocated_budget = self.initial_search_battery * 0.20 
        rospy.loginfo(f"Search Battery Locked: {self.initial_search_battery:.1f} | Fixed Budget: {self.allocated_budget:.1f}")
        
        rl_target = None
        self.needs_reroute = True 
        
        while not rospy.is_shutdown() and self.battery > 5.0:
            current_xy = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
            active_targets = list(self.unvisited_macro_cubes + self.unvisited_micro_cubes)
            
            if self.needs_reroute or len(self.current_aco_route) <= 1:
                rl_target = self.rl_explorer.get_best_sector()
                self.current_rl_target = rl_target 
                
                active_scores = list(self.unvisited_macro_scores + self.unvisited_micro_scores)
                
                if rl_target is not None:
                    active_targets.append(rl_target)
                    active_scores.append(1.0) 
                    
                    if not any(np.allclose(rl_target, pt) for pt in self.rl_history):
                        self.rl_history.append(rl_target)
                    
                energy_burned = self.initial_search_battery - self.battery
                aco_budget = self.allocated_budget - energy_burned
                
                if len(active_targets) == 0 or aco_budget <= 0:
                    if len(active_targets) == 0:
                        rospy.loginfo("Map fully explored and cleared! ACO mapping final route to Home.")
                    else:
                        rospy.logwarn("Allocated ACO budget exhausted! ACO mapping final route to Home.")
                    active_targets = []
                    active_scores = []
                    aco_budget = 9999.0 
                
                self.current_aco_route = self.aco.compute_route(
                    current_xy, 
                    np.array(active_targets), 
                    np.array(active_scores), 
                    self.wind_dir, 
                    self.wind_speed, 
                    aco_budget, 
                    self.home_xy
                )
                self.needs_reroute = False 
            
            if len(self.current_aco_route) > 1:
                next_waypoint = self.current_aco_route[1] 
                
                status = self.fly_route(next_waypoint)
                
                if status == "INTERRUPTED":
                    rospy.loginfo("Rerouting ACO due to new target discovery or target physically covered...")
                    continue 
                    
                elif status == "REACHED":
                    self.current_aco_route.pop(0) 

            current_pos = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
            if np.linalg.norm(current_pos - self.home_xy) < 1.0 and len(self.current_aco_route) <= 1:
                rospy.loginfo("Quadrotor reached home proximity. Proceeding to precision alignment.")
                break

        rospy.loginfo("Aligning precisely over home coordinates (0, 0)...")
        align_pose = PoseStamped()
        align_pose.pose.position.x = self.home_xy[0]
        align_pose.pose.position.y = self.home_xy[1]
        align_pose.pose.position.z = self.current_pose.pose.position.z
        
        align_rate = rospy.Rate(20)
        timeout = 0
        while not rospy.is_shutdown() and timeout < 200: 
            self.local_pos_pub.publish(align_pose)
            dist_to_home = math.hypot(self.current_pose.pose.position.x - self.home_xy[0], 
                                      self.current_pose.pose.position.y - self.home_xy[1])
            if dist_to_home < 0.15: 
                rospy.loginfo(f"Precision alignment achieved (Error: {dist_to_home:.2f}m).")
                break
            timeout += 1
            align_rate.sleep()

        rospy.loginfo("Initiating OFFBOARD precision descent to maintain strict X/Y lock...")
        self.mission_state = "LANDING"
        
        landing_rate = rospy.Rate(20)
        target_z = self.current_pose.pose.position.z
        
        while not rospy.is_shutdown() and self.current_pose.pose.position.z > 0.3:
            target_z -= 0.05
            align_pose.pose.position.z = max(-0.2, target_z) 
            self.local_pos_pub.publish(align_pose)
            landing_rate.sleep()
            
        rospy.loginfo("Touchdown proximity achieved. Handing over to AUTO.LAND to cut motors.")
        self.set_mode_client(0, 'AUTO.LAND')
        rospy.sleep(2.0) 
            
        rospy.loginfo("Mission Complete. Generating end-of-mission assets...")
        self.mission_state = "COMPLETED"
        self.generate_reports() 

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    try:
        mission = AutonomousMission()
        mission.run()
    except rospy.ROSInterruptException:
        pass
