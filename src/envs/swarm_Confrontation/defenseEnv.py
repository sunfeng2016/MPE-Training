# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2024-05-17
# @Description: Implementation of Agent Class

import os
import pygame
import imageio
import numpy as np

from baseEnv import BaseEnv
from scipy.spatial import distance

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
# os.environ["DISPLAY"] = ":11"

class DefenseEnv(BaseEnv):
    def __init__(self):
        super(DefenseEnv, self).__init__()
        
        # red base
        self.red_core = {
            'center': np.array([2250.0, 0.0]),
            'radius': 25.0
        }

        self.red_base = {
            'center': np.array([2250.0, 0.0]),
            'radius': 1250.0
        }

        self.red_lines = np.array([
            [[1366.0,  884.0], [1750.0,  500.0]],
            [[1750.0,  500.0], [1750.0, -500.0]],
            [[1750.0, -500.0], [1366.0, -884.0]],
            [[3134.0,  884.0], [2750.0,  500.0]],
            [[2750.0,  500.0], [2750.0, -500.0]],
            [[2750.0, -500.0], [3134.0, -884.0]],
        ])

        self.red_lines_vec = self.red_lines[:, 1, :] - self.red_lines[:, 0, :]
        self.red_lines_len = np.linalg.norm(self.red_lines_vec, axis=1)
        self.red_lines_unitvec = self.red_lines_vec / self.red_lines_len[:, np.newaxis]

        self.red_base_center = self.red_base['center']
        self.red_base_radius = self.red_base['radius']
        self.red_square_size = 1000.0 / 2
        
        # 左侧威胁区
        self.left_sector_pos1 = np.array([1366.0, 884.0])
        self.left_sector_pos2 = np.array([1366.0, -884.0])
        self.left_sector_theta1, self.left_sector_theta2 = calculate_sector_theta(
            self.left_sector_pos1, self.left_sector_pos2, self.red_base_center)
        self.left_threat_x = self.red_base_center[0] - self.red_square_size

        # 右侧威胁区
        self.right_sector_pos1 = np.array([3134.0, -884.0])
        self.right_sector_pos2 = np.array([3134.0, 884.0])
        self.right_sector_theta1, self.right_sector_theta2 = calculate_sector_theta(
            self.right_sector_pos1, self.right_sector_pos2, self.red_base_center)
        self.right_threat_x = self.red_base_center[0] + self.red_square_size

        # blue base
        self.blue_bases = [
            {'center': np.array([1500.0, 1500.0]), 'radius': 500.0},    # 上右
            {'center': np.array([1500.0, -1500.0]), 'radius': 500.0},   # 下右
            {'center': np.array([500.0, 1500.0]), 'radius': 500.0},     # 上左
            {'center': np.array([500.0, -1500.0]), 'radius': 500.0},    # 下左
        ]

        # max in threat zone time
        self.max_in_threat_zone_time = 10

        self.interval = 5

        self.explode_radius = 100

    def reset(self):
        super().reset()

        self.in_threat_zone_times = np.zeros(self.n_blues)

    def step(self, actions):
        # Get red actions
        at = self.acc_actions[actions[:, 0]]
        pt = self.heading_actions[actions[:, 1]]
        attack_t = actions[:, 2]

        # pt *= 0

        # Perform attack actions
        # for i in range(self.n_reds):
        #     if not self.red_alives[i]:
        #         continue

        #     if attack_t[i] == 0:    # no attack
        #         continue
        #     elif attack_t[i] == 1:
        #         continue
        #         self.explode(i)   # explode
        #     elif attack_t[i] == 2:  # collide
        #         flag = self.collide(i)
        #         if not flag:
        #             pt[i] = 0
        #     elif attack_t[i] == 3: # soft kill
        #         self.soft_kill(i)
        #     else:
        #         raise ValueError
            
        # Perform move actions
        self.red_directions += pt * self.max_angular_vel
        self.red_directions = (self.red_directions + np.pi) % (2 * np.pi) - np.pi
        self.red_velocities += at * self.dt_time
        self.red_velocities = np.clip(self.red_velocities, self.red_min_vel, self.red_max_vel)
        self.red_positions += np.column_stack((self.red_velocities * np.cos(self.red_directions),
                                               self.red_velocities * np.sin(self.red_directions))) * self.dt_time
        
        self.blue_step()
        self.merge_state()

        self.check_boundaries()

        # Update step counter
        self._total_steps += 1
        self._episode_steps += 1

    def merge_state(self):
        self.positions = np.vstack([self.red_positions, self.blue_positions])
        self.directions = np.hstack([self.red_directions, self.blue_directions])
        self.velocities = np.hstack([self.red_velocities, self.blue_velocities])
        self.alives = np.hstack([self.red_alives, self.blue_alives])

    def flee_explode_zone(self, target_positions):
        distances_red2blue = distance.cdist(self.red_positions, self.blue_positions)
        valid_mask = self.red_alives[:, np.newaxis] & self.blue_alives[np.newaxis, :]
        distances_red2blue = np.where(valid_mask, distances_red2blue, np.inf)
        distances_blue2red = distances_red2blue.T

        nearest_id = np.argmin(distances_blue2red, axis=1)

        is_in_explode = distances_red2blue[nearest_id, :] < self.explode_radius

        flee_or_not = np.sum(is_in_explode, axis=1) > 2

        flee_directions = self.blue_positions - self.red_positions[nearest_id, :]
        flee_angles = np.arctan2(flee_directions[:, 1], flee_directions[:, 0])      

        dx = np.cos(flee_angles)
        dy = np.sin(flee_angles)

        offsets = np.stack([dx, dy], axis=1) * self.explode_radius

        targets = self.red_positions[nearest_id, :] + offsets

        target_positions[flee_or_not] = targets[flee_or_not]

        return target_positions

    def flee_threat_zone(self, is_in_threat, target_positions):
        # 计算智能体当前位置到线段起点的向量
        pos_vec = self.blue_positions[:, np.newaxis, :] - self.red_lines[:, 0, :][np.newaxis, :, :]
        # 将点向量按照线段长度缩放
        pos_unitvec = pos_vec / self.red_lines_len[:, np.newaxis]
        # 计算投影长度t
        t = np.einsum('nij,nij->ni', self.red_lines_unitvec[np.newaxis, :, :], pos_unitvec)
        # 限制 t 的范围在[0,1]之间
        t = np.clip(t, 0.0, 1.0)
        # 计算线段上距离智能体最近的坐标点
        nearest = self.red_lines[:, 0, :][np.newaxis, :, :] + t[:, :, np.newaxis] * self.red_lines_vec[np.newaxis, :, :]

        # 计算距离
        distance = np.linalg.norm(self.blue_positions[:, np.newaxis, :] - nearest, axis=2)

        nearest_id = np.argmin(distance, axis=1)

        nearest_target = nearest[range(self.n_blues), nearest_id]

        target_positions[is_in_threat] = nearest_target[is_in_threat]

        return target_positions
    
    def around_threat_zone(self, will_in_threat, target_positions):
        target_angles = np.random.uniform(self.right_sector_theta2, self.left_sector_theta1, size=self.n_blues)
        positions_y = self.blue_positions[:, 1]
        target_angles = np.where(positions_y > 0, target_angles, -target_angles)

        # 计算目标位置
        dx = np.cos(target_angles)
        dy = np.sin(target_angles)
        offsets = np.stack([dx, dy], axis=1) * self.red_base_radius
        new_targets = self.red_base_center + offsets

        # 更新威胁区域内智能体的目标位置
        target_positions[will_in_threat] = new_targets[will_in_threat]

        return target_positions
    
    def is_hit_core_zone(self):
        # 判断智能体是否在红方高价值区域内
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_core['center'], axis=1)
        in_red_core = dists_to_center < self.red_core['radius']

        self.blue_alives[in_red_core] = False

    def is_in_threat_zone(self):
        # 1. 判断蓝方智能体是否在红方圆形基地内
        dists_to_center = np.linalg.norm(self.blue_positions - self.red_base_center, axis=1)
        in_red_base = dists_to_center < self.red_base_radius

        # 2. 判断蓝方智能体是否在 x 轴的威胁区域内
        x_positions = self.blue_positions[:, 0]
        x_in_left = x_positions < self.left_threat_x
        x_in_right = x_positions > self.right_threat_x

        # 3. 判断蓝方智能体是否在两个扇形区域内
        vectors_to_center = self.blue_positions - self.red_base_center
        angles = np.arctan2(vectors_to_center[:, 1], vectors_to_center[:, 0])
        angles = np.mod(angles + 2*np.pi, 2*np.pi)

        # 左边扇形区域角度判断
        left_sector_angle_range = np.logical_or(
            (self.left_sector_theta1 <= self.left_sector_theta2) & (angles > self.left_sector_theta1) & (angles < self.left_sector_theta2),
            (self.left_sector_theta1 > self.left_sector_theta2) & ((angles > self.left_sector_theta1) | (angles < self.left_sector_theta2))
        )

        # 右边扇形区域角度判断
        right_sector_angle_range = np.logical_or(
            (self.right_sector_theta1 <= self.right_sector_theta2) & (angles > self.right_sector_theta1) & (angles < self.right_sector_theta2),
            (self.right_sector_theta1 > self.right_sector_theta2) & ((angles > self.right_sector_theta1) | (angles < self.right_sector_theta2))
        )

        # 当前在威胁区域内的：在 left/right 两个扇形区域的角度范围内 且在红方基地的范围内 且x轴坐标在left_threat_x左侧/right_threat_x右侧
        in_threat_zone = (left_sector_angle_range & x_in_left & in_red_base) | (right_sector_angle_range & x_in_right & in_red_base)
        # 将会在威胁区域内的：在 left/right 两个扇形区域的角度范围内 且在红方基地的范围外
        # will_in_threat_zone = (left_sector_angle_range & ~in_red_base) | (right_sector_angle_range & ~in_red_base)
        will_in_threat_zone = ~in_red_base

        self.in_threat_zone_times[in_threat_zone] += 1
        self.in_threat_zone_times[~in_threat_zone] = 0

        self.blue_alives[self.in_threat_zone_times >= self.max_in_threat_zone_time] = False

        return in_threat_zone, will_in_threat_zone

    def blue_step(self):
        # 计算多波次的mask
        mask = np.zeros(self.n_blues, dtype=bool)
        # 第1波次派出0号组
        if self._episode_steps <= self.interval:
            valid_num = self.group_sizes[0]
        # 第2波次派出1和2号组
        elif self.interval < self._episode_steps <= self.interval * 2:
            valid_num = sum(self.group_sizes[:3])
        # 第3波次派出3和4号组
        else:
            valid_num = self.n_blues
        mask[:valid_num] = True
        
        # 初始化每个智能体的目标点坐标
        target_positions = np.tile(self.red_base_center, (self.n_blues, 1))

        # 判断智能体是否在警戒区，是否即将进入警戒区
        is_in_threat, will_in_threat = self.is_in_threat_zone()

        if self._episode_steps > self.interval * 2:
            # 对于即将进入警戒区的智能体，绕飞警戒区
            target_positions = self.around_threat_zone(will_in_threat, target_positions)
            # 对于已经在警戒区的智能体，逃离警戒区
            target_positions = self.flee_threat_zone(is_in_threat, target_positions)

        # 对于在红方自爆范围内的智能体，逃离自爆范围
        target_positions = self.flee_explode_zone(target_positions)

        # 计算期望方向
        desired_directions = np.arctan2(target_positions[:, 1] - self.blue_positions[:, 1],
                                        target_positions[:, 0] - self.blue_positions[:, 0])
        
        # 计算当前方向到期望方向的角度
        angles_diff = desired_directions - self.blue_directions

        # 将角度差规范化到[-pi,pi] 区间内
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi

        # 确保转向角度不超过最大角速度
        angles_diff = np.clip(angles_diff, -self.max_angular_vel, self.max_angular_vel)

        # 更新当前方向
        self.blue_directions[mask] += angles_diff[mask]
        self.blue_directions = (self.blue_directions + np.pi) % (2 * np.pi) - np.pi
        self.blue_positions[mask] += (np.column_stack((self.blue_velocities * np.cos(self.blue_directions),
                                                       self.blue_velocities * np.sin(self.blue_directions))) * self.dt_time)[mask]

        # 判断蓝方智能体是否进入核心区域
        self.is_hit_core_zone()

    def distribute_red_agents(self):
        n_out_bases = int(self.n_reds * np.random.uniform(0.1, 0.2))
        n_in_bases = int(self.n_reds - n_out_bases)

        return n_in_bases, n_out_bases
    
    def generate_red_positions(self):
        n_in_bases, n_out_bases = self.distribute_red_agents()

        # 使用 numpy 生成随机角度和半径
        angles = np.random.uniform(0, 2 * np.pi, n_in_bases)
        radii = self.red_base_radius * np.sqrt(np.random.uniform(0, 1, n_in_bases))

        # 计算智能体的位置
        x = self.red_base_center[0] + radii * np.cos(angles)
        y = self.red_base_center[1] + radii * np.sin(angles)
        in_base_positions = np.vstack([x, y]).T

        out_base_positions = (np.random.rand(n_out_bases, 2) - 0.5) * np.array([self.size_x, self.size_y])

        positions = np.vstack([in_base_positions, out_base_positions])
        directions = (np.random.rand(self.n_reds) - 0.5) * 2 * np.pi

        return positions, directions

    def distribute_blue_agents(self):
        # 随机分成 n_groups 组, 总和为 n_agents
        n_agents = self.n_blues
        n_groups = len(self.blue_bases) + 1
        group_sizes = np.random.multinomial(n_agents, np.ones(n_groups) / n_groups)

        return group_sizes

    def generate_blue_positions(self):
        group_sizes = self.distribute_blue_agents()
        agent_positions = []

        blue_bases = [self.red_base] + self.blue_bases
        self.group_sizes = group_sizes

        # Initialize agent positions in each group
        for group_idx, group_size in enumerate(group_sizes):
            center = blue_bases[group_idx]['center']
            radius = blue_bases[group_idx]['radius']
            
            # 使用 numpy 生成随机角度和半径
            if group_idx == 0:
                angles = np.random.uniform(0, 2 * np.pi, group_size)
                radii = radius + 20
            else:
                angles = np.random.uniform(0, 2 * np.pi, group_size)
                radii = radius * np.sqrt(np.random.uniform(0, 1, group_size))

            # 计算智能体的位置
            x = center[0] + radii * np.cos(angles)
            y = center[1] + radii * np.sin(angles)
            positions = np.vstack([x, y]).T

            agent_positions.append(positions)
        
        agent_positions = np.vstack(agent_positions)
        agent_directions = np.arctan2(self.red_base_center[1] - agent_positions[:, 1], 
                                      self.red_base_center[0] - agent_positions[:, 0])
        agent_directions += np.random.uniform(-np.pi/18, np.pi/18, self.n_blues)
            
        return agent_positions, agent_directions

    def init_positions(self):
        
        red_positions, red_directions = self.generate_red_positions()
        blue_positions, blue_directions = self.generate_blue_positions()

        positions = np.vstack([red_positions, blue_positions])
        directions = np.hstack([red_directions, blue_directions])
        velocities = np.hstack([np.ones(self.n_reds) * self.red_max_vel, np.ones(self.n_blues) * self.blue_max_vel])

        return positions, directions, velocities
    
    def transform_lines(self):
        # 将世界坐标转换为屏幕坐标
        half_size_x = self.size_x / 2
        half_size_y = self.size_y / 2

        self.transformed_lines = np.zeros_like(self.red_lines)
        self.transformed_lines[:, :, 0] = ((self.red_lines[:, :, 0] + half_size_x) / self.size_x * self.screen_width).astype(int)
        self.transformed_lines[:, :, 1] = ((self.red_lines[:, :, 1] + half_size_y) / self.size_y * self.screen_height).astype(int)

    def transform_circles(self):
        self.transformed_circles_center = []
        self.transformed_circles_radius = []

        circles = [self.red_core, self.red_base] + self.blue_bases

        for circle in circles:
            self.transformed_circles_center.append(self.transform_position(circle['center']))
            self.transformed_circles_radius.append(circle['radius'] / self.size_x * self.screen_width)
            
        
    def render(self, frame_num=0, save_frames=False):

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            red_plane_img = pygame.image.load('./png/red_plane_s.png').convert_alpha()
            blue_plane_img = pygame.image.load('./png/blue_plane_s.png').convert_alpha()

            # 缩放飞机贴图
            scale_factor = 0.2  # 调整缩放比例
            self.red_plane_img = pygame.transform.scale(red_plane_img, (int(red_plane_img.get_width() * scale_factor), 
                                                                        int(red_plane_img.get_height() * scale_factor)))
            self.blue_plane_img = pygame.transform.scale(blue_plane_img, (int(blue_plane_img.get_width() * scale_factor), 
                                                                          int(blue_plane_img.get_height() * scale_factor)))

            pygame.display.set_caption("Swarm Confrontation")

            self.transform_lines()
            self.transform_circles()

            self.num_circles = len(self.blue_bases) + 2

            # 初始化字体
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))
        self.transform_positions()
        angles = np.degrees(self.directions)

        # 渲染基地
        for i in range(self.num_circles):
            width = 0 if i == 0 else 2
            color = (255, 0, 0) if i <= 1 else (0, 0, 255)
            pygame.draw.circle(self.screen, color, self.transformed_circles_center[i], self.transformed_circles_radius[i], width=width)

        # 渲染突防通道
        for line in self.transformed_lines:
            pygame.draw.line(self.screen, (255, 0, 0), (line[0,0], line[0,1]), (line[1,0], line[1,1]), 2)
        
        # 渲染飞机
        for i in range(self.n_agents):
            if self.alives[i]:
                image = self.red_plane_img if i < self.n_reds else self.blue_plane_img
                # cache = self.red_img_cache if i < self.n_reds else self.blue_img_cache

                rotated_img = pygame.transform.rotate(image, -angles[i])
                # rotated_img = self.get_rotated_image(image, angles[i], cache, i)
                new_rect = rotated_img.get_rect(center=self.transformed_positions[i])
                self.screen.blit(rotated_img, new_rect)

        # 计算存活的智能体数量
        red_alive = sum(self.alives[:self.n_reds])
        blue_alive = sum(self.alives[self.n_reds:])

        # 渲染存活数量文本
        red_text = self.font.render(f'Red Alive: {red_alive}', True, (255, 0, 0))
        blue_text = self.font.render(f'Blue Alive: {blue_alive}', True, (0, 0, 255))
        self.screen.blit(red_text, (10, 10))
        self.screen.blit(blue_text, (10, 50))

        pygame.display.flip()

        if save_frames:
            frame_path = f"frames/frame_{frame_num:04d}.png"
            pygame.image.save(self.screen, frame_path)
            

def create_gif(frame_folder, output_path,  fps=10):
    images = []
    for file_name in sorted(os.listdir(frame_folder)):
        if file_name.endswith('.png'):
            file_path = os.path.join(frame_folder, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_path, images, fps=fps)

def calculate_sector_theta(pos1, pos2, center):
    theta1 = np.arctan2(pos1[1] - center[1], pos1[0] - center[0])
    theta2 = np.arctan2(pos2[1] - center[1], pos2[0] - center[0])
    
    # Normalize theta to the range[0, 2*pi]
    theta1 = (theta1 + 2 * np.pi) % (2 * np.pi)
    theta2 = (theta2 + 2 * np.pi) % (2 * np.pi)

    return theta1, theta2
            
def angle_within_range(p, start_angle, end_angle, center):
    angles = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    
    # Normalize angles to the range [0, 2*pi]
    angles = np.mod(angles + 2*np.pi, 2*np.pi)

    if start_angle <= end_angle:
        return (angles >= start_angle) & (angles <= end_angle)
    else:
        return (angles >= start_angle) | (angles <= end_angle)


if __name__ == "__main__":

    world = DefenseEnv()

    import time
    world.reset()
    num_frames = 100

    time_list = []
    world.render(frame_num=0, save_frames=True)

    for i in range(1, num_frames):
        print('-'* 30)
        start_time = time.time()
        last_time = time.time()

        # world.update_matrices()
        # print("更新矩阵: {:.5f}".format(time.time() - last_time))
        # last_time = time.time()

        # world.get_obs()
        # print("获取观测: {:.5f}".format(time.time() - last_time))
        # last_time = time.time()

        actions = world.scripted_policy_red()
        print("脚本策略: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.step(actions)
        print("环境更新: {:.5f}".format(time.time() - last_time))
        last_time = time.time()
        
        world.render(frame_num=i, save_frames=True)
        print("环境渲染: {:.5f}".format(time.time() - last_time))
        time_list.append(time.time() - start_time)
    
    time_list = np.array(time_list)

    print(time_list.mean(), time_list.std())

    world.close()

    create_gif("frames", "output.gif", fps=100)


    


        





        



