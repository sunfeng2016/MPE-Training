# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2023-12-11
# @Description: Implementation of the MPE Environment.

import random
import numpy as np

from .plane import Plane
from VISUALIZE.mcom import mcom
from smac.env.multiagentenv import MultiAgentEnv

from absl import logging

class MultiPlaneEnv(MultiAgentEnv):
    def __init__(
            self,
            n_reds=50,
            n_blues=50,
            id_embedding_size=2,
            max_time_step=500,
            time_per_step=6,
            frame_per_step=6,
            battlefield_height=4.0,
            view_rad=400.0,
            view_ang=np.pi/3,
            communicate_rad = 200.0,
            collision_distance=10.0,
            cold_boot_step=5,
            max_observed_allies=5,
            max_observed_enemies=5,
            render=False,
            battlefield_size_x = 1000.0,
            battlefield_size_y = 1000.0,
            move_amount=5.0,
            replay_path="TEMP/render",
            debug=False,
            obs_last_action=False,
            obs_timestep_number=False,
            obs_id_embedding=True,
            obs_instead_of_state=False,
            state_last_action=True,
            state_timestep_number=False,
            state_include_enemy=False,
            state_id_embedding=False,
            reward_sparse=False,
            reward_win=200,
            reward_defeat=0,
            reward_scale=True,
            reward_scale_rate=20,
            reward_death_value=10,
            reward_death_scale=0.5,
        ):
        
        # Set the number of red and blue planes
        self.n_reds = n_reds
        self.n_blues = n_blues

        # Set the number of agent and enemies
        self.n_agents = self.n_reds
        self.n_enemies = self.n_blues

        # Calculate the total number of planes
        self.n_planes = self.n_reds + self.n_blues

        # Initialize epsido and time step counters
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0

        # Set the maximum time steps, time per step, and frames per step
        self.episode_limit = max_time_step
        self.time_per_step = time_per_step
        self.frame_per_step = frame_per_step

        # Set the move amount per step for each plane
        self.move_amount = move_amount

        # Set the height and size of the battlefield
        self.height = battlefield_height
        self.size_x = battlefield_size_x
        self.size_y = battlefield_size_y

        # Set the view range
        self.view_rad = view_rad
        self.view_ang = view_ang
        
        # Set the communicate range
        self.communicate_rad = communicate_rad

        # Set the boundaries of the battlefield
        self.wall_pos = np.array([
            -self.size_x / 2,
            self.size_x / 2,
            -self.size_y / 2,
            self.size_y / 2
        ])
        self.max_pos = np.array([
            self.size_x / 2,
            self.size_y / 2,
        ])

        # Set the initial distance between red and blue planes
        self.init_dis = self.size_x * 0.50

        # Calculate the y range for red and blue planes
        self.red_y_range = self.get_y_range(self.n_reds)
        self.blue_y_range = self.get_y_range(self.n_blues)

        # Set the collision distances
        self.collision_distance = collision_distance

        # Set the params for scripted policy
        self.cold_boot_step = cold_boot_step

        # Set the max observed num
        self.max_observed_allies = max_observed_allies
        self.max_observed_enemies = max_observed_enemies
        
        # Set the observation parameters
        self.obs_last_action = obs_last_action
        self.obs_timestep_number = obs_timestep_number
        self.obs_id_embedding = obs_id_embedding

        # Set the state parameters
        self.obs_instead_of_state = obs_instead_of_state
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        self.state_include_enemy = state_include_enemy
        self.state_id_embedding = state_id_embedding

        # Set the reward parameters
        self.reward_sparse = reward_sparse
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_death_value = reward_death_value
        self.reward_death_scale = reward_death_scale
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )

        # Set the action dimension
        self.n_actions_move = 7
        self.n_actions_attack = self.max_observed_enemies
        self.n_actions = self.n_actions_move + self.n_actions_attack

        # Ser the last action array
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Set the render and replay path
        self.render_or_not = render
        self.replay_path = replay_path

        # Set the parameters to record the game result
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0

        # Get the ID embeddings for the planes
        self.id_embedding_size = id_embedding_size
        self.plane_embbeddings = self.get_position_embeddings()

        # Create the list of planes in the environment
        self.red_planes = [Plane(iden=i, index=i, color='red') for i in range(self.n_reds)]
        self.blue_planes = [Plane(iden=i+self.n_reds, index=i, color='blue') for i in range(self.n_blues)]
        self.planes = self.red_planes + self.blue_planes

        # Set Debug
        self.debug = debug

        # Set the other arrtibutes of planes
        for i, plane in enumerate(self.planes):
            plane.id_embedding = self.plane_embbeddings[i]
            plane.view_rad = self.view_rad
            plane.view_ang = self.view_ang
            plane.communicate_rad = self.communicate_rad
            plane.frame_per_step = self.frame_per_step
            plane.time_per_step = self.time_per_step / self.frame_per_step
            plane.max_observed_allies = self.max_observed_allies
            plane.max_observed_enemies = self.max_observed_enemies
            plane.initial_vel = self.move_amount
    
    # -----------------------------------------------------reset-----------------------------------------------------
    def reset(self):
        """
        Reset the environment.
        """
        # Reset time step
        self._episode_steps = 0

        # Reset num of alive planes
        self.n_red_alive = self.n_reds
        self.n_blue_alive = self.n_blues

        # Reset win_counted and defeat_counted
        self.win_counted = False
        self.defeat_counted = False

        # Reset the distances matrix and angle matrix
        self.distances_matrix_red2blue = None
        self.distances_matrix_red2red = None
        self.angles_matrix_red2blue = None
        self.angles_matrix_blue2red = None

        # Reset obs dims
        self.move_feats_dim = self.get_obs_move_feats_size()
        self.enemy_feats_dim = self.get_obs_enemy_feats_size()
        self.ally_feats_dim = self.get_obs_ally_feats_size()
        self.own_feats_dim = self.get_obs_own_feats_size()
        self.obs_size = np.prod(self.move_feats_dim) + np.prod(self.ally_feats_dim) \
              + np.prod(self.own_feats_dim) + np.prod(self.enemy_feats_dim)

        # Reset state dims
        self.enemy_state_dim = (self.n_enemies, (7 if self.state_id_embedding \
                                                 else 5) if self.state_include_enemy else 0)
        self.ally_state_dim = (self.n_agents, 7 if self.state_id_embedding else 5)

        # Reset the last action array
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Reset the positions of the planes
        self.init_planes()
        
        # Update matrices
        self.update_matrices()

        # return self.get_obs(), self.get_state()

    def init_planes(self):
        """
        Initialize the positions and other properties of the planes.
        """
        # Generate a random angle between -pi and pi
        theta = (2 * np.random.rand() - 1) * np.pi

        # Create rotation matrix
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Get the wall position
        xMin, xMax, yMin, yMax = self.wall_pos
        xMid = (xMin + xMax) / 2
        yMid = (yMin + yMax) / 2

        # Init each plane
        for plane in self.planes:
            plane.reset()

            # Set the initial position and direction for blue planes
            side_multiplier = -1 if plane.red else 1
            y_range = self.red_y_range if plane.red else self.blue_y_range

            # Starting position
            x_start = xMid + side_multiplier * self.init_dis
            y_start = y_range / (self.n_reds if plane.red else self.n_blues) * (plane.index + 0.5) - y_range / 2
            xy = np.array([x_start, y_start])

            # Introduce random noise for initial position
            xy += (np.random.randn(2, ) - 0.5) / 10

            # Make slight adjustments to the initial position when the number of planes is greater than 50
            if (self.n_reds > 50 and plane.red) or (self.n_blues > 50 and not plane.red):
                centering = np.array([xMid + side_multiplier * self.init_dis * 0.9, yMid])
                ratio = 1 if plane.iden % 3 == 2 else 0.5 if plane.iden % 3 == 0 else 0.75
                xy = centering + (xy - centering) * ratio

            # Rotate the initial position
            xy = np.dot(xy, rotate.T)

            # Set the initial position
            plane.state.pos = xy.copy()

            # Set the initial direction
            p_ang = theta + (np.random.rand() - 0.5) * np.pi / 12 if plane.red else \
                (theta + np.pi) + (np.random.rand() - 0.5) * np.pi / 12
            plane.state.dir = plane.rotate_dir(p_ang)

            plane.backup_state()

    # -----------------------------------------------------step-------------------------------------------------------
    def step(self, actions_red):
        """
        Perform a step in the game environment.

        Args:
            actions_red (np.ndarray): List of red actions.

        Returns:
            tuple: A tuple containing the reward, terminated flag, and additional info.
        """

        actions_int = [int(a) for a in actions_red]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Get the actions of blue team by the scripted policy
        actions_blue = self.scripted_policy_blue()

        # Combine red and blue actions
        actions = np.concatenate((actions_red, actions_blue))
         
        # Perform actions for each frame per step
        for i in range(self.frame_per_step):
            self.meta_step(actions)
            if self.render_or_not and i < self.frame_per_step - 1:
                self.render(actions)

        # Update distance and angles matrices
        self.update_matrices()

        # Update collision state of game objects
        self.update_collision()

        # Render
        if self.render_or_not:
            self.render(actions)

        self._total_steps += 1
        self._episode_steps += 1

        terminated = False
        reward = self.reward_battle()
        info = {'battle_won': False}
        game_end_code = self.game_result()

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code ==1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info['battle_won'] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            self.battles_game += 1
            self.timeouts += 1

        if terminated:
            self._episode_count += 1
        
        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        return reward, terminated, info

    # -----------------------------------------------------game result---------------------------------------------------
    def game_result(self):
        """
        Get the result of the game.
        """
        if (self.n_red_alive == 0 and self.n_blue_alive > 0):
            return -1 # lost
        if (self.n_red_alive >= 0 and self.n_blue_alive == 0):
            return 1 # won

        return None

    def meta_step(self, actions):
        """
        Perform a meta step in the game simulation.

        Args:
            actions (list): List of actions for each plane to perform.
        """

        # Plane fly step
        for plane, action in zip(self.planes, actions):
            if not plane.alive:
                continue
            # Move the plane based on the fly action
            plane.fly(action)
    
    # --------------------------------------------------update matrices-----------------------------------------------------
    def get_plane_info(self, planes):
        """
        Get the positions, angles, and alive status of each plane.

        Args:
            planes (list): List of Plane objects.

        Returns:
            tuple: A tuple containing three numpy arrays - positions, angles, and alive status.
        """
        # Create an array of positions by extracting the pos attribute from each plane object
        positions = np.array([plane.state.pos for plane in planes])
        
        # Create an array of angles by extracting the angle attribute from each plane object
        angles = np.array([plane.state.angle for plane in planes])
        
        # Create an array of alive statuses by extracting the alive attribute from each plane object
        alive = np.array([plane.alive for plane in planes])

        return positions, angles, alive

    def calculate_distances(self, positions1, positions2):
        delta = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
        return np.linalg.norm(delta, axis=-1)
    
    def update_matrices(self):
        """
        Updates the distances and angles matrices based on the current plane positions and angles.
        """
        # Get the plane information for the red and blue planes
        red_positions, red_angles, red_alive = self.get_plane_info(self.red_planes)
        blue_positions, blue_angles, blue_alive = self.get_plane_info(self.blue_planes)

        # Calculate the distances and angles between the red and blue planes
        delta = red_positions[:, np.newaxis, :] - blue_positions[np.newaxis, :, :]                    
        distances = np.linalg.norm(delta, axis=2) 
        angles_red2blue = red_angles[:, np.newaxis] - np.arctan2(-delta[:, :, 1], -delta[:, :, 0]) 
        angles_blue2red = blue_angles[np.newaxis, :] - np.arctan2(delta[:, :, 1], delta[:, :, 0])  

        # Create a mask for the alive planes
        valid_mask = red_alive[:, np.newaxis] & blue_alive[np.newaxis, :]

        # Update the distances matrix and angles matrix
        self.distances_matrix_red2blue = np.where(valid_mask, distances, np.inf)
        self.angles_matrix_red2blue = np.where(valid_mask, angles_red2blue, np.inf)
        self.angles_matrix_blue2red = np.where(valid_mask, angles_blue2red, np.inf).T

        # Calculate the distances between the red and red planes
        distances_red2red = self.calculate_distances(red_positions, red_positions)
        self.distances_matrix_red2red = np.where(red_alive[:, np.newaxis], distances_red2red, np.inf)
    
    # ------------------------------------------------update collision-------------------------------------------------
    def update_collision(self):
        """
        Update collision information for collided pairs of planes.
        """

        # Find the indices of collided pairs
        red_indices, blue_indices = np.where(self.distances_matrix_red2blue < self.collision_distance)

        if len(red_indices) == 0:
            return

        # Update collision state for collided pairs
        for red_index, blue_index in zip(red_indices, blue_indices):
            red_plane = self.red_planes[red_index]
            blue_plane = self.blue_planes[blue_index]

            # Set planes as not alive and collided
            red_plane.alive = False
            blue_plane.alive = False
            red_plane.just_died = True
            blue_plane.just_died = True

            if red_plane.target is not None and red_plane.target.index == blue_plane.index:
                red_plane.collided = True

        # Remove duplicates
        red_indices = np.unique(red_indices)
        blue_indices = np.unique(blue_indices)

        # Update number of alive planes
        self.n_red_alive -= len(red_indices)
        self.n_blue_alive -= len(blue_indices)

        # Set distances and angles to infinity for the collided pairs
        self.distances_matrix_red2blue[red_indices, :] = np.inf
        self.distances_matrix_red2blue[:, blue_indices] = np.inf

        self.distances_matrix_red2red[red_indices, :] = np.inf
        self.distances_matrix_red2red[:, red_indices] = np.inf

        self.angles_matrix_red2blue[red_indices, :] = np.inf
        self.angles_matrix_red2blue[:, blue_indices] = np.inf
        
        self.angles_matrix_blue2red[red_indices, :] = np.inf
        self.angles_matrix_blue2red[:, blue_indices] = np.inf
    
    # -------------------------------------------------get observation---------------------------------------------------
    def get_obs_move_feats_size(self):
        """
        Return the size of the vector containing the agents's movement-related features.
        """
        move_feats = self.n_actions_move

        return move_feats
    
    def get_obs_own_feats_size(self):
        """
        Return the size of the vector containing the agents's own features.
        """
        own_feats = 0
        if self.obs_id_embedding:
            own_feats += self.id_embedding_size
        if self.obs_timestep_number:
            own_feats += 1
        
        return own_feats

    def get_obs_ally_feats_size(self):
        """
        Return the size of the vector containing the agents's ally features.
        """
        nf_ally = 6
        if self.obs_id_embedding:
            nf_ally += self.id_embedding_size
        if self.obs_last_action:
            nf_ally += self.n_actions

        return self.max_observed_allies, nf_ally
    
    def get_obs_enemy_feats_size(self):
        """
        Return the size of the vector containing the agents's enemy features.
        """
        nf_enemy = 6
        if self.obs_id_embedding:
            nf_enemy += self.id_embedding_size
        
        return self.max_observed_enemies, nf_enemy
    
    def update_observed_enemies(self):
        """
        Update the observed enemies for each plane.
        """
        for plane, distances, angles in zip(self.red_planes, self.distances_matrix_red2blue, self.angles_matrix_red2blue):
            if plane.alive:
                plane.update_observed_enemies(distances, angles, self.blue_planes)

    def update_observed_allies(self):
        """
        Update the observed allies for each plane.
        """
        for plane, distances in zip(self.red_planes, self.distances_matrix_red2red):
            if plane.alive:
                plane.update_observed_allies(distances, self.red_planes)

    def get_obs_agent(self, agent):
        """
        Return the observation for the given agent.
        """
        move_feats = np.zeros(self.move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(self.enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(self.ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(self.own_feats_dim, dtype=np.float32)

        if agent.alive:

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent)
            move_feats = avail_actions[self.n_actions_attack:]

            # Enemy features
            for idx, enemy in enumerate(agent.observed_enemies):
                if enemy is None:
                    break

                enemy_feats[idx, 0] = 1     # visible
                ind = 1

                if self.obs_id_embedding:
                    enemy_feats[idx, ind:ind+self.id_embedding_size] = enemy.id_embedding  # id_embedding
                    ind += self.id_embedding_size
                
                enemy_feats[idx, ind] = self.distances_matrix_red2blue[agent.index, enemy.index] / self.view_rad # distance
                enemy_feats[idx, ind+1:ind+3] = (enemy.state.pos - agent.state.pos) / self.view_rad  # relative position
                enemy_feats[idx, ind+3:ind+5] = enemy.state.dir - agent.state.dir  # relative direction

            # Ally features
            for idx, ally in enumerate(agent.observed_allies):
                if ally is None:
                    break

                ally_feats[idx, 0:1] = 1  # visible
                ind = 1

                if self.obs_id_embedding:
                    ally_feats[idx, ind:ind+self.id_embedding_size] = ally.id_embedding  # id_embedding
                    ind += self.id_embedding_size

                ally_feats[idx, ind] = self.distances_matrix_red2red[agent.index, ally.index] / self.communicate_rad # distance
                ally_feats[idx, ind+1:ind+3] = (ally.state.pos - agent.state.pos) / self.communicate_rad  # relative position
                ally_feats[idx, ind+3:ind+5] = ally.state.dir - agent.state.dir  # relative direction

                if self.obs_last_action:
                    ally_feats[idx, ind+5:] = self.last_action[ally.index]  # last action

            # Own features
            if self.obs_id_embedding:
                own_feats[0:2] = agent.id_embedding
            if self.obs_timestep_number:
                own_feats[-1] = self._episode_steps / self.episode_limit
    
        agent_obs = np.concatenate(
            (
                move_feats,
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats
            )
        )

        return agent_obs


    def get_obs(self):
        """
        Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        self.update_observed_allies()
        self.update_observed_enemies()

        agents_obs = [self.get_obs_agent(plane) for plane in self.red_planes]

        return agents_obs

    # -------------------------------------------------get avail actions---------------------------------------------------
      
    def rotate_dir(self, direction, theta=0):
        """
        Rotate a direction vector by a given angle in radians.

        Parameters:
            direction (np.array): The input direction vector as a numpy array.
            theta (float): The angle in radians by which the direction vector should be rotated. Default is 0.

        Returns:
            np.array: The rotated direction vector as a numpy array.
        """
        # Calculate the cosine and sine of theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Calculate the new x and y components of the direction vector
        x = direction[0] * cos_theta - direction[1] * sin_theta
        y = direction[0] * sin_theta + direction[1] * cos_theta
        
        # Update the direction vector with the new components
        return np.array([x, y])

    def check_bounds(self, position):
        """
        Check if the given position is within the bounds defined by the wall positions.

        Args:
            position (tuple): The position to check in the format (x, y).

        Returns:
            bool: True if the position is within the bounds, False otherwise.
        """
        x, y = position
        xMin, xMax, yMin, yMax = 1.2 * self.wall_pos
        return (xMin < x < xMax) and (yMin < y < yMax)
    
    def can_move(self, agent, angle):
        """
        Determine if the agent can move in a given angle.

        Args:
            agent (Agent): The agent object.
            angle (float): The angle at which the agent wants to move.

        Returns:
            bool: True if the agent can move in the given angle, False otherwise.
        """
        position = agent.state.pos.copy()
        direction = agent.state.dir.copy()

        # Iterate for each frame in a step
        for _ in range(self.frame_per_step):
            # Rotate the direction by the given angle divided by the number of frames per step
            direction = self.rotate_dir(direction, angle / self.frame_per_step)

            # Update the position by adding the direction multiplied by the agent's velocity and the time per step
            position += direction * self.move_amount * (self.time_per_step / self.frame_per_step)

            # Check if the updated position is within the bounds
            # if not self.check_bounds(position):
            #     import time
            #     return False

        if self.check_bounds(position):
            return True
        else:
            return False
        
        # return True

    def get_avail_agent_actions(self, agent):
        """
        Returns a list indicating the availability of actions for the given agent.
        
        Args:
            agent (Agent): The agent for which to get the available actions.
            
        Returns:
            list: A list of integers indicating the availability of each action.
                Each element in the list is either 0 or 1, where 0 indicates
                that the action is not available and 1 indicates that the
                action is available.
        """
        # TODO
        avail_actions = [1] * self.n_actions
        return avail_actions
        
    
        # Create a list of zeros with length equal to the number of actions
        avail_actions = [0] * self.n_actions

        if agent.alive:
            # Define the angles for move actions
            angles = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, np.pi, -np.pi]

            # Check if each observed enemy is present and set the corresponding
            # action as available if the enemy is present
            for idx, enemy in enumerate(agent.observed_enemies):
                if enemy is None:
                    break
                avail_actions[idx] = 1
            
            # Check if the agent can move in each angle and set the corresponding
            # move action as available if the agent can move in that angle
            for i, angle in enumerate(angles):
                if self.can_move(agent, angle):
                    avail_actions[self.n_actions_attack+i] = 1

        return avail_actions

    def get_avail_actions(self):
        """
        Returns a list of available actions for each plane.

        Returns:
            avail_actions (list): A list of available actions for each plane.
        """
        # Initialize an empty list to store the available actions
        avail_actions = []

        # Iterate over the red_planes list
        for agent in self.red_planes:
            # Get the available actions for the current plane
            avail_action = self.get_avail_agent_actions(agent)
            
            # Append the available actions to the list
            avail_actions.append(avail_action)

        # Return the list of available actions
        return avail_actions
    
    # -------------------------------------------------get state---------------------------------------------------
    def get_state(self):
        """
        Return the global state.
        NOTE: This function should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat
        
        ally_state = np.zeros(self.ally_state_dim)
        enemy_state = np.zeros(self.enemy_state_dim)

        for idx, agent in enumerate(self.red_planes):
            if not agent.alive:
                continue

            ally_state[idx, 0] = 1  # alive
            ind = 1

            if self.state_id_embedding:
                ally_state[idx, ind:ind+self.id_embedding_size] = agent.id_embedding   # id_embedding
                ind += self.id_embedding_size
            
            ally_state[idx, ind  :ind+2] = agent.state.pos / self.max_pos     # normalized position
            ally_state[idx, ind+2:ind+5] = agent.state.dir  # direction

        if self.state_include_enemy:
            for idx, enemy in enumerate(self.blue_planes):
                if not enemy.alive:
                    continue

                enemy_state[idx, 0] = 1  # alive
                ind = 1

                if self.state_id_embedding:
                    enemy_state[idx, ind:ind+self.id_embedding_size] = enemy.id_embedding   # id_embedding
                    ind += self.id_embedding_size

                enemy_state[idx, ind  :ind+2] = enemy.state.pos / self.max_pos     # normalized position
                enemy_state[idx, ind+2:ind+5] = enemy.state.dir  # direction

        state = np.append(ally_state.flatten(), enemy_state.flatten())

        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        state = state.astype(np.float32)

        return state

    # ------------------------------------------------compute reward---------------------------------------------------
    def reward_battle(self):
        """
        Reward function for the battle.
        
        Returns:
            int: The reward value.
        """
        # Initialize the reward
        reward = 0
        
        # Check if the reward is sparse
        if self.reward_sparse:
            return 0
        
        # Calculate the reward for each agent
        for agent in self.red_planes:
            # Skip the agent if it has not died
            if not agent.just_died:
                continue
            
            # Reset the just_died flag
            agent.just_died = False
            
            # Check if the agent collided with the target
            if agent.collided:
                reward += self.reward_death_value
            else:
                reward += self.reward_death_value * self.reward_death_value

        return reward
    
    # ------------------------------------------------ render ---------------------------------------------------
    def render(self, actions):
        """
        Render the scene.

        This function initializes the visual bridge and sets up the 3D gemometry and skybox.
        """
        # Check if the visual bridge has been initialized
        if not hasattr(self, 'visual_bridge'):
            # Initialize the visual bridge with specified parameters
            self.visual_bridge = mcom(
                path=self.replay_path,
                digit=8,
                rapid_flush=False,
                draw_mode='Threejs'
            )
            # Initialize v2d
            self.visual_bridge.v2d_init()
            # Set font style
            self.visual_bridge.set_style(
                'font', 
                fontPath='/examples/fonts/ttf/FZYTK.TTF', 
                fontLineHeight=1500
            )
            # Set skybox style
            # self.visual_bridge.set_style(
            #     'skybox6side',
            #     posx='/wget/sky_textures/right_posx.jpg',
            #     negx='/wget/sky_textures/left_negx.jpg',
            #     posy='/wget/sky_textures/top_posy.jpg',
            #     negy='/wget/sky_textures/bottom_negy.jpg',
            #     posz='/wget/sky_textures/front_posz.jpg',
            #     negz='/wget/sky_textures/back_negz.jpg'
            # )
            self.visual_bridge.set_style(
                'skybox6side',
                posx='/wget/battleground_textures/right.jpg',
                negx='/wget/battleground_textures/left.jpg',
                posy='/wget/battleground_textures/up.jpg',
                negy='/wget/battleground_textures/down.jpg',
                posz='/wget/battleground_textures/front.jpg',
                negz='/wget/battleground_textures/back.jpg'
            )
            # self.visual_bridge.set_style(
            #     'skybox6side',
            #     posx='/wget/white_textures/px.jpg',
            #     negx='/wget/white_textures/nx.jpg',
            #     posy='/wget/white_textures/py.jpg',
            #     negy='/wget/white_textures/ny.jpg',
            #     posz='/wget/white_textures/pz.jpg',
            #     negz='/wget/white_textures/nz.jpg'
            # )
            # self.visual_bridge.set_style(
            #     'skybox6side',
            #     posx='/wget/black_textures/px.jpg',
            #     negx='/wget/black_textures/nx.jpg',
            #     posy='/wget/black_textures/py.jpg',
            #     negy='/wget/black_textures/ny.jpg',
            #     posz='/wget/black_textures/pz.jpg',
            #     negz='/wget/black_textures/nz.jpg'
            # )
            # Initialize plane geometry
            self.visual_bridge.advanced_geometry_rotate_scale_translate(
                'plane',                            # Declare the 3D geometry
                'fbx=/examples/files/plane.fbx',    # Use an fbx model
                -np.pi/2, 0, np.pi/2,               # 3D rotation
                2, 2, 2,                            # 3D scale
                0, 0, 0                             # 3D translation
            )
            # Initialize missile geometry
            self.visual_bridge.advanced_geometry_rotate_scale_translate(
                'missile',                          # Declare the 3D geometry
                'fbx=/examples/files/missile.fbx',  # Use an fbx model
                0, 0, -np.pi/2,                     # 3D rotation
                0.1, 0.1, 0.1,                      # 3D scale
                0, 0, 0                             # 3D translation
            )
            # Initialize tower geometry
            self.visual_bridge.advanced_geometry_rotate_scale_translate(
                'tower', 
                'BoxGeometry(1,1,1)',
                0, 0, 0,  
                0, 0, 5, 
                0, 0,-4
            ) # 长方体

            # Wait for the fonts to load
            for _ in range(50):
                self.visual_bridge.空指令()
                self.visual_bridge.v2d_show()

        # Get the number of alive planes
        n_Red_Planes = self.n_red_alive
        n_Blue_Planes = self.n_blue_alive


        # Determine who is winning
        if n_Red_Planes > n_Blue_Planes:
            who_is_winning = '红方领先'
        elif n_Blue_Planes > n_Red_Planes:
            who_is_winning = '蓝方领先'
        else:
            who_is_winning = '双方持平'

        # Create the label string
        label = '红方剩余飞机: {} 蓝方剩余飞机: {}\n'.format(n_Red_Planes, n_Blue_Planes) + \
                '当前战况：{}\n'.format(who_is_winning) + \
                '当前时间步: {}\n'.format(self._episode_steps + 1) + \
                '当前回合: {}\n'.format(self._episode_count + 1)
                
        self.visual_bridge.v2dx(
            'tower|99999|Gray|10', 
            10, 0, 50, 
            ro_x=0, ro_y=0, ro_z=0,
            label=label, 
            label_color='Aqua', opacity=0)

        
        # Render the walls
        self.visual_bridge.line3d(
            'fat|{}|{}|{}'.format(100001, 'gray', 3),
            x_arr = np.array([self.wall_pos[0], self.wall_pos[0]]),
            y_arr = np.array([self.wall_pos[2], self.wall_pos[3]]),
            z_arr = np.array([self.height, self.height]),
            tension = 0,
            opacity = 1
        )
        self.visual_bridge.line3d(
            'fat|{}|{}|{}'.format(100002, 'Gray', 3),
            x_arr = np.array([self.wall_pos[0], self.wall_pos[1]]),
            y_arr = np.array([self.wall_pos[3], self.wall_pos[3]]),
            z_arr = np.array([self.height, self.height]),
            tension = 0,
            opacity = 1
        )
        self.visual_bridge.line3d(
            'fat|{}|{}|{}'.format(100003, 'Gray', 3),
            x_arr = np.array([self.wall_pos[1], self.wall_pos[1]]),
            y_arr = np.array([self.wall_pos[3], self.wall_pos[2]]),
            z_arr = np.array([self.height, self.height]),
            tension = 0,
            opacity = 1
        )
        self.visual_bridge.line3d(
            'fat|{}|{}|{}'.format(100004, 'Gray', 3),
            x_arr = np.array([self.wall_pos[1], self.wall_pos[0]]),
            y_arr = np.array([self.wall_pos[2], self.wall_pos[2]]),
            z_arr = np.array([self.height, self.height]),
            tension = 0,
            opacity = 1
        )
        
        # Render the planes
        for i, plane in enumerate(self.planes):
            if not plane.alive and not plane.just_died:
                continue

            plane_size = 1.5 if plane.alive else 0.0
            x, y = plane.state.pos
            yaw = np.arctan2(plane.state.dir[1], plane.state.dir[0])
            roll = self.get_roll(plane)

            target = plane.target if plane.alive else None
            target_id = 0 if target is None else target.iden

            self.visual_bridge.v2dx(
                'plane|{}|{}|{}'.format(plane.iden, plane.color, plane_size),
                x, y, self.height,
                ro_x = roll, ro_y = 0, ro_z = yaw,
                ro_order = 'ZYX',
                renderOrder = 0,
                opacity = 1,
                label = str(plane.index) +'-'+ str(actions[i])+'-' + str(target_id),
                label_color = 'white',
                track_n_frame = 0
            )
            if target is not None and target.alive:
                self.visual_bridge.line3d(
                    'simple|{}|{}|{}'.format(-plane.iden, plane.color, 0),
                    x_arr = np.array([plane.state.pos[0], target.state.pos[0]]),
                    y_arr = np.array([plane.state.pos[1], target.state.pos[1]]),
                    z_arr = np.array([self.height, self.height]),
                    tension = 0,
                    opacity = 0.5
                )
            else:
                self.visual_bridge.line3d(
                    'simple|{}|{}|{}'.format(-plane.iden, plane.color, 0),
                    x_arr = np.array([0, 0]),
                    y_arr = np.array([0, 0]),
                    z_arr = np.array([self.height, self.height]),
                    tension = 0,
                    opacity = 0
                )
    
        # Show the current frame
        self.visual_bridge.v2d_show()
    
    def get_roll(self, plane):
        """
        Calculate the roll angle of the plane based on its current and previous direction.

        Args:
            plane (Plane): The plane object.

        Returns:
            float: The roll angle of the plane.
        """

        # Calculate the current and previous angle
        curr_angle = np.arctan2(plane.state.dir[1], plane.state.dir[0])
        prev_angle = np.arctan2(plane.state.prev_dir[1], plane.state.prev_dir[0])

        # Calculate the angular displacement
        dis_ange = curr_angle - prev_angle
        dis_ange = (dis_ange + np.pi) % (2 * np.pi) - np.pi

        # Determine the roll angle based on the angular displacement
        if dis_ange > 0:
            return -np.pi / 3
        elif dis_ange < 0:
            return np.pi / 3
        else:
            return 0
    
    # ----------------------------------------------scripted policy-------------------------------------------------
    def sample_target(self, plane):
        """
        Selects a target enemy from the observed enemies of the plane based on their weights.

        Args:
            plane (Plane): The plane object containing the observed enemies.

        Returns:
            Enemy: The selected target enemy.
        """
        # Get the indices of non-None enemies
        non_none_indices = [i for i, enemy in enumerate(plane.observed_enemies) if enemy is not None]

        # Calculate the number of non-None enemies
        num_non_none = len(non_none_indices)

        if num_non_none == 1:
            return plane.observed_enemies[0], 0

        # Calculate the weights for each enemy
        weights = [num_non_none - i for i in range(num_non_none)] + [0] * (len(plane.observed_enemies) - num_non_none)

        # Calculate the total weight
        total_weight = sum(weights)

        # Normalize the weights
        normalized_weights = [weight / total_weight for weight in weights]

        # Select a target enemy based on the normalized weights
        target_index = np.random.choice(range(len(plane.observed_enemies)), p=normalized_weights)

        # Return the selected target enemy
        return plane.observed_enemies[target_index], target_index
    
    def scripted_policy_red(self):
        actions = np.zeros(self.n_reds, dtype=np.int32)

        actions[:] = self.n_actions_attack

        # for i, plane in enumerate(self.red_planes):
        #     if not plane.alive:
        #         continue
        #     avail_action = self.get_avail_agent_actions(plane)
        #     if sum(avail_action) == 0: # TODO
        #         plane.alive = False
        #         self.n_red_alive -= 1
        #         continue
        #     valid_action = np.where(np.array(avail_action) == 1)[0]
        #     actions[i] = random.choice(valid_action)

        return actions

    def scripted_policy_blue(self):
        actions = np.zeros(self.n_blues, dtype=np.int32)

        actions[:] = self.n_actions_attack

        # for i, plane in enumerate(self.blue_planes):
        #     if not plane.alive: # TODO
        #         continue
        #     avail_action = self.get_avail_agent_actions(plane)
        #     if sum(avail_action) == 0:
        #         plane.alive = False
        #         self.n_blue_alive -= 1
        #         continue

        #     valid_action = np.where(np.array(avail_action) == 1)[0]
        #     actions[i] = random.choice(valid_action)

        return actions

    # ------------------------------------------------others--------------------------------------------------
    def get_y_range(self, num_plans):
        """
        Get the y range based on the number of plans.
        
        Args:
            num_plans (int): The number of plans.
        
        Returns:
            float: The calculated y range.
        """
        # Initialize y range
        y_range = self.size_y * 0.5
        
        # Adjust y range based on number of plans
        if num_plans in range(0, 60):
            y_range *= 0.5
        if num_plans in range(60, 100):
            y_range *= 0.8
        
        return y_range

    def get_position_embeddings(self, dtype=np.float32):
        """
        Generate position embeddings for each plane in the environment.

        Args:
            n_bits (int): Number of bits for encoding each position.
            dtype (np.dtype): Data type of the position embeddings array.

        Returns:
            position_embeddings (np.ndarray): Array of position embeddings.
        """

        n_bits = self.id_embedding_size
        
        # Generate an array of shape (n_reds + n_blues, 1) representing the plane ids
        ids = np.arange(self.n_planes)[:, np.newaxis]

        # Calculate the denominator part of the sine and cosine functions
        div_term = np.exp(np.arange(0, n_bits, 2) * (-np.log(10000.0) / n_bits))

        # Initialize an array of shape (n_reds + n_blues, n_bits) to store the position embeddings
        position_embeddings = np.zeros((self.n_planes, n_bits), dtype=dtype)

        # Assign the results of the sine function to the even indexed positions and the cosine function to the odd indexed positions in the array
        position_embeddings[:, 0::2] = np.sin(ids * div_term)
        position_embeddings[:, 1::2] = np.cos(ids * div_term)

        return position_embeddings