# -*- coding: utf-8 -*-
# @Author: Feng Sun
# @Date: 2023-09-21
# @Description: Implementation of Plane Class

import numpy as np

class PlaneState(object):
    def __init__(self):
        """
        Initializes the PlaneState object.
        """
        self.pos = None         # Current Position (x, y)
        self.dir = None         # Current Direction (x, y)
        self.prev_pos = None    # Previous Position (x, y)
        self.prev_dir = None    # Previous Direction (x, y)
        self.p_vel = None       # Physical Velocity (scalar)

    @property
    def angle(self):
        return np.arctan2(self.dir[1], self.dir[0])

class Plane(object):
    def __init__(self, iden=0, index=0, color='red') -> None:
        """
        Initialize the Plane object.
        
        Args:
            iden (int, optional): The ID of the plane. Defaults to 0.
            index (int, optional): The index of the plane in the team list. Defaults to 0.
            id_embedding (None, optional): The ID embedding of the plane. Defaults to 0.
        """
        # Plane properties
        self.iden = iden
        self.index = index
        self.id_embedding = None
        self.color = color
        self.red = True if color == 'red' else False

        # Plane characteristics
        self.initial_vel = 5.0
        self.track_accleration = 1.0

        # Plane state
        self.state = PlaneState()               # The state of the plane
        
        # Plane status
        self.collided = False
        self.alive = True
        self.just_died = False

        # Time and frame settings
        self.time_per_step = 1
        self.frame_per_step = 6
        
        # Field of view settings
        self.view_rad = 200.0
        self.view_ang = np.pi / 3
        
        # Range of communication
        self.communicate_rad = 200.0

        # Enemy observation settings
        self.max_observed_enemies = 5
        self.max_observed_allies = 5
        self.observed_enemies = []
        self.observed_allies = []

        # Bounds
        self.bounds = None

        # Tracking settings
        self.target = None

        # Init Actions num
        self.n_actions_move = 7
        self.n_actions_attack = 5
    
    def reset(self):
        """
        Reset the state and attributes of the plane.
        """

        # Reset plane attributes
        self.alive = True
        self.just_died = False
        self.collided = False
        self.target = None
        
        # Reset plane state, including position, direction and velocity
        self.state.pos = np.array([0.0, 0.0])
        self.state.dir = np.array([1.0, 0.0])
        self.state.p_vel = self.initial_vel

        # Backup plane position and direction
        self.backup_state()

        # Clear the observed enemies list and fill it with None values
        self.observed_enemies = [None] * self.max_observed_enemies
        self.observed_enemies_num = 0

        # Clear the observed allies list and fill it with None values
        self.observed_allies = [None] * self.max_observed_allies

        # Rotation angle of different fly actions
        self.rotation_mapping = {
            i: 0 for i in range(self.n_actions_attack)
        }

        self.rotation_mapping.update({
            self.n_actions_attack: 0,                                           # forward
            self.n_actions_attack + 1: np.pi / (self.frame_per_step * 4),       # turn left pi/4
            self.n_actions_attack + 2: -np.pi / (self.frame_per_step * 4),      # turn right pi/4
            self.n_actions_attack + 3: np.pi / (self.frame_per_step * 2),       # turn left pi/2
            self.n_actions_attack + 4: -np.pi / (self.frame_per_step * 2),      # turn right pi/2
            self.n_actions_attack + 5: np.pi / self.frame_per_step,             # turn left pi
            self.n_actions_attack + 6: -np.pi / self.frame_per_step             # turn right pi
        })
    
        # get the bounds of the buffer field
        self.xMin, self.xMax, self.yMin, self.yMax = self.bounds * 0.9

        # set the direction of bounds
        self.dir_px = np.array([-1, 0])
        self.dir_nx = np.array([1, 0])
        self.dir_py = np.array([0, -1])
        self.dir_ny = np.array([0, 1])

        self.dir_list = (self.dir_nx, self.dir_px, self.dir_ny, self.dir_py)


    def get_avail_actions(self, angle):
        actions = [0] * self.n_actions_move

        if -np.pi <= angle < -np.pi/2:
            actions[4] = 1      # right90
            actions[6] = 1      # right180
        elif -np.pi/2 <= angle < -np.pi/4:
            actions[2] = 1      # right45
            actions[4] = 1      # right90
        elif -np.pi/4 <= angle < 0:
            actions[0] = 1      # forward
            actions[2] = 1      # right45
        elif 0 <= angle < np.pi/4:
            actions[0] = 1      # forward
            actions[1] = 1      # left45
        elif np.pi/4 <= angle < np.pi/2:
            actions[1] = 1      # left45
            actions[3] = 1      # left90
        elif np.pi/2 <= angle <= np.pi:
            actions[3] = 1      # left90
            actions[5] = 1      # left180
        else:
            ValueError(f"Invalid angle: {angle}")
    
        return actions

    def get_avail_move_actions(self):
        x, y = self.state.pos

        in_buffer_cond = (
            x <= self.xMin,     # bound -x
            x >= self.xMax,     # bound +x
            y <= self.yMin,     # bound -y
            y >= self.yMax      # bound +y
        )

        for i in range(4):
            if in_buffer_cond[i]:
                angle = self.angle_between_direction(self.state.dir, self.dir_list[i])
                return self.get_avail_actions(angle)
        
        return [1] * self.n_actions_move

    def fly(self, fly_action):
        # Backup the current state of the plane
        self.backup_state()
        self.target = None
        self.state.p_vel = self.initial_vel
        
        # Check if the fly action is valid
        if fly_action not in self.rotation_mapping:
            raise ValueError(f"Invalid fly action: {fly_action}")
        
        # Get the rotation angle
        rotation_angle = self.rotation_mapping[fly_action]

        # Check if the fly action need track
        if fly_action < self.n_actions_attack:
            target = self.observed_enemies[fly_action]
            assert target is not None
            self.target = target
            self.state.p_vel = self.track_accleration * self.initial_vel
            if target.alive:
                direction = target.state.pos - self.state.pos
                self.state.dir = direction / np.linalg.norm(direction)

        # Update the direction of the plane
        self.state.dir = self.rotate_dir(rotation_angle)
        
        # Update the position of the plane
        self.state.pos += self.state.dir * self.state.p_vel * self.time_per_step

    def update_observed_allies(self, distances, allies):
        """
        Updates the list of observed allies based on their distances from the agent.

        Args:
            distances (np.ndarray): Array of distances between the agent and allies.
            allies (List[Any]): List of ally objects.

        """
        # Check if the allies are within the field of view
        in_radius = distances < self.communicate_rad

        # Get the indices of allies that are within the field of view
        valid_indices = np.where(in_radius)[0]

        if len(valid_indices) == 0:
            # If no allies are within the field of view, set observed allies to None
            self.observed_allies = [None] * self.max_observed_allies
            return

        # Sort the indices based on the distances
        sorted_indices = valid_indices[np.argsort(distances[valid_indices])]

        # Slice the sorted indices to get the observed allies
        observed_allies = [allies[i] for i in sorted_indices]

        # If the number of observed allies is less than the maximum number allowed,
        # add None values to the list to fill it up.
        observed_allies += [None] * max(0, self.max_observed_allies - len(observed_allies))

        # Set the observed allies to the updated list
        self.observed_allies = observed_allies[:self.max_observed_allies]

    def update_observed_enemies(self, distances, angles, enemies):
        """
        Update the list of observed enemies based on the given distances, angles, and enemy positions.

        Args:
            distances (numpy.ndarray): Array of distances to the enemies.
            angles (numpy.ndarray): Array of angles to the enemies.
            enemies (list): List of enemy positions.

        Returns:
            None
        """
        # Check if the enemies are within the field of view and view radius
        in_fov = np.abs(angles) < self.view_ang / 2
        in_radius = distances < self.view_rad

        # Get the indices of enemies that are both within the field of view and view radius
        valid_indices = np.where(in_fov & in_radius)[0]

        if len(valid_indices) == 0:
            # If no enemies are within the field of view and view radius, set observed_enemies to None
            self.observed_enemies = [None] * self.max_observed_enemies
            self.observed_enemies_num = 0
            return

        # Sort the indices based on the distances
        sorted_indices = valid_indices[np.argsort(distances[valid_indices])]
        
        # Slice the sorted indices to get the observed enemies
        observed_enemies = [enemies[i] for i in sorted_indices]

        # Update the observed_enemies_num
        self.observed_enemies_num = min(self.max_observed_enemies, len(observed_enemies))

        # If the number of observed enemies is less than the maximum number allowed,
        # add None values to the list to fill it up.
        observed_enemies += [None] * max(0, self.max_observed_enemies - len(observed_enemies))

        # Update the observed_enemies list
        self.observed_enemies = observed_enemies[:self.max_observed_enemies]

    def nearest_observed_enemy(self, distances, angles, enemies):
        # Check if the enemies are within the field of view and view radius
        in_fov = np.abs(angles) < self.view_ang / 2
        in_radius = distances < self.view_rad

        # Get the indices of enemies that are both within the field of view and view radius
        valid_indices = np.where(in_fov & in_radius)[0]

        if not valid_indices.size:
            # If no enemies are within the field of view and view radius, set observed_enemies to None
            return None
        
        # Sort the indices based on the distances
        nearest_index = valid_indices[np.argmin(distances[valid_indices])]

        # return the nearest observed enemy
        return enemies[nearest_index]

    def backup_state(self):
        """
        Backup the current state of the plane.
        """
        self.state.prev_pos = self.state.pos.copy()
        self.state.prev_dir = self.state.dir.copy()


    def rotate_dir(self, theta=0):
        """
        Rotates the direction vector of the state by the given angle.

        Args:
            theta (float): The angle in radians by which to rotate the direction vector.

        Returns:
            numpy.array: The new direction vector after rotation.
        """
        # Calculate the cosine and sine of theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Calculate the new x and y components of the direction vector
        x = self.state.dir[0] * cos_theta - self.state.dir[1] * sin_theta
        y = self.state.dir[0] * sin_theta + self.state.dir[1] * cos_theta
        
        # Update the direction vector with the new components
        return np.array([x, y])

    def angle_between_direction(self, dir1, dir2):
        """
        Calculate the angle between two directions.

        Args:
            dir1 (np.ndarray): The first direction vector.
            dir2 (np.ndarray): The second direction vector.

        Returns:
            float: The angle between the two directions in radians.
        """
        # Calculate the dot product of the two direction vectors
        dot_product = np.dot(dir1, dir2)

        # Calculate the angle in radians using the arccosine function
        angle_rad = np.arccos(dot_product)

        # Calculate the cross product of the two direction vectors
        cross_product = np.cross(dir1, dir2)

        # Check if the cross product is negative
        if cross_product < 0:
            # Reverse the sign of the angle
            angle_rad = -angle_rad
        
        return angle_rad
    