import numpy as np
from matplotlib import pyplot as plt
from utils import getpolicy, getvalue

from world import World

class RocketWorld(World):
    """
    RocketWorld is a subclass of World representing a specific environment for reinforcement learning.
        - The environment is a continuous 2D space where a rocket must learn to hover in the center.
        - The state space consists of the rocket's position and velocity, while the action space includes on/off
          commands for left, right, and center thruster.
        - The state space is a discretized 5D space combining position and velocity dimensions, as well as angle.
        - The state is more granular near the center to facilitate learning.
        - The action space consists of 8 discrete actions representing all combinations of thruster commands.
        - A positive rewards of 1 is given each time step the rocket hovers within a small radius of the center,
          while a negative reward of -0.1 is given for moving outside the defined boundaries. Touching the boundaries
          gives a large negative reward of -10 and ends the episode.

    The following methods are inherited from the World base class and must be implemented:
        - getWorldSize: Returns the size of the state space as a tuple
        - getDimensionNames: Returns a list of names the state dimensions
        - getActions: Returns a list of action names
        - init: Initialize the state for a new epoch
        - getState: Retruns the current state of the World, and if this state is terminal
        - doAction: Performes the specified action and updates the state
        - draw: Updates any visual information
    """

    def __init__(self):
        # Define state space: position (x, y) and velocity (vx, vy)
        self._pos_bin_edges = [-1.0, -0.4, -0.1, 0.1, 0.4, 1.0]
        self._vel_bin_edges = [-2.0, -0.3, -0.05, 0.05, 0.3, 2.0]
        # Make angle bins more granular when pointing upwards, as this is more critical for balancing
        self._angle_bin_edges = [-180, -45, -7, 0, 7, 45, 180]
        # Angle velocity bins, more granular around 0
        self._angle_vel_bin_edges = [-450, -360, -60, 0, 60, 360, 450]

        # Define action space: combinations of left, right, and center thruster on/off
        self._actions = ["N", "L", "R", "C"]

        # State variable for discrete state
        self._state = None
        self._isTerminal = False
        self._lastAction = None
        self._lastReward = None

        # Internal variables for continuous rocket state
        self.pos = None
        self.vel = None
        self.angle = None
        self.angleVel = None

    def discretizeState(self, pos, vel, angle, angleVel):
        pos_x_idx     = np.digitize(pos[0]  , self._pos_bin_edges      ) - 1
        pos_y_idx     = np.digitize(pos[1]  , self._pos_bin_edges      ) - 1
        vel_x_idx     = np.digitize(vel[0]  , self._vel_bin_edges      ) - 1
        vel_y_idx     = np.digitize(vel[1]  , self._vel_bin_edges      ) - 1
        angle_idx     = np.digitize(angle   , self._angle_bin_edges    ) - 1
        angle_vel_idx = np.digitize(angleVel, self._angle_vel_bin_edges) - 1
        state = (pos_x_idx, pos_y_idx, vel_x_idx, vel_y_idx, angle_idx, angle_vel_idx)
        state = tuple(np.clip(state, 0, np.array(self.getWorldSize()) - 1))
        return state
        
    def getWorldSize(self):
        return (len(self._pos_bin_edges)-1, len(self._pos_bin_edges)-1, len(self._vel_bin_edges)-1, len(self._vel_bin_edges)-1, len(self._angle_bin_edges)-1, len(self._angle_vel_bin_edges)-1)
    
    def getDimensionNames(self):
        return ["xPosition", "yPosition", "xVelocity", "yVelocity", "angle", "angleVelocity"]
    
    def getActions(self):
        return self._actions
    
    def getState(self):
        return self._state, self._isTerminal

    def init(self):
        # Randomly choose between a "stable" and "unstable" start configuration
        if np.random.rand() < 0.9:
            # Stable start: near center, low velocity and angle
            self.pos = np.random.uniform(-0.3, 0.3, size=2)
            self.vel = np.random.uniform(-0.1, 0.1, size=2)
            self.angle = np.random.uniform(-15, 15)
            self.angleVel = np.random.uniform(-60, 60)
        else:
            # Unstable start: further from center, higher velocity and angle
            self.pos = np.random.uniform(-1, 1, size=2)
            self.vel = np.random.uniform(-1, 1, size=2)
            self.angle = np.random.uniform(-180, 180)
            self.angleVel = np.random.uniform(-360, 360)
        self._state = self.discretizeState(self.pos, self.vel, self.angle, self.angleVel)

    def doAction(self, act):
        # Define thruster effects, where the center only provides forward thrust,
        # While left and right thrusters provide less forward thrust and some rotational force.
        # Only relative Scaling is not imortant here, we do fine tuning later.
        linear_thrust_map = {
            "N": 0.0,
            "L": 1.2,
            "R": 1.2,
            "C": 2.5
        }
        angular_thrust_map = {
            "N": 0.0,
            "L": 1.0,
            "R": -1.0,
            "C": 0.0
        }

        # Store last action for potential use in rendering
        self._lastAction = act
        
        # Define time step
        dt = 0.015

        # Magic scalars
        forward_scale = 9.0
        rotational_scale = 50*180.0
        
        # Define gravity
        gravity = 9.0

        # Define air resistance factor, i.e. friction
        friction = 0.02
        angle_friction = 10.00

        # Get thrust based on action and magic constant scaling factors which were determined through trial and error
        forward_thrust    = linear_thrust_map[act] * forward_scale
        rotational_thrust = angular_thrust_map[act] * rotational_scale

        # Update angle using Verlet integration based on rotational thrust from left/right thrusters and friction
        torque_vector = rotational_thrust - angle_friction * self.angleVel
        self.angle += self.angleVel * dt + 0.5 * torque_vector * dt * dt
        self.angle = (self.angle + 180) % 360 - 180  # Keep angle within [-180, 180]
        self.angleVel += torque_vector * dt
        
        # Calculate rocket's direction vector based on current angle
        rad_angle = np.deg2rad(self.angle)
        
        # Update velocity using Verlet integration based on thrust, gravity, and friction
        thrust_vector = np.array([np.sin(rad_angle), np.cos(rad_angle)]) * forward_thrust
        force_vector = thrust_vector - gravity * np.array([0,1]) - friction * self.vel
        self.pos += self.vel * dt + 0.5 * force_vector * dt * dt
        self.vel += force_vector * dt
        
        # Update discrete state
        self._state = self.discretizeState(self.pos, self.vel, self.angle, self.angleVel)

        # Check for terminal conditions, i.e. if the rocket goes out of bounds
        # Else check if it is within the target hover zone, i.e. the center rectangle of size 0.2x0.2
        # Else give no reward for being outside the target zones but within bounds
        reward = 0.0
        self._isTerminal = False
        if np.any(np.abs(self.pos) > 1):
            self._isTerminal = True
            reward = -0.1
        elif np.all(np.abs(self.pos) < 0.1):
            reward = 1.0

        # Store last reward for potential use in rendering
        self._lastReward = reward
        
        # Return the reward, as well as constant True since all actions are valid in this world
        return True, reward
        
    def draw(self, epoch=None, Q=None, sleepTime=0.01, params={}):
        self._drawPre()

        # Draw boundaries
        boundary_square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='k', linestyle='--')
        plt.gca().add_artist(boundary_square)

        if params.get("DrawState", True):
            # Draw position bins as a faint grid
            for x in self._pos_bin_edges[1:-1]:
                plt.plot([x, x], [-1, 1], color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            for y in self._pos_bin_edges[1:-1]:
                plt.plot([-1, 1], [y, y], color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Color the boundary of the current bin in which the rocket is located
            pos_x_idx, pos_y_idx, _, _, _, _ = self._state
            bin_x_min = self._pos_bin_edges[pos_x_idx  ] if pos_x_idx > 0 else -1
            bin_x_max = self._pos_bin_edges[pos_x_idx+1] if pos_x_idx < len(self._pos_bin_edges)-2 else 1

            bin_y_min = self._pos_bin_edges[pos_y_idx  ] if pos_y_idx > 0 else -1
            bin_y_max = self._pos_bin_edges[pos_y_idx+1] if pos_y_idx < len(self._pos_bin_edges)-2 else 1

            plt.plot([bin_x_min, bin_x_max], [bin_y_min, bin_y_min], color='gray', linewidth=2)
            plt.plot([bin_x_min, bin_x_max], [bin_y_max, bin_y_max], color='gray', linewidth=2)
            plt.plot([bin_x_min, bin_x_min], [bin_y_min, bin_y_max], color='gray', linewidth=2)
            plt.plot([bin_x_max, bin_x_max], [bin_y_min, bin_y_max], color='gray', linewidth=2)

            # Add the angle bins as a separate circle on the right side of the plot,
            # with radial lines indicating the angle bin edges.
            center = (1.6, 0.6)
            radius = 0.3
            circle = plt.Circle(center, radius, fill=False, color='k', linestyle='-')
            plt.gca().add_artist(circle)
            plt.text(center[0], center[1] + radius + 0.05, "Angle", horizontalalignment='center')
            for angle in self._angle_bin_edges[:-1]:
                rad = np.deg2rad(angle)
                plt.plot([center[0], center[0] + np.sin(rad)*radius], [center[1], center[1] + np.cos(rad)*radius], color='gray', linestyle='-.', linewidth=0.5, alpha=0.5)

            # Highlight the current angle bin edges
            angle_idx = self._state[4]
            angle_min = self._angle_bin_edges[angle_idx  ] if angle_idx > 0 else -180
            angle_max = self._angle_bin_edges[angle_idx+1] if angle_idx < len(self._angle_bin_edges)-2 else 180
            rad_min = np.deg2rad(angle_min)
            rad_max = np.deg2rad(angle_max)
            plt.plot([center[0], center[0] + np.sin(rad_min)*radius], [center[1], center[1] + np.cos(rad_min)*radius], color='gray', linewidth=2)
            plt.plot([center[0], center[0] + np.sin(rad_max)*radius], [center[1], center[1] + np.cos(rad_max)*radius], color='gray', linewidth=2)

            # Add a final radial line indicating the rocket's current angle
            rad_angle = np.deg2rad(self.angle)
            plt.plot([center[0], center[0] + np.sin(rad_angle)*radius], [center[1], center[1] + np.cos(rad_angle)*radius], color='blue', linewidth=2)

            # Draw a similar circlular slider for angle velocity below the angle circle
            # Do not draw the full circle, just a 75% arc to indicate range (like a speedometer)
            center = (1.6, -0.7)
            radius = 0.3
            arc_angles = np.linspace(-135, 135, 100)
            arc_x = center[0] + np.sin(np.deg2rad(arc_angles)) * radius
            arc_y = center[1] + np.cos(np.deg2rad(arc_angles)) * radius
            plt.plot(arc_x, arc_y, color='k', linestyle='-')
            plt.text(center[0], center[1] + radius + 0.05, "Angular Velocity", horizontalalignment='center')
            for angle_vel_edge in self._angle_vel_bin_edges[1:-1]:
                # Scale min-max to -135 to 135 for display purposes
                angle_vel_edge_scaled = (angle_vel_edge - self._angle_vel_bin_edges[0]) / (self._angle_vel_bin_edges[-1] - self._angle_vel_bin_edges[0]) * 270 - 135
                rad = np.deg2rad(angle_vel_edge_scaled)
                plt.plot([center[0], center[0] + np.sin(rad)*radius], [center[1], center[1] + np.cos(rad)*radius], color='gray', linestyle='-.', linewidth=0.5, alpha=0.5)

            # Highlight the current angle velocity bin edges
            angle_vel_idx = self._state[5]
            angle_vel_min = self._angle_vel_bin_edges[angle_vel_idx  ] if angle_vel_idx > 0 else self._angle_vel_bin_edges[0]
            angle_vel_max = self._angle_vel_bin_edges[angle_vel_idx+1] if angle_vel_idx < len(self._angle_vel_bin_edges)-2 else self._angle_vel_bin_edges[-1]
            angle_vel_min_scaled = (angle_vel_min - self._angle_vel_bin_edges[0]) / (self._angle_vel_bin_edges[-1] - self._angle_vel_bin_edges[0]) * 270 - 135
            angle_vel_max_scaled = (angle_vel_max - self._angle_vel_bin_edges[0]) / (self._angle_vel_bin_edges[-1] - self._angle_vel_bin_edges[0]) * 270 - 135
            rad_min = np.deg2rad(angle_vel_min_scaled)
            rad_max = np.deg2rad(angle_vel_max_scaled)

            # Do not plot the min/max lines if this is the extreme bins
            if angle_vel_idx > 0:
                plt.plot([center[0], center[0] + np.sin(rad_min)*radius], [center[1], center[1] + np.cos(rad_min)*radius], color='gray', linewidth=2)
            if angle_vel_idx < len(self._angle_vel_bin_edges)-2:
                plt.plot([center[0], center[0] + np.sin(rad_max)*radius], [center[1], center[1] + np.cos(rad_max)*radius], color='gray', linewidth=2)
            
            # Draw a final radial line indicating the rocket's current angle velocity, clipped to the min/max range
            angle_vel_scaled = (self.angleVel - self._angle_vel_bin_edges[0]) / (self._angle_vel_bin_edges[-1] - self._angle_vel_bin_edges[0]) * 270 - 135
            angle_vel_scaled = np.clip(angle_vel_scaled, -135, 135)
            rad_angle_vel = np.deg2rad(angle_vel_scaled)
            plt.plot([center[0], center[0] + np.sin(rad_angle_vel)*radius], [center[1], center[1] + np.cos(rad_angle_vel)*radius], color='blue', linewidth=2)
            
            # Draw the velocity and velocity range/bins as a "slider" UI element to the right and bottom of the boundary square
            # X-velocity slider, with label below
            plt.plot([-1,1], [-1.1, -1.1], color='black', linewidth=0.5)
            for v_edge in self._vel_bin_edges[1:-1]:
                # Scale min-max to [-1,1] for display purposes
                v_edge = (v_edge - self._vel_bin_edges[0]) / (self._vel_bin_edges[-1] - self._vel_bin_edges[0]) * 2 - 1
                plt.plot([v_edge, v_edge], [-1.07, -1.13], color='black', linewidth=0.5)
            v = (self.vel[0] - self._vel_bin_edges[0]) / (self._vel_bin_edges[-1] - self._vel_bin_edges[0]) * 2 - 1
            plt.plot(np.clip([v, v], -1, 1), [-1.05, -1.15], color='blue', linewidth=2)
            plt.text(0, -1.2, "X Velocity", horizontalalignment='center', verticalalignment='center')

            # Y-velocity slider, with verical orientation and label to the right
            plt.plot([1.1,1.1], [-1,1], color='black', linewidth=0.5)
            for v_edge in self._vel_bin_edges[1:-1]:
                # Scale min-max to [-1,1] for display purposes
                v_edge = (v_edge - self._vel_bin_edges[0]) / (self._vel_bin_edges[-1] - self._vel_bin_edges[0]) * 2 - 1
                plt.plot([1.07, 1.13], [v_edge, v_edge], color='black', linewidth=0.5)
            v = (self.vel[1] - self._vel_bin_edges[0]) / (self._vel_bin_edges[-1] - self._vel_bin_edges[0]) * 2 - 1
            plt.plot([1.05, 1.15], np.clip([v, v], -1, 1), color='blue', linewidth=2)
            plt.text(1.2, 0, "Y Velocity", rotation=-90, verticalalignment='center', horizontalalignment='center')

        # Draw target hover zone, set color based on whether the rocket is within the zone
        col = "#40FF40" if np.all(np.abs(self.pos) < 0.1) else 'r'
        hover_zone = plt.plot([-0.1, 0.1, 0.1, -0.1, -0.1], [-0.1, -0.1, 0.1, 0.1, -0.1], color=col, linewidth=2, linestyle='--')
        
        # Draw the rocket as a triangle with orientation based on angle
        rocket_length = 0.1
        rad_angle = np.deg2rad(self.angle)
        tip = self.pos + np.array([np.sin(rad_angle), np.cos(rad_angle)]) * rocket_length
        left = self.pos + np.array([np.sin(rad_angle + 140 * np.pi / 180), np.cos(rad_angle + 140 * np.pi / 180)]) * (rocket_length / 2)
        right = self.pos + np.array([np.sin(rad_angle - 140 * np.pi / 180), np.cos(rad_angle - 140 * np.pi / 180)]) * (rocket_length / 2)
        rocket_shape = np.array([tip, left, right, tip])
        plt.plot(rocket_shape[:, 0], rocket_shape[:, 1], 'b-')

        # Add thrusters as three lines depending on which thrusters are active
        # The center line is a bit longer to indicate stronger thrust, but all three lines are draw parallel to the rocket's
        # orientation from the base of the triangle
        thruster_length = 0.05
        
        aligned_vector = np.array([np.sin(rad_angle), np.cos(rad_angle)])
        perpendicular_vector = np.array([-np.cos(rad_angle), np.sin(rad_angle)])
        center_thruster_start = self.pos - aligned_vector * (rocket_length / 2)
        center_thruster_end = center_thruster_start - aligned_vector * thruster_length
        left_thruster_start = center_thruster_start + perpendicular_vector * (rocket_length / 6)
        left_thruster_end = left_thruster_start - aligned_vector * thruster_length * 0.7
        right_thruster_start = center_thruster_start - perpendicular_vector * (rocket_length / 6)
        right_thruster_end = right_thruster_start - aligned_vector * thruster_length * 0.7

        if 'C' in self._lastAction:
            plt.plot([center_thruster_start[0], center_thruster_end[0]], [center_thruster_start[1], center_thruster_end[1]], 'r-')
        if 'L' in self._lastAction:
            plt.plot([left_thruster_start[0], left_thruster_end[0]], [left_thruster_start[1], left_thruster_end[1]], 'r-')
        if 'R' in self._lastAction:
            plt.plot([right_thruster_start[0], right_thruster_end[0]], [right_thruster_start[1], right_thruster_end[1]], 'r-')
        
        # Format the plot
        plt.axis("square")
        plt.xlim(-1.2, 2.0)
        plt.ylim(-1.3, 1.2)
        
        # Add title and labels
        if params.get("DrawState", True):
            plt.title(f"Rocket World - Epoch {epoch}\n" + 
                    f"Position: ({self.pos[0]:4.1f}, {self.pos[1]:4.1f})\n" +
                    f"Velocity: ({self.vel[0]:5.2f}, {self.vel[1]:5.2f})\n" +
                    f"Angle: {self.angle:6.1f}°\n" + 
                    f"Angular velocity: {self.angleVel:6.2f}\n" +
                    f"State: {[int(s) for s in self._state]}, Terminal: {self._isTerminal}\n" +
                    f"Last Action: {self._lastAction:>3}, Reward: {self._lastReward:6.2f}", family='monospace')
        else:
            plt.title(f"Rocket World - Epoch {epoch}" if epoch is not None else "Rocket World")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        self._drawPost(sleepTime)