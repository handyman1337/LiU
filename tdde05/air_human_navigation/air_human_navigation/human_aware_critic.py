import air_navigation
import math
import json
import os

#where the human_tracker node dumps detected human positions
HUMAN_FILE = "/tmp/humans.json"

#Tuning constants, change these to adjust avoidance behavior--------------

#personal-space gaussian - the discomfort zone around a human
PS_AMPLITUDE   = 20.0  #peak cost right at the human's position
PS_SIGMA_FRONT = 1.5   #how far in front the zone extends, keep this wide
PS_SIGMA_BACK  = 0.2   #very narrow behind so the robot can sneak past from behind
PS_SIGMA_Y     = 1.0   #lateral width, wider = robot keeps more side distance to humans

#frontal approach gaussian - extra penalty for heading in front of a human
FA_AMPLITUDE = 5.0  #extra cost on top of personal space when approaching in front
FA_SIGMA_X   = 3.5  #how far ahead the frontal penalty kicks in
FA_SIGMA_Y   = 0.6  #keeps this narrow so trajectories to the humans' sides aren't affected

#weights that scale each cost component in the final score
PS_WEIGHT = 5.0  #weight on personal space cost
FA_WEIGHT = 3.0  #weight on frontal approach cost

#slow down near humans but don't stop completely
SPEED_SCALE = 0.5  #cost = this * personal_cost * velocity

#punish standing still for too long - forces the planner to eventually commit to a trajectory
STUCK_ACTIVATION = 2.0   #personal space cost threshold before this kicks in
STUCK_VEL_THRESH = 0.10  #velocities below this count as "stuck"
STUCK_SCALE      = 7.0   #how hard to punish being stuck


class HumanAwareCritic(air_navigation.TrajectoryCritic):

    def __init__(self):
        super().__init__()
        #starts empty, gets filled each planning cycle from the json file
        self.humans = []

    def onInit(self):
        print("HumanAwareCritic initialized")

    def prepare(self, pose, vel, goal, global_plan):
        #reload human positions before scoring each batch of trajectories
        try:
            with open(HUMAN_FILE) as f:
                self.humans = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  #keep old list if file isn't ready yet
        return True

    def reset(self):
        pass

    def debrief(self, vel):
        pass

    def scoreTrajectory(self, traj):
        max_personal = 0.0
        max_frontal  = 0.0

        #skip the first pose since it's the robot's current position,
        poses = traj.poses[1:] if len(traj.poses) > 1 else traj.poses

        for pose in poses:
            for human in self.humans:
                pc = self.personal_space_cost(pose.x, pose.y, human)
                if pc > max_personal:
                    max_personal = pc

                fc = self.frontal_approach_cost(pose.x, pose.y, human)
                if fc > max_frontal:
                    max_frontal = fc

        return (
            self.comfort_zone_cost(max_personal)
            + self.frontal_zone_cost(max_frontal)
            + self.speed_cost(traj, max_personal)
            + self.stuck_cost(traj, max_personal)
        )


    #Modular cost components-----------------------------

    def personal_space_cost(self, x, y, human):
        #asymmetric gaussian based on lecture 11
        #wider in front, very narrow behind so passing from behind is cheap
        dx = x - human["x"]
        dy = y - human["y"]
        if dx * dx + dy * dy > 16.0:
            return 0.0  #human outside 4m range, so don't even compute

        #rotate into human's local frame so front/back/side make sense
        rel_x = human["cos_h"] * dx + human["sin_h"] * dy
        rel_y = -human["sin_h"] * dx + human["cos_h"] * dy

        sigma_x = PS_SIGMA_FRONT if rel_x >= 0 else PS_SIGMA_BACK

        return PS_AMPLITUDE * math.exp(
            -((rel_x ** 2) / (2 * sigma_x ** 2) + (rel_y ** 2) / (2 * PS_SIGMA_Y ** 2))
        )

    def frontal_approach_cost(self, x, y, human):
        #extra cost for being in front of the human, zero behind them
        #wider than personal space so the robot starts steering earlier
        dx = x - human["x"]
        dy = y - human["y"]
        if dx * dx + dy * dy > 16.0:
            return 0.0

        rel_x = human["cos_h"] * dx + human["sin_h"] * dy
        rel_y = -human["sin_h"] * dx + human["cos_h"] * dy

        if rel_x <= 0:
            return 0.0  #behind the human, no penalty here

        return FA_AMPLITUDE * math.exp(
            -((rel_x ** 2) / (2 * FA_SIGMA_X ** 2) + (rel_y ** 2) / (2 * FA_SIGMA_Y ** 2))
        )

    def comfort_zone_cost(self, max_personal):
        return PS_WEIGHT * max_personal

    def frontal_zone_cost(self, max_frontal):
        return FA_WEIGHT * max_frontal

    def speed_cost(self, traj, max_personal):
        #robot should slow down when near humans and not go at full speed
        return SPEED_SCALE * max_personal * abs(traj.velocity.x)

    def stuck_cost(self, traj, max_personal):
        #if the robot is near a human and considers just standing still,
        #this makes that trajectory lose so it picks a moving one instead
        if max_personal < STUCK_ACTIVATION:
            return 0.0
        return STUCK_SCALE * max_personal * max(0.0, STUCK_VEL_THRESH - abs(traj.velocity.x))
