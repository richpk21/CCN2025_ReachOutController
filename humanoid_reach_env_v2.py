from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import torch as th
import stable_baselines3 as sb3
from stable_baselines3 import PPO 
import mujoco.viewer 

# --------------------------- Environment Config -------------------------------
@dataclass
class SpotFRReachConfig:
    xml_path: str = "./mujoco_models/humanoid_fixed_body.xml"
    render_mode:str = "none" # rgb_array
    # episode_length: int = 300
    # action_limit: float = 40.0
    # action_reg: float = 1e-4
    # success_thresh: float = 0.02
    success_thresh: float = 0.05
    success_bonus: float = 2.0
    terminate_on_success: bool = False

    # Target sampling (world frame) — tuned for menagerie Spot workspace near FR foot
    target_center: Tuple[float, float, float] = (0.45, -0.20, 0.05) #(fl_hip)
    # target_range: Tuple[float, float, float]  = (0.10, 0.10, 0.05)
    target_range: Tuple[float, float, float]  = (0.5, 0.5, 0.5)








# ------- render util -----------
def viewer_render_sphere(viewer: mujoco.viewer.Handle,  pos:np.ndarray, size:np.ndarray, color: np.ndarray):
    # geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom=viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE, #type
        # type=mujoco.mjtGeom.mjGEOM_CAPSULE, #type
        # type=mujoco.mjtGeom.mjGEOM_ARROW, #type
        size=size,   # sphere radius
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=color #np.array([0, 1, 0, 0.5])           # green
    )
    viewer.user_scn.ngeom += 1


def viewer_render_arrow(viewer: mujoco.viewer.Handle, pos:np.ndarray, size: np.ndarray, mat:np.ndarray, color: np.ndarray):
    # geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

    # size_x = np.linalg.norm(vec)
    mujoco.mjv_initGeom(
        geom=viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW, #type 
        # type=mujoco.mjtGeom.mjGEOM_CAPSULE, #type 
        size=size,   # sphere radius
        pos=pos,
        mat=mat,
        rgba=color #np.array([0, 1, 0, 0.5])           # green
    )
    viewer.user_scn.ngeom += 1



class HumanoidReachEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "viewer", "rgb_array"], "render_fps": 60}

    def __init__(self, config: Optional[SpotFRReachConfig] = None, render_mode:str = "none"):
        super().__init__()
        self.cfg = SpotFRReachConfig()
        if config:
            self.cfg = config
        self.model : mujoco.MjStruct = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data: mujoco.MjStruct = mujoco.MjData(self.model)
        self.render_mode = render_mode

        n_act = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        # Observation: qpos, qvel, and target/foot positions
        obs_dim = 3*4 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        print(f"observation_space: {self.observation_space.shape}")
        self.num_steps = 0


        

        # id
        self.hand_left_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand_left")
        self.gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hand_left")

        # values
        self.body_qpos = self.data.qpos[:3] # xyz
        self.left_arm_qpos = self.data.qpos[-3:] # torque control
        self.left_arm_qvel = self.data.qvel[-3:] # torque control
        self.left_arm_qacc = self.data.qacc[-3:] # torque control
        self.hand_pos = self.data.site_xpos[int(self.hand_left_sid)].copy()
        self.action = np.zeros(3)
        self.go_cue = np.zeros(1)
        self.init_pos = np.array([0.19, 0.35 ,1.0])
        self.target_pos = self.init_pos
                
        self.dist = 0.0
        self.reward = 0.0

        self.g_sensor = np.array([0, 0, -9.8]) # self.model.opt.gravity
        self.g_sensor_enabled = False
        self.gravity_setup = {
            "mode": 0, 
            "gravity": np.array([0,0, -9.8]),
            "gravity-x-range": [0, 0],
            "gravity-y-range": [0, 0],
            "gravity-z-range": [0, 0],
            "en": False,
        }
        self._sample_gravity()

        self.prev_state = { 
            "qpos" : self.left_arm_qpos, 
            "qvel":self.left_arm_qvel,
            "qacc":self.left_arm_qacc,
            "pos": self.hand_pos,
            "dist": self.dist,
            "action": self.action,
            "go_cue" : self.go_cue,
        }

        self._viewer = None
        if self.render_mode in ("human", "viewer"):
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)


    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.action = np.zeros(3)

        self.terminated = False
        self.truncated = False
        self.t = 0.0
        self.num_steps = 0
        self.dist = 0.0
        self.reward = 0.0
        self.go_cue = np.zeros(1)

        # self.disable_pd_controller()

        # self.target_pos
        self._sample_target()
        self._sample_gravity()

        mujoco.mj_forward(self.model, self.data)

        self.left_arm_qpos = self.data.qpos[-3:] # torque control
        self.left_arm_qvel = self.data.qpos[-3:] # torque control
        self.left_arm_qacc = self.data.qpos[-3:] # torque control
        self.hand_pos = self.hand_pos = self.data.site_xpos[int(self.hand_left_sid)]

        

        self.prev_state = { 
            "qpos" : self.left_arm_qpos, 
            "qvel":self.left_arm_qvel,
            "qacc":self.left_arm_qacc,
            "pos": self.hand_pos,
            "dist": self.dist,
            "action": self.action,
            "go_cue":self.go_cue,
        }

        return self._get_obs(), self._get_info()


    # def set_gravity(self, g:tuple, enable_sensor:bool=False):
    #     if len(g) != 3:
    #         print("Warning: set_gravity len(g) != 3.")
    #         return
    #     self.model.opt.gravity = np.array(g)
    #     self.g_sensor_enabled = enable_sensor

    # ----------------------------------------------------------
    def _get_obs(self):
        # 22
        z3 = np.zeros(3)
        # n = 3 * 4 + 1
        c = np.concatenate([self.left_arm_qpos, self.left_arm_qvel, self.action, (self.hand_pos - self.target_pos), self.go_cue]).astype(np.float32)
        # c = np.concatenate([z3, self.left_arm_qvel, self.action, (self.hand_pos - self.target_pos), self.go_cue]).astype(np.float32)

        return c

    def _get_info(self):
        info = {"distance": self.dist,
                "reward": self.reward,
                "ep_length": self.num_steps,
                }
        return info

    # ----------------------------------------------------------
    def _calc_dist(self, a, b) -> float:
        return float(np.sqrt( np.sum(np.square(a - b)) ))

    # ___________________________
    def _get_reward(self) -> float:
        # dist = self._calc_dist(self.hand_pos,  self.target_pos)
        # self.dist = dist
        # dist =  (self.go_cue) * dist_init + (1-self.go_cue) * self.dist # 2


        target_pos = (self.go_cue) * self.target_pos + (1- self.go_cue) * self.init_pos  #3 
        dist = self._calc_dist(self.hand_pos,  target_pos)
        vel = self.prev_state["qvel"]

        reward = 0.0

        # distance loss
        reward += (np.exp( -dist**2) -1)
        # reward += (np.exp( -dist**4) -1)

        # velocity term
        # reward += np.exp( -dist**2) * (np.exp( -np.linalg.norm(vel)**2) -1)
        # reward += (dist**2) * (np.exp( -np.linalg.norm(vel)**2) -1)

        reward += np.exp( -dist**2) * (-np.linalg.norm(vel)**2)

        # reward += 5e-9 -np.sum(np.square(vel))
        jerk = (self.prev_state["qacc"] - self.left_arm_qacc) / self.model.opt.timestep # jerk = (a_0 - a_1) / delta_t
        reward += 1e-10 * -np.sum(np.square(jerk)) 
        reward += 1e-12 * -np.sum(np.square(self.action))


        reward = float(reward)

        return reward


    # ----------------------------------- step(self, action)-----------------------
    def step(self, action:np.ndarray):

        if self.num_steps >= 500-1:  # about 1 seconds
            self.go_cue = np.ones(1)
        if self.num_steps > 2500:  # about 5 seconds
            self.truncated = True
        self.num_steps+=1

        
        self.action = action
        self.data.ctrl[-3:] = action

        mujoco.mj_step(self.model, self.data)

        dt = self.model.opt.timestep # 0.002 by default
        self.t += dt
  
        # visualize here
        self.render()
        

        self.left_arm_qpos = self.data.qpos[-3:] # torque control
        self.left_arm_qvel = self.data.qpos[-3:] # torque control
        self.left_arm_qacc = self.data.qpos[-3:] # torque control
        self.hand_pos = self.data.site_xpos[int(self.hand_left_sid)]
        self.dist = self._calc_dist(self.hand_pos,  self.target_pos)

        self.prev_state = { 
            "qpos" : self.left_arm_qpos, 
            "qvel":self.left_arm_qvel,
            "qacc":self.left_arm_qacc,
            "pos": self.hand_pos,
            "dist": self.dist,
            "go_cue": self.go_cue,
            }

        obs = self._get_obs()
        reward = self._get_reward()
        self.reward = reward
        info = self._get_info()
        
       

        info["reward"] = reward
    

        # info = {"distance": dist, "target":self._target, "pos":self._foot_pos()}
        
        # print([obs, reward, terminated, truncated, info])
        return obs, reward, self.terminated, self.truncated, info
    




    


    
    # def disable_pd_controller(self):
    #     for j in range(12):
    #         self.model.actuator_gainprm[j] = 0.0
    #         self.model.actuator_biasprm[j] = 0

    # -------------------
    def render(self):
        # visualize 
        # --- keep the passive viewer updated ---
        if (self._viewer is not None) and (self.render_mode in ["human", "viewer"]):
            try:
                self._viewer.user_scn.ngeom = 0
                viewer_render_sphere(viewer=self._viewer, pos = np.array(self.target_pos),   size=np.array([0.05, 0.05, 0.05]), color = np.array([0, 1, 0,0.2]) , )
                # viewer_render_sphere(viewer=self._viewer, pos = np.array(self._fr_foot_xpos()), size=np.array([0.03, 0.03, 0.03]), color = np.array([0, 0, 1,0.2]) , )
                # imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
                # imu_pos =self.data.site_xpos[int(imu_site_id)].copy()

                # body_mat = self.body_orientation_matrix("body")
                # viewer_render_arrow(viewer=self._viewer, pos = imu_pos, size=np.array([0.001, 0.001, 0.3]),mat=body_mat,color = np.array([1, 0, 1, 0.9]) )


                self._viewer.sync()
                # print(f"viewer sync")
            except Exception as e:
                print(f"Exception: {e}")
        # end visualize




    # ------------------------
    #   
    # sample target
    def _sample_target(self):
        self.target_pos = np.array([
                self.np_random.uniform(0.19, 0.50),
                self.np_random.uniform(0.00, 0.35),
                self.np_random.uniform(1.00, 1.35),
            ], dtype=np.float32)
        #mujoco.mj_forward(self.model, self.data) # site

    # sample gravity
    def _sample_gravity(self):
        # print(f"_sample_gravity!")
        gravity = self.gravity_setup["gravity"]
        x = self.gravity_setup["gravity-x-range"]
        y = self.gravity_setup["gravity-y-range"]
        z = self.gravity_setup["gravity-z-range"]

        self.g_sensor_enabled = self.gravity_setup["en"]

        if self.gravity_setup["mode"] == 1:
            gravity += np.array([
                    self.np_random.uniform(x[0], x[0]),
                    self.np_random.uniform(y[0], y[0]),
                    self.np_random.uniform(z[0], z[0]),
            ], dtype=np.float32)

        self.model.opt.gravity = gravity
        if self.g_sensor_enabled:
            self.g_sensor = gravity

    def set_gravity(self, config:dict):
        # print(f"set_gravity!")
        self.gravity_setup["mode"] = config["mode"]
        self.gravity_setup["gravity"] = config["gravity"]
        self.gravity_setup["gravity-x-range"] = config["gravity-x-range"]
        self.gravity_setup["gravity-y-range"] = config["gravity-y-range"]
        self.gravity_setup["gravity-z-range"] = config["gravity-z-range"]
        self.gravity_setup["en"] = config["en"]
        self._sample_gravity()



    # ----------------------------------------------------------
    def close(self):
        pass


register(
    id="humanoid-reach-env-v2",
    # entry_point="my_spot_env:SpotFRLegReachEnv",
    entry_point="humanoid_reach_env_v2:HumanoidReachEnv",
)
