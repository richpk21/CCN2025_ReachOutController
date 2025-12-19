from __future__ import annotations
from typing import List, Optional, Tuple

import os, sys

from dataclasses import dataclass
import gymnasium as gym
import mujoco
import mujoco.viewer 
# from mujoco import mjv_initGeom
import numpy as np
from gymnasium import spaces
# from utils import _names, _find_first, _find_all, _require_fixed_base_or_raise, make_fixed_base_xml 
import math
# from networks import GRUPolicy
# from agents import GRUAgent
#imu_site
import torch as th
import time
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing as mp
from humanoid_reach_env_v2 import HumanoidReachEnv
from collections import defaultdict
# from gru_policy import GRUActorCriticPolicy, GRUHiddenStateCallback, GRULikeLSTM



class ActivationCatcher:
    def __init__(self, policy: th.nn.Module, prefixes=("mlp_extractor.policy_net", "mlp_extractor.value_net"),
                 only_linear=True):
        self.acts = {}     # name -> Tensor on CPU
        self.handles = []

        for name, module in policy.named_modules():
            if not any(name.startswith(p) for p in prefixes):
                continue
            if only_linear and not isinstance(module, th.nn.Linear):
                continue

            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.acts[name] = output.detach().cpu()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class RewardRecorderCallback(BaseCallback):
    """
    Logs episode rewards to TensorBoard and stores them in memory.
    Works with (Vec)Monitor-wrapped envs or envs that fill info["episode"].
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.running_mean = None

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        # infos = self.locals.get("infos", None)
        if infos is not None:
            self.episode_rewards.extend(rewards.tolist())
            mean_r = float(np.mean(rewards))
            if self.running_mean is None:
                self.running_mean = mean_r
            else:
                alpha = 0.01 # smoothing factor
                self.running_mean = alpha * mean_r + (1-alpha) * self.running_mean
            self.logger.record("custom/step_reward_mean", self.running_mean)

        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    ep_info = infos[i].get("episode")
                    if ep_info is not None:
                        ep_r = ep_info["r"]
                        ep_len = ep_info["l"]
                        self.logger.record("custom/ep_reward", ep_r)
                        self.logger.record("custom/ep_length", ep_len)

        return True
    
    

class GymnasiumToSB3(gym.Wrapper):
    """
    Wrap a Gymnasium env to look like old Gym API:
    - reset() -> obs
    - step() -> obs, reward, done, info   where done = terminated or truncated
    """
    def reset(self, **kwargs):
        try:
            print("Try-g2sb-reset")
            obs, info = self.env.reset(**kwargs)
            print("Try-g2sb3-reset-done")
        except Exception as e:
            print(f"Try-g2sb3-reset-Error: {e}")
        print(f"obs:{type(obs)}, {obs}")
        return obs

    def step(self, action):
        print("Try-g2sb3-step")
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info



def make_env(env_id: str, seed: int):
    """
    Factory for DummyVecEnv. For macOS this is nice and safe.
    """
    def _init():
        env = gym.make(env_id)
        env = GymnasiumToSB3(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


        

if __name__ == "__main__":

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set
        pass   

    ENV_ID = "humanoid-reach-env-v2"
    N_BATCH = 64
    N_ENVS = 12
    # N_ENVS = 1
    N_STEPS = 5000 *  N_ENVS 
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    # vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    vec_env = VecMonitor(vec_env)

    gravity_setup = {
                    "mode": 0, 
                    "gravity": np.array([0.0,0.0, -9.8]),
                    "gravity-x-range": [-9.8*3, 9.8*10],
                    "gravity-y-range": [-9.8*3, 9.8*10],
                    "gravity-z-range": [-9.8*10, 9.8*10],
                    "en": False,
                }
    
    vec_env.env_method("set_gravity", gravity_setup) 

    policy_kwargs = dict(
        net_arch=[64]
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=0,
        n_steps=N_STEPS,
        batch_size= N_BATCH, # N_STEPS * N_ENVS,
        gamma=0.9,
        learning_rate = 1e-3, #1e-4 , #* 1e-2,
        tensorboard_log="tb"
        # device="mps" # for custom policy
    )



    # file =  "humanoid-reach-env-v2-64-001"  # ppo9  # 64, 16
    # file =  "humanoid-reach-env-v2-64-001 copy 2"  # ppo9  # 64, 16
    # file = "humanoid-reach-env-v2-64-002-"  # ppo9  # 64, 16
    file = "humanoid-reach-env-v2-64-002"  # ppo9  # 64, 16
    # file = "humanoid-reach-env-v2-64-003"  # ppo9  # 64, 16


    if os.path.exists(file + ".zip"):
        print(f"{file+'.zip'} exists.")
        model = PPO.load(path=file+".zip", print_system_info=True)
        model.set_env(vec_env)
        # model.load(path=file+".zip")
    else:
        print(f"{file+'.zip'} does not exists.")
        
    
    train = False
    eval = True
    eval2 = False
    plot_activity=False

    if train:
        # filetowrite =  "humanoid-reach-env-v2-64-002" #  003:ppo10
        filetowrite =  "humanoid-reach-env-v2-64-003" #  003:ppo10

        continue_training = False
        if file == filetowrite:
            continue_training = True
        

        try:
            print("try")
            callback = RewardRecorderCallback(verbose=1)
            # model.learn(total_timesteps=1e8, callback=RewardLoggingCallback)
            model.learn(total_timesteps=1e9, callback=callback, progress_bar=True, reset_num_timesteps= not continue_training)
            # model.learn(total_timesteps=1e9, callback=callback, progress_bar=True, reset_num_timesteps= True)

            print('model.learn(total..)')
            model.save(filetowrite)
            # model.policy.save(file+"_policy")
            print('model.saved try-1')
        except KeyboardInterrupt:
            model.save(filetowrite)
            # model.policy.save(file+"_policy")
            print('model.saved keyboard')
            print("keyboard interrupt")
        # except Exception as e:
        #     model.save(file)
        #     # model.policy.save(file+"_policy")
        #     print('model.saved err')
        #     print(f"Try-Error: {e}")

        print(f"what?")


    if eval:
            try:
                print("Eval")
                # model.load("SpotFRLegReach-v0_captian_america_v0_001")
                # Quick evaluation rollout
                eval_env = gym.make(ENV_ID, render_mode="human")

                gravity_setup = {
                    "mode": 0, 
                    "gravity": np.array([0.0,0.0, -9.8]),
                    "gravity-x-range": [0.0, 0.0],
                    "gravity-y-range": [0.0, 0.0],
                    "gravity-z-range": [0.0, 0.0],
                    "en": True,
                }
                
                # obs_action_enable = [0,0,0]
                # eval_env.unwrapped.set_obs_action_enable(obs_action_enable=obs_action_enable)
                print(f"call eval_env.set_gravity!")

                # eval_env.env_method("set_gravity", gravity_setup) 
                # eval_env.set_gravity(gravity_setup)
                eval_env.unwrapped.set_gravity(gravity_setup)


                obs, info = eval_env.reset() # options={"fr_toe_target": np.array([0,0,0])} )
                
                # g = np.array([0,0,-8], dtype=np.float32)
                # eval_env.unwrapped.model.opt.gravity = g

                
                # eval_env.unwrapped.set_gravity(g)
                # print(f"gravity: ", eval_env.unwrapped.model.opt.gravity )

                
                # for _ in range(int(10/0.002)):
                for _ in range(int(50000)):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    print("obs action", obs[12:15] )
                    print('go cue:', eval_env.unwrapped.go_cue)
                    if terminated or truncated:
                        print("episode ended.")
                        obs, info = eval_env.reset()
                    time.sleep(1/60)

            except Exception as e:
                print(f"Eval-Error: {e}")




    if eval2:
            
            # model: PPO = PPO.load(file+".zip")  # or your trained model
            policy = model.policy
            catcher = ActivationCatcher(policy, only_linear=True)

            # obs = ...  # shape (obs_dim,) or (n_envs, obs_dim)
            # obs_tensor, _ = policy.obs_to_tensor(obs)   # handles device + preprocessing

            # with th.no_grad():
            #     # forward pass through the full actor-critic; hooks will populate catcher.acts
            #     _actions, _values, _logp = policy(obs_tensor)

            # for k, v in catcher.acts.items():
            #     print(k, v.shape)  # each is the activation output of that Linear layer

            # catcher.close()


            try:
                print("Eval")
                # model.load("SpotFRLegReach-v0_captian_america_v0_001")
                # Quick evaluation rollout
                eval_env = gym.make(ENV_ID, render_mode="human")
                obs, info = eval_env.reset() # options={"fr_toe_target": np.array([0,0,0])} )
                # obs_tensor, _ = policy.obs_to_tensor(obs) 
                for k, v in catcher.acts.items():
                    print(k, v.shape)  # each is the activation output of that Linear layer

                
                # g = np.array([0,0,-8], dtype=np.float32)
                # eval_env.unwrapped.model.opt.gravity = g
                # eval_env.unwrapped.set_gravity(g)
                # print(f"gravity: ", eval_env.unwrapped.model.opt.gravity )

                
                # for _ in range(int(10/0.002)):

                hidden_activities={
                    "mlp_extractor.policy_net.0": [], # 64
                    "mlp_extractor.policy_net.2": [], #32
                                   }
                for _ in range(int(5000)):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)

                    # obs_tensor, _ = policy.obs_to_tensor(obs) 
                    for k, v in catcher.acts.items():
                        v = v.detach().cpu().numpy()
                        hidden_activities[k].append(v)
                        print(k, v.shape, type(v))  # each is the activation output of that Linear layer

                    if terminated or truncated:
                        print("episode ended.")
                        obs, info = eval_env.reset()
                    # time.sleep(1/60)

                catcher.close()



                h0 = np.concatenate(hidden_activities["mlp_extractor.policy_net.0"], axis=0)
                # h2 = np.concatenate(hidden_activities["mlp_extractor.policy_net.2"], axis=0)

                print(f"hidden_activities-0: {h0.shape} ")
                # print(f"hidden_activities-2: {h0.shape} ")

                t = np.linspace(0, 10, 5000)
                print(t[0], t[-1], len(t))
                
                np.save("h0", h0)
                # np.save("h2", h2)

            except Exception as e:
                print(f"Eval-Error: {e}")


    if plot_activity:
        import matplotlib.pyplot as plt

        h0 = np.load("h0.npy")
        h2 = np.load("h2.npy")

        t = np.linspace(0, 10, 5000)
        print(t[0], t[-1], len(t))

        plt.plot(t, h0[:,0])
        plt.plot(t, h2[:,0])

        plt.show()
