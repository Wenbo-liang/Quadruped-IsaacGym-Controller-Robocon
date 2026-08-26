import isaacgym
assert isaacgym

import os
import torch
import imageio
import numpy as np
from tqdm import trange

from robodog_gym.envs import *
from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv

def record(mode="walk", num_frames=300, output_name="walk.mp4"):
    print(f"\n--- 开始录制: {output_name} ---")
    
    Cfg.env.num_envs = 1
    Cfg.env.record_video = True
    Cfg.env.recording_width_px = 640
    Cfg.env.recording_height_px = 480
    Cfg.domain_rand.push_robots = False
    
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)
    obs = env.reset()

    # 加载已训练的策略模型
    model_path = "runs/rsl_exteroceptive_simple_mybot_v3/2026-08-21_13-55-14.639110/checkpoints/body_latest.jit"
    policy = torch.jit.load(model_path, map_location="cuda:0")

    frames = []
    for _ in trange(num_frames, desc=f"Rendering {mode}"):
        if mode == "stand":
            env.commands[:, 0] = 0.0  # 速度 0: 稳定站立
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
        elif mode == "walk":
            env.commands[:, 0] = 0.8  # 前进 0.8m/s: 直线行走
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

        with torch.no_grad():
            policy_obs = obs[0] if isinstance(obs, tuple) else obs
            actions = policy(policy_obs.detach())
        obs, rew, done, info = env.step(actions)

        frame = env.render(mode="rgb_array")
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frames.append(frame.astype(np.uint8))

    imageio.mimsave(output_name, frames, fps=30)
    print(f"成功保存视频: {output_name}")

if __name__ == '__main__':
    # 1. 录制稳定站立视频 (约 6 秒)
    record(mode="stand", num_frames=200, output_name="robot_stand.mp4")
    # 2. 录制直线行走视频 (约 10 秒)
    record(mode="walk", num_frames=300, output_name="robot_walk.mp4")
