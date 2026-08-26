import isaacgym
from isaacgym import gymapi
assert isaacgym

import argparse
import importlib
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import trange

from robodog_gym.envs import *
from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.go1_config import config_go1
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='mybot_v3')
    parser.add_argument('--load_run', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()

def run_env(args):
    if args.robot == 'go1':
        config_go1(Cfg)
    else:
        try:
            mod = importlib.import_module(f"robodog_gym.envs.robodog.{args.robot}_config")
            for fn_name in dir(mod):
                if fn_name.startswith("config"):
                    getattr(mod, fn_name)(Cfg)
                    break
        except Exception as e:
            print(f"⚠️ 动态加载 {args.robot} 配置提示: {e}，使用默认配置")

    missing_attrs = [
        'randomize_motor_offset', 'randomize_motor_strength', 'randomize_Kp_factor', 
        'randomize_Kd_factor', 'randomize_rigids_after_start', 'randomize_com_displacement'
    ]
    for attr in missing_attrs:
        if not hasattr(Cfg.domain_rand, attr):
            setattr(Cfg.domain_rand, attr, False)

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.9
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.9
    Cfg.commands.distributional_commands = True

    Cfg.env.num_envs = 1
    Cfg.env.record_video = True

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "actuator_net"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=args.headless, cfg=Cfg)
    env.reset()

    policy = None
    if args.load_run:
        model_path = f"runs/rsl_exteroceptive_simple_{args.robot}/{args.load_run}/checkpoints/body_latest.jit"
        try:
            policy = torch.jit.load(model_path, map_location="cuda:0")
            print(f"✅ 成功加载神经网络权重: {model_path}")
        except Exception as e:
            print(f"❌ 加载模型权重失败，请检查路径: {e}")

    history_queue = []

    def get_obs_1200():
        nonlocal history_queue
        for obj in [getattr(env, 'observations', None), env]:
            if obj is None:
                continue
            for attr in dir(obj):
                try:
                    val = getattr(obj, attr)
                    if isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[1] == 1200:
                        return val
                except Exception:
                    pass

        current_obs = env.policy_obs_buf
        steps_needed = int(np.ceil(1200 / current_obs.shape[1]))
        
        if not history_queue:
            history_queue = [current_obs.clone() for _ in range(steps_needed)]
        else:
            history_queue.pop(0)
            history_queue.append(current_obs.clone())

        concat_obs = torch.cat(history_queue, dim=-1)

        if concat_obs.shape[1] > 1200:
            return concat_obs[:, :1200]
        elif concat_obs.shape[1] < 1200:
            return F.pad(concat_obs, (0, 1200 - concat_obs.shape[1]))
        return concat_obs

    def record_behavior(mode="stand", num_frames=200, output_name="out.mp4"):
        print(f"\n🎥 开始渲染行为视频: {output_name} ({mode})")
        frames = []
        for i in trange(num_frames, desc=f"Rendering {mode}"):
            if mode == "stand":
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
            elif mode == "walk":
                env.commands[:, 0] = 0.8
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0

            if policy is not None:
                with torch.no_grad():
                    obs_1200 = get_obs_1200()
                    actions = policy(obs_1200.detach())
            else:
                actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)

            env.step(actions)

            try:
                cam_handle = getattr(env, 'rendering_camera', None)
                if cam_handle is None:
                    cam_handle = getattr(env, 'camera_handle', None)
                    
                if cam_handle is not None and hasattr(env, 'envs') and len(env.envs) > 0:
                    base_pos = env.root_states[0, :3].cpu().numpy()
                    cam_pos = gymapi.Vec3(float(base_pos[0] + 1.6), float(base_pos[1] + 1.6), float(base_pos[2] + 0.8))
                    cam_target = gymapi.Vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2] + 0.2))
                    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)
            except Exception:
                pass

            frame = env.render(mode="rgb_array")
            if frame is not None:
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                frames.append(frame.astype(np.uint8))

        if len(frames) > 0:
            imageio.mimsave(output_name, frames, fps=30)
            print(f"✓ 成功导出视频: {output_name}")

    record_behavior(mode="stand", num_frames=200, output_name="mybot_v3_stand.mp4")
    record_behavior(mode="walk", num_frames=300, output_name="mybot_v3_walk.mp4")
    print("✨ 所有行为视频导出完毕！")

if __name__ == '__main__':
    args = get_args()
    run_env(args)
