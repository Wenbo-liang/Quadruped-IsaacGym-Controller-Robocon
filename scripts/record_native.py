import os

# 强制限制 Lavapipe 线程，避免过早崩溃
os.environ["LP_NUM_THREADS"] = "1"
os.environ["MESA_VK_WINSYS"] = "headless"
os.environ["VK_ICD_FILENAMES"] = os.path.expanduser("~/lvp_icd.json")
os.environ["VK_LOADER_LAYERS_DISABLE"] = "~all~"

# 自动挂载底层动态链接库
libffi_path = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
libvulkan_path = "/lib/x86_64-linux-gnu/libvulkan.so.1"
curr_preload = os.environ.get("LD_PRELOAD", "")
if libffi_path not in curr_preload:
    os.environ["LD_PRELOAD"] = f"{libffi_path} {libvulkan_path} {curr_preload}".strip()

import isaacgym
from isaacgym import gymapi
assert isaacgym

import torch
import torch.nn.functional as F
import imageio
import importlib
import numpy as np
from tqdm import trange

try:
    mod = importlib.import_module("robodog_gym.envs.robodog.mybot_v3_config")
    for fn_name in dir(mod):
        if fn_name.startswith("config"):
            from robodog_gym.envs.base.legged_robot_config import Cfg
            getattr(mod, fn_name)(Cfg)
            break
except Exception as e:
    pass

from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv


def record_native_video(mode="stand", num_frames=150, output_name="walk.mp4"):
    print(f"\n>>> 正在启动最原初的单进程直录模式 (边跑边存): {output_name} ({mode}) <<<")

    # 1. 补齐 domain_rand 缺失属性，防止 AttributeError
    missing_attrs = [
        'randomize_motor_offset', 'randomize_motor_strength', 'randomize_Kp_factor', 
        'randomize_Kd_factor', 'randomize_rigids_after_start', 'randomize_com_displacement'
    ]
    for attr in missing_attrs:
        if not hasattr(Cfg.domain_rand, attr):
            setattr(Cfg.domain_rand, attr, False)

    # 2. 基础画面与环境配置
    Cfg.env.num_envs = 1
    Cfg.env.record_video = True
    Cfg.env.recording_width_px = 640
    Cfg.env.recording_height_px = 480
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "actuator_net"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg, eval_cfg=None)
    env.reset()

    # 加载训练权重 (请确保路径正确)
    model_path = "runs/rsl_exteroceptive_simple_mybot_v3/2026-08-21_13-55-14.639110/checkpoints/body_latest.jit"
    policy = torch.jit.load(model_path, map_location="cuda:0")

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

    # 核心修改：实时写入视频，这样即使中途崩溃 (Segfault)，已经渲染的帧也会完好保存在本地！
    writer = imageio.get_writer(output_name, fps=30)

    try:
        for i in trange(num_frames, desc=f"Rendering {mode}"):
            if mode == "stand":
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
            elif mode == "walk":
                env.commands[:, 0] = 0.8
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0

            with torch.no_grad():
                obs_1200 = get_obs_1200()
                actions = policy(obs_1200.detach())

            env.step(actions)

            # 锁定相机视角，避免拍到灰色天空
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

            # 获取图像并立刻写入视频文件
            frame = env.render(mode="rgb_array")
            if frame is not None:
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                writer.append_data(frame.astype(np.uint8))
                
    except Exception as e:
        print(f"\n录制中断: {e}")
    finally:
        # 关闭写入器，保存文件
        writer.close()
        print(f"✓ 视频文件已安全保存: {output_name}")

if __name__ == '__main__':
    record_native_video(mode="stand", num_frames=100, output_name="native_robot_stand.mp4")
    record_native_video(mode="walk", num_frames=100, output_name="native_robot_walk.mp4")
