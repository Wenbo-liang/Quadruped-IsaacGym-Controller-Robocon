from typing import Union

from params_proto import Meta

from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.mybot_v2_1_config import config_mybot_v2_1


def config_mybot_v3(Cnfg: Union[Cfg, Meta]):
    config_mybot_v2_1(Cnfg)

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf'

    _ = Cnfg.reward_scales
    # 方案：加大姿态倾斜惩罚（从 v2_1 继承的 -5.0 改为 -25.0，放大 5 倍惩罚）
    _.orientation = -25.0
    # 加大电机力矩惩罚（从 v2_1 继承的 -0.0001 改为 -0.001，防止电机过载）
    _.torques = -0.001
