from typing import Union, Tuple

from td3_bc.td3_bc import TD3BC, TD3BC_Refine, TD3BC_Online, TD3BC_Config, TD3BC_Refine_Config, TD3BC_Online_Config
from td3_bc.td3_bc_ftd import TD3BC_FTD, TD3BC_FTD_Config


def get_td3_bc_agent(
    obs_shape: Union[int, Tuple[int]],
    action_dim: int,
    max_action: float,
    train_steps: int,
    cfg: Union[
        TD3BC_Config,
        TD3BC_Refine_Config,
        TD3BC_Online_Config,
    ],
    device: str,
) -> Union[TD3BC, TD3BC_Refine, TD3BC_Online]:
    if type(cfg) is TD3BC_Config:
        return TD3BC(obs_shape, action_dim, max_action, cfg, device)
    elif type(cfg) is TD3BC_Refine_Config:
        return TD3BC_Refine(obs_shape, action_dim, max_action, cfg, device)
    elif type(cfg) is TD3BC_Online_Config:
        return TD3BC_Online(obs_shape, action_dim, max_action, train_steps, cfg, device)
    elif type(cfg) is TD3BC_FTD_Config:
        return TD3BC_FTD(obs_shape, action_dim, max_action, cfg, device)
    else:
        raise ValueError(f"Unsupported configuration type: {type(cfg)}")
