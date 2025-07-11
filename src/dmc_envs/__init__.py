from typing import Optional, Dict

import gymnasium.envs
from gymnasium.envs.registration import register
import gymnasium.envs.registration

import dm_control.suite as suite

OBS_TYPE = "both"  # 'state', 'pixels', or 'both'


def make(
    domain_name: str,
    task_name: str,
    seed: int = 1,
    visualize_reward: bool = True,
    obs_type: str = "state",
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_skip: int = 1,
    episode_length: int = 1000,
    environment_kwargs: Optional[Dict] = None,
    time_limit: Optional[float] = None,
    channels_first: bool = True,
    disctracting_control: bool = False,
):
    assert obs_type in ["state", "pixels", "both"], "obs_type must be one of: state, pixels, both"

    if disctracting_control:
        assert obs_type in ["both", "pixels"], "disctracting control only supports state or pixels observation types"
        env_id = "dmc_distraction_%s_%s_%s-v1" % (domain_name, task_name, seed)
    else:
        env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

    if obs_type in ["pixels", "both"]:
        assert not visualize_reward, "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gymnasium.envs.registration.registry:
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        register(
            id=env_id,
            entry_point="dmc_envs.dmc2gym:DistractionDMCWrapper"
            if disctracting_control
            else "dmc_envs.dmc2gym:DMCWrapper",
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                obs_type=obs_type,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )


for domain_name, task_name in suite._get_tasks(tag=None):
    make(
        domain_name=domain_name,
        task_name=task_name,
        seed=1,
        visualize_reward=False,
        obs_type=OBS_TYPE,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=False,
        disctracting_control=False,
    )
    make(
        domain_name=domain_name,
        task_name=task_name,
        seed=1,
        visualize_reward=False,
        obs_type=OBS_TYPE,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=False,
        disctracting_control=True,
    )
