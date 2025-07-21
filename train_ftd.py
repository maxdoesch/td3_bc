import draccus
from functools import partial
import gymnasium as gym
from td3_bc.trainer import get_trainer, TrainerConfig
import dmc_envs # noqa: F401
from td3_bc.segmentation import MobileSAMV2Config
from td3_bc.wrappers import FTDObservationWrapper, FTDObservationWrapperConfig

def make_vec(env_id: str):
    env = gym.make(env_id, obs_type='pixels', channels_first=True, height=128, width=128)
    env_config = FTDObservationWrapperConfig(
        sam_config=MobileSAMV2Config(
            image_size=128,
            confidence_threshold=0.4
        ),
        add_original_frame=True,
    )
    env = FTDObservationWrapper(env, config=env_config)
    return env

@draccus.wrap()
def main(cfg: TrainerConfig):
    envs = gym.vector.SyncVectorEnv([partial(make_vec, env_id=cfg.env_name) for _ in range(cfg.num_envs)])

    trainer = get_trainer(cfg, envs=envs)
    trainer.train()


if __name__ == "__main__":
    main()
