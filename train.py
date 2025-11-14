import draccus

from td3_bc.trainer import get_trainer, TrainerConfig
import dmc_env  # noqa: F401


@draccus.wrap()
def main(cfg: TrainerConfig):
    trainer = get_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
