import draccus
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(filename)s:%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

from src.td3_bc.trainer import get_trainer, TrainerConfig

@draccus.wrap()
def main(cfg: TrainerConfig):
    trainer = get_trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
