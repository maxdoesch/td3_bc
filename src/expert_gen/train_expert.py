import os
import argparse

from sbx import PPO



def main():
    parser = argparse.ArgumentParser(description="Train an expert agent using PPO.")
    parser.add_argument("--env_id", type=str, default="dmc_cheetah_run_1-v1",
                        help="Environment ID to train on.")

if __name__ == "__main__":
    main()