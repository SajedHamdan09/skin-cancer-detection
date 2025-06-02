"""Main entry point of the program.
Parses a config file and runs the trainer.
"""

import argparse
from senn.trainer import init_trainer


def main():
    """
    Entry point to the trainer.
    Parses command line args for config file, then initializes and runs the Trainer.
    """
    parser = argparse.ArgumentParser(description="Run SENN trainer with specified config.")
    parser.add_argument(
        '--config',
        default="config.json",
        help='Path to experiment config file (JSON)'
    )
    args = parser.parse_args()

    trainer = init_trainer(args.config)
    trainer.run()
    trainer.finalize()


if __name__ == "__main__":
    main()
