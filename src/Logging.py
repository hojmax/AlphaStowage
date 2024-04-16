import wandb
import os


def init_wandb_group() -> None:
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()


def init_wandb_run(config: dict) -> None:
    wandb.init(
        entity="hojmax",
        project="AlphaStowage",
        config=config,
        save_code=True,
    )
