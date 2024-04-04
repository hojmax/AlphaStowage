import wandb
import os
import time
from Buffer import ReplayBuffer
import torch.multiprocessing as mp


def logging_process(buffer: ReplayBuffer, queue: mp.Queue, config: dict) -> None:
    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    episode_count = 1

    while True:
        if not queue.empty():
            value, reshuffles = queue.get()
            log_episode(
                episode_count,
                value,
                reshuffles,
                config,
            )
            episode_count += 1
        else:
            time.sleep(1)


def init_wandb_group() -> None:
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()


def init_wandb_run(config: dict) -> None:
    wandb.init(
        entity="hojmax",
        project="AlphaStowage",
        config=config,
        save_code=True,
    )


def log_episode(
    episode: int, final_value: int, final_reshuffles: int, config: dict
) -> None:
    if config["train"]["log_wandb"]:
        wandb.log(
            {
                "episode": episode,
                "value": final_value,
                "reshuffles": final_reshuffles,
            }
        )
    else:
        print(
            f"*Episode {episode}* Value: {final_value}, Reshuffles: {final_reshuffles}"
        )


def log_batch(
    i: int,
    avg_loss: float,
    avg_value_loss: float,
    avg_cross_entropy: float,
    lr: float,
    config: dict,
):
    if config["train"]["log_wandb"]:
        wandb.log(
            {
                "loss": avg_loss,
                "value_loss": avg_value_loss,
                "cross_entropy": avg_cross_entropy,
                "batch": i,
                "lr": lr,
            }
        )
    else:
        print(
            f"*Batch {i}* Loss: {avg_loss}, Value Loss: {avg_value_loss}, Cross Entropy: {avg_cross_entropy}, LR: {lr}"
        )
