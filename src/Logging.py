import wandb
import os
import time
from Buffer import ReplayBuffer
import torch.multiprocessing as mp


def logging_process(queue: mp.Queue, config: dict) -> None:
    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    episode_logger = EpisodeLogger(config)
    episode_count = 1

    while True:
        if not queue.empty():
            value, reshuffles = queue.get()
            episode_logger.log(episode_count, value, reshuffles, config)
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


class EpisodeLogger:
    """Averages out over log_interval episodes before logging"""

    def __init__(self, config: dict):
        self.avg_value = 0
        self.avg_reshuffles = 0
        self.count = 0
        self.log_interval = config["inference"]["log_interval"]

    def log(self, episode: int, value: int, reshuffles: int, config: dict):
        self.avg_value += value
        self.avg_reshuffles += reshuffles
        self.count += 1

        if self.count == self.log_interval:
            self.avg_value /= self.log_interval
            self.avg_reshuffles /= self.log_interval

            log_episode(
                episode,
                self.avg_value,
                self.avg_reshuffles,
                config,
            )

            self.avg_value = 0
            self.avg_reshuffles = 0
            self.count = 0


class BatchLogger:
    """Averages out over log_interval batches before logging"""

    def __init__(self, config: dict):
        self.avg_loss = 0
        self.avg_value_loss = 0
        self.avg_cross_entropy = 0
        self.avg_lr = 0
        self.count = 0
        self.log_interval = config["train"]["log_interval"]

    def log(
        self,
        i: int,
        loss: float,
        value_loss: float,
        cross_entropy: float,
        lr: float,
        config: dict,
    ):
        self.avg_loss += loss
        self.avg_value_loss += value_loss
        self.avg_cross_entropy += cross_entropy
        self.avg_lr += lr
        self.count += 1

        if self.count == self.log_interval:
            self.avg_loss /= self.log_interval
            self.avg_value_loss /= self.log_interval
            self.avg_cross_entropy /= self.log_interval
            self.avg_lr /= self.log_interval

            log_batch(
                i,
                self.avg_loss,
                self.avg_value_loss,
                self.avg_cross_entropy,
                self.avg_lr,
                config,
            )

            self.avg_loss = 0
            self.avg_value_loss = 0
            self.avg_cross_entropy = 0
            self.avg_lr = 0
            self.count = 0


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
