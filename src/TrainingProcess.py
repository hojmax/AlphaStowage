from Train import init_model, PretrainedModel, get_optimizer, get_scheduler, train_batch
import torch
from Buffer import ReplayBuffer
import torch.multiprocessing as mp
import numpy as np
import wandb
from Logging import init_wandb_run
from StepLogger import StepLogger
import time
import subprocess


class TrainingProcess:
    def __init__(
        self,
        buffer: ReplayBuffer,
        gpu_update_event: mp.Event,
        device: torch.device,
        pretrained: PretrainedModel,
        config: dict,
    ) -> None:
        if config["wandb"]["should_log"]:
            init_wandb_run(config)

        self.device = device
        self.buffer = buffer
        self.gpu_update_event = gpu_update_event
        self.config = config
        self.logger = StepLogger(
            n=self.config["train"]["log_interval"],
            step_name="batch",
            log_wandb=self.config["wandb"]["should_log"],
        )
        self.model = init_model(config, device, pretrained)
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.model.train()
        self.batch = 1

    def loop(self) -> None:
        self._wait_for_buffer()

        while True:
            if self._should_swap():
                self._swap_over()

            self._handle_batch()
            self.batch += 1

    def _handle_batch(self) -> None:
        loss, value_loss, cross_entropy_loss = train_batch(
            self.model, self.buffer, self.optimizer, self.scheduler, self.config
        )
        self.logger.log(
            {
                "loss": loss,
                "value_loss": value_loss,
                "cross_entropy_loss": cross_entropy_loss,
                "lr": self.scheduler.current_lr(),
            }
        )

    def _should_swap(self) -> bool:
        return self.batch % self.config["train"]["swap_and_save_interval"] == 0

    def _swap_over(self) -> None:
        torch.save(self.model.state_dict(), f"shared_model.pt")

        if self.config["wandb"]["should_log"]:
            wandb.save("shared_model.pt")

        self.config["inference"]["can_only_add"] = False
        self.gpu_update_event.set()

    def _wait_for_buffer(self):
        while len(self.buffer) < self.config["train"]["batch_size"]:
            time.sleep(1)
