import torch
import time
from replay_buffer import ReplayBuffer
from logger import log_step, init_wandb_run
import numpy as np
from network import NeuralNetwork
import copy
from shared_storage import SharedStorage
import ray
import logging


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, config: dict, checkpoint: dict):
        self.config = config
        self.checkpoint = checkpoint

        # Fix random generator seed
        np.random.seed(self.config["train"]["seed"])
        torch.manual_seed(self.config["train"]["seed"])

        # Initialize the model
        self.model = NeuralNetwork(self.config)
        self.model.set_weights(copy.deepcopy(self.checkpoint["weights"]))
        self.device = "cuda:0" if torch.cuda.is_available() else "mps"
        self.model.to(self.device)
        self.model.train()

        self.training_step = 0

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["l2_weight_reg"],
        )

        # Initializer scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            config["train"]["scheduler_step_size_in_batches"],
            config["train"]["scheduler_gamma"],
        )

    def training_loop(
        self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer
    ) -> None:

        if self.config["train"]["log_wandb"]:
            init_wandb_run(self.config)

        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            logging.info("Waiting for games to be played before training...")
            time.sleep(1)

        while self.training_step < self.config["train"][
            "training_steps"
        ] and not ray.get(shared_storage.get_info.remote("terminate")):
            batch = ray.get(
                replay_buffer.sample.remote(self.config["train"]["batch_size"])
            )

            avg_loss, avg_value_loss, avg_cross_entropy = self.train_batch(batch)
            current_lr = self.scheduler.get_last_lr()[0]

            shared_storage.set_info.remote("training_step", self.training_step)
            if self.training_step % self.config["train"]["checkpoint_interval"] == 0:
                log_step(
                    self.training_step,
                    avg_loss,
                    avg_value_loss,
                    avg_cross_entropy,
                    current_lr,
                    self.config,
                )
                shared_storage.set_info.remote(
                    "weights", copy.deepcopy(self.model.get_weights())
                )

    def train_batch(self, batch):

        # Batch is a list of tuples, convert it to a tensor
        bay, flat_T, prob, value = zip(*batch)

        bay = torch.stack(bay).squeeze(2)
        flat_T = torch.stack(flat_T)
        prob = torch.stack(prob)
        value = torch.stack(value)

        bay = bay.to(self.device)
        flat_T = flat_T.to(self.device)
        prob = prob.to(self.device)
        value = value.to(self.device)

        pred_prob, pred_value = self.model(bay, flat_T)
        loss, value_loss, cross_entropy = self.calculate_loss(
            pred_value=pred_value,
            value=value,
            pred_prob=pred_prob,
            prob=prob,
        )
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        self.training_step += self.config["train"]["batch_size"]

        return loss.item(), value_loss.item(), cross_entropy.item()

    def calculate_loss(self, pred_value, value, pred_prob, prob) -> tuple:
        value_error = torch.mean(torch.square(value - pred_value))
        cross_entropy = (
            -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
        )
        loss = self.config["train"]["value_scaling"] * value_error + cross_entropy
        return loss, value_error, cross_entropy
