from Train import (
    get_config,
    create_testset,
    play_episode,
    test_network,
    get_optimizer,
    get_scheduler,
    train_batch,
    get_env,
    save_model,
)
import torch
from NeuralNetwork import NeuralNetwork
from tqdm import tqdm
import wandb
from Node import TruncatedEpisodeError
import time
import warnings
from Buffer import ReplayBuffer
from Logging import log_batch, log_eval, log_episode, init_wandb_run, init_wandb_group
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import numpy as np
from checkpoint_config import CheckpointConfig
from self_play import SelfPlay


class Evaluator:
    def __init__(
        self, model: NeuralNetwork, update_events: list[mp.Event], config: dict
    ):
        self.model = model
        self.update_events = update_events
        self.config = config
        self.test_set = create_testset(config)
        avg_value, avg_reshuffles = test_network(self.model, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)
        self.best_avg_value = avg_value

    def eval_loop(self, stop_event: mp.Event) -> None:
        batch = 0
        while not stop_event.is_set():
            if self.should_eval(batch):
                self.run_eval(batch)
            batch += 1

    def should_eval(self, batch: int) -> bool:
        return batch % self.config["train"]["batches_before_eval"] == 0

    def update_inference_params(self) -> None:
        self.config["train"]["can_only_add"] = False

        torch.save(self.model.state_dict(), "shared_model.pt")

        for event in self.update_events:
            event.set()

    def run_eval(self, batch: int) -> None:
        avg_value, avg_reshuffles = test_network(self.model, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, self.config, batch)

        if avg_value >= self.best_avg_value:
            self.best_avg_value = avg_value
            self.update_inference_params()
            save_model(self.model, self.config, batch)

    def close(self) -> None:
        for env in self.test_set:
            env.close()


def training_loop(
    device: torch.device,
    pretrained: CheckpointConfig,
    buffer: ReplayBuffer,
    stop_event: mp.Event,
    config: dict,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = init_model(config, device, pretrained)

    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    model.train()

    while len(buffer) < config["train"]["batch_size"]:
        print("Waiting for buffer to fill up...")
        time.sleep(1)

    batches = tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1))

    for batch in batches:
        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model, buffer, optimizer, scheduler, config
        )
        current_lr = scheduler.get_last_lr()[0]
        log_batch(
            batch, avg_loss, avg_value_loss, avg_cross_entropy, current_lr, config
        )

    stop_event.set()


def get_model_weights_path(pretrained: CheckpointConfig):
    api = wandb.Api()
    run = api.run(pretrained["wandb_run"])
    file = run.file(pretrained["wandb_model"])
    file.download(replace=True)

    return pretrained["wandb_model"]


def init_model(
    config: dict, device: torch.device, pretrained: CheckpointConfig
) -> NeuralNetwork:
    model = NeuralNetwork(config).to(device)

    if pretrained["wandb_model"] and pretrained["wandb_run"]:
        model_weights_path = get_model_weights_path(pretrained)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))

    return model


def run_processes(config, pretrained):
    buffer = ReplayBuffer(config)
    stop_event = mp.Event()
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    print("GPUs:", torch.cuda.device_count(), "CPUs", mp.cpu_count())

    update_events = [mp.Event() for _ in range(config["train"]["num_workers"])]

    processes = [
        Process(
            target=training_loop,
            args=(device, pretrained, buffer, stop_event, config),
        )
    ]

    for i in range(config["train"]["num_workers"]):
        self_play_worker = SelfPlay(config, seed=i)
        self_play_worker_process = Process(
            target=self_play_worker.selfplay_loop,
            args=(stop_event, update_events[i], buffer),
        )
        processes.append(self_play_worker_process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pretrained = CheckpointConfig(wandb_run=None, wandb_model=None)
    config = get_config("config.json")
    if config["train"]["log_wandb"]:
        init_wandb_group()
    run_processes(config, pretrained)
