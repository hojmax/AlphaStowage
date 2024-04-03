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
from typing import TypedDict
import warnings
from Buffer import ReplayBuffer
from Logging import log_batch, log_eval, log_episode, init_wandb_run, init_wandb_group
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import numpy as np
from multiprocessing.connection import Connection
from GPUProcess import gpu_process


class PretrainedModel(TypedDict):
    """Optional way of specifying models from previous runs (to continue training, testing etc.)
    Example:
    wandb_run: "alphastowage/AlphaStowage/camwudzo"
    wandb_model: "model20000.pt"
    """

    wandb_run: str = None
    wandb_model: str = None


class Evaluator:
    def __init__(self, conn: Connection, update_events: list[mp.Event], config: dict):
        self.conn = conn
        self.update_events = update_events
        self.config = config
        self.test_set = create_testset(config)
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)
        self.best_avg_value = avg_value

    def should_eval(self, batch: int) -> bool:
        return batch % self.config["train"]["batches_before_eval"] == 0

    def update_inference_params(self) -> None:
        self.config["train"]["can_only_add"] = False

        torch.save(self.model.state_dict(), "shared_model.pt")

        for event in self.update_events:
            event.set()

    def run_eval(self, batch: int) -> None:
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, self.config, batch)

        if avg_value >= self.best_avg_value:
            self.best_avg_value = avg_value
            self.update_inference_params()
            save_model(self.model, self.config, batch)

    def close(self) -> None:
        for env in self.test_set:
            env.close()


def inference_loop(
    id: int,
    buffer: ReplayBuffer,
    stop_event: mp.Event,
    conn: Connection,
    config: dict,
) -> None:
    torch.manual_seed(id)
    np.random.seed(id)

    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    avg_value = 0
    avg_reshuffles = 0
    avg_over = 2
    i = 0
    while not stop_event.is_set():

        env = get_env(config)
        env.reset(np.random.randint(1e9))

        try:
            observations, final_value, final_reshuffles = play_episode(
                env, conn, config, deterministic=False
            )
            i += 1
            env.close()
        except TruncatedEpisodeError:
            warnings.warn("Episode was truncated in training.")
            env.close()
            continue

        for bay, flat_T, prob, value in observations:
            buffer.extend(bay, flat_T, prob, value)

        avg_value += final_value / avg_over
        avg_reshuffles += final_reshuffles / avg_over

        if i % avg_over == 0:
            log_episode(
                buffer.increment_episode(),
                avg_value,
                avg_reshuffles,
                config,
            )
            avg_value = 0
            avg_reshuffles = 0
        else:
            buffer.increment_episode()


def training_loop(
    device: torch.device,
    pretrained: PretrainedModel,
    buffer: ReplayBuffer,
    stop_event: mp.Event,
    update_events: list[mp.Event],
    conn: Connection,
    config: dict,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = init_model(config, device, pretrained)

    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    evaluator = Evaluator(conn, update_events, config)
    model.train()

    while len(buffer) < config["train"]["batch_size"]:
        print("Waiting for buffer to fill up...")
        time.sleep(1)

    batches = tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1))

    for batch in batches:
        if evaluator.should_eval(batch):
            evaluator.run_eval(batch)

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model, buffer, optimizer, scheduler, config
        )
        current_lr = scheduler.get_last_lr()[0]
        log_batch(
            batch, avg_loss, avg_value_loss, avg_cross_entropy, current_lr, config
        )

    evaluator.close()
    stop_event.set()


def get_model_weights_path(pretrained: PretrainedModel):
    api = wandb.Api()
    run = api.run(pretrained["wandb_run"])
    file = run.file(pretrained["wandb_model"])
    file.download(replace=True)

    return pretrained["wandb_model"]


def init_model(
    config: dict, device: torch.device, pretrained: PretrainedModel
) -> NeuralNetwork:
    model = NeuralNetwork(config, device).to(device)

    if pretrained["wandb_model"] and pretrained["wandb_run"]:
        model_weights_path = get_model_weights_path(pretrained)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))

    return model


def run_processes(config, pretrained):
    buffer = ReplayBuffer(config)
    stop_event = mp.Event()
    # training_device = "mps"
    # gpu_device = "mps"
    # gpu_update_event = mp.Event()
    # training_pipe = mp.Pipe()
    # inference_pipes = [mp.Pipe() for _ in range(16)]
    inference_processes = 60
    # training_device = "cuda:0"
    gpu_device = "cuda:0"
    gpu_update_event = mp.Event()
    training_pipe = mp.Pipe()
    inference_pipes = [mp.Pipe() for _ in range(inference_processes)]

    processes = [
        # Process(
        #     target=training_loop,
        #     args=(
        #         training_device,
        #         pretrained,
        #         buffer,
        #         stop_event,
        #         gpu_update_event,
        #         training_pipe[1],
        #         config,
        #     ),
        # ),
        Process(
            target=gpu_process,
            args=(
                gpu_device,
                stop_event,
                gpu_update_event,
                config,
                inference_pipes + [training_pipe],
            ),
        ),
    ] + [
        Process(
            target=inference_loop,
            args=(
                id + 1,
                buffer,
                stop_event,
                conn,
                config,
            ),
        )
        for id, (_, conn) in enumerate(inference_pipes)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pretrained = PretrainedModel(wandb_run=None, wandb_model=None)
    config = get_config("config.json")
    if config["train"]["log_wandb"]:
        init_wandb_group()
    run_processes(config, pretrained)
