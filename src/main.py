from Train import (
    get_config,
    play_episode,
    get_optimizer,
    get_scheduler,
    train_batch,
    get_env,
)
import torch
from NeuralNetwork import NeuralNetwork
import wandb
from Node import TruncatedEpisodeError
import time
from typing import TypedDict
import warnings
from Buffer import ReplayBuffer
from Logging import log_batch, logging_process, init_wandb_run, init_wandb_group
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


def inference_loop(
    id: int,
    buffer: ReplayBuffer,
    conn: Connection,
    episode_queue: mp.Queue,
    config: dict,
) -> None:
    torch.manual_seed(id)
    np.random.seed(id)

    i = 0
    while True:

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

        episode_queue.put((final_value, final_reshuffles))


def swap_over(
    batch: int,
    model: NeuralNetwork,
    config: dict,
    gpu_update_event: mp.Event,
) -> None:
    model_path = f"model{batch}.pt"
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), f"shared_model.pt")

    if config["train"]["log_wandb"]:
        wandb.save(model_path)

    config["inference"]["can_only_add"] = False
    gpu_update_event.set()


def training_loop(
    device: torch.device,
    pretrained: PretrainedModel,
    buffer: ReplayBuffer,
    gpu_update_event: mp.Event,
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

    batch = 1
    while True:
        if batch % config["eval"]["batch_interval"] == 0:
            swap_over(batch, model, config, gpu_update_event)

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model, buffer, optimizer, scheduler, config
        )
        current_lr = scheduler.get_last_lr()[0]
        log_batch(
            batch, avg_loss, avg_value_loss, avg_cross_entropy, current_lr, config
        )
        batch += 1


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
    training_device = "cuda:0" if torch.cuda.is_available() else "mps"
    gpu_device = "cuda:1" if torch.cuda.is_available() else "mps"
    gpu_update_event = mp.Event()
    inference_pipes = [mp.Pipe() for _ in range(config["inference"]["n_processes"])]
    episode_queue = mp.Queue()

    processes = [
        Process(
            target=training_loop,
            args=(
                training_device,
                pretrained,
                buffer,
                gpu_update_event,
                config,
            ),
        ),
        Process(
            target=gpu_process,
            args=(
                gpu_device,
                gpu_update_event,
                config,
                inference_pipes,
            ),
        ),
        Process(
            target=logging_process,
            args=(
                buffer,
                episode_queue,
                config,
            ),
        ),
    ] + [
        Process(
            target=inference_loop,
            args=(
                id,
                buffer,
                conn,
                episode_queue,
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
    config = get_config(
        "config.json" if torch.cuda.is_available() else "local_config.json"
    )

    if config["train"]["log_wandb"]:
        init_wandb_group()

    run_processes(config, pretrained)
