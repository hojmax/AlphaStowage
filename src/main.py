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
from Evaluator import evaluator_loop


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


def swap_over(
    model_nr: mp.Value,
    batch: int,
    model: NeuralNetwork,
    config: dict,
    gpu_update_event: mp.Event,
) -> None:
    model_path = f"model{batch}.pt"
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), f"shared_model.pt")
    model_nr.value = batch

    if config["train"]["log_wandb"]:
        wandb.save(model_path)

    config["inference"]["can_only_add"] = False
    gpu_update_event.set()


def training_loop(
    device: torch.device,
    pretrained: PretrainedModel,
    buffer: ReplayBuffer,
    stop_event: mp.Event,
    model_nr: mp.Value,
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

    batches = tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1))

    for batch in batches:
        if batch % config["eval"]["batch_interval"] == 0:
            swap_over(model_nr, batch, model, config, gpu_update_event)

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model, buffer, optimizer, scheduler, config
        )
        current_lr = scheduler.get_last_lr()[0]
        log_batch(
            batch, avg_loss, avg_value_loss, avg_cross_entropy, current_lr, config
        )

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
    training_device = "cuda:0" if torch.cuda.is_available() else "mps"
    gpu_device = "cuda:1" if torch.cuda.is_available() else "mps"
    model_nr = mp.Value("i", 0)
    gpu_update_event = mp.Event()
    evaluation_pipe = mp.Pipe()
    inference_pipes = [mp.Pipe() for _ in range(config["inference"]["n_processes"])]

    processes = [
        Process(
            target=training_loop,
            args=(
                training_device,
                pretrained,
                buffer,
                stop_event,
                model_nr,
                gpu_update_event,
                config,
            ),
        ),
        Process(
            target=evaluator_loop,
            args=(
                evaluation_pipe[1],
                model_nr,
                stop_event,
                config,
            ),
        ),
        Process(
            target=gpu_process,
            args=(
                gpu_device,
                stop_event,
                gpu_update_event,
                config,
                inference_pipes + [evaluation_pipe],
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
    config = get_config(
        "config.json" if torch.cuda.is_available() else "local_config.json"
    )
    if config["train"]["log_wandb"]:
        init_wandb_group()
    run_processes(config, pretrained)
