from Train import (
    get_config,
    play_episode,
    get_optimizer,
    get_scheduler,
    train_batch,
    get_env,
    init_model,
    PretrainedModel,
)
import torch
from NeuralNetwork import NeuralNetwork
import wandb
from Node import TruncatedEpisodeError
import time
import warnings
from Buffer import ReplayBuffer
from Logging import (
    logging_process,
    init_wandb_run,
    init_wandb_group,
    BatchLogger,
)
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import numpy as np
from multiprocessing.connection import Connection
from GPUProcess import gpu_process


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

        del observations, final_value, final_reshuffles


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
    batch_logger = BatchLogger(config)

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

        loss, value_loss, cross_entropy = train_batch(
            model, buffer, optimizer, scheduler, config
        )
        current_lr = scheduler.get_last_lr()[0]
        batch_logger.log(batch, loss, value_loss, cross_entropy, current_lr, config)
        batch += 1


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
                pretrained,
                gpu_device,
                gpu_update_event,
                config,
                inference_pipes,
            ),
        ),
        Process(
            target=logging_process,
            args=(
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
    mp.set_sharing_strategy("file_system")
    pretrained = PretrainedModel(
        wandb_run="hojmax/AlphaStowage/0mh7t6hv", wandb_model="model504000.pt"
    )
    config = get_config(
        "config.json" if torch.cuda.is_available() else "local_config.json"
    )

    if config["train"]["log_wandb"]:
        init_wandb_group()

    run_processes(config, pretrained)
