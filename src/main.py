from InferenceLoggerProcess import InferenceLoggerProcess
from InferenceProcess import InferenceProcess
from TrainingProcess import TrainingProcess
from GPUProcess import GPUProcess
from Logging import init_wandb_group
import torch.multiprocessing as mp
from Train import PretrainedModel
from Buffer import ReplayBuffer
import torch
import json


def start_process_loop(process_class, *args, **kwargs):
    process = process_class(*args, **kwargs)
    process.loop()


def get_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)

    return config


def run_processes(config: dict, pretrained: PretrainedModel):
    buffer = ReplayBuffer(config)
    n_gpus = torch.cuda.device_count()
    training_device = "cuda:0" if n_gpus >= 1 else "mps"
    gpu_device = "cuda:1" if n_gpus >= 2 else ("cuda:0" if n_gpus >= 1 else "mps")
    gpu_update_event = mp.Event()
    inference_pipes = [mp.Pipe() for _ in range(config["inference"]["n_processes"])]
    episode_queue = mp.Queue()

    processes = [
        mp.Process(
            target=start_process_loop,
            args=(
                TrainingProcess,
                buffer,
                gpu_update_event,
                training_device,
                pretrained,
                config,
            ),
        ),
        mp.Process(
            target=start_process_loop,
            args=(
                GPUProcess,
                inference_pipes,
                gpu_update_event,
                gpu_device,
                pretrained,
                config,
            ),
        ),
        mp.Process(
            target=start_process_loop,
            args=(
                InferenceLoggerProcess,
                episode_queue,
                config,
            ),
        ),
    ] + [
        mp.Process(
            target=start_process_loop,
            args=(
                InferenceProcess,
                seed,
                buffer,
                conn,
                episode_queue,
                config,
            ),
        )
        for seed, (_, conn) in enumerate(inference_pipes)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = get_config(
        "config.json" if torch.cuda.is_available() else "local_config.json"
    )
    pretrained = PretrainedModel(
        wandb_run=config["wandb"]["pretrained_run"],
        wandb_model=config["wandb"]["pretrained_model"],
        artifact=config["wandb"]["artifact"],
    )

    if config["wandb"]["should_log"]:
        init_wandb_group()

    run_processes(config, pretrained)
