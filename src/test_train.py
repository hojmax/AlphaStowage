from InferenceControllerProcess import InferenceControllerProcess
from TrainingProcess import TrainingProcess
from multiprocessing import Array
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
    training_device = "cuda:0" if torch.cuda.is_available() else "mps"
    template_event = mp.Event()

    process = mp.Process(
        target=start_process_loop,
        args=(
            TrainingProcess,
            buffer,
            template_event,
            training_device,
            pretrained,
            config,
        ),
    )

    process.start()
    process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = get_config(
        "config.json" if torch.cuda.is_available() else "local_config.json"
    )
    config["train"]["swap_interval"] = int(1e9)

    pretrained = PretrainedModel(
        wandb_run=config["wandb"]["pretrained_run"],
        wandb_model=config["wandb"]["pretrained_model"],
        artifact=config["wandb"]["artifact"],
        local_model=config["wandb"].get("local_model"),
    )

    if config["wandb"]["should_log"]:
        init_wandb_group()

    run_processes(config, pretrained)
