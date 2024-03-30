import threading
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
from Logging import log_batch, log_eval, log_episode


class PretrainedModel(TypedDict):
    """Optional way of specifying models from previous runs (to continue training, testing etc.)
    Example:
    wandb_run: "alphastowage/AlphaStowage/camwudzo"
    wandb_model: "model20000.pt"
    """

    wandb_run: str = None
    wandb_model: str = None


class Evaluator:
    def __init__(
        self, model: NeuralNetwork, inference_models: list[NeuralNetwork], config: dict
    ):
        self.model = model
        self.inference_models = inference_models
        self.config = config
        self.test_set = create_testset(config)
        avg_value, avg_reshuffles = test_network(self.model, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)
        self.best_avg_value = avg_value

    def should_eval(self, batch: int) -> bool:
        return batch % self.config["train"]["batches_before_eval"] == 0

    def update_inference_params(self) -> None:
        for inference_model in self.inference_models:
            inference_model.load_state_dict(self.model.state_dict())

    def run_eval(self, batch: int) -> None:
        avg_value, avg_reshuffles = test_network(self.model, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch)

        if avg_value > self.best_avg_value:
            self.best_avg_value = avg_value
            self.update_inference_params()
            save_model(self.model, config, batch)

    def close(self) -> None:
        for env in self.test_set:
            env.close()


def inference_loop(
    model: NeuralNetwork,
    buffer: ReplayBuffer,
    stop_event: threading.Event,
    config: dict,
) -> None:
    model.eval()

    while not stop_event.is_set():
        env = get_env(config)
        env.reset()

        try:
            observations, final_value, final_reshuffles = play_episode(
                env, model, config, model.device, deterministic=False
            )
            env.close()
        except TruncatedEpisodeError:
            warnings.warn("Episode was truncated in training.")
            env.close()
            continue

        for bay, flat_T, prob, value in observations:
            buffer.extend(bay, flat_T, prob, value)

        log_episode(buffer.increment_episode(), final_value, final_reshuffles, config)


def training_loop(
    model: NeuralNetwork,
    inference_models: list[NeuralNetwork],
    buffer: ReplayBuffer,
    stop_event: threading.Event,
    config: dict,
) -> None:
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    evaluator = Evaluator(model, inference_models, config)
    model.train()

    while len(buffer) == 0:
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


def create_threads(config: dict, pretrained: PretrainedModel) -> list[threading.Thread]:
    """Creates N threads for N available GPUs. First thread is for training the model, the rest are for inference."""
    buffer = ReplayBuffer(config)
    stop_event = threading.Event()
    devices = (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu", "cpu"]  # For local testing
    )
    models = [init_model(config, device, pretrained) for device in devices]

    return [
        threading.Thread(
            target=training_loop,
            args=(models[0], models[1:], buffer, stop_event, config),
        )
    ] + [
        threading.Thread(
            target=inference_loop,
            args=(model, buffer, stop_event, config),
        )
        for model in models[1:]
    ]


def run_threads(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def init_wandb(config: dict) -> None:
    wandb.init(
        entity="alphastowage", project="AlphaStowage", config=config, save_code=True
    )


if __name__ == "__main__":
    pretrained = PretrainedModel(wandb_model=None, wandb_run=None)
    config = get_config("config.json")

    if config["train"]["log_wandb"]:
        init_wandb(config)

    threads = create_threads(config, pretrained)
    run_threads(threads)
