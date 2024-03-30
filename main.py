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


def update_inference_params(
    model: NeuralNetwork, inference_models: list, config: dict
) -> None:
    config["train"]["use_baseline_policy"] = False
    for inference_model in inference_models:
        inference_model.load_state_dict(model.state_dict())


def save_model(model: NeuralNetwork, config: dict, i: int) -> None:
    model_path = f"model{i}.pt"
    torch.save(model.state_dict(), model_path)

    if config["train"]["log_wandb"]:
        wandb.save(model_path)


def inference_function(
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
            warnings.warn("Episode was truncated.")
            env.close()
            continue

        for bay, flat_T, prob, value in observations:
            buffer.extend(bay, flat_T, prob, value)

        log_episode(buffer.increment_episode(), final_value, final_reshuffles, config)


def training_function(
    model: NeuralNetwork,
    inference_models: list[NeuralNetwork],
    buffer: ReplayBuffer,
    stop_event: threading.Event,
    config: dict,
) -> None:
    test_set = create_testset(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    best_model_score, initial_reshuffles = test_network(
        model, test_set, config, model.device
    )
    log_eval(best_model_score, initial_reshuffles, 0, config)

    while len(buffer) < config["train"]["batch_size"]:
        print("Waiting for buffer to fill up")
        time.sleep(1)

    model.train()
    for i in tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1)):
        if i % config["train"]["batches_before_eval"] == 0:
            avg_error, avg_reshuffles = test_network(
                model, test_set, config, model.device
            )
            log_eval(avg_error, avg_reshuffles, i, config)
            if avg_error > best_model_score:
                best_model_score = avg_error
                update_inference_params(model, inference_models, config)
                save_model(model, config, i)

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model,
            buffer,
            config["train"]["batch_size"],
            optimizer,
            scheduler,
            config["train"]["value_scaling"],
            model.device,
        )
        current_lr = scheduler.get_last_lr()[0]
        log_batch(i, avg_loss, avg_value_loss, avg_cross_entropy, current_lr, config)

    for env in test_set:
        env.close()

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
    """Creates N threads. First thread is for training the model, the rest are for inference."""
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
            target=training_function,
            args=(models[0], models[1:], buffer, stop_event, config),
        )
    ] + [
        threading.Thread(
            target=inference_function,
            args=(model, buffer, stop_event, config),
        )
        for model in models[1:]
    ]


def start_threads(threads: list[threading.Thread]) -> None:
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
    config = get_config()

    if config["train"]["log_wandb"]:
        init_wandb(config)

    threads = create_threads(config, pretrained)
    start_threads(threads)
