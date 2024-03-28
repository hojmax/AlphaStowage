import threading
from Train import (
    get_config,
    create_testset,
    play_episode,
    test_network,
    get_optimizer,
    get_scheduler,
    train_batch,
)
from test_model import transform_benchmarking_data, get_benchmarking_data
import torch
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from tqdm import tqdm
import numpy as np
import wandb
from Node import TruncatedEpisodeError
import time
import random

log_wandb = True


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.episode = 0

    def extend(self, items):
        with self.lock:
            self.buffer.extend(items)
            self.buffer = self.buffer[-self.capacity :]
            self.episode += 1
            return self.episode

    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)

        with self.lock:
            batch_indices = np.random.choice(
                len(self.buffer), batch_size, replace=False
            )
            batch = [self.buffer[i] for i in batch_indices]
            bay_batch = torch.stack([bay.squeeze(0) for (bay, _, _), _, _ in batch])
            flat_T_batch = torch.stack(
                [flat_T.squeeze(0) for (_, flat_T, _), _, _ in batch]
            )
            mask_batch = torch.stack([mask.squeeze(0) for (_, _, mask), _, _ in batch])
            prob_batch = torch.stack([probs for _, probs, _ in batch])
            value_batch = torch.stack(
                [torch.tensor([value], dtype=torch.float32) for _, _, value in batch]
            )
            return bay_batch, flat_T_batch, mask_batch, prob_batch, value_batch

    def __len__(self):
        return len(self.buffer)


def inference_function(model, device, buffer, stop_event):
    model.eval()

    while not stop_event.is_set():
        # if (
        #     config["use_baseline_policy"]
        #     and len(buffer.buffer) >= config["train"]["max_data"]
        # ):
        #     time.sleep(1)
        #     continue
        env = Env(
            # config["env"]["R"],
            # config["env"]["C"],
            # config["env"]["N"],
            random.choice(range(6, config["env"]["R"] + 1, 2)),
            random.choice(range(2, config["env"]["C"] + 1, 2)),
            random.choice(range(4, config["env"]["N"] + 1, 2)),
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset()
        try:
            episode_data, value, reshuffles = play_episode(
                env, model, config, device, deterministic=False
            )
            episode = buffer.extend(episode_data)
        except TruncatedEpisodeError:
            pass
        env.close()
        if log_wandb:
            wandb.log({"episode": episode, "value": value, "reshuffles": reshuffles})


def update_inference_params(model, inference_models, config):
    config["use_baseline_policy"] = False
    for inference_model in inference_models:
        inference_model.load_state_dict(model.state_dict())


def log_eval(avg_error, avg_reshuffles, i):
    wandb.log(
        {
            "eval_moves": avg_error,
            "eval_reshuffles": avg_reshuffles,
            "batch": i,
        }
    )


def training_function(model, device, inference_models, buffer, stop_event):
    # test_set = create_testset(config)
    combinations = [
        (6, 2, 6),
        (6, 4, 8),
    ]

    test_set = get_benchmarking_data("benchmark/set_2")
    test_set = [
        e
        for combination in combinations
        for e in test_set
        if e["N"] == combination[2]
        and e["R"] == combination[0]
        and e["C"] == combination[1]
    ]
    test_set = transform_benchmarking_data(test_set)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    best_model_score, initial_reshuffles = test_network(model, test_set, config, device)
    if log_wandb:
        wandb.log(
            {
                "eval_moves": best_model_score,
                "eval_reshuffles": initial_reshuffles,
                "batch": 0,
            }
        )

    while len(buffer) < config["train"]["batch_size"]:
        print("Waiting for buffer to fill up")
        time.sleep(1)

    model.train()
    for i in tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1)):
        if i % config["train"]["batches_before_eval"] == 0:
            avg_error, avg_reshuffles = test_network(model, test_set, config, device)
            if log_wandb:
                log_eval(avg_error, avg_reshuffles, i)
            else:
                print(f"Eval moves: {avg_error}, Eval reshuffles: {avg_reshuffles}")
            if avg_error > best_model_score:
                best_model_score = avg_error
                update_inference_params(model, inference_models, config)
                if log_wandb:
                    torch.save(model.state_dict(), f"model{i}.pt")
                    wandb.save(f"model{i}.pt")

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model,
            buffer,
            config["train"]["batch_size"],
            optimizer,
            scheduler,
            config["train"]["value_scaling"],
            device,
        )
        if log_wandb:
            wandb.log(
                {
                    "loss": avg_loss,
                    "value_loss": avg_value_loss,
                    "cross_entropy": avg_cross_entropy,
                    "batch": i,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        else:
            print(
                f"Batch: {i}, Loss: {avg_loss}, Value Loss: {avg_value_loss}, Cross Entropy: {avg_cross_entropy}, LR: {optimizer.param_groups[0]['lr']}"
            )

    for env in test_set:
        env.close()

    stop_event.set()


def get_model_weights_path(wandb_run, wandb_model):
    api = wandb.Api()
    run = api.run(wandb_run)
    file = run.file(wandb_model)
    file.download(replace=True)

    return wandb_model


if __name__ == "__main__":
    # use_prev_model = {
    #     "wandb_run": "alphastowage/AlphaStowage/camwudzo",
    #     "wandb_model": "model20000.pt",
    # }
    use_prev_model = None
    config = get_config()
    buffer = ReplayBuffer(config["train"]["max_data"])
    stop_event = threading.Event()
    training_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    inference_device1 = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    inference_device2 = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"

    training_model = NeuralNetwork(config).to(training_device)
    inference_model1 = NeuralNetwork(config).to(inference_device1)
    inference_model2 = NeuralNetwork(config).to(inference_device2)

    if use_prev_model:
        # config["use_baseline_policy"] = False
        model_weights_path = get_model_weights_path(
            use_prev_model["wandb_run"], use_prev_model["wandb_model"]
        )
        training_model.load_state_dict(
            torch.load(model_weights_path, map_location=training_device)
        )
        inference_model1.load_state_dict(
            torch.load(model_weights_path, map_location=inference_device1)
        )
        inference_model2.load_state_dict(
            torch.load(model_weights_path, map_location=inference_device2)
        )

    if log_wandb:
        wandb.init(
            entity="alphastowage", project="AlphaStowage", config=config, save_code=True
        )

    inference_thread1 = threading.Thread(
        target=inference_function,
        args=(inference_model1, inference_device1, buffer, stop_event),
    )
    inference_thread2 = threading.Thread(
        target=inference_function,
        args=(inference_model2, inference_device2, buffer, stop_event),
    )

    training_thread = threading.Thread(
        target=training_function,
        args=(
            training_model,
            training_device,
            [inference_model1, inference_model2],
            buffer,
            stop_event,
        ),
    )

    inference_thread1.start()
    inference_thread2.start()
    training_thread.start()

    training_thread.join()
    inference_thread2.join()
    inference_thread1.join()

    if log_wandb:
        torch.save(training_model.state_dict(), "final_model.pt")
        wandb.save("final_model.pt")
