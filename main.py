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
import torch
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from tqdm import tqdm
import numpy as np
import wandb
from Node import TruncatedEpisodeError
import time
import logging


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()

    def extend(self, items):
        with self.lock:
            self.buffer.extend(items)
            self.buffer = self.buffer[-self.capacity :]

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
    i = 1
    while not stop_event.is_set():
        if (
            config["use_baseline_policy"]
            and len(buffer.buffer) >= config["train"]["max_data"]
        ):
            time.sleep(1)
            continue
        env = Env(
            config["env"]["R"],
            config["env"]["C"],
            config["env"]["N"],
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset()
        try:
            episode_data, value = play_episode(
                env, model, config, device, deterministic=False
            )
            buffer.extend(episode_data)
        except TruncatedEpisodeError:
            pass
        env.close()
        wandb.log({"episode": i, "value": value})
        i += 1


def update_inference_params(model, inference_model, config):
    config["use_baseline_policy"] = False
    inference_model.load_state_dict(model.state_dict())


def log_model(model, test_set, config, device, i):
    wandb.log(
        {
            "eval_score": test_network(model, test_set, config, device),
            "batch": i,
        }
    )
    torch.save(model.state_dict(), f"model{i}.pt")
    wandb.save(f"model{i}.pt")


def training_function(model, device, inference_model, buffer, stop_event):
    while len(buffer.buffer) == 0:
        time.sleep(1)

    test_set = create_testset(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    model.train()
    for i in tqdm(range(1, int(config["train"]["train_for_n_batches"]) + 1)):
        if i % config["train"]["batches_before_swap"] == 0:
            update_inference_params(model, inference_model, config)
        if i % config["train"]["batches_before_eval"] == 0:
            log_model(model, test_set, config, device, i)

        avg_loss, avg_value_loss, avg_cross_entropy = train_batch(
            model,
            buffer,
            config["train"]["batch_size"],
            optimizer,
            scheduler,
            config["train"]["value_scaling"],
            device,
        )
        wandb.log(
            {
                "loss": avg_loss,
                "value_loss": avg_value_loss,
                "cross_entropy": avg_cross_entropy,
                "batch": i,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    for env in test_set:
        env.close()

    stop_event.set()


if __name__ == "__main__":
    logging.basicConfig(filename="main.log", level=logging.INFO)
    config = get_config()
    buffer = ReplayBuffer(config["train"]["max_data"])
    stop_event = threading.Event()
    training_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    inference_device1 = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    inference_device2 = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"

    training_model = NeuralNetwork(config).to(training_device)
    inference_model = NeuralNetwork(config).to(inference_device1)

    inference_thread1 = threading.Thread(
        target=inference_function,
        args=(inference_model, inference_device1, buffer, stop_event),
    )
    inference_thread2 = threading.Thread(
        target=inference_function,
        args=(inference_model, inference_device2, buffer, stop_event),
    )

    training_thread = threading.Thread(
        target=training_function,
        args=(
            training_model,
            training_device,
            inference_model,
            buffer,
            stop_event,
        ),
    )
    wandb.init(project="multi-thread", config=config)

    inference_thread1.start()
    inference_thread2.start()
    training_thread.start()

    training_thread.join()
    inference_thread2.join()
    inference_thread1.join()

    torch.save(training_model.state_dict(), "final_model.pt")
    wandb.save("final_model.pt")
