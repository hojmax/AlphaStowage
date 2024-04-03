from Train import (
    create_testset,
    test_network,
    save_model,
)
import torch
from Logging import log_eval
import torch.multiprocessing as mp
import numpy as np
from multiprocessing.connection import Connection
import subprocess
import time
import wandb


class Evaluator:
    def __init__(
        self,
        conn: Connection,
        gpu_update_event: mp.Event,
        config: dict,
    ) -> None:
        self.conn = conn
        self.gpu_update_event = gpu_update_event
        self.config = config
        self.test_set = create_testset(config)
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)
        self.best_avg_value = avg_value

    def update_inference_params(self, batch: int) -> None:
        model_path = f"model{batch}.pt"

        # After the first evaluation, we now allow removes in inference
        self.config["inference"]["can_only_add"] = False

        if self.config["train"]["log_wandb"]:
            wandb.save(model_path)

        subprocess.run(
            [
                "mv",
                model_path,
                "shared_model.pt",
            ]
        )

        self.gpu_update_event.set()

    def run_eval(self, batch: int) -> None:
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, self.config, batch)

        if avg_value >= self.best_avg_value:
            self.best_avg_value = avg_value
            self.update_inference_params(batch)

    def __del__(self) -> None:
        for env in self.test_set:
            env.close()


def evaluator_loop(
    conn: Connection,
    model_nr: mp.Value,
    gpu_update_event: mp.Event,
    stop_event: mp.Event,
    config: dict,
) -> None:

    evaluator = Evaluator(conn, gpu_update_event, config)
    old_model_nr = model_nr.value

    while not stop_event.is_set():
        if old_model_nr == model_nr.value:
            time.sleep(1)
            continue

        evaluator.run_eval(model_nr.value)
