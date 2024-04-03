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
from Logging import init_wandb_run


class Evaluator:
    def __init__(
        self,
        conn: Connection,
        config: dict,
    ) -> None:
        self.conn = conn
        self.config = config
        self.test_set = create_testset(config)
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)

    def run_eval(self, batch: int) -> None:
        avg_value, avg_reshuffles = test_network(self.conn, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, self.config, batch)

    def __del__(self) -> None:
        for env in self.test_set:
            env.close()


def evaluator_loop(
    conn: Connection,
    model_nr: mp.Value,
    stop_event: mp.Event,
    config: dict,
) -> None:
    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    evaluator = Evaluator(conn, config)
    old_model_nr = model_nr.value

    while not stop_event.is_set():
        if old_model_nr == model_nr.value:
            time.sleep(1)
            continue

        evaluator.run_eval(model_nr.value)
