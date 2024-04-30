from train import create_testset
import torch
from network import NeuralNetwork
import wandb
from replay_buffer import ReplayBuffer
import numpy as np
from self_play import SelfPlay
from shared_storage import SharedStorage
from evaluator import Evaluator
import ray
import copy
from trainer import Trainer
import time
import uuid


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    """Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU"""

    def get_initial_weights(self, config):
        return NeuralNetwork(config).get_weights()


class AlphaZero:
    def __init__(self, config: dict):
        self.config = config
        self.test_set = create_testset(config)

        # Fix seed
        np.random.seed(self.config["train"]["seed"])
        torch.manual_seed(self.config["train"]["seed"])

        # Initialize checkpoint
        self.checkpoint = {
            "training_step": 0,
            "num_played_games": 0,
            "steps_played": 0,
            "terminate": False,
        }
        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"] = copy.deepcopy(ray.get(cpu_weights))
        self.checkpoint["best_weights"] = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, num_workers: int) -> None:

        id = str(uuid.uuid4())

        # Initialize workers
        self.training_worker = Trainer.options(num_cpus=1, num_gpus=0).remote(
            self.config, self.checkpoint
        )

        self.shared_storage_worker = SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(self.config)

        self._launch_self_play_workers(num_workers=num_workers, id=id)

        self.training_worker.training_loop.remote(
            self.shared_storage_worker, self.replay_buffer_worker, id
        )

        self.test_worker = Evaluator.options(num_cpus=1, num_gpus=0).remote(
            self.config, self.checkpoint, self.config["train"]["seed"]
        )
        self.test_worker.eval_loop.remote(self.shared_storage_worker, id)

        self.logging_loop()

        # Send terminate signal
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )

    def self_play(self) -> None:

        id = str(uuid.uuid4())

        self._launch_self_play_workers(
            num_workers=self.config["train"]["num_workers"], id=id
        )

        keys = ["training_step", "terminate"]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while (
                info["training_step"] < self.config["train"]["training_steps"]
                and not info["terminate"]
            ):
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                time.sleep(5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

    def _launch_self_play_workers(self, num_workers: int, id: str):
        self.self_play_workers = [
            SelfPlay.options(num_cpus=1, num_gpus=0).remote(
                self.config, self.checkpoint, seed=i
            )
            for i in range(num_workers)
        ]

        [
            self_play.self_play_loop.remote(
                self.shared_storage_worker, self.replay_buffer_worker, id
            )
            for self_play in self.self_play_workers
        ]

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def logging_loop(self) -> None:
        keys = ["num_played_games", "training_step", "steps_played"]

        info = ray.get(self.shared_storage_worker.get_info.remote(keys))

        try:
            while info["training_step"] < self.config["train"]["training_steps"]:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                # Calculate steps per second
                steps = info["steps_played"] or 0
                print(
                    f"Step {info['training_step']}/{self.config['train']['training_steps']}, "
                    f"Games {info['num_played_games']}, "
                    f"Steps played {steps}, "
                )

                time.sleep(5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

    def load_checkpoint(self, wandb_run: str, wandb_model: str) -> None:
        api = wandb.Api()
        run = api.run(wandb_run)
        file = run.file(wandb_model)
        file.download(replace=True)
        self.checkpoint = torch.load(wandb_model, map_location="cpu")
