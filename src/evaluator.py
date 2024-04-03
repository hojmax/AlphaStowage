from train import create_testset, test_network
import torch
from network import NeuralNetwork
from logger import log_eval
import ray
from shared_storage import SharedStorage
import numpy as np
import copy


@ray.remote
class Evaluator:
    def __init__(self, config: dict, checkpoint: dict, seed: int = 0):
        self.config = config
        self.test_set = create_testset(config)

        # Initialize the model
        self.model = NeuralNetwork(config)
        self.model.set_weights(checkpoint["weights"])
        self.model.to("cpu")
        self.model.eval()

        # Fix seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize the best average value
        avg_value, avg_reshuffles = test_network(self.model, self.test_set, self.config)
        log_eval(avg_value, avg_reshuffles, config, batch=0)
        self.best_avg_value = avg_value

    def eval_loop(self, shared_storage: SharedStorage) -> None:
        training_step = ray.get(shared_storage.get_info.remote("training_step"))
        while training_step < self.config["train"]["training_steps"] and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            # Load the weights
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            avg_value, avg_reshuffles = test_network(
                self.model, self.test_set, self.config
            )
            log_eval(avg_value, avg_reshuffles, self.config, training_step)

            if avg_value >= self.best_avg_value:
                self.best_avg_value = avg_value
                shared_storage.set_info.remote(
                    "best_weights", copy.deepcopy(self.model.get_weights())
                )

    def __del__(self) -> None:
        for env in self.test_set:
            env.close()
