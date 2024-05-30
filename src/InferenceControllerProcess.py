import time
from Logging import init_wandb_run
from StepLogger import StepLogger
from multiprocessing import Queue
import torch.multiprocessing as mp
import time


class InferenceControllerProcess:
    """Collects episode data from inference processes and logs it.
    The episode data includes: value, reshuffles, seconds/episode and removes/episode.
    """

    def __init__(self, queue: Queue, config: dict, current_env_size: mp.Array):
        self.queue = queue
        self.config = config
        self.current_env_size = current_env_size
        self.sum_reshuffles = 0
        self.count = 0
        self.previous_avg = None

        if self.config["wandb"]["should_log"]:
            init_wandb_run(self.config)

        self.loggers = {}

    def loop(self) -> None:
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                tag = data.pop("tag")
                self.log_for_tag(data, tag)
                self.log_for_tag(data, "all")
                self.increment_reshuffle(data["reshuffles"])
            else:
                time.sleep(5)

    def log_for_tag(self, data, tag):
        if tag not in self.loggers:
            self.loggers[tag] = StepLogger(
                n=self.config["inference"]["log_interval"],
                step_name="episode",
                log_wandb=self.config["wandb"]["should_log"],
                tag=tag,
            )
        self.loggers[tag].log(data)

    def increment_reshuffle(self, reshuffles):
        self._update_reshuffle_stats(reshuffles)
        if self._should_average():
            self._process_average()

    def _update_reshuffle_stats(self, reshuffles):
        self.sum_reshuffles += reshuffles
        self.count += 1

    def _should_average(self):
        return self.count % self.config["train"]["episodes_to_avg_over"] == 0

    def _process_average(self):
        next_avg = self._calculate_next_avg()
        if self._is_worse(next_avg):
            self.increment_N()
            self.previous_avg = None
        else:
            self.previous_avg = next_avg

        self._reset_stats()

    def _calculate_next_avg(self):
        return self.sum_reshuffles / self.count

    def _is_worse(self, next_avg):
        return self.previous_avg is not None and next_avg < self.previous_avg

    def _reset_stats(self):
        self.sum_reshuffles = 0
        self.count = 0

    def increment_N(self):
        self.current_env_size[2] = min(
            self.current_env_size[2] + self.config["env"]["increment_N"],
            self.config["env"]["N"],
        )
