import time
from Logging import init_wandb_run
from StepLogger import StepLogger
from multiprocessing import Queue
import time


class InferenceLoggerProcess:
    """Collects episode data from inference processes and logs it.
    The episode data includes: value, reshuffles, seconds/episode and removes/episode.
    """

    def __init__(self, queue: Queue, config: dict):
        self.queue = queue
        self.config = config

        if self.config["wandb"]["should_log"]:
            init_wandb_run(self.config)

        self.logger = StepLogger(
            n=self.config["inference"]["log_interval"],
            step_name="episode",
            log_wandb=self.config["wandb"]["should_log"],
        )

    def loop(self):
        while True:
            if not self.queue.empty():
                self.logger.log(self.queue.get())
            else:
                time.sleep(5)
