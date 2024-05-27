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

        self.loggers = {}

    def loop(self) -> None:
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                tag = data.pop("tag")
                self.log_for_tag(data, tag)
                self.log_for_tag(data, "all")
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
