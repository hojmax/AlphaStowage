from collections import defaultdict
import wandb


class StepLogger:
    def __init__(self, n, step_name, log_wandb):
        self.step_name = step_name
        self.n = n
        self.count = 0
        self.log_wandb = log_wandb
        self.sum_dict = defaultdict(float)

    def log(self, data: dict) -> None:
        self._increment_with(data)

        if self._should_log():
            self._log_avg()
            self._reset()

    def _increment_with(self, data):
        for key, value in data.items():
            self.sum_dict[key] += value

        self.count += 1

    def _should_log(self):
        return self.count % self.n == 0

    def _log_avg(self):
        avg_dict = {key: value / self.n for key, value in self.sum_dict.items()}
        avg_dict[self.step_name] = self.count
        if self.log_wandb:
            wandb.log(avg_dict)
        else:
            print(avg_dict)

    def _reset(self):
        self.sum_dict.clear()
