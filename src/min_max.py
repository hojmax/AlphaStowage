import numpy as np


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (values - self.minimum) / (self.maximum - self.minimum)
        return values
