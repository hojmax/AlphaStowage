import numpy as np


class MinMaxStats:
    """
    A class that holds the min-max values of the tree using np.float16.
    """

    def __init__(self):
        self.maximum = np.float16(-np.inf)
        self.minimum = np.float16(np.inf)

    def update(self, value):
        value = np.float16(value)
        self.maximum = np.maximum(self.maximum, value)
        self.minimum = np.minimum(self.minimum, value)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.float16(values)
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (values - self.minimum) / (self.maximum - self.minimum)
        return values
