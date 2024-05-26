from torch.optim.lr_scheduler import ExponentialLR
import numpy as np


class ExponentialLRWithMinLR(ExponentialLR):
    def __init__(self, optimizer, step_size, gamma, min_lr, last_epoch=-1):
        self.min_lr = min_lr
        exponential = np.exp(np.log(gamma) / step_size)
        super(ExponentialLRWithMinLR, self).__init__(optimizer, exponential, last_epoch)

    def get_lr(self):
        lr_list = super().get_lr()
        lr_list = [max(lr, self.min_lr) for lr in lr_list]
        return lr_list

    def current_lr(self):
        return self.get_last_lr()[0]
