from torch.optim.lr_scheduler import StepLR


class StepLRWithMinLR(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1):
        self.min_lr = min_lr
        super(StepLRWithMinLR, self).__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        lr_list = super().get_lr()
        lr_list = [max(lr, self.min_lr) for lr in lr_list]
        return lr_list
