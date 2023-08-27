# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 0:08
# @Author  : zhoujun
import math
from functools import partial
from torch.optim import lr_scheduler


class StepLR(object):
    def __init__(self,
                 step_each_epoch,
                 step_size,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(StepLR, self).__init__()
        self.step_size = step_each_epoch * step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** (current_step // self.step_size)


class MultiStepLR(object):
    def __init__(self,
                 step_each_epoch,
                 milestones,
                 warmup_epoch=0,
                 gamma=0.1,
                 last_epoch=-1,
                 **kwargs):
        super(MultiStepLR, self).__init__()
        self.milestones = [step_each_epoch * e for e in milestones]
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return self.gamma ** len([m for m in self.milestones if m <= current_step])

class ConstLR(object):
    def __init__(self,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(ConstLR, self).__init__()
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1.0, self.warmup_epoch))
        return 1.0


class LinearLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(LinearLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        return max(0.0, float(self.epochs - current_step) / float(max(1, self.epochs - self.warmup_epoch)))


class CosineAnnealingLR(object):
    def __init__(self,
                 epochs,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step, num_cycles=0.5):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        progress = float(current_step - self.warmup_epoch) / float(max(1, self.epochs - self.warmup_epoch))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class PolynomialLR(object):
    def __init__(self,
                 step_each_epoch,
                 epochs,
                 lr_end=1e-7,
                 power=1.0,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(PolynomialLR, self).__init__()
        self.lr_end = lr_end
        self.power = power
        self.epochs = epochs * step_each_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch
        self.last_epoch = last_epoch

    def __call__(self, optimizer):
        lr_lambda = partial(
            self.lambda_func,
            lr_init=optimizer.defaults["lr"],
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda, self.last_epoch)

    def lambda_func(self, current_step, lr_init):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        elif current_step > self.epochs:
            return self.lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - self.lr_end
            decay_steps = self.epochs - self.warmup_epoch
            pct_remaining = 1 - (current_step - self.warmup_epoch) / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init