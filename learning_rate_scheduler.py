import math
from enum import Enum

import numpy as np
import torch
import torch.optim as optim


class SchedulerType(Enum):
    STEP_SCHEDULER = "step",
    MULTI_STEP_SCHEDULER = "multi_step",
    EXPONENTIAL_SCHEDULER = "exponential",
    COSINE_ANNEALING_SCHEDULER = "cosine_annealing",
    LINEAR_WARMUP_THEN_POLY_SCHEDULER = "linear_warmup_then_poly"


class StepScheduler:
    """
        optimizer: 优化器
        step_size: 每间隔多少步，就去计算优化器的学习率并将其更新
        gamma: lr_(t+1) = lr_(t) * gamma
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """
    def __init__(self, optimizer, step_size=30, gamma=0.1, verbose=False):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=-1,
            verbose=self.verbose
        )


    """
        调用学习率调度器
    """
    def step(self):
        self.lr_scheduler.step()



    """
        获得学习率调度器的状态
    """
    def get_state_dict(self):
        return self.lr_scheduler.state_dict()

    """
        加载学习率调度器的状态字典
    """
    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)


class MultiStepScheduler:
    """
        optimizer: 优化器
        milestones: 列表，列表内的数据必须是整数且递增，每一个数表示调度器被执行了对应次数后，就更新优化器的学习率
        gamma: lr_(t+1) = lr_(t) * gamma
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """
    def __init__(self, optimizer, milestones, gamma, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.milestones,
            gamma=gamma,
            last_epoch=-1,
            verbose=self.verbose
        )

    """
        调用学习率调度器
    """
    def step(self):
        self.lr_scheduler.step()


    """
        获得学习率调度器的状态
    """
    def get_state_dict(self):
        return self.lr_scheduler.state_dict()


    """
        加载学习率调度器的状态字典
    """
    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)


class ExponentialScheduler:

    """
        optimizer: 优化器
        gamma: lr_(t+1) = lr_(t) * gamma, 每一次调用，优化器的学习率都会更新
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """
    def __init__(self, optimizer, gamma=0.95, verbose=False):
        self.optimizer = optimizer
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.gamma,
            last_epoch=-1,
            verbose=self.verbose
        )

        """
            调用学习率调度器
        """

        def step(self):
            self.lr_scheduler.step()

        """
            获得学习率调度器的状态
        """

        def get_state_dict(self):
            return self.lr_scheduler.state_dict()

        """
            加载学习率调度器的状态字典
        """

        def load_state_dict(self, state_dict: dict):
            self.lr_scheduler.load_state_dict(state_dict)


class CosineAnnealingScheduler:

    """
        optimizer: 优化器，优化器中有一个已经设定的初始学习率，这个初始学习率就是调度器能达到的最大学习率(max_lr)
        t_max: 周期，调度器每被调用2 * t_max，优化器的学习率就会从max_lr -> min_lr -> max_lr
        min_lr: 最小学习率
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """
    def __init__(self, optimizer, t_max=5, min_lr=0, verbose=False):
        self.optimizer = optimizer
        self.t_max = t_max
        self.min_lr = min_lr
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.t_max,
            eta_min=self.min_lr,
            last_epoch=-1,
            verbose=self.verbose
        )

    """
        调用学习率调度器
    """
    def step(self):
        self.lr_scheduler.step()


    """
        获得学习率调度器的状态
    """
    def get_state_dict(self):
        return self.lr_scheduler.state_dict()


    """
        加载学习率调度器的状态字典
    """
    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)

class LinearWarmupThenPolyScheduler:

    """
        预热阶段采用Linear，之后采用Poly
        optimizer: 优化器
        warmup_iters: 预热步数
        total_iters: 总训练步数
        min_lr: 最低学习率
    """
    def __init__(self, optimizer, warmup_iters=1500, total_iters=2000, warmup_ratio=1e-6, min_lr=0., power=1.):
        self.optimizer = optimizer
        self.current_iters = 0
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_ration = warmup_ratio
        self.min_lr = min_lr
        self.power = power

        self.base_lr = None
        self.regular_lr = None
        self.warmup_lr = None

    def get_base_lr(self):
        return np.array([param_group.setdefault("initial_lr", param_group["lr"]) for param_group in self.optimizer.param_groups])

    def get_lr(self):
        coeff = (1 - self.current_iters / self.total_iters) ** self.power
        return (self.base_lr - np.full_like(self.base_lr, self.min_lr)) * coeff + np.full_like(self.base_lr, self.min_lr)

    def get_regular_lr(self):
        return self.get_lr()

    def get_warmup_lr(self):
        k = (1 - self.current_iters / self.warmup_iters) * (1 - self.warmup_ration)
        return (1 - k) * self.regular_lr

    def update(self):
        assert 0 <= self.current_iters < self.total_iters
        self.current_iters = self.current_iters + 1
        self.base_lr = self.get_base_lr()
        self.regular_lr = self.get_regular_lr()
        self.warmup_lr = self.get_warmup_lr()

    def set_lr(self):
        if self.current_iters <= self.warmup_iters:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.warmup_lr[idx]
        elif self.current_iters <= self.total_iters:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.regular_lr[idx]

    def step(self):
        self.update()
        self.set_lr()









"""
    获取学习率调度器
    optimizer: 使用学习率调度器的优化器
    scheduler_type: 要获取的调度器的类型
    kwargs: 参数字典，作用于调度器
    
    需要改变优化器的参数，在该方法中调整
"""
def get_lr_scheduler(optimizer: optim, scheduler_type: SchedulerType, kwargs=None):
    if kwargs is None:
        # 返回默认设置的调度器
        if scheduler_type == SchedulerType.STEP_SCHEDULER:
            return StepScheduler(
                optimizer=optimizer,
                step_size=30,
                gamma=0.1,
                verbose=False
            )
        elif scheduler_type == SchedulerType.MULTI_STEP_SCHEDULER:
            return MultiStepScheduler(
                optimizer=optimizer,
                milestones=[30, 60, 90],
                gamma=0.1,
                verbose=False
            )
        elif scheduler_type == SchedulerType.EXPONENTIAL_SCHEDULER:
            return ExponentialScheduler(
                optimizer=optimizer,
                gamma=0.95,
                verbose=False
            )
        elif scheduler_type == SchedulerType.COSINE_ANNEALING_SCHEDULER:
            return CosineAnnealingScheduler(
                optimizer=optimizer,
                t_max=5,
                min_lr=0,
                verbose=False
            )
        elif scheduler_type == SchedulerType.LINEAR_WARMUP_THEN_POLY_SCHEDULER:
            return LinearWarmupThenPolyScheduler(
                optimizer=optimizer,
                warmup_iters=1500,
                total_iters=2000,
                warmup_ratio=1e-6,
                min_lr=0.,
                power=1.
            )
    else:
        # 返回自定义设置的调度器
        if scheduler_type == SchedulerType.STEP_SCHEDULER:
            return StepScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.MULTI_STEP_SCHEDULER:
            return MultiStepScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.EXPONENTIAL_SCHEDULER:
            return ExponentialScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.COSINE_ANNEALING_SCHEDULER:
            return CosineAnnealingScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.LINEAR_WARMUP_THEN_POLY_SCHEDULER:
            return LinearWarmupThenPolyScheduler(
                optimizer=optimizer,
                **kwargs
            )