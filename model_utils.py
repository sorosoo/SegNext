import copy
import math
import os.path
from pathlib import Path
import torch.nn as nn
import torch
import yaml
import model
import json
import re
import torch.optim as optim
import learning_rate_scheduler

"""
    获取模型
    @:param train: 是否获取模型进行训练
                   如果为True，使用模型进行训练；
                   如果为False，使用模型进行预测。
    @:param model_config: 模型配置文件路径
    @:param train_config: 训练配置文件路径
    @:param predict_config: 预测配置文件路径
    @:return 实例化模型
"""
def get_model(
        train: bool,
        model_config=Path("config") / "model.yaml",
        train_config=Path("config") / "train.yaml",
        predict_config=Path("config") / "predict.yaml"
):
    with model_config.open("r", encoding="utf-8") as mcf:
        model_config = yaml.load(mcf, Loader=yaml.FullLoader)

        nmf2d_config = model_config["nmf2d_config"]
        if train:
            with train_config.open("r", encoding="utf-8") as tcf:
                train_config = yaml.load(tcf, Loader=yaml.FullLoader)
                device = train_config["device"]
        else:
            with predict_config.open("r", encoding="utf-8") as pcf:
                predict_config = yaml.load(pcf, Loader=yaml.FullLoader)
                device = predict_config["device"]
        nmf2d_config["device"] = device

        net = model.SegNeXt(
            embed_dims=model_config["embed_dims"],
            expand_rations=model_config["expand_rations"],
            depths=model_config["depths"],
            drop_prob_of_encoder=model_config["drop_prob_of_encoder"],
            drop_path_prob=model_config["drop_path_prob"],
            hidden_channels=model_config["channels_of_hamburger"],
            out_channels=model_config["channels_of_hamburger"],
            classes_num=len(model_config["classes"]),
            drop_prob_of_decoder=model_config["drop_prob_of_decoder"],
            nmf2d_config=json.dumps(nmf2d_config)
        ).to(device=device)
        return net

"""
    分割模型中的参数
    named_parameters: 带名称的参数
    regex_expr: 正则表达式(r"")
    
    返回值：
        target, left
        target: 表示符合正则表达式的参数
        left: 表示不符合正则表达式的参数
"""
def split_parameters(named_parameters, regex_expr):
    target = []
    left = []

    pattern = re.compile(regex_expr)
    for name, param in named_parameters:
        if pattern.fullmatch(name):
            target.append((name, param))
        else:
            left.append((name, param))

    return target, left


"""
    获取优化器
    @:param net: 网络模型
    @:param optimizer_config: 优化器配置文件路径
    @:return 优化器
"""
def get_optimizer(
        net,
        optimizer_config=Path("config") / "optimizer.yaml"
):
    with optimizer_config.open("r", encoding="utf-8") as f:
        optimizer_config = yaml.load(f, Loader=yaml.FullLoader)

        base_config = optimizer_config["base_config"]
        lr = eval(base_config["kwargs"])["lr"]
        weight_decay = eval(base_config["kwargs"])["weight_decay"]


        parameters_config = optimizer_config["parameters"][1:]
        left = net.named_parameters()
        parameters = []

        for params_config in parameters_config[1:]:
            params, left = split_parameters(
                named_parameters=left,
                regex_expr=r'' + next(iter(params_config.values()))["regex_expr"]
            )
            params = list(
                map(
                    lambda tp: tp[-1], params
                )
            )
            parameters.append(params)

        parameters = [
            list(
                map(
                    lambda tp: tp[-1], left
                )
            ),
            *parameters
        ]
        params = [
                {
                    'params': param,
                    'lr': lr * next(iter(params_config.values())).setdefault('lr_mult', 1.0),
                    'weight_decay': weight_decay * next(iter(params_config.values())).setdefault('weight_decay', 0.)
                }
                for idx, params_config in enumerate(parameters_config) for param in parameters[idx]
            ]

        optimizer = eval(f"optim.{base_config['optim_type']}")(params, **eval(base_config["kwargs"]))
    return optimizer

"""
    获取学习率调度器
    @:param optimizer: 优化器
    @:param lr_scheduler_config: 学习率调度器配置文件路径
    @:return 学习率调度器
"""
def get_lr_scheduler(
        optimizer,
        lr_scheduler_config=Path("config") / "lr_scheduler.yaml"
):
    lr_scheduler = None
    with lr_scheduler_config.open("r", encoding="utf-8") as f:
        lr_scheduler_config = yaml.load(f, yaml.FullLoader)
        lr_scheduler = learning_rate_scheduler.get_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=eval(f"learning_rate_scheduler.SchedulerType.{lr_scheduler_config['scheduler_type']}"),
            kwargs=eval(lr_scheduler_config["kwargs"])
        )
    return lr_scheduler


"""
    搜寻模型权重文件和自己创建的模型中第一个不同的参数
    left: 元组，("模型名称": state_dict)
    right: 元组，("模型名称": state_dict)
    ignore_counts: 忽略不同的数目
    列表：
        {
            "row_num": 0,
            "模型名称1": "name1",
            "模型名称2": "name2"
        }
"""
def first_diff(left: tuple, right: tuple, ignore_counts=0):
    left = copy.deepcopy(left)
    left_name, left_state = left
    left_state = list(left_state.keys())
    left_ord = 0

    right = copy.deepcopy(right)
    right_name, right_state = right
    right_state = list(right_state.keys())
    right_ord = 0

    response = None

    while left_ord < len(left_state) and right_ord < len(right_state):
        left_sign = left_state[left_ord].split(".")[-1]
        right_sign = right_state[right_ord].split(".")[-1]
        print(f"{left_ord}: {left_state[left_ord]} --> {right_state[right_ord]}")
        if left_sign != right_sign:
            if ignore_counts != 0:
                ignore_counts -= 1
                left_ord += 1
                right_ord += 1
                continue

            assert left_ord == right_ord
            response = {
                "row_num": left_ord,
                left_name: left_state[left_ord],
                right_name: right_state[right_ord]
            }
            return response

        left_ord += 1
        right_ord += 1

    while ignore_counts:
        left_ord += 1
        right_ord += 1
        ignore_counts -= 1

    if left_ord < len(left_state) and right_ord >= len(right_state):
        response = {
            "row_num": left_ord,
            left_name: left_state[left_ord],
            right_name: "None"
        }
    if left_ord >= len(left_state) and right_ord < len(right_state):
        response = {
            "row_num": right_ord,
            left_name: "None",
            right_name: right_state[right_ord]
        }
    if left_ord >= len(left_state) and right_ord >= len(right_state):
        response = {
            "row_num": -1,
            left_name: "same",
            right_name: "same"
        }
    print(f"{response['row_num']}: {response[left_name]} --> {response[right_name]}")
    return response


"""
    初始化模型
    @:param train: 
        True表示，初始化用来训练的网络；
        False表示，初始化用来预测的网络.
    net: 网络模型
    optimizer: 优化器
    pretrained: 是否加载预训练权重
    @:param train_config: 训练配置文件路径
"""
def init_model(
        train,
        net,
        optimizer=None,
        train_config=Path("config") / "train.yaml",
        predict_config=Path("config") / "predict.yaml"
):
    # 初始化权重
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            if m.weight is not None:
                nn.init.normal_(m.weight, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.)

    if train:
        with train_config.open("r", encoding="utf-8") as tcf:
            config = yaml.load(tcf, yaml.FullLoader)
    else:
        with predict_config.open("r", encoding="utf-8") as pcf:
            config = yaml.load(pcf, yaml.FullLoader)

    mode = config["mode"]
    if mode == -1:
        return

    checkpoint = torch.load(os.path.sep.join(config["checkpoint"]))
    if mode == 0:
        for regex_expr in config["regex_expr"]:
            checkpoint["state_dict"] = {
                tp[0]: tp[-1]
                for tp in zip(net.state_dict().keys(), checkpoint["state_dict"].values())
                if re.compile(r"" + regex_expr).fullmatch(tp[0])
            }
        checkpoint["optimizer"]["state"] = dict()

    net.load_state_dict(checkpoint["state_dict"], strict=False)
    if train:
        optimizer.load_state_dict(checkpoint["optimizer"])