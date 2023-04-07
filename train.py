import math
import os.path
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import utils
import data_utils
import model_utils
from torch.utils.data import DataLoader
import losses
from datetime import datetime

"""
    1 epoch train
    @:param epochs: 总共的epoch数
    @:param epoch: 当前epoch
    @:param net: 神经网络模型
    @:param train_data_loader: 训练数据加载器
    @:param image_size: 图片大小
    @:param classes_num: 类别数
    @:param loss_fn: 损失函数
    @:param lr_scheduler: 学习率调度器
    @:param optimizer: 优化器
    @:param device: 运行场地
    @:return 1 epoch train avg loss, 1 epoch train avg scores
"""
def fit(
        epochs,
        epoch,
        net,
        train_data_loader,
        image_size,
        classes_num,
        loss_fn,
        lr_scheduler,
        optimizer,
        device="cuda"
):
    matrix = data_utils.ConfusionMatrix(classes_num)
    scores_list = []
    loss_list = []
    progress_bar = tqdm(train_data_loader)
    for idx, data in enumerate(progress_bar):
        images, labels = data
        lr_scheduler.step()
        optimizer.zero_grad()
        predictions = torch.transpose(net(images), -2, -1).view(-1, classes_num, *image_size)
        matrix.update(labels, data_utils.inv_one_hot_of_outputs(predictions, device), device)
        scores = matrix.get_scores()
        matrix.reset()
        scores_list.append(scores)

        loss = loss_fn(
            predictions,
            torch.squeeze(labels, dim=1).to(dtype=torch.long)
        )
        loss_value = loss.item()
        if np.isnan(loss_value):
            loss_value = max(loss_list) if len(loss_list) != 0 else 1.0
        loss_list.append(loss_value)

        loss.backward()
        optimizer.step()

        progress_bar.set_description(
            f"train --> Epoch {epoch + 1} / {epochs}, batch_loss: {loss_value:.3f}, batch_iou: {scores['avg_iou']:.3f}, batch_accuracy: {scores['accuracy']:.3f}"
        )
    progress_bar.close()
    return sum(loss_list) / len(loss_list), utils.avg_confusion_matrix_scores_list(scores_list)

"""
    1 epoch train
    @:param epochs: 总共的epoch数
    @:param epoch: 当前epoch
    @:param net: 神经网络模型
    @:param train_data_loader: 验证数据加载器
    @:param image_size: 图片大小
    @:param classes_num: 类别数
    @:param loss_fn: 损失函数
    @:param device: 运行场地
    @:return val avg loss, val avg scores
"""
@torch.no_grad()
def val(
        epochs,
        epoch,
        net,
        val_data_loader,
        image_size,
        classes_num,
        loss_fn,
        device="cuda"
):
    matrix = data_utils.ConfusionMatrix(classes_num)
    scores_list = []
    loss_list = []
    progress_bar = tqdm(val_data_loader)
    for idx, data in enumerate(progress_bar):
        images, labels = data
        predictions = torch.transpose(net(images), -2, -1).view(-1, classes_num, *image_size)
        matrix.update(labels, data_utils.inv_one_hot_of_outputs(predictions, device), device)
        scores = matrix.get_scores()
        matrix.reset()
        scores_list.append(scores)

        loss = loss_fn(
            predictions,
            torch.squeeze(labels, dim=1).to(dtype=torch.long)
        )
        loss_value = loss.item()
        if np.isnan(loss_value):
            loss_value = max(loss_list) if len(loss_list) != 0 else 1.0
        loss_list.append(loss_value)

        progress_bar.set_description(
            f"val ---> Epoch {epoch + 1} / {epochs}, batch_loss: {loss_value:.3f}, batch_iou: {scores['avg_iou']:.3f}, batch_accuracy: {scores['accuracy']:.3f}"
        )
    progress_bar.close()
    return sum(loss_list) / len(loss_list), utils.avg_confusion_matrix_scores_list(scores_list)


"""
    模型训练
    
    net: 网络模型
    optimizer: 优化器,
    lr_scheduler: 学习率调度器,
    weight: 每一类的权重
    root_path: 存储训练数据和验证数据的根目录
    train_dir_names: 存储训练数据的目录，元组形式(images_path, labels_path)
    val_dir_names: 存储验证数据的目录, 元组形式(images_path, labels_path)
    classes_num: 类别数量
    yaml_path: 配置文件路径
"""
def train(
        net,
        optimizer,
        lr_scheduler,
        train_config=Path("config") / "train.yaml",
        model_config=Path("config") / "model.yaml"
):
    with model_config.open("r", encoding="utf-8") as mcf:
        model_config = yaml.load(mcf, yaml.FullLoader)
        classes_num = len(model_config["classes"])

    with train_config.open("r", encoding="utf-8") as tcf:
        train_config = yaml.load(tcf, Loader=yaml.Loader)
        device = train_config["device"]
        epochs = train_config["epochs"]

        train_images_dataset = data_utils.Pic2PicDataset(
            root=os.path.sep.join(train_config["root"]),
            x_dir_name=Path(os.path.sep.join(train_config["train_dir_name"])) / train_config["images_dir_name"],
            y_dir_name=Path(os.path.sep.join(train_config["train_dir_name"])) / train_config["labels_dir_name"]
        )
        train_data_loader = DataLoader(
            dataset=train_images_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["workers"]
        )

        val_images_dataset = data_utils.Pic2PicDataset(
            root=os.path.sep.join(train_config["root"]),
            x_dir_name=Path(os.path.sep.join(train_config["val_dir_name"])) / train_config["images_dir_name"],
            y_dir_name=Path(os.path.sep.join(train_config["val_dir_name"])) / train_config["labels_dir_name"]
        )
        val_data_loader = DataLoader(
            dataset=val_images_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            num_workers=train_config["workers"]
        )

        image_height, image_width = train_config["image_height"], train_config["image_width"]
        weight = torch.tensor(train_config["weight"]) if len(train_config["weight"]) != 1 else torch.ones(classes_num)
        loss_fn = losses.FocalLoss(
            weight=weight.to(device)
        )

        max_train_iou, max_val_iou = -np.inf, -np.inf
        best_train_model, best_val_model = None, None

        for epoch in range(0, epochs):
            # 训练
            net.train()
            train_avg_loss, train_avg_scores = fit(
                epochs=epochs,
                epoch=epoch,
                net=net,
                train_data_loader=train_data_loader,
                image_size=(image_height, image_width),
                classes_num=classes_num,
                loss_fn=loss_fn,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                device=device
            )
            print()
            print(utils.confusion_matrix_scores2table(train_avg_scores))
            print(f"train_avg_loss: {train_avg_loss:.3f}")

            if max_train_iou < train_avg_scores["avg_iou"]:
                max_train_iou = train_avg_scores["avg_iou"]
                best_train_model = {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "avg_iou": max_train_iou
                }



            # 验证
            if (epoch + 1) % train_config["eval_every_n_epoch"] == 0:
                net.eval()
                val_avg_loss, val_avg_scores = val(
                    epochs=epochs,
                    epoch=epoch,
                    net=net,
                    val_data_loader=val_data_loader,
                    image_size=(image_height, image_width),
                    classes_num=classes_num,
                    loss_fn=loss_fn,
                    device=device
                )
                print()
                print(utils.confusion_matrix_scores2table(val_avg_scores))
                print(f"val_avg_loss: {val_avg_loss:.3f}")

                if max_val_iou < val_avg_scores["avg_iou"]:
                    max_val_iou = val_avg_scores["avg_iou"]
                    best_val_model = {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "avg_iou": max_val_iou
                    }



                m = {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "avg_iou": val_avg_scores["avg_iou"]
                }

                torch.save(
                    obj=m,
                    f=f"{os.path.sep.join(train_config['save_path'])}_Iou{100 * best_val_model['avg_iou']:.3f}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.pth"
                )


        torch.save(
            obj=best_train_model,
            f=f"{os.path.sep.join(train_config['save_path'])}_train_Iou{100 * best_train_model['avg_iou']:.3f}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.pth"
        )
        torch.save(
            obj=best_train_model,
            f=f"{os.path.sep.join(train_config['save_path'])}_val_Iou{100 * best_val_model['avg_iou']:.3f}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.pth"
        )





if __name__ == "__main__":
    net = model_utils.get_model(True)
    optimizer = model_utils.get_optimizer(net)
    lr_scheduler = model_utils.get_lr_scheduler(optimizer=optimizer)
    model_utils.init_model(
        train=True,
        net=net,
        optimizer=optimizer
    )
    train(
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )