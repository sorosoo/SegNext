import json
import math

from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data.dataset import T_co
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import arguments
import utils
import torch.nn.functional as F


"""
    处理labels中1~224的像素，即进行如下处理：
        224 -> 1
        223 -> 2
        ...
    labels: 标签集合/模型预测集合，[batch_size, channels=1, height, width]

    返回值：
        labels, [batch_size, channels=1, height, width]
"""
@torch.no_grad()
def converge_labels(labels: torch.Tensor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    assert len(labels.shape) == 4 and labels.shape[1] == 1
    labels = labels.to(device)
    for num in range(254, 127, -1):
        labels[labels == num] = 255 - num
    return labels


"""
    对labels进行独热编码
    classes_num: 编码的类别数量
    labels: 标签集合, [batch_size, channels=1, height, width]

    返回值：独热编码后的矩阵, [batch_size, height * width, classes_num]
"""
@torch.no_grad()
def one_hot(
        classes_num: int,
        labels: torch.Tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    assert len(labels.shape) == 4 and labels.shape[1] == 1
    labels = labels.to(device)
    # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
    labels = torch.flatten(labels, start_dim=-2)
    # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
    labels = torch.transpose(labels, -2, -1)
    assert labels.shape[-1] == 1
    # (batch_size, height * width, channels) -> (batch_size, height * width)
    labels = torch.squeeze(labels, dim=-1).long()
    # (batch_size, height * width, classes_num)
    one_hot_labels = torch.zeros(*labels.shape, classes_num).to(device)
    return torch.scatter(input=one_hot_labels, dim=-1, index=torch.unsqueeze(labels, -1), value=1.)

"""
    将模型的输出反独热编码
    outputs: [batch_size, classes_num, height, width]
    
    返回值：
        反独热编码后的张量, [batch_size, 1, height, width]
"""
@torch.no_grad()
def inv_one_hot_of_outputs(
        outputs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    assert len(outputs.shape) == 4

    result = torch.argmax(
        F.log_softmax(
            input=outputs.to(device).permute(0, 2, 3, 1),
            dim=-1
        ),
        dim=-1,
        keepdim=True
    ).permute(0, 3, 1, 2)
    return result

"""
    将PIL读取格式的图片或np转换为tensor格式，同时将维度顺序和数量进行转换

    返回值：[channels, height, width]
"""
@torch.no_grad()
def pil2tensor(pil, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    to_tensor = transforms.ToTensor()
    return to_tensor(pil).to(device)


class Pic2PicDataset(Dataset):
    """
        root: 数据集存放的目录，该目录中存放了数据(x)及其对应的标签(y)
        x_dir_name: root下数据(x)所处的目录名
        y_dir_name: root下标签(y)所处的目录名

    """
    def __init__(self, root: str, x_dir_name="images", y_dir_name="labels", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Pic2PicDataset, self).__init__()

        self.device = device
        x_paths = (Path(root) / x_dir_name).glob(pattern="*")
        y_paths = (Path(root) / y_dir_name).glob(pattern="*")

        self.x2y_paths = list(zip(x_paths, y_paths))


    def __len__(self):
        return len(self.x2y_paths)

    def __getitem__(self, index) -> T_co:
        item = self.x2y_paths[index]
        x_path, y_path = item
        x = Image.open(x_path)
        y = Image.open(y_path)
        y_np = np.array(y)
        y.close()
        y = converge_labels(torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0), device=self.device)
        return pil2tensor(x, self.device), y.squeeze(0)

class ConfusionMatrix:

    def __init__(self, classes_num):
        self.classes_num = classes_num
        # matrix的维度：[classes_num, classes_num]
        self.matrix = None


    """
        计算混淆矩阵
        labels: 真实标签，[batch_size, channels=1, height, width]
        labels已经经过converge_labels()处理，其中的像素值都是类别对应的较小label
        
        predictions: 预测值，[batch_size, channels=1, height, width]
        predictions也已经经过converge_labels()处理，其中的像素值也已经被处理为类别对应的较小label
    """
    @torch.no_grad()
    def update(self, labels, predictions, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        assert len(labels.shape) == 4 and len(predictions.shape) == 4 and labels.shape[1] == 1 and predictions.shape[1] == 1
        if self.matrix is None:
            labels = labels.to(device)
            predictions = predictions.to(device)
            # [batch_size, channels=1, height, width] -> [batch_size, height, width]
            labels = torch.squeeze(labels, dim=1)
            # [batch_size, channels=1, height, width] -> [batch_size, height, width]
            predictions = torch.squeeze(predictions, dim=1)
            # mask: [batch_size, height, width]
            mask = (labels < self.classes_num) | (predictions < self.classes_num)
            # labels_masked: [batch_size, height, width]
            labels_masked = labels[mask]
            # predictions_masked: [batch_size, height, width]
            predictions_masked = predictions[mask]
            assert labels_masked.shape == predictions_masked.shape

            # matrix: [classes_num, classes_num], all ele is 0
            self.matrix = torch.zeros(self.classes_num, self.classes_num, dtype=torch.float32, device=device)

            for row in range(0, self.classes_num):
                for col in range(0, self.classes_num):
                    cnt = torch.sum((labels_masked == row) & (predictions_masked == col))
                    self.matrix[row, col] = cnt

    """
        清空混淆矩阵
    """
    def reset(self):
        self.matrix = None

    """
        获取计算出的混淆矩阵
    """
    def get_confusion_matrix(self):
        assert self.matrix is not None
        return self.matrix

    """
        计算某一个标签对应的类别的精度
        
        label_of_cls: 类别的标签值
        返回值：
            (cls_name, precision)
    """
    @torch.no_grad()
    def adjust_cls_precision(self, label_of_cls):
        assert self.matrix is not None and 0 <= label_of_cls < self.classes_num
        result = (
                utils.get_cls_of_label(arguments.classes, label_of_cls),
                (self.matrix[label_of_cls, label_of_cls] / torch.sum(self.matrix[:, label_of_cls])).item()
            )
        return result if not np.isnan(result[-1]) else (utils.get_cls_of_label(arguments.classes, label_of_cls), 0.)

    """
        计算所有类别的精度
        
        返回值：
            列表, [(cls_name, precision), ...]
    """
    @torch.no_grad()
    def adjust_classes_precision(self):
        cls_precision_list = []
        # 0是background(背景)的标签值
        for label_of_cls in range(0, self.classes_num):
            cls_precision_list.append(self.adjust_cls_precision(label_of_cls))
        return cls_precision_list


    """
        计算平均预测精度

        返回值：
            precision
    """
    @torch.no_grad()
    def adjust_avg_precision(self):
        assert self.matrix is not None
        try:
            return math.fsum([tp[-1] for tp in self.adjust_classes_precision()]) / self.classes_num
        except ZeroDivisionError as e:
            return 0.


    """
        计算某一个标签对应的类别的召回率
        
        返回值：
            (cls_name, recall)
    """
    @torch.no_grad()
    def adjust_cls_recall(self, label_of_cls):
        assert self.matrix is not None and 0 <= label_of_cls < self.classes_num
        result = (
                utils.get_cls_of_label(arguments.classes, label_of_cls),
                (self.matrix[label_of_cls, label_of_cls] / torch.sum(self.matrix[label_of_cls, :])).item()
            )

        return result if not np.isnan(result[-1]) else (utils.get_cls_of_label(arguments.classes, label_of_cls), 0.)


    """
        计算所有类别的召回率
        
        返回值：
            列表, [(cls_name, recall), ...]
    """
    @torch.no_grad()
    def adjust_classes_recall(self):
        cls_recall_list = []
        # 0是background(背景)的标签值
        for label_of_cls in range(0, self.classes_num):
            cls_recall_list.append(self.adjust_cls_recall(label_of_cls))
        return cls_recall_list


    """
        计算平均召回率

        返回值：
            recall
    """
    @torch.no_grad()
    def adjust_avg_recall(self):
        assert self.matrix is not None
        try:
            return math.fsum([tp[-1] for tp in self.adjust_classes_recall()]) / self.classes_num
        except ZeroDivisionError as e:
            return 0.

    """
        计算准确率
    """
    @torch.no_grad()
    def adjust_accuracy(self):
        assert self.matrix is not None
        try:
            return (torch.sum(torch.diag(self.matrix)) / torch.sum(self.matrix)).item()
        except ZeroDivisionError as e:
            return 0.


    """
        计算某一个标签对应的类别的iou
        
        返回值：
            (cls_name, iou)
    """
    @torch.no_grad()
    def adjust_cls_iou(self, label_of_cls):
        assert self.matrix is not None and 0 <= label_of_cls < self.classes_num
        result = (
                utils.get_cls_of_label(arguments.classes, label_of_cls),
                (self.matrix[label_of_cls, label_of_cls] /
                 (torch.sum(
                    torch.cat(
                        [
                            self.matrix[label_of_cls, :].view(-1),
                            self.matrix[:, label_of_cls].view(-1)
                        ]
                    )
                 ) - self.matrix[label_of_cls, label_of_cls])).item()
            )
        return result if not np.isnan(result[-1]) else (utils.get_cls_of_label(arguments.classes, label_of_cls), 0.)


    """
        计算所有类别的iou
        
        返回值：
            列表, [(cls_name, iou), ...]
    """
    @torch.no_grad()
    def adjust_classes_iou(self):
        cls_iou_list = []
        # 0是background(背景)的标签值
        for label_of_cls in range(0, self.classes_num):
            cls_iou_list.append(self.adjust_cls_iou(label_of_cls))
        return cls_iou_list

    """
        计算平均iou
    
        返回值：
            iou
    """

    @torch.no_grad()
    def adjust_avg_iou(self):
        assert self.matrix is not None
        try:
            return math.fsum([tp[-1] for tp in self.adjust_classes_iou()]) / self.classes_num
        except ZeroDivisionError as e:
            return 0.

    """
        返回评价指标
        一个函数全部包括
        
        返回值：
            字典
            {
                "classes_precision": [(cls_name, precision), ...],
                "avg_precision": precision,
                "classes_recall": [(cls_name, recall), ...],
                "avg_recall": recall,
                "classes_iou": [(cls_name, iou), ...],
                "avg_iou": iou,
                "accuracy": accuracy
            }
    """
    @torch.no_grad()
    def get_scores(self):
        return {
            "classes_precision": self.adjust_classes_precision(),
            "avg_precision": self.adjust_avg_precision(),
            "classes_recall": self.adjust_classes_recall(),
            "avg_recall": self.adjust_avg_recall(),
            "classes_iou": self.adjust_classes_iou(),
            "avg_iou": self.adjust_avg_iou(),
            "accuracy": self.adjust_accuracy()
        }


"""
    对图片的每个通道进行标准化
    result = (pixel_value - mean) / std 
    
    images: 输入的图像, [batch_size, channels, height, width]
    
    返回值：
        标准化后的张量, std: [batch_size=1, channels, height, width], mean: [batch_size=1, channels, height, width]
"""
@torch.no_grad()
def normalize_channels(images):
    assert len(images.shape) == 4

    std_mean_tuple = torch.std_mean(
        input=images,
        dim=0
    )

    images = (images - std_mean_tuple[0]) / std_mean_tuple[1]

    return images, *std_mean_tuple


if __name__ == "__main__":
    pass
    # labels = torch.tensor(
    #     [
    #         [
    #             [
    #                 [1, 2, 3, 4],
    #                 [3, 3, 4, 0]
    #             ]
    #         ],
    #         [
    #             [
    #                 [1, 2, 3, 3],
    #                 [2, 0, 4, 4]
    #             ]
    #         ]
    #     ]
    # )
    #
    # predictions = torch.tensor(
    #     [
    #         [
    #             [
    #                 [1, 4, 3, 2],
    #                 [2, 2, 4, 3]
    #             ]
    #         ],
    #         [
    #             [
    #                 [1, 4, 4, 2],
    #                 [0, 1, 4, 3]
    #             ]
    #         ]
    #     ]
    # )
    #
    # print(labels.shape)
    # print(predictions.shape)
    #
    # cm = ConfusionMatrix(classes_num=5)
    # cm.update(labels, predictions)
    # scores = cm.get_scores()
    #
    # utils.confusion_matrix_scores2table(scores)
    #
    # utils.avg_confusion_matrix_scores_list(
    #     [scores, scores]
    # )
    # utils.confusion_matrix_scores2table(scores)

    # data = torch.ones(2, 3, 4, 5).to(device="cuda", dtype=torch.float32)
    # print(normalize_channels(data)[0])
    # a = np.ones((224, 224, 3))
    # print(pil2tensor(a).shape)