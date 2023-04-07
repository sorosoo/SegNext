import os

import numpy as np
import yaml
from PIL import Image
import data_utils
import torch
from pathlib import Path
import model_utils
import utils
from matplotlib import pyplot as plt



"""
    预测
    @:param net: 网络模型
    @:param image: 图像
    @:param cls_name: 类别名
    @:param predict_config: 预测配置文件路径
    @:param model_config: 模型配置文件路径
    
    @:return mask: [image_height, image_width]，元素类型为bool
"""
def predict(
        net,
        image: Image,
        cls_name,
        predict_config=Path("config") / "predict.yaml",
        model_config=Path("config") / "model.yaml"
):
    with model_config.open("r", encoding="utf-8") as mcf:
        model_config = yaml.load(mcf, Loader=yaml.FullLoader)
        classes = model_config["classes"]

    with predict_config.open("r", encoding="utf-8") as pcf:
        predict_config = yaml.load(pcf, yaml.FullLoader)
        device = predict_config["device"]
        image = data_utils.pil2tensor(image, device)
        if len(image.shape) == 3:
            image = torch.unsqueeze(image, dim=0)
        batch_size, _, image_height, image_width = image.shape

        prediction = data_utils.inv_one_hot_of_outputs(
            torch.transpose(
                net(image),
                -2,
                -1
            ).reshape(batch_size, len(classes), image_height, image_width),
            device
        )

        mask = torch.squeeze(
            prediction == utils.get_label_of_cls(classes, cls_name)[0]
        )

        return mask


"""
    将预测结果与原图混合
    
    @:param net: 神经网络模型
    @:param image: 原图
    @:param mask: predict的对应某一类别的mask
    @:param mask: 神经网络的预测结果
    @:param classes: 所有类别
    @:param cls_name: 类别
    @:param colors: 所有类别对应的颜色列表
    @:return 混合后的图像
"""
def blend(
        image: Image,
        mask,
        classes,
        cls_name,
        colors
):
    mask = mask.to(device="cpu").numpy()
    new_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    new_image[mask] = utils.get_color_of_cls(classes, colors, cls_name)
    new_image = Image.fromarray(new_image)
    blend_image = Image.blend(image, new_image, 0.5)
    return blend_image



"""
    展示图像
    @:param 需要进行展示的图像，图像尺寸应为[height, width, channels=3]
"""
def show_image(image):
    plt.imshow(image)
    plt.show()




if __name__ == "__main__":
    with Path(os.path.sep.join(["config", "model.yaml"])).open("r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        classes = model_config["classes"]

    colors = utils.get_colors(len(classes))

    image_path = os.path.sep.join([
        "dataset", "test", "biomass_image_train_0233_8.jpg"
    ])

    cls_name = "leaf"
    net = model_utils.get_model(False)
    model_utils.init_model(False, net)
    image = Image.open(image_path)
    mask = predict(net, image, cls_name)
    show_image(blend(image, mask, classes, cls_name, colors))