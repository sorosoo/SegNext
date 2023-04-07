import colorsys
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from tabulate import tabulate
from torchvision.transforms import transforms, InterpolationMode

"""
    生成num种颜色
    返回值: color list，返回的color list的第一个数值永远是(0, 0, 0)
"""
def get_colors(num: int):
    assert num >= 1
    if num <= 21:
        colors = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
            (128, 64, 12)
        ]
    else:
        hsv_tuples = [(x / num, 1., 1.) for x in range(0, num - 1)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        if (0, 0, 0) in colors:
            colors.remove((0, 0, 0))
        colors = [(0, 0, 0), *colors]
    return colors

"""
    获取某种颜色对应的标签
    返回值：标签值
"""
def get_label_of_color(colors, color):
    low_label = colors.index(color)
    return low_label, 255 - low_label

"""
    获取某个标签值对应的颜色
    返回值：元组(r, g, b)
"""
def get_color_of_label(colors, label):
    low_label = label if label < 255 - label else 255 - label
    return colors[low_label]

"""
    获取某种类别对应的标签
    返回值：标签值
"""
def get_label_of_cls(classes, cls):
    low_label = classes.index(cls)
    return low_label, 255 - low_label

"""
    获取某个标签值对应的类别
    返回值：类别
"""
def get_cls_of_label(classes, label):
    low_label = label if label < 255 - label else 255 - label
    return classes[low_label]

"""
    获取某种颜色对应的类别
    返回值：类别
    color: (r, g, b)
"""
def get_cls_of_color(classes, colors, color):
    idx = colors.index(color)
    return get_cls_of_label(classes, idx)

"""
    获取某种类别对应的颜色
    返回值：颜色，(r, g, b)
"""
def get_color_of_cls(classes, colors, cls):
    idx = classes.index(cls)
    return get_color_of_label(colors, idx)


def draw_mask(draw, points, shape_type, label, out_line_value, line_width=10, point_width=5):
    points = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(points) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = points
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=out_line_value, fill=label)
    elif shape_type == 'rectangle':
        assert len(points) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(points, outline=out_line_value, fill=label)
    elif shape_type == 'line':
        assert len(points) == 2, 'Shape of shape_type=line must have 2 points'
        greater_label = out_line_value
        draw.line(xy=points, fill=greater_label, width=line_width)
    elif shape_type == 'linestrip':
        greater_label = out_line_value
        draw.line(xy=points, fill=greater_label, width=line_width)
    elif shape_type == 'point':
        assert len(points) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = points[0]
        r = point_width
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=out_line_value, fill=label)
    else:
        assert len(points) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=points, outline=out_line_value, fill=label)

"""
    负责将labelme的标记转换成(mask)图像
    classes: 类别列表
"""
def labelme_json2mask(classes, json_path: str, mask_saved_path: str):
    assert classes is not None and classes[0] == "background"

    json_path = Path(json_path)
    if json_path.exists() and json_path.is_file():
        with json_path.open(mode="r") as f:
            json_data = json.load(f)
            image_height = json_data["imageHeight"]
            image_width = json_data["imageWidth"]
            image_path = json_data["imagePath"]
            shapes = json_data["shapes"]

            cls_info_list = []
            for shape in shapes:
                cls_name_in_json = shape["label"]
                assert cls_name_in_json in classes
                points = shape["points"]
                shape_type = shape["shape_type"]
                label_of_cls = classes.index(cls_name_in_json)
                cls_info_list.append(
                    {
                        "cls_name": cls_name_in_json,
                        "label": label_of_cls,
                        "points": points,
                        "shape_type": shape_type
                    }
                )

            mask = np.zeros(shape=(image_height, image_width), dtype=np.uint8)
            mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask)
            for cls_info in cls_info_list:
                points = cls_info["points"]
                shape_type = cls_info["shape_type"]
                label = cls_info["label"]
                draw_mask(draw, points, shape_type, label, 255 - label)

            mask = np.array(mask)
            mask = Image.fromarray(mask)
            mask.save(str(Path(mask_saved_path) / (str(image_path).split(".")[0] + ".png")))

        os.remove(json_path)

"""
    将root_path下labelme生成的json文件全部进行处理：
        1. 有原图匹配的json文件，会转换成mask，存储到mask_saved_path路径下
        2. 没有原图，但是有json文件的，直接删除该json文件
        3. 有原图，但是没有json文件的，会在mask_saved_path下生成一个纯黑背景图片
    root_path: 存储着原图和json文件，原图后缀名尽量为.jpg
"""
def convert_labelme_jsons2masks(classes, root_path: str, mask_saved_path: str, original_image_suffix=".jpg"):
    assert 0 < len(classes) <= 128
    original_images = set(
        map(
            lambda name: str(name).split(".")[0],
            Path(root_path).glob(pattern=f"*{original_image_suffix}")
        )
    )
    json_files = Path(root_path).glob(pattern="*.json")
    for json_file in json_files:
        filename = str(json_file).split(".")[0]
        if filename in original_images:
            labelme_json2mask(classes, str(json_file), mask_saved_path)
            original_images.remove(filename)
        else:
            os.remove(json_file)

    if len(original_images) != 0:
        for image_filename in original_images:
            image_path = image_filename + f"{original_image_suffix}"
            image = Image.open(image_path)
            height, width = image.height, image.width
            image.close()
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = Image.fromarray(mask)
            mask.save(str(Path(mask_saved_path) / (os.path.basename(image_filename) + ".png")))

"""
    将混淆矩阵得到的尺度(scores)组合成表格形式输出到控制台
    scores: 混淆矩阵的尺度(scores)
"""
def confusion_matrix_scores2table(scores):
   assert scores is not None and isinstance(scores, dict)

   classes = [tp[0] for tp in scores["classes_precision"]]
   cls_precision_list = [tp[-1] for tp in scores["classes_precision"]]
   cls_recall_list = [tp[-1] for tp in scores["classes_recall"]]
   cls_iou_list = [tp[-1] for tp in scores["classes_iou"]]
   table1 = tabulate(
       tabular_data=np.concatenate(
           (
               np.asarray(classes).reshape(-1, 1),
               np.asarray(cls_precision_list).reshape(-1, 1),
               np.asarray(cls_recall_list).reshape(-1, 1),
               np.asarray(cls_iou_list).reshape(-1, 1)
           ), 1
       ),
       headers=["classes", "precision", "recall", "iou"],
       tablefmt="grid"
   )

   avg_precision = scores["avg_precision"]
   avg_recall = scores["avg_recall"]
   avg_iou = scores["avg_iou"]
   accuracy = scores["accuracy"]
   table2 = tabulate(
       tabular_data=[(avg_precision, avg_recall, avg_iou, accuracy)],
       headers=["avg_precision", "avg_recall", "avg_iou", "accuracy"],
       tablefmt="grid"
   )

   table = tabulate(
       tabular_data=np.concatenate(
           (
               np.asarray(["single", "overall"]).reshape(-1, 1),
               np.asarray([table1, table2]).reshape(-1, 1)
           ), 1
       ),
       headers=["table type", "table"],
       tablefmt="grid"
   )

   return table


"""
    相加混淆矩阵得到的两个scores
    
    返回值：
        相加后的混淆矩阵
"""
def sum_2_confusion_matrix_scores(scores_left: dict, scores_right: dict):
    scores_left["classes_precision"] = [
        (tp[0][0], tp[0][-1] + tp[-1][-1]) for tp in zip(scores_left["classes_precision"], scores_right["classes_precision"])
    ]
    scores_left["classes_recall"] = [
        (tp[0][0], tp[0][-1] + tp[-1][-1]) for tp in zip(scores_left["classes_recall"], scores_right["classes_recall"])
    ]
    scores_left["classes_iou"] = [
        (tp[0][0], tp[0][-1] + tp[-1][-1]) for tp in zip(scores_left["classes_iou"], scores_right["classes_iou"])
    ]

    scores_left["avg_precision"] = scores_left["avg_precision"] + scores_right["avg_precision"]
    scores_left["avg_recall"] = scores_left["avg_recall"] + scores_right["avg_recall"]
    scores_left["avg_iou"] = scores_left["avg_iou"] + scores_right["avg_iou"]
    scores_left["accuracy"] = scores_left["accuracy"] + scores_right["accuracy"]

    return scores_left

"""
    将混淆矩阵列表内的scores进行相加
    @:param scores_list: 得分列表
    @:return 相加后的得分
"""
def sum_confusion_matrix_scores_list(scores_list):
    if len(scores_list) == 1:
        return scores_list[0]

    result = scores_list[0]
    for i in range(1, len(scores_list)):
        result = sum_2_confusion_matrix_scores(result, scores_list[i])
    return result

"""
    对混淆矩阵得出的scores_list相加后求平均
    
    返回值：
        相加后求平均的scores
"""
def avg_confusion_matrix_scores_list(scores_list):
    assert scores_list is not None and len(scores_list) >= 1
    result = sum_confusion_matrix_scores_list(scores_list)

    result["classes_precision"] = [
        (tp[0], tp[-1] / len(scores_list)) for tp in result["classes_precision"]
    ]
    result["classes_recall"] = [
        (tp[0], tp[-1] / len(scores_list)) for tp in result["classes_recall"]
    ]
    result["classes_iou"] = [
        (tp[0], tp[-1] / len(scores_list)) for tp in result["classes_iou"]
    ]

    result["avg_precision"] = result["avg_precision"] / len(scores_list)
    result["avg_recall"] = result["avg_recall"] / len(scores_list)
    result["avg_iou"] = result["avg_iou"] / len(scores_list)
    result["accuracy"] = result["accuracy"] / len(scores_list)

    return result

"""
    对原始作为x的输入图像进行增强预处理，产生相同大小的图片(旋转、翻转、亮度调整)
    ts是pytorch工具包，经过该工具包处理后图像如果和原本的不同，
    就会保存在磁盘上，以达到增强数据的目的，请先执行该函数之后，再对原始数
    据图像进行人工标注。
    root_path目录下的数据只有图片，且图片后缀名一致
    
    root_path: 作为x的原始输入图像所在目录
    ts: 预处理策略
"""
def augment_raw_images2(
        root_path,
        ts=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            ]
        )
):
    image_paths = Path(root_path).glob(pattern="*")
    for image_path in image_paths:
        counter = 0
        image_filename, image_suffix = os.path.splitext(image_path)

        image = Image.open(image_path)
        image_np = np.array(image)


        for transform in ts.transforms:
            new_image = transform(Image.fromarray(image_np))
            new_image_np = np.array(new_image)

            if not np.array_equal(image_np, new_image_np):
                new_image_copy = Image.fromarray(new_image_np)
                new_image_copy.save(str(Path(f"{image_filename}_{counter}{image_suffix}")))
                new_image_copy.close()
                counter += 1

            new_image.close()

        image.close()



"""
    对原始作为x的输入图像进行增强预处理，产生image_cropped_shape大小的图片
    现将图像resize为image_resized_shape大小，然后进行1次裁剪和1次随机裁剪，裁剪的图像保留下来，原始图像放入to_path中
    ts是pytorch工具包，经过该工具包处理后图像如果和原本的不同，
    就会保存在磁盘上，以达到增强数据的目的，请先执行该函数之后，再对原始数
    据图像进行人工标注。
    from_path目录下的数据只有图片，且图片后缀名一致
    
    from_path: 作为x的原始输入图像所在目录
    to_path: 处理后的原始图像放入哪里，如果为None，就删除原始图像
    image_resized_shape: 图像resize之后的大小, image_cropped_shape每个维度必须小于image_resized_shape
    image_cropped_shape: 图像裁剪后的大小，image_cropped_shape每个维度必须小于image_resized_shape
    ts: 预处理策略
"""
def augment_raw_images(
        from_path,
        to_path="to/path",
        image_resized_shape=(256, 256),
        image_cropped_shape=(224, 224),
        ts=None
):
    if ts is None:
        ts = transforms.Compose(
            [
                transforms.Resize(image_resized_shape, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomCrop(image_cropped_shape),
                transforms.RandomResizedCrop(image_cropped_shape)
            ]
        )
    image_paths = Path(from_path).glob("*")
    for image_path in image_paths:
        counter = 0
        image_filename, image_suffix = os.path.splitext(image_path)
        with Image.open(image_path) as image:
            image = ts.transforms[0](image)
            image_copy_np = copy.deepcopy(np.array(image))
            for transform in ts.transforms[0:]:
                image = transform(image)
                image_np = np.array(image)
                if not np.array_equal(image_np, image_copy_np):
                    image.save(str(Path(f"{image_filename}_{counter}{image_suffix}")))
                    counter = counter + 1
                    image.close()
                    image = Image.fromarray(image_copy_np)
        if to_path:
            Path(image_path).rename(Path(to_path) / f"{os.path.basename(image_path)}")
        else:
            Path(image_path).unlink()


"""
    对验证数据集中的图片进行大小的统一，以便其拥有统一的大小，可以进行批次训练
    from_path: 验证数据集所在的目录
    to_path: 原始数据应该转移到哪里
    resized_shape: (height, width), resize后的大小
"""
def resize_val_images(from_path, to_path, resized_shape):
    image_paths = Path(from_path).glob(pattern="*")
    for image_path in image_paths:
        original_image = Image.open(image_path)
        original_image_np = np.array(original_image)
        resized_image = Image.fromarray(original_image_np).resize(resized_shape)
        original_image.close()

        if not to_path:
            Path(image_path).unlink(missing_ok=True)
        else:
            Path(image_path).rename(Path(to_path) / os.path.basename(image_path))

        resized_image.save(image_path)
        resized_image.close()


"""
    将一张图片按照尺寸裁剪为多张图片
    @:param image: 图片
    @:param crop_size: 裁剪尺寸，为tuple(image_height, image_width)
    
    @:return 裁剪之后的图片列表
"""
def crop_image2images(image: Image, crop_size):
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:-1]
    left_image_height, left_image_width = image_np.shape[:-1]
    crop_height, crop_width = crop_size
    left_upper = (0, 0)
    right_lower = (crop_width, crop_height)
    image_list = []

    while left_image_width / crop_width >= 1 or left_image_height / crop_height >= 1:
        if left_image_width / crop_width >= 1 and left_image_height / crop_height >= 1:
            new_image = image.crop((*left_upper, *right_lower))
            left_image_width -= crop_width
            left_upper = (left_upper[0] + crop_width, left_upper[-1])
            right_lower = (right_lower[0] + crop_width, right_lower[-1])
            image_list.append(new_image)
        elif left_image_height / crop_height >= 1:
            left_image_width = image_width
            left_image_height -= crop_height
            left_upper = (0, image_height - left_image_height)
            right_lower = (crop_width, image_height - left_image_height + crop_height)
        else:
            break
    return image_list

"""
    将目录下的所有图片进行裁剪
    @:param root_path: 图片的目录
    @:param to: 原图片应该转移到哪里
    @:param crop_size: 裁剪大小, tuple(crop_height, crop_width)
"""
def crop_images2small_images(root_path, to, crop_size):
    image_paths = Path(root_path).glob(pattern="*")
    for image_path in image_paths:
        image = Image.open(image_path)
        image_cropped_list = crop_image2images(image, crop_size)
        for idx, image_cropped in enumerate(image_cropped_list):
            image_cropped.save(
                f"_{idx}".join(os.path.splitext(image_path))
            )
            image_cropped.close()
        image.close()
        if to is None:
            Path(image_path).unlink(missing_ok=True)
        else:
            Path(image_path).rename(
                str(
                    Path(to) / os.path.basename(image_path)
                )
            )

"""
    判断是否能够多gpu分布式并行运算
"""
def distributed_enabled():
    return torch.cuda.is_available() and torch.cuda.device_count() > 1 and torch.__version__ >= "0.4.0"

if __name__ == "__main__":
    # crop_images2small_images(
    #     root_path="dataset/train/images",
    #     to=None,
    #     crop_size=(512, 512)
    # )


    # augment_raw_images2(root_path="dataset/train/images")

    crop_images2small_images(
        root_path="dataset/test",
        to=None,
        crop_size=(512, 512)
    )

    # augment_raw_images2(root_path="dataset/val/images")

    # resize_val_images(
    #     from_path="dataset/test",
    #     to_path=None,
    #     resized_shape=(1024, 1024)
    # )

    # convert_labelme_jsons2masks(
    #     classes=[
    #         "background",
    #         "leaf"
    #     ],
    #     root_path="dataset/train/images",
    #     mask_saved_path="dataset/train/labels"
    # )