import yaml
from pathlib import Path
import utils
import torch

if __name__ == "__main__":
    model_config = Path("config") / "model.yaml"
    with model_config.open("r", encoding="utf-8") as f:
        model_config = yaml.load(f, yaml.FullLoader)

        # 类别
        classes = model_config["classes"]

        # 类别对应的语义颜色，按照顺序对应
        colors = utils.get_colors(len(classes))


    train_config = Path("config") / "train.yaml"
    with train_config.open("r", encoding="utf-8") as f:
        train_config = yaml.load(f, yaml.FullLoader)

        # 类别对应的权重
        weight = torch.tensor(train_config["weight"]) if len(train_config["weight"]) != 1 else torch.ones(len(classes))