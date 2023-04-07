import yaml
from pathlib import Path

model_config = Path("config") / "model.yaml"
with model_config.open("r", encoding="utf-8") as f:
    model_config = yaml.load(f, yaml.FullLoader)
    # 类别
    classes = model_config["classes"]