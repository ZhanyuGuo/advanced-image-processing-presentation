from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
import os.path as osp
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

cfg = Config.fromfile("configs/mmdet_homework_cfg2.py")
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 6
cfg.work_dir = "checkpoints"
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
print(f"Config:\n{cfg.pretty_text}")

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
