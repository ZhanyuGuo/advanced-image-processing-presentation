_base_ = "mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py"

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1))
)
device = 'cuda'

# dataset settings
classes = ("balloon",)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        ann_file="balloon_dataset/balloon/train/coco_train.json",
        img_prefix="balloon_dataset/balloon/train/",
    ),
    val=dict(
        classes=classes,
        ann_file="balloon_dataset/balloon/val/coco_val.json",
        img_prefix="balloon_dataset/balloon/val/",
    ),
    test=dict(
        classes=classes,
        ann_file="balloon_dataset/balloon/val/coco_val.json",
        img_prefix="balloon_dataset/balloon/val/",
    ),
)

# optimizer
optimizer = dict(lr=0.02 / 8)
lr_config = dict(warmup=None)
log_config = dict(interval=10)

checkpoint_config = dict(interval=12)

load_from = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
