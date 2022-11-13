import os
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette, mean_iou

file_path = os.path.dirname(__file__)

# choose to use a config and initialize the segmentor
# config_name = "fcn_r50-d8_512x1024_80k_cityscapes"
config_name = "fcn_d6_r101-d16_512x1024_80k_cityscapes"

config_path = os.path.abspath(
    os.path.join(file_path, "../configs/fcn/{}.py".format(config_name))
)

# setup a checkpoint file to load
# chkpt_name = "fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d"
chkpt_name = "fcn_d6_r101-d16_512x1024_80k_cityscapes_20210308_102747-cb336445"

chkpt_path = os.path.abspath(
    os.path.join(file_path, "../checkpoints/{}.pth".format(chkpt_name))
)


device = "cuda"
# device='cpu'

# initialize the segmentor
model = init_segmentor(config_path, chkpt_path, device=device)

# use the segmentor to do inference
# img_name = "demo"
# img_name = "dog"
# img_name = "grab"
# img_name = "img"
# img_name = "lindau_000058_000019_leftImg8bit"
# img_name = "weimar_000046_000019_leftImg8bit"
img_name = "zurich_000070_000019_leftImg8bit"

img_path = os.path.abspath(
    os.path.join(file_path, "../assets/{}.png".format(img_name))
)

result = inference_segmentor(model, img_path)

palette = "cityscapes"
opacity = 0.5
out_file = os.path.abspath(os.path.join(file_path, "../assets/{}_out.png".format(img_name)))

# show image
show_result_pyplot(
    model,
    img_path,
    result,
    get_palette(palette),
    opacity=opacity,
    out_file=out_file,
)

# calculate iou
result = result[0]
label_path = os.path.abspath(os.path.join(file_path, "../assets/{}_label.png".format(img_name[:-12])))
num_classes = 19
ignore_index = 255
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
ret_metrics = mean_iou(result, label, num_classes, ignore_index, nan_to_num=0)
print(ret_metrics['IoU'])

IoUs = []
for i in ret_metrics['IoU']:
    if i == 0:
        continue

    IoUs.append(i)

mIoU = sum(IoUs) / len(IoUs)
print("mIoU = ", mIoU)
