import os
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette, mean_iou, get_classes

file_path = os.path.dirname(__file__)

# choose to use a config and initialize the segmentor
config = os.path.abspath(
    os.path.join(file_path, "../configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py")
)

# setup a checkpoint file to load
checkpoint = os.path.abspath(
    os.path.join(
        file_path,
        "../checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth",
    )
)

device = "cuda"
# device='cpu'

# initialize the segmentor
model = init_segmentor(config, checkpoint, device=device)

# use the segmentor to do inference
# img = os.path.abspath(os.path.join(file_path, "../demo/demo.png"))
# img = os.path.abspath(os.path.join(file_path, "../demo/lindau_000058_000019_leftImg8bit.png"))
img = os.path.abspath(os.path.join(file_path, "../demo/lindau_000058_000019_leftImg8bit_resize.png"))
# img = os.path.abspath(os.path.join(file_path, "../demo/img.png"))

result = inference_segmentor(model, img)

palette = "cityscapes"
opacity = 0.5
out_file = os.path.abspath(os.path.join(file_path, "../demo/result.png"))

# show image
show_result_pyplot(
    model,
    img,
    result,
    get_palette(palette),
    opacity=opacity,
    out_file=out_file,
)

result = result[0]
# dct = {}
# for row in result:
#     for col in row:
#         if col in dct.keys():
#             dct[col] += 1
#         else:
#             dct[col] = 0

# print(dct)

# label_path = os.path.abspath(os.path.join(file_path, "../demo/lindau_000058_000019_label.png"))
label_path = os.path.abspath(os.path.join(file_path, "../demo/lindau_000058_000019_label_resize.png"))
num_classes = 19
ignore_index = 255
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
ret_metrics = mean_iou(result, label, num_classes, ignore_index, nan_to_num=0)
print(ret_metrics['IoU'])

IoU_list = []
for i in ret_metrics['IoU']:
    if i == 0:
        continue

    IoU_list.append(i)

mIoU = sum(IoU_list) / len(IoU_list)
print(mIoU)
