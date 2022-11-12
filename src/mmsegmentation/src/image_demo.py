import os
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
img = os.path.abspath(os.path.join(file_path, "../demo/lindau_000058_000019_leftImg8bit.png"))
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
# result = result[0]
# num_classes = 19
# ignore_index = 255
# pred_size = result.shape
# label = np.random.randint(0, num_classes, size=pred_size)

# ret_metrics = mean_iou(result, label, num_classes, ignore_index)
# print(ret_metrics)
