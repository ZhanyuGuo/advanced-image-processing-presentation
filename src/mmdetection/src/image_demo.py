import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

file_path = os.path.dirname(__file__)

# choose to use a config and initialize the detector
config_name = "mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco"

config_path = os.path.abspath(
    os.path.join(file_path, "../configs/mask_rcnn/{}.py".format(config_name))
)

# setup a checkpoint file to load
chkpt_name = "mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295"

chkpt_path = os.path.abspath(
    os.path.join(file_path, "../checkpoints/{}.pth".format(chkpt_name))
)

device = "cuda"
# device='cpu'

# initialize the detector
model = init_detector(config_path, chkpt_path, device=device)

# use the detector to do inference
img_name = "demo"
# img_name = "dog"
# img_name = "dog2"
# img_name = "zurich_000070_000019_leftImg8bit"

img_path = os.path.abspath(
    os.path.join(file_path, "../assets/{}.png".format(img_name))
)
out_path = os.path.abspath(os.path.join(file_path, "../assets/{}_out.png".format(img_name)))

result = inference_detector(model, img_path)

# show image
show_result_pyplot(model, img_path, result)
show_result_pyplot(model, img_path, result, out_file=out_path)
